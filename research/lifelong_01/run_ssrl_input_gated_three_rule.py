"""
research/lifelong_01/run_ssrl_input_gated_three_rule.py -- Procedural Runner for SSRL-14.

Evaluates SSRLInputGatedMLP (6,658 params: zero-init mask-to-feature input gate + 6,402 backbone)
with L2-Init (lambda=0.01) + M=64/rule DER++ (alpha=0.5) on Rule 1 -> Rule 2 -> Rule 3 (2,000 steps/rule)
across Seeds 102, 103, 104 alongside a matched 2,000-step/rule bare SSRLInstructionMLP DER++ control.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from research.lifelong_01.ssrl_checkpoints import capture_and_verify_checkpoint, compute_tensor_fingerprint_from_state_dict
from research.lifelong_01.ssrl_model import (
    SSRLInstructionMLP, SSRLInputGatedMLP, get_expected_parameter_count, get_input_gated_parameter_count,
)
from research.lifelong_01.ssrl_generator import generate_pilot_dataset_dict
from research.lifelong_01.ssrl_counterfactual_probe import validate_counterfactual_probe_contract
from research.lifelong_01.run_ssrl_joint_reference import (
    load_and_verify_dataset, reserve_run_directory, atomic_write_json, make_json_serializable,
    DeadlineExceededError, SpecificationApprovalError, CheckpointVerificationError,
)
from research.lifelong_01.run_ssrl_two_rule import build_minibatch_schedule
from research.lifelong_01.run_ssrl_paired_replication import compute_schedule_hash
from research.lifelong_01.run_ssrl_l2_init_plasticity import (
    LearnabilityGateError, _assert_learnability_gate, clone_frozen_init_params,
    compute_l2_init_penalty, compute_mechanism_snapshot, compact_population_eval, _summarize_scalar_list,
)
from research.lifelong_01.run_ssrl_l2init_der import (
    EVAL_MODE_DETERMINISTIC, _teacher_dict_fingerprint, capture_frozen_teacher_logits,
    compute_der_logit_loss, evaluate_buffer_logit_drift,
)
from research.lifelong_01.run_ssrl_three_rule_stream import (
    SHA256_RULE_1_DEFAULT, SHA256_RULE_2_DEFAULT, SHA256_RULE_3_DEFAULT, SHA256_PROBE_DEFAULT,
    select_multi_rule_stratified_memory, build_multi_rule_cyclic_replay_schedule, compact_three_rule_battery_eval,
)


PROPOSED_INPUT_GATED_THREE_RULE_CONFIG: Dict[str, Any] = {
    "study_id": "SSRL-14",
    "phase": "input_gated_three_rule_stream",
    "classification": "bilinear_input_gated_l2init_der_m64_three_seed",
    "gate_approval_status": "APPROVED",
    "primary_model_class": "SSRLInputGatedMLP",
    "expected_gated_parameter_count": 6658,
    "bare_control_model_class": "SSRLInstructionMLP",
    "expected_bare_parameter_count": 6402,
    "gate_gain": 4.0,
    "seeds": [102, 103, 104],
    "minibatch_sampler_seed_r1": 2026,
    "minibatch_sampler_seed_r2": 2027,
    "minibatch_sampler_seed_r3": 2028,
    "rule_1_id": 1, "rule_1_indices": [0, 1, 2, 3],
    "rule_2_id": 2, "rule_2_indices": [4, 5, 6, 7],
    "rule_3_id": 3, "rule_3_indices": [8, 9, 10, 11],
    "train_pool_size": 384, "batch_size": 16,
    "step_budget_r1": 2000, "step_budget_r2": 2000, "step_budget_r3": 2000,
    "replay_buffer_size_per_rule": 64, "replay_batch_size": 16, "replay_ratio_new_to_old": [4, 1],
    "milestone_r3_steps": [0, 25, 50, 100, 250, 500, 750, 1000, 1500, 2000],
    "learning_rate": 0.001, "betas": [0.9, 0.999], "eps": 1e-08, "weight_decay": 0.0,
    "l2_init_lambda": 0.01, "der_alpha": 0.5, "gate_g_lc_threshold": 0.90,
    "enforce_post_r1_learnability_gate": True,
    "enforce_bare_post_r1_learnability_gate": True,
    "enforce_fresh_r3_learnability_gate": True,
    "wall_clock_ceiling_training_seconds_per_stage": 60.0,
    "wall_clock_ceiling_eval_seconds_per_stage": 60.0,
}


def verify_input_gated_three_rule_approval(config: Dict[str, Any]) -> None:
    """AGENTS.md Section 2 / Section 8: Strictly blocks unapproved execution."""
    status = config.get("gate_approval_status")
    if status != "APPROVED":
        raise SpecificationApprovalError(
            f"SSRL-14 spec status is '{status}', not 'APPROVED'. Execution is forbidden before explicit authorization."
        )


def _instantiate_model(model_kind: str, gate_gain: float) -> nn.Module:
    if model_kind == "gated":
        assert get_input_gated_parameter_count() == 6658
        return SSRLInputGatedMLP(feature_dim=16, mask_dim=16, hidden_dim=64, output_dim=2, gate_gain=gate_gain)
    if model_kind == "bare":
        assert get_expected_parameter_count() == 6402
        return SSRLInstructionMLP(input_dim=32, hidden_dim=64, output_dim=2)
    raise ValueError(f"Unknown model_kind '{model_kind}'")


def compute_input_gate_mechanism_snapshot(
    model: SSRLInputGatedMLP, init_params: Dict[str, torch.Tensor], d3: Dict[str, Any], cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Records input-gate selectivity across Rules 1/2/3, norms, and Rule 3 eval_seen diagnostics."""
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count == get_input_gated_parameter_count() == 6658
    fp_before = compute_tensor_fingerprint_from_state_dict(model.state_dict())
    model.eval()
    assert not model.training, "Mechanism snapshot requires eval mode"

    rule_coords = {1: list(cfg["rule_1_indices"]), 2: list(cfg["rule_2_indices"]), 3: list(cfg["rule_3_indices"])}
    with torch.no_grad():
        diffs = [
            (p.detach().to(torch.float32) - init_params[n].to(p.device, dtype=torch.float32)).reshape(-1)
            for n, p in model.named_parameters()
        ]
        dist_init_l2 = float(torch.linalg.vector_norm(torch.cat(diffs, dim=0), ord=2).item())

        gates: Dict[int, torch.Tensor] = {}
        rule_gate_stats: Dict[str, Dict[str, float]] = {}
        for rid, active in rule_coords.items():
            mask_vec = torch.zeros((1, 16), dtype=torch.float32)
            mask_vec[0, active] = 1.0
            g = model.compute_input_gate(mask_vec).squeeze(0)
            gates[rid] = g
            other_rules = [idx for orid, coords in rule_coords.items() if orid != rid for idx in coords]
            all_inactive = [idx for idx in range(16) if idx not in set(active)]
            rule_gate_stats[f"rule_{rid}"] = {
                "active_gain_mean": float(g[active].mean().item()),
                "prior_or_other_rules_gain_mean": float(g[other_rules].mean().item()),
                "all_distractor_gain_mean": float(g[all_inactive].mean().item()),
            }

        cos_12 = float(F.cosine_similarity(gates[1].unsqueeze(0), gates[2].unsqueeze(0), dim=1).item())
        cos_13 = float(F.cosine_similarity(gates[1].unsqueeze(0), gates[3].unsqueeze(0), dim=1).item())
        cos_23 = float(F.cosine_similarity(gates[2].unsqueeze(0), gates[3].unsqueeze(0), dim=1).item())

        seen_r3 = d3["eval_seen_examples"]
        assert len(seen_r3) == 192, f"Expected 192 Rule 3 seen examples, got {len(seen_r3)}"
        x3 = torch.tensor([ex["features"] + ex["mask"] for ex in seen_r3], dtype=torch.float32)
        y3 = torch.tensor([int(ex["target_class"]) for ex in seen_r3], dtype=torch.long)
        feat3, mask3 = x3[:, : model.feature_dim], x3[:, model.feature_dim :]
        u_gated = torch.cat([feat3 * model.compute_input_gate(mask3), mask3], dim=-1)
        h1 = model.relu1(model.fc1(u_gated))
        h2 = model.relu2(model.fc2(h1))
        logits = model.fc3(h2)
        assert torch.equal(logits, model(x3)), "Input-gated forward diagnostic mismatch"

        ce_loss_r3 = float(nn.CrossEntropyLoss()(logits, y3).item())
        margin_r3 = float(torch.abs(logits[:, 1] - logits[:, 0]).mean().item())
        h1_dead = int((~(h1 > 0.0).any(dim=0)).sum().item())
        h2_dead = int((~(h2 > 0.0).any(dim=0)).sum().item())

    assert compute_tensor_fingerprint_from_state_dict(model.state_dict()) == fp_before
    return {
        "mode": EVAL_MODE_DETERMINISTIC,
        "parameter_count": param_count,
        "rule_3_seen_denominator": len(seen_r3),
        "distance_from_init_l2": dist_init_l2,
        "fc_in_gate_weight_l2": float(torch.linalg.vector_norm(model.fc_in_gate.weight.detach().to(torch.float32), ord=2).item()),
        "fc1_weight_l2": float(torch.linalg.vector_norm(model.fc1.weight.detach().to(torch.float32), ord=2).item()),
        "fc2_weight_l2": float(torch.linalg.vector_norm(model.fc2.weight.detach().to(torch.float32), ord=2).item()),
        "fc3_weight_l2": float(torch.linalg.vector_norm(model.fc3.weight.detach().to(torch.float32), ord=2).item()),
        "rule_gate_gains": rule_gate_stats,
        "pairwise_gate_cosines": {"cos_r1_r2": cos_12, "cos_r1_r3": cos_13, "cos_r2_r3": cos_23},
        "ce_loss_r3": ce_loss_r3,
        "mean_abs_logit_margin_r3": margin_r3,
        "h1_dead_relu_count": h1_dead,
        "h2_dead_relu_count": h2_dead,
    }


def train_generic_multi_rule_l2init_der(
    model_kind: str, initial_model_state: Dict[str, torch.Tensor], init_params: Dict[str, torch.Tensor],
    tagged_schedule: List[Tuple[str, int, List[Dict[str, Any]]]], d3: Dict[str, Any], sha3: str,
    config: Dict[str, Any], seed: int, arm_tag: str, train_deadline: float, eval_ceiling: float,
    expected_start_fp: str, expected_init_fp: str, evaluate_r3_milestones: bool = False,
    teacher_by_rule: Optional[Dict[int, Dict[int, torch.Tensor]]] = None,
    expected_teacher_fp_by_rule: Optional[Dict[int, str]] = None,
) -> Tuple[nn.Module, float, Dict[str, Any], str, Dict[str, Any]]:
    """Trains SSRLInputGatedMLP or SSRLInstructionMLP with Adam, L2-Init, and per-rule DER++."""
    if time.monotonic() > train_deadline:
        raise DeadlineExceededError(f"Training deadline exceeded before {arm_tag}")
    if compute_tensor_fingerprint_from_state_dict(init_params) != expected_init_fp:
        raise CheckpointVerificationError(f"init_params fingerprint mismatch before {arm_tag}")
    if teacher_by_rule is not None:
        if expected_teacher_fp_by_rule is None:
            raise ValueError(f"expected_teacher_fp_by_rule required for {arm_tag}")
        for rid, td in teacher_by_rule.items():
            if rid not in expected_teacher_fp_by_rule or _teacher_dict_fingerprint(td) != expected_teacher_fp_by_rule[rid]:
                raise CheckpointVerificationError(f"Teacher logit fingerprint mismatch for rule {rid} before {arm_tag}")

    model = _instantiate_model(model_kind, float(config["gate_gain"]))
    exp_params = int(config["expected_gated_parameter_count"] if model_kind == "gated" else config["expected_bare_parameter_count"])
    assert sum(p.numel() for p in model.parameters()) == exp_params
    model.load_state_dict(initial_model_state)
    loaded_fp = compute_tensor_fingerprint_from_state_dict(model.state_dict())
    if loaded_fp != expected_start_fp:
        raise CheckpointVerificationError(f"Initial fingerprint mismatch for {arm_tag}: {loaded_fp} != {expected_start_fp}")

    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], betas=tuple(config["betas"]), eps=config["eps"], weight_decay=config["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()
    l2_lambda, der_alpha = float(config["l2_init_lambda"]), float(config["der_alpha"])
    counters: Dict[str, Any] = {
        "total_optimizer_steps": 0, "total_presentations": 0, "rule_steps_by_tag": {}, "replay_updates_by_rule": {},
        "der_loss_steps": 0, "der_loss_sum": 0.0, "final_ce_loss": None, "final_l2_init_penalty": None, "final_der_mse_loss": None,
    }
    milestones_set = set(config["milestone_r3_steps"]) if evaluate_r3_milestones else set()
    milestone_records: Dict[str, Any] = {}
    t_start = time.monotonic()

    if 0 in milestones_set:
        t_eval_dl = time.monotonic() + eval_ceiling
        model.eval()
        milestone_records["0"] = {"rule_3_updates": 0, "total_steps": 0, **compact_population_eval(model, d3["eval_seen_examples"], f"{arm_tag}_r3step0", seed, 0, sha3, "rule_3_seen")}
        if time.monotonic() > t_eval_dl:
            raise DeadlineExceededError(f"Milestone 0 eval deadline exceeded for {arm_tag}")

    model.train()
    for tag, rule_id, batch in tagged_schedule:
        if time.monotonic() > train_deadline:
            raise DeadlineExceededError(f"Training deadline exceeded at step {counters['total_optimizer_steps'] + 1} for {arm_tag}")
        inputs = torch.stack([torch.tensor(ex["features"] + ex["mask"], dtype=torch.float32) for ex in batch], dim=0)
        targets = torch.tensor([int(ex["target_class"]) for ex in batch], dtype=torch.long)

        optimizer.zero_grad()
        logits = model(inputs)
        ce_loss = loss_fn(logits, targets)
        l2_penalty = compute_l2_init_penalty(model, init_params, l2_lambda)
        loss = ce_loss + l2_penalty
        if tag == "replay":
            if teacher_by_rule is None or rule_id not in teacher_by_rule:
                raise ValueError(f"Missing teacher logits for replay rule {rule_id} in {arm_tag}")
            assert all(int(ex["rule_id"]) == rule_id for ex in batch)
            der_mse = compute_der_logit_loss(logits, batch, teacher_by_rule[rule_id])
            loss = loss + (der_alpha * der_mse)
            counters["der_loss_steps"] += 1
            counters["der_loss_sum"] += float(der_mse.item())
            counters["final_der_mse_loss"] = float(der_mse.item())
            rk = str(rule_id)
            counters["replay_updates_by_rule"][rk] = counters["replay_updates_by_rule"].get(rk, 0) + 1

        loss.backward()
        optimizer.step()
        counters["total_optimizer_steps"] += 1
        counters["total_presentations"] += len(batch)
        counters["rule_steps_by_tag"][tag] = counters["rule_steps_by_tag"].get(tag, 0) + 1
        counters["final_ce_loss"], counters["final_l2_init_penalty"] = float(ce_loss.item()), float(l2_penalty.item())

        if tag == "rule_3" and counters["rule_steps_by_tag"].get("rule_3", 0) in milestones_set:
            r3_step = counters["rule_steps_by_tag"]["rule_3"]
            t_eval_dl = time.monotonic() + eval_ceiling
            model.eval()
            milestone_records[str(r3_step)] = {
                "rule_3_updates": r3_step, "total_steps": counters["total_optimizer_steps"],
                **compact_population_eval(model, d3["eval_seen_examples"], f"{arm_tag}_r3step{r3_step}", seed, r3_step, sha3, "rule_3_seen"),
            }
            if time.monotonic() > t_eval_dl:
                raise DeadlineExceededError(f"Milestone {r3_step} eval deadline exceeded for {arm_tag}")
            model.train()

    assert counters["total_optimizer_steps"] > 0 and counters["total_presentations"] > 0
    assert compute_tensor_fingerprint_from_state_dict(init_params) == expected_init_fp
    if teacher_by_rule is not None and expected_teacher_fp_by_rule is not None:
        for rid, td in teacher_by_rule.items():
            assert _teacher_dict_fingerprint(td) == expected_teacher_fp_by_rule[rid]
    counters["mean_der_mse_loss"] = counters["der_loss_sum"] / counters["der_loss_steps"] if counters["der_loss_steps"] > 0 else None
    return model, time.monotonic() - t_start, counters, loaded_fp, milestone_records


def _eval_mech_ckpt(
    model_kind: str, model: nn.Module, init_params: Dict[str, torch.Tensor], expected_start_fp: str,
    d1: Dict[str, Any], d2: Dict[str, Any], d3: Dict[str, Any], dp: Dict[str, Any],
    seed: int, step: int, arm_tag: str, ckpt_name: str, abs_run_dir: str,
    sha1: str, sha2: str, sha3: str, cfg: Dict[str, Any], t_eval_ceil: float,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    ev = compact_three_rule_battery_eval(model, d1, d2, d3, dp, seed, step, arm_tag, sha1, sha2, sha3, time.monotonic() + t_eval_ceil)
    mech = compute_input_gate_mechanism_snapshot(model, init_params, d3, cfg) if model_kind == "gated" else compute_mechanism_snapshot(model, init_params, d3)
    ckpt = capture_and_verify_checkpoint(model.state_dict(), abs_run_dir, ckpt_name)
    assert ckpt["tensor_fingerprint"] != expected_start_fp, f"Arm {ckpt_name} had zero parameter delta"
    return ev, mech, ckpt


def _build_gated_vs_bare_comparisons(
    seed: int, cfg: Dict[str, Any], miles_fresh: Dict[str, Any], miles_seq: Dict[str, Any],
    miles_der: Dict[str, Any], miles_bare_der: Dict[str, Any],
    ev_freeze_r1: Dict[str, Any], ev_seq_r3: Dict[str, Any],
    ev_der_r2: Dict[str, Any], ev_der_r3: Dict[str, Any],
    ev_bare_post_r1: Dict[str, Any], ev_bare_der_r2: Dict[str, Any], ev_bare_der_r3: Dict[str, Any],
) -> Dict[str, Any]:
    """Computes aligned R3 transfer deltas, steps-to-90%, forgetting, and gated-vs-bare gains."""
    milestone_deltas, steps_to_90 = {}, {"fresh_r3": None, "seq_r3": None, "stream_der_r3": None, "bare_stream_der_r3": None}
    thresh = cfg["gate_g_lc_threshold"]
    for step_k in cfg["milestone_r3_steps"]:
        k_str = str(step_k)
        fm = miles_fresh[k_str]["metric"]
        sm = miles_seq[k_str]["metric"]
        dm = miles_der[k_str]["metric"]
        bm = miles_bare_der[k_str]["metric"]
        assert fm.denominator == sm.denominator == dm.denominator == bm.denominator == 192
        milestone_deltas[k_str] = {
            "rule_3_updates": step_k, "fresh_r3_numerator": fm.numerator, "fresh_r3_pct": fm.pct,
            "seq_r3_numerator": sm.numerator, "seq_r3_pct": sm.pct,
            "stream_der_r3_numerator": dm.numerator, "stream_der_r3_pct": dm.pct,
            "bare_stream_der_r3_numerator": bm.numerator, "bare_stream_der_r3_pct": bm.pct,
            "delta_seq_minus_fresh_pp": ((sm.numerator - fm.numerator) / 192.0) * 100.0,
            "delta_stream_der_minus_fresh_pp": ((dm.numerator - fm.numerator) / 192.0) * 100.0,
            "delta_bare_der_minus_fresh_pp": ((bm.numerator - fm.numerator) / 192.0) * 100.0,
            "delta_gated_minus_bare_r3_pp": ((dm.numerator - bm.numerator) / 192.0) * 100.0,
        }
        for key, m in (("fresh_r3", fm), ("seq_r3", sm), ("stream_der_r3", dm), ("bare_stream_der_r3", bm)):
            if steps_to_90[key] is None and m.rate >= thresh:
                steps_to_90[key] = step_k

    def n192(ev: Dict[str, Any], key: str) -> int:
        m = ev["metrics"][key]
        assert m.denominator == 192
        return m.numerator

    r1_base, r1_bare_base = n192(ev_freeze_r1, "acc_rule_1_seen"), n192(ev_bare_post_r1, "acc_rule_1_seen")
    r1_seq, r2_seq, r3_seq = n192(ev_seq_r3, "acc_rule_1_seen"), n192(ev_seq_r3, "acc_rule_2_seen"), n192(ev_seq_r3, "acc_rule_3_seen")
    r1_der_p2, r2_der_p2 = n192(ev_der_r2, "acc_rule_1_seen"), n192(ev_der_r2, "acc_rule_2_seen")
    r1_der, r2_der, r3_der = n192(ev_der_r3, "acc_rule_1_seen"), n192(ev_der_r3, "acc_rule_2_seen"), n192(ev_der_r3, "acc_rule_3_seen")
    r2_bare_p2 = n192(ev_bare_der_r2, "acc_rule_2_seen")
    r1_bare, r2_bare, r3_bare = n192(ev_bare_der_r3, "acc_rule_1_seen"), n192(ev_bare_der_r3, "acc_rule_2_seen"), n192(ev_bare_der_r3, "acc_rule_3_seen")
    final_k = str(cfg["step_budget_r3"])
    return {
        "seed": seed, "milestone_deltas": milestone_deltas, "steps_to_90pct_r3": steps_to_90,
        "final_transfer_delta_stream_der_pp": milestone_deltas[final_k]["delta_stream_der_minus_fresh_pp"],
        "final_transfer_delta_bare_der_pp": milestone_deltas[final_k]["delta_bare_der_minus_fresh_pp"],
        "r1_forgetting_total_seq_pp": ((r1_base - r1_seq) / 192.0) * 100.0,
        "r1_forgetting_total_stream_der_pp": ((r1_base - r1_der) / 192.0) * 100.0,
        "r1_forgetting_total_bare_der_pp": ((r1_bare_base - r1_bare) / 192.0) * 100.0,
        "r1_forgetting_phase3_stream_der_pp": ((r1_der_p2 - r1_der) / 192.0) * 100.0,
        "r2_forgetting_phase3_stream_der_pp": ((r2_der_p2 - r2_der) / 192.0) * 100.0,
        "r2_forgetting_phase3_bare_der_pp": ((r2_bare_p2 - r2_bare) / 192.0) * 100.0,
        "gated_minus_bare_r1_pp": ((r1_der - r1_bare) / 192.0) * 100.0,
        "gated_minus_bare_r2_pp": ((r2_der - r2_bare) / 192.0) * 100.0,
        "gated_minus_bare_r3_pp": ((r3_der - r3_bare) / 192.0) * 100.0,
        "stream_minus_seq_r1_pp": ((r1_der - r1_seq) / 192.0) * 100.0,
        "stream_minus_seq_r2_pp": ((r2_der - r2_seq) / 192.0) * 100.0,
        "stream_minus_seq_r3_pp": ((r3_der - r3_seq) / 192.0) * 100.0,
    }


def run_input_gated_three_rule_study(
    dataset_1_path: str, sha256_1: str, dataset_2_path: str, sha256_2: str,
    dataset_3_path: str, sha256_3: str, probe_dataset_path: str, sha256_probe: str,
    destination: str, run_dir: str, config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Master procedural entry point for SSRL-14 Input-Gated Three-Rule Sequential Stream Study."""
    cfg = copy.deepcopy(config or PROPOSED_INPUT_GATED_THREE_RULE_CONFIG)
    verify_input_gated_three_rule_approval(cfg)
    assert get_input_gated_parameter_count() == cfg["expected_gated_parameter_count"] == 6658
    assert get_expected_parameter_count() == cfg["expected_bare_parameter_count"] == 6402
    assert len(set(cfg["rule_1_indices"]) | set(cfg["rule_2_indices"]) | set(cfg["rule_3_indices"])) == 12
    abs_run_dir, abs_dest = reserve_run_directory(run_dir), os.path.abspath(destination)

    d1 = load_and_verify_dataset(dataset_1_path, expected_sha256=sha256_1, approved_config={**cfg, "study_id": "SSRL-01", "phase": "learnability_pilot", "seed": 101, "partition_seed": 42, "rule_id": cfg["rule_1_id"], "rule_indices": cfg["rule_1_indices"]})
    d2 = load_and_verify_dataset(dataset_2_path, expected_sha256=sha256_2, approved_config={**cfg, "study_id": "SSRL-02", "phase": "two_rule_diagnostic", "seed": 101, "partition_seed": 43, "rule_id": cfg["rule_2_id"], "rule_indices": cfg["rule_2_indices"]})
    d3 = load_and_verify_dataset(dataset_3_path, expected_sha256=sha256_3, approved_config={**cfg, "study_id": "SSRL-13", "phase": "three_rule_sequential_stream", "seed": 101, "partition_seed": 44, "rule_id": cfg["rule_3_id"], "rule_indices": cfg["rule_3_indices"]})
    if d3["metadata"].get("study_id") == "SSRL-13":
        assert d3 == generate_pilot_dataset_dict(tuple(cfg["rule_3_indices"]), cfg["rule_3_id"], 101, partition_seed=44, study_id="SSRL-13", phase="three_rule_sequential_stream")

    with open(probe_dataset_path, "rb") as f:
        actual_probe_sha = hashlib.sha256(f.read()).hexdigest()
    if actual_probe_sha != sha256_probe:
        raise ValueError(f"Probe SHA mismatch: expected {sha256_probe}, got {actual_probe_sha}")
    with open(probe_dataset_path, "r", encoding="utf-8") as f:
        dp = json.load(f)
    validate_counterfactual_probe_contract(dp, d1, d2)

    diagnostics: Dict[str, Any] = {
        "study_id": cfg["study_id"], "phase": cfg["phase"], "classification": cfg["classification"],
        "status": "IN_PROGRESS", "current_stage": "INITIALIZATION", "execution_mode": EVAL_MODE_DETERMINISTIC,
        "dataset_hashes": {"sha256_1": sha256_1, "sha256_2": sha256_2, "sha256_3": sha256_3, "sha256_probe": sha256_probe},
        "environment": {
            "python_version": sys.version.split()[0], "torch_version": torch.__version__, "device": "cpu",
            "gate_gain": cfg["gate_gain"], "l2_init_lambda": cfg["l2_init_lambda"], "der_alpha": cfg["der_alpha"],
            "replay_buffer_size_per_rule": cfg["replay_buffer_size_per_rule"], "replay_batch_size": cfg["replay_batch_size"],
            "step_budgets": [cfg["step_budget_r1"], cfg["step_budget_r2"], cfg["step_budget_r3"]],
        },
        "seeds_status": {s: "NOT_RUN" for s in cfg["seeds"]}, "seeds_data": {},
        "paired_gated_vs_bare_comparisons": [], "run_directory": abs_run_dir, "destination": abs_dest,
    }
    diag_file = os.path.join(abs_run_dir, "failure_diagnostics.json")

    try:
        t_train_ceil, t_eval_ceil = float(cfg["wall_clock_ceiling_training_seconds_per_stage"]), float(cfg["wall_clock_ceiling_eval_seconds_per_stage"])
        s1 = [("rule_1", 1, b) for b in build_minibatch_schedule(d1["train_examples"], cfg["step_budget_r1"], cfg["batch_size"], cfg["minibatch_sampler_seed_r1"])]
        raw_s2 = build_minibatch_schedule(d2["train_examples"], cfg["step_budget_r2"], cfg["batch_size"], cfg["minibatch_sampler_seed_r2"])
        raw_s3 = build_minibatch_schedule(d3["train_examples"], cfg["step_budget_r3"], cfg["batch_size"], cfg["minibatch_sampler_seed_r3"])
        s2_seq, s3_seq = [("rule_2", 2, b) for b in raw_s2], [("rule_3", 3, b) for b in raw_s3]

        buf_r1 = select_multi_rule_stratified_memory(d1["train_examples"], cfg["rule_1_indices"], cfg["replay_buffer_size_per_rule"], cfg["replay_batch_size"])
        buf_r2 = select_multi_rule_stratified_memory(d2["train_examples"], cfg["rule_2_indices"], cfg["replay_buffer_size_per_rule"], cfg["replay_batch_size"])
        rb, ra = cfg["replay_ratio_new_to_old"]
        s2_der = build_multi_rule_cyclic_replay_schedule(raw_s2, "rule_2", 2, {1: buf_r1}, cfg["replay_batch_size"], rb, ra)
        s3_der = build_multi_rule_cyclic_replay_schedule(raw_s3, "rule_3", 3, {1: buf_r1, 2: buf_r2}, cfg["replay_batch_size"], rb, ra)
        diagnostics["buffer_ids_r1"], diagnostics["buffer_ids_r2"] = [int(x["example_id"]) for x in buf_r1], [int(x["example_id"]) for x in buf_r2]
        diagnostics["schedule_hashes"] = {k: compute_schedule_hash(v) for k, v in [("r1", s1), ("r2_seq", s2_seq), ("r2_der", s2_der), ("r3_seq", s3_seq), ("r3_der", s3_der)]}

        probe_x = torch.stack([torch.tensor(ex["features"] + ex["mask"], dtype=torch.float32) for ex in d1["eval_seen_examples"][:4]], dim=0)
        for seed in cfg["seeds"]:
            diagnostics["seeds_status"][seed] = "IN_PROGRESS"
            seed_rec: Dict[str, Any] = {"seed": seed}
            diagnostics["seeds_data"][str(seed)] = seed_rec

            torch.manual_seed(seed)
            m_gated_init = _instantiate_model("gated", float(cfg["gate_gain"]))
            torch.manual_seed(seed)
            m_bare_init = _instantiate_model("bare", float(cfg["gate_gain"]))
            m_gated_init.eval(); m_bare_init.eval()
            with torch.no_grad():
                assert torch.equal(m_gated_init(probe_x), m_bare_init(probe_x)), f"Seed {seed} Step-0 neutrality check failed"

            init_g, fp_init_g = clone_frozen_init_params(m_gated_init)
            init_b, fp_init_b = clone_frozen_init_params(m_bare_init)
            seed_rec["fresh_init_fingerprint_gated"], seed_rec["fresh_init_fingerprint_bare"] = fp_init_g, fp_init_b

            # 1. Gated Phase R1 + Freeze + Fresh R3 + Seq R2->R3 + DER R2->R3
            diagnostics["current_stage"] = f"seed_{seed}_post_r1"
            m_r1, t_r1, cnt_r1, _, _ = train_generic_multi_rule_l2init_der("gated", m_gated_init.state_dict(), init_g, s1, d3, sha256_3, cfg, seed, f"ig3_r1_s{seed}", time.monotonic() + t_train_ceil, t_eval_ceil, fp_init_g, fp_init_g)
            ev_r1 = compact_population_eval(m_r1, d1["eval_seen_examples"], f"ig3_post_r1_s{seed}", seed, cfg["step_budget_r1"], sha256_1, "rule_1_seen")
            if cfg["enforce_post_r1_learnability_gate"]:
                _assert_learnability_gate(ev_r1["metric"], seed, "post_r1", cfg["gate_g_lc_threshold"])
            ckpt_r1 = capture_and_verify_checkpoint(m_r1.state_dict(), abs_run_dir, f"checkpoint_ig3_post_r1_s{seed}.pt")
            fp_r1 = ckpt_r1["tensor_fingerprint"]
            assert fp_r1 != fp_init_g
            t_r1_logits, t_r1_fp, t_r1_floats = capture_frozen_teacher_logits(m_r1, buf_r1, fp_r1)
            seed_rec["post_r1"] = {"elapsed_train_seconds": t_r1, "counters": cnt_r1, "checkpoint": ckpt_r1, "mechanism": compute_input_gate_mechanism_snapshot(m_r1, init_g, d3, cfg), "post_r1_seen": ev_r1, "teacher_logits_fp_r1": t_r1_fp, "teacher_logits_bytes_r1": t_r1_floats * 4}

            diagnostics["current_stage"] = f"seed_{seed}_freeze_after_r1"
            ev_freeze_r1 = compact_three_rule_battery_eval(m_r1, d1, d2, d3, dp, seed, 0, f"ig3_freeze_s{seed}", sha256_1, sha256_2, sha256_3, time.monotonic() + t_eval_ceil)
            seed_rec["freeze_after_r1"] = {"fingerprint": fp_r1, "eval": ev_freeze_r1}

            diagnostics["current_stage"] = f"seed_{seed}_fresh_r3"
            m_fr, t_fr, cnt_fr, _, miles_fr = train_generic_multi_rule_l2init_der("gated", m_gated_init.state_dict(), init_g, s3_seq, d3, sha256_3, cfg, seed, f"ig3_fresh_s{seed}", time.monotonic() + t_train_ceil, t_eval_ceil, fp_init_g, fp_init_g, evaluate_r3_milestones=True)
            if cfg["enforce_fresh_r3_learnability_gate"]:
                _assert_learnability_gate(miles_fr[str(cfg["step_budget_r3"])]["metric"], seed, "fresh_r3", cfg["gate_g_lc_threshold"])
            ev_fr, mech_fr, ckpt_fr = _eval_mech_ckpt("gated", m_fr, init_g, fp_init_g, d1, d2, d3, dp, seed, len(s3_seq), f"ig3_fresh_s{seed}", f"checkpoint_ig3_fresh_r3_s{seed}.pt", abs_run_dir, sha256_1, sha256_2, sha256_3, cfg, t_eval_ceil)
            seed_rec["fresh_r3"] = {"elapsed_train_seconds": t_fr, "counters": cnt_fr, "checkpoint": ckpt_fr, "mechanism": mech_fr, "milestone_evaluations": miles_fr, "final_eval": ev_fr}

            diagnostics["current_stage"] = f"seed_{seed}_seq_post_r2"
            m_sr2, t_sr2, cnt_sr2, _, _ = train_generic_multi_rule_l2init_der("gated", m_r1.state_dict(), init_g, s2_seq, d3, sha256_3, cfg, seed, f"ig3_sr2_s{seed}", time.monotonic() + t_train_ceil, t_eval_ceil, fp_r1, fp_init_g)
            ev_sr2, mech_sr2, ckpt_sr2 = _eval_mech_ckpt("gated", m_sr2, init_g, fp_r1, d1, d2, d3, dp, seed, len(s2_seq), f"ig3_sr2_s{seed}", f"checkpoint_ig3_seq_post_r2_s{seed}.pt", abs_run_dir, sha256_1, sha256_2, sha256_3, cfg, t_eval_ceil)
            fp_sr2 = ckpt_sr2["tensor_fingerprint"]
            t_sr2_logits, _, _ = capture_frozen_teacher_logits(m_sr2, buf_r2, fp_sr2)
            seed_rec["seq_post_r2"] = {"elapsed_train_seconds": t_sr2, "counters": cnt_sr2, "checkpoint": ckpt_sr2, "mechanism": mech_sr2, "final_eval": ev_sr2}

            diagnostics["current_stage"] = f"seed_{seed}_seq_r3"
            m_sr3, t_sr3, cnt_sr3, _, miles_sr3 = train_generic_multi_rule_l2init_der("gated", m_sr2.state_dict(), init_g, s3_seq, d3, sha256_3, cfg, seed, f"ig3_sr3_s{seed}", time.monotonic() + t_train_ceil, t_eval_ceil, fp_sr2, fp_init_g, evaluate_r3_milestones=True)
            ev_sr3, mech_sr3, ckpt_sr3 = _eval_mech_ckpt("gated", m_sr3, init_g, fp_sr2, d1, d2, d3, dp, seed, len(s3_seq), f"ig3_sr3_s{seed}", f"checkpoint_ig3_seq_r3_s{seed}.pt", abs_run_dir, sha256_1, sha256_2, sha256_3, cfg, t_eval_ceil)
            seed_rec["seq_r3"] = {"elapsed_train_seconds": t_sr3, "counters": cnt_sr3, "checkpoint": ckpt_sr3, "mechanism": mech_sr3, "milestone_evaluations": miles_sr3, "final_eval": ev_sr3, "buffer_logit_drift_r1": evaluate_buffer_logit_drift(m_sr3, buf_r1, t_r1_logits), "buffer_logit_drift_r2": evaluate_buffer_logit_drift(m_sr3, buf_r2, t_sr2_logits)}

            diagnostics["current_stage"] = f"seed_{seed}_der_post_r2"
            m_dr2, t_dr2, cnt_dr2, _, _ = train_generic_multi_rule_l2init_der("gated", m_r1.state_dict(), init_g, s2_der, d3, sha256_3, cfg, seed, f"ig3_dr2_s{seed}", time.monotonic() + t_train_ceil, t_eval_ceil, fp_r1, fp_init_g, teacher_by_rule={1: t_r1_logits}, expected_teacher_fp_by_rule={1: t_r1_fp})
            ev_dr2, mech_dr2, ckpt_dr2 = _eval_mech_ckpt("gated", m_dr2, init_g, fp_r1, d1, d2, d3, dp, seed, len(s2_der), f"ig3_dr2_s{seed}", f"checkpoint_ig3_der_post_r2_s{seed}.pt", abs_run_dir, sha256_1, sha256_2, sha256_3, cfg, t_eval_ceil)
            fp_dr2 = ckpt_dr2["tensor_fingerprint"]
            t_dr2_logits, t_dr2_fp, t_dr2_floats = capture_frozen_teacher_logits(m_dr2, buf_r2, fp_dr2)
            seed_rec["der_post_r2"] = {"elapsed_train_seconds": t_dr2, "counters": cnt_dr2, "checkpoint": ckpt_dr2, "mechanism": mech_dr2, "final_eval": ev_dr2, "teacher_logits_fp_r2": t_dr2_fp, "teacher_logits_bytes_r2": t_dr2_floats * 4}

            diagnostics["current_stage"] = f"seed_{seed}_stream_der_r3"
            m_dr3, t_dr3, cnt_dr3, _, miles_dr3 = train_generic_multi_rule_l2init_der("gated", m_dr2.state_dict(), init_g, s3_der, d3, sha256_3, cfg, seed, f"ig3_dr3_s{seed}", time.monotonic() + t_train_ceil, t_eval_ceil, fp_dr2, fp_init_g, evaluate_r3_milestones=True, teacher_by_rule={1: t_r1_logits, 2: t_dr2_logits}, expected_teacher_fp_by_rule={1: t_r1_fp, 2: t_dr2_fp})
            ev_dr3, mech_dr3, ckpt_dr3 = _eval_mech_ckpt("gated", m_dr3, init_g, fp_dr2, d1, d2, d3, dp, seed, len(s3_der), f"ig3_dr3_s{seed}", f"checkpoint_ig3_stream_der_r3_s{seed}.pt", abs_run_dir, sha256_1, sha256_2, sha256_3, cfg, t_eval_ceil)
            seed_rec["stream_der_r3"] = {"elapsed_train_seconds": t_dr3, "counters": cnt_dr3, "checkpoint": ckpt_dr3, "mechanism": mech_dr3, "milestone_evaluations": miles_dr3, "final_eval": ev_dr3, "buffer_logit_drift_r1": evaluate_buffer_logit_drift(m_dr3, buf_r1, t_r1_logits), "buffer_logit_drift_r2": evaluate_buffer_logit_drift(m_dr3, buf_r2, t_dr2_logits)}

            # 2. Matched Bare Control Arm (2,000 steps/rule)
            diagnostics["current_stage"] = f"seed_{seed}_bare_post_r1"
            mb_r1, tb_r1, cnt_br1, _, _ = train_generic_multi_rule_l2init_der("bare", m_bare_init.state_dict(), init_b, s1, d3, sha256_3, cfg, seed, f"ig3_br1_s{seed}", time.monotonic() + t_train_ceil, t_eval_ceil, fp_init_b, fp_init_b)
            ev_br1_pop = compact_population_eval(mb_r1, d1["eval_seen_examples"], f"ig3_bare_r1_s{seed}", seed, cfg["step_budget_r1"], sha256_1, "rule_1_seen")
            if cfg["enforce_bare_post_r1_learnability_gate"]:
                _assert_learnability_gate(ev_br1_pop["metric"], seed, "bare_post_r1", cfg["gate_g_lc_threshold"])
            ev_br1_bat, mech_br1, ckpt_br1 = _eval_mech_ckpt("bare", mb_r1, init_b, fp_init_b, d1, d2, d3, dp, seed, cfg["step_budget_r1"], f"ig3_bare_r1_s{seed}", f"checkpoint_ig3_bare_post_r1_s{seed}.pt", abs_run_dir, sha256_1, sha256_2, sha256_3, cfg, t_eval_ceil)
            fp_br1 = ckpt_br1["tensor_fingerprint"]
            tb_r1_logits, tb_r1_fp, _ = capture_frozen_teacher_logits(mb_r1, buf_r1, fp_br1)
            seed_rec["bare_post_r1"] = {"elapsed_train_seconds": tb_r1, "counters": cnt_br1, "checkpoint": ckpt_br1, "mechanism": mech_br1, "final_eval": ev_br1_bat, "teacher_logits_fp_r1": tb_r1_fp}

            diagnostics["current_stage"] = f"seed_{seed}_bare_der_post_r2"
            mb_dr2, tb_dr2, cnt_bdr2, _, _ = train_generic_multi_rule_l2init_der("bare", mb_r1.state_dict(), init_b, s2_der, d3, sha256_3, cfg, seed, f"ig3_bdr2_s{seed}", time.monotonic() + t_train_ceil, t_eval_ceil, fp_br1, fp_init_b, teacher_by_rule={1: tb_r1_logits}, expected_teacher_fp_by_rule={1: tb_r1_fp})
            ev_bdr2, mech_bdr2, ckpt_bdr2 = _eval_mech_ckpt("bare", mb_dr2, init_b, fp_br1, d1, d2, d3, dp, seed, len(s2_der), f"ig3_bdr2_s{seed}", f"checkpoint_ig3_bare_der_post_r2_s{seed}.pt", abs_run_dir, sha256_1, sha256_2, sha256_3, cfg, t_eval_ceil)
            fp_bdr2 = ckpt_bdr2["tensor_fingerprint"]
            tb_r2_logits, tb_r2_fp, _ = capture_frozen_teacher_logits(mb_dr2, buf_r2, fp_bdr2)
            seed_rec["bare_der_post_r2"] = {"elapsed_train_seconds": tb_dr2, "counters": cnt_bdr2, "checkpoint": ckpt_bdr2, "mechanism": mech_bdr2, "final_eval": ev_bdr2, "teacher_logits_fp_r2": tb_r2_fp}

            diagnostics["current_stage"] = f"seed_{seed}_bare_stream_der_r3"
            mb_dr3, tb_dr3, cnt_bdr3, _, miles_bdr3 = train_generic_multi_rule_l2init_der("bare", mb_dr2.state_dict(), init_b, s3_der, d3, sha256_3, cfg, seed, f"ig3_bdr3_s{seed}", time.monotonic() + t_train_ceil, t_eval_ceil, fp_bdr2, fp_init_b, evaluate_r3_milestones=True, teacher_by_rule={1: tb_r1_logits, 2: tb_r2_logits}, expected_teacher_fp_by_rule={1: tb_r1_fp, 2: tb_r2_fp})
            ev_bdr3, mech_bdr3, ckpt_bdr3 = _eval_mech_ckpt("bare", mb_dr3, init_b, fp_bdr2, d1, d2, d3, dp, seed, len(s3_der), f"ig3_bdr3_s{seed}", f"checkpoint_ig3_bare_stream_der_r3_s{seed}.pt", abs_run_dir, sha256_1, sha256_2, sha256_3, cfg, t_eval_ceil)
            seed_rec["bare_stream_der_r3"] = {"elapsed_train_seconds": tb_dr3, "counters": cnt_bdr3, "checkpoint": ckpt_bdr3, "mechanism": mech_bdr3, "milestone_evaluations": miles_bdr3, "final_eval": ev_bdr3, "buffer_logit_drift_r1": evaluate_buffer_logit_drift(mb_dr3, buf_r1, tb_r1_logits), "buffer_logit_drift_r2": evaluate_buffer_logit_drift(mb_dr3, buf_r2, tb_r2_logits)}

            comp = _build_gated_vs_bare_comparisons(seed, cfg, miles_fr, miles_sr3, miles_dr3, miles_bdr3, ev_freeze_r1, ev_sr3, ev_dr2, ev_dr3, ev_br1_bat, ev_bdr2, ev_bdr3)
            seed_rec["gated_vs_bare_comparison"] = comp
            diagnostics["paired_gated_vs_bare_comparisons"].append(comp)
            diagnostics["seeds_status"][seed] = "COMPLETED"

        comps = diagnostics["paired_gated_vs_bare_comparisons"]
        summary: Dict[str, Any] = {f"transfer_delta_stream_der_r3_step{k}_pp": _summarize_scalar_list([c["milestone_deltas"][str(k)]["delta_stream_der_minus_fresh_pp"] for c in comps]) for k in cfg["milestone_r3_steps"]}
        for key in ("gated_minus_bare_r1_pp", "gated_minus_bare_r2_pp", "gated_minus_bare_r3_pp", "r1_forgetting_total_seq_pp", "r1_forgetting_total_stream_der_pp", "r1_forgetting_total_bare_der_pp", "r2_forgetting_phase3_stream_der_pp", "r2_forgetting_phase3_bare_der_pp", "stream_minus_seq_r1_pp", "stream_minus_seq_r2_pp", "stream_minus_seq_r3_pp"):
            summary[key] = _summarize_scalar_list([c[key] for c in comps])
        for arm_name in ("stream_der_r3", "bare_stream_der_r3"):
            for rk in ("r1", "r2"):
                summary[f"buffer_logit_drift_mse_{rk}_{arm_name}"] = _summarize_scalar_list([diagnostics["seeds_data"][str(s)][arm_name][f"buffer_logit_drift_{rk}"]["mse_logit_drift"] for s in cfg["seeds"]])
        diagnostics["summary_statistics"] = summary
        diagnostics["status"], diagnostics["current_stage"] = "COMPLETED", "COMPLETED"
        payload = {"config": cfg, "diagnostics": diagnostics}
        atomic_write_json(make_json_serializable(payload), abs_dest, refuse_overwrite=True)
        return payload

    except Exception as exc:
        diagnostics["status"], diagnostics["error_type"], diagnostics["error_message"] = "FAILED", type(exc).__name__, str(exc)
        for s in cfg["seeds"]:
            if diagnostics["seeds_status"][s] == "IN_PROGRESS":
                diagnostics["seeds_status"][s] = "FAILED"
            elif diagnostics["seeds_status"][s] != "COMPLETED":
                diagnostics["seeds_status"][s] = "NOT_RUN"
        try:
            atomic_write_json(make_json_serializable({"diagnostics": diagnostics}), diag_file, refuse_overwrite=False)
        except Exception as write_err:
            print(f"[Error writing failure diagnostics: {write_err}]", file=sys.stderr)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="SSRL-14 Input-Gated 3-Rule Sequential Stream Runner")
    for arg, default in [("dataset_1", None), ("dataset_2", None), ("dataset_3", None), ("probe_dataset", None), ("run_dir", None), ("destination", None), ("sha256_1", SHA256_RULE_1_DEFAULT), ("sha256_2", SHA256_RULE_2_DEFAULT), ("sha256_3", SHA256_RULE_3_DEFAULT), ("sha256_probe", SHA256_PROBE_DEFAULT)]:
        parser.add_argument(f"--{arg}", required=(default is None), default=default)
    args = parser.parse_args()
    run_input_gated_three_rule_study(args.dataset_1, args.sha256_1, args.dataset_2, args.sha256_2, args.dataset_3, args.sha256_3, args.probe_dataset, args.sha256_probe, args.destination, args.run_dir)


if __name__ == "__main__":
    main()
