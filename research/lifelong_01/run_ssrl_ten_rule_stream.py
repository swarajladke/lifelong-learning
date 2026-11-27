"""
research/lifelong_01/run_ssrl_ten_rule_stream.py -- Procedural Runner for SSRL-16.

Evaluates the full PLAN.md 10-rule sequential continual learning stream (Stages 1..10: disjoint R1..R4
plus overlapping R5, R8..R12 uniquely identifying all 16 coordinates), 5 isolated 500-step target
transfer probes (P1..P5), and zero-shot composition (Comp-6, Comp-7) on SSRLInputGatedMLP (6,658 params)
vs freeze_after_r1, seq_r10, and matched bare_stream_der_r10 (6,402 params).
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
import torch.optim as optim

from research.lifelong_01.ssrl_checkpoints import capture_and_verify_checkpoint, compute_tensor_fingerprint_from_state_dict
from research.lifelong_01.ssrl_model import (
    SSRLInputGatedMLP, get_expected_parameter_count, get_input_gated_parameter_count,
)
from research.lifelong_01.ssrl_generator import generate_pilot_dataset_dict
from research.lifelong_01.ssrl_counterfactual_probe import validate_counterfactual_probe_contract, evaluate_counterfactual_probe
from research.lifelong_01.run_ssrl_joint_reference import (
    load_and_verify_dataset, reserve_run_directory,
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
    select_multi_rule_stratified_memory, build_multi_rule_cyclic_replay_schedule,
)
from research.lifelong_01.run_ssrl_input_gated_three_rule import _instantiate_model
from research.lifelong_01.run_ssrl_five_rule_stream import (
    SHA256_RULE_4_DEFAULT, SHA256_RULE_5_DEFAULT, SHA256_COMP_6_DEFAULT, SHA256_COMP_7_DEFAULT,
    atomic_write_compact_json,
)


DEFAULT_DATASET_SHA256: Dict[str, str] = {
    "ssrl_pilot_t1_s101.json": SHA256_RULE_1_DEFAULT,
    "ssrl_pilot_t2_s101.json": SHA256_RULE_2_DEFAULT,
    "ssrl_pilot_t3_s101.json": SHA256_RULE_3_DEFAULT,
    "ssrl_pilot_t4_s101.json": SHA256_RULE_4_DEFAULT,
    "ssrl_pilot_t5_s101.json": SHA256_RULE_5_DEFAULT,
    "ssrl_pilot_t6_comp_s101.json": SHA256_COMP_6_DEFAULT,
    "ssrl_pilot_t7_comp_s101.json": SHA256_COMP_7_DEFAULT,
    "ssrl_pilot_s6_r8_s101.json": "1d73af311189672ec0a889cb51cb77dbc28208d18c4852cffe51637a1320bd56",
    "ssrl_pilot_s7_r9_s101.json": "180a0211f24d933f6ee555dc6190dc38af95e2b694e1194d3883b6463db0a407",
    "ssrl_pilot_s8_r10_s101.json": "30b6a4b3ae78136cb43db2775013acca43cea1bc55b5686b3b50624352fd8da9",
    "ssrl_pilot_s9_r11_s101.json": "d80cc06fffcc3a095a671a986cca463c2a37c4e52febf63b4412a85c4213a76a",
    "ssrl_pilot_s10_r12_s101.json": "5dd7618cbf7fb66dfe06636b8b044955816e2fb8396a6af7246ebe8b8e239837",
    "ssrl_pilot_p1_r13_s101.json": "c584c008ef517839825bc131a8fde7b4ca57bff60889612ae4b2bae1f5baa128",
    "ssrl_pilot_p2_r14_s101.json": "6a725176573da9c81c1077e1e69f4c925004a722313c53de8c3795dc11ce7683",
    "ssrl_pilot_p3_r15_s101.json": "e802b6110784dfeb3e9076427f7f333818be98b756984395cabfbc21986e953d",
    "ssrl_pilot_p4_r16_s101.json": "1042dd6bd1b4f585556ec186ccd5cafc512e19bd8f693cdef4af6d70b021f8ff",
    "ssrl_pilot_p5_r17_s101.json": "18837f62b8261e34b07c4513ecbf00463685b076b323448d342f78f582da84ef",
    "ssrl_probe_counterfactual_128.json": SHA256_PROBE_DEFAULT,
}

PROPOSED_TEN_RULE_STREAM_CONFIG: Dict[str, Any] = {
    "study_id": "SSRL-16", "phase": "ten_rule_continual_and_transfer",
    "classification": "ten_rule_plan_capstone_input_gated_vs_bare_three_seed",
    "gate_approval_status": "APPROVED",
    "primary_model_class": "SSRLInputGatedMLP", "expected_gated_parameter_count": 6658,
    "bare_control_model_class": "SSRLInstructionMLP", "expected_bare_parameter_count": 6402,
    "gate_gain": 4.0, "seeds": [102, 103, 104],
    "stream_stages": [
        {"stage": 1, "rule_id": 1, "coords": [0, 1, 2, 3], "sampler_seed": 2026, "pseed": 42, "study": "SSRL-01", "phase": "learnability_pilot", "file": "ssrl_pilot_t1_s101.json"},
        {"stage": 2, "rule_id": 2, "coords": [4, 5, 6, 7], "sampler_seed": 2027, "pseed": 43, "study": "SSRL-02", "phase": "two_rule_diagnostic", "file": "ssrl_pilot_t2_s101.json"},
        {"stage": 3, "rule_id": 3, "coords": [8, 9, 10, 11], "sampler_seed": 2028, "pseed": 44, "study": "SSRL-13", "phase": "three_rule_sequential_stream", "file": "ssrl_pilot_t3_s101.json"},
        {"stage": 4, "rule_id": 4, "coords": [12, 13, 14, 15], "sampler_seed": 2029, "pseed": 45, "study": "SSRL-15", "phase": "five_rule_overlapping_stream", "file": "ssrl_pilot_t4_s101.json"},
        {"stage": 5, "rule_id": 5, "coords": [0, 4, 8, 12], "sampler_seed": 2030, "pseed": 46, "study": "SSRL-15", "phase": "five_rule_overlapping_stream", "file": "ssrl_pilot_t5_s101.json"},
        {"stage": 6, "rule_id": 8, "coords": [2, 6, 10, 14], "sampler_seed": 2031, "pseed": 49, "study": "SSRL-16", "phase": "ten_rule_continual_and_transfer", "file": "ssrl_pilot_s6_r8_s101.json"},
        {"stage": 7, "rule_id": 9, "coords": [3, 7, 11, 15], "sampler_seed": 2032, "pseed": 50, "study": "SSRL-16", "phase": "ten_rule_continual_and_transfer", "file": "ssrl_pilot_s7_r9_s101.json"},
        {"stage": 8, "rule_id": 10, "coords": [0, 2, 5, 7], "sampler_seed": 2033, "pseed": 51, "study": "SSRL-16", "phase": "ten_rule_continual_and_transfer", "file": "ssrl_pilot_s8_r10_s101.json"},
        {"stage": 9, "rule_id": 11, "coords": [8, 10, 13, 15], "sampler_seed": 2034, "pseed": 52, "study": "SSRL-16", "phase": "ten_rule_continual_and_transfer", "file": "ssrl_pilot_s9_r11_s101.json"},
        {"stage": 10, "rule_id": 12, "coords": [1, 6, 9, 12], "sampler_seed": 2035, "pseed": 53, "study": "SSRL-16", "phase": "ten_rule_continual_and_transfer", "file": "ssrl_pilot_s10_r12_s101.json"},
    ],
    "composition_specs": [
        {"key": "comp_6", "rule_id": 6, "coords": [1, 5, 9, 13], "pseed": 47, "study": "SSRL-15", "phase": "five_rule_overlapping_stream", "file": "ssrl_pilot_t6_comp_s101.json"},
        {"key": "comp_7", "rule_id": 7, "coords": [0, 1, 4, 5], "pseed": 48, "study": "SSRL-15", "phase": "five_rule_overlapping_stream", "file": "ssrl_pilot_t7_comp_s101.json"},
    ],
    "transfer_probe_specs": [
        {"probe_idx": 1, "rule_id": 13, "coords": [0, 5, 10, 15], "sampler_seed": 3001, "pseed": 54, "study": "SSRL-16", "phase": "ten_rule_continual_and_transfer", "file": "ssrl_pilot_p1_r13_s101.json"},
        {"probe_idx": 2, "rule_id": 14, "coords": [3, 6, 9, 12], "sampler_seed": 3002, "pseed": 55, "study": "SSRL-16", "phase": "ten_rule_continual_and_transfer", "file": "ssrl_pilot_p2_r14_s101.json"},
        {"probe_idx": 3, "rule_id": 15, "coords": [2, 3, 12, 13], "sampler_seed": 3003, "pseed": 56, "study": "SSRL-16", "phase": "ten_rule_continual_and_transfer", "file": "ssrl_pilot_p3_r15_s101.json"},
        {"probe_idx": 4, "rule_id": 16, "coords": [4, 6, 8, 11], "sampler_seed": 3004, "pseed": 57, "study": "SSRL-16", "phase": "ten_rule_continual_and_transfer", "file": "ssrl_pilot_p4_r16_s101.json"},
        {"probe_idx": 5, "rule_id": 17, "coords": [1, 3, 7, 14], "sampler_seed": 3005, "pseed": 58, "study": "SSRL-16", "phase": "ten_rule_continual_and_transfer", "file": "ssrl_pilot_p5_r17_s101.json"},
    ],
    "train_pool_size": 384, "batch_size": 16, "step_budget_per_stream_rule": 2000,
    "transfer_probe_steps": 500, "transfer_probe_milestones": [0, 25, 50, 100, 250, 500],
    "replay_buffer_size_per_rule": 64, "replay_batch_size": 16, "replay_ratio_new_to_old": [4, 1],
    "learning_rate": 0.001, "betas": [0.9, 0.999], "eps": 1e-08, "weight_decay": 0.0,
    "l2_init_lambda": 0.01, "der_alpha": 0.5, "gate_g_lc_threshold": 0.90,
    "enforce_post_r1_learnability_gate": True, "enforce_bare_post_r1_learnability_gate": True,
    "wall_clock_ceiling_training_seconds_per_stage": 60.0, "wall_clock_ceiling_eval_seconds_per_stage": 60.0,
}


def verify_ten_rule_stream_approval(config: Dict[str, Any]) -> None:
    """AGENTS.md Section 2 / Section 8: Strictly blocks unapproved execution."""
    status = config.get("gate_approval_status")
    if status != "APPROVED":
        raise SpecificationApprovalError(
            f"SSRL-16 spec status is '{status}', not 'APPROVED'. Execution is forbidden before explicit authorization."
        )


def compact_ten_rule_and_comp_battery_eval(
    model: nn.Module, stage_ds: Dict[int, Dict[str, Any]], comp_ds: Dict[str, Dict[str, Any]],
    shas: Dict[str, str], cfg: Dict[str, Any], dp: Dict[str, Any],
    seed: int, step: int, arm_tag: str, eval_deadline: float,
) -> Dict[str, Any]:
    """Evaluates Stages 1..10 + zero-shot Comp-6 & Comp-7 (seen/unseen) and counterfactual probe."""
    if time.monotonic() > eval_deadline:
        raise DeadlineExceededError(f"Evaluation deadline exceeded before {arm_tag}")
    model.eval()
    assert not model.training
    t_start = time.monotonic()
    metrics, per_unit = {}, {}
    targets: List[Tuple[str, Dict[str, Any], str]] = [
        (f"stage_{sp['stage']}", stage_ds[int(sp["stage"])], shas[sp["file"]]) for sp in cfg["stream_stages"]
    ] + [(cs["key"], comp_ds[cs["key"]], shas[cs["file"]]) for cs in cfg["composition_specs"]]

    for prefix, ds, sha in targets:
        for suffix, exp_n in (("seen", 192), ("unseen", 64)):
            exs = ds[f"eval_{suffix}_examples"]
            assert len(exs) == exp_n, f"{prefix}_{suffix} expected {exp_n}, got {len(exs)}"
            res = compact_population_eval(model, exs, arm_tag, seed, step, sha, f"{prefix}_{suffix}")
            metrics[f"acc_{prefix}_{suffix}"] = res["metric"]
            per_unit[f"{prefix}_{suffix}"] = res["per_unit_outcomes"]

    probe_full = evaluate_counterfactual_probe(model=model, probe_examples=dp["probe_examples"], device=torch.device("cpu"))
    prs = probe_full["pair_records"]
    assert len(prs) == 128
    pair_outcomes = {
        "probe_ids": [int(r["probe_id"]) for r in prs],
        "correct_a": [bool(r["predicted_class_a"] == r["target_class_a"]) for r in prs],
        "correct_b": [bool(r["predicted_class_b"] == r["target_class_b"]) for r in prs],
        "switched": [bool(r["is_switched"]) for r in prs],
    }
    if time.monotonic() > eval_deadline:
        raise DeadlineExceededError(f"Evaluation deadline exceeded after {arm_tag}")
    return {
        "elapsed_eval_seconds": time.monotonic() - t_start,
        "metrics": metrics,
        "per_unit_populations": per_unit,
        "counterfactual_probe_r1_r2": {"overall_metrics": probe_full["overall_metrics"], "strata_metrics": probe_full["strata_metrics"], "pair_outcomes": pair_outcomes},
    }


def compute_ten_rule_gate_snapshot(
    model: SSRLInputGatedMLP, init_params: Dict[str, torch.Tensor], d10: Dict[str, Any], cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Records W_gate diag vs off-diag stats and gains across Stages 1..10, Comp 6/7, and Probes 1..5."""
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count == get_input_gated_parameter_count() == 6658
    fp_before = compute_tensor_fingerprint_from_state_dict(model.state_dict())
    model.eval()
    assert not model.training

    all_masks: Dict[str, List[int]] = {
        **{f"stage_{sp['stage']}": list(sp["coords"]) for sp in cfg["stream_stages"]},
        **{cs["key"]: list(cs["coords"]) for cs in cfg["composition_specs"]},
        **{f"probe_{ps['probe_idx']}": list(ps["coords"]) for ps in cfg["transfer_probe_specs"]},
    }
    with torch.no_grad():
        diffs = [(p.detach().to(torch.float32) - init_params[n].to(p.device, dtype=torch.float32)).reshape(-1) for n, p in model.named_parameters()]
        dist_init_l2 = float(torch.linalg.vector_norm(torch.cat(diffs, dim=0), ord=2).item())
        w_gate = model.fc_in_gate.weight.detach().to(torch.float32)
        diag_mask = torch.eye(16, dtype=torch.bool)
        w_diag, w_offdiag = w_gate[diag_mask], w_gate[~diag_mask]

        gates, mask_gains = {}, {}
        for name, active in all_masks.items():
            mv = torch.zeros((1, 16), dtype=torch.float32)
            mv[0, active] = 1.0
            g = model.compute_input_gate(mv).squeeze(0)
            gates[name] = g
            inactive = [idx for idx in range(16) if idx not in set(active)]
            mask_gains[name] = {"active_gain_mean": float(g[active].mean().item()), "all_distractor_gain_mean": float(g[inactive].mean().item())}

        seen_s10 = d10["eval_seen_examples"]
        assert len(seen_s10) == 192
        x10 = torch.tensor([ex["features"] + ex["mask"] for ex in seen_s10], dtype=torch.float32)
        y10 = torch.tensor([int(ex["target_class"]) for ex in seen_s10], dtype=torch.long)
        feat10, mask10 = x10[:, : model.feature_dim], x10[:, model.feature_dim :]
        u_gated = torch.cat([feat10 * model.compute_input_gate(mask10), mask10], dim=-1)
        h1 = model.relu1(model.fc1(u_gated))
        h2 = model.relu2(model.fc2(h1))
        logits = model.fc3(h2)
        assert torch.equal(logits, model(x10))
        ce_s10 = float(nn.CrossEntropyLoss()(logits, y10).item())
        margin_s10 = float(torch.abs(logits[:, 1] - logits[:, 0]).mean().item())
        h1_dead = int((~(h1 > 0.0).any(dim=0)).sum().item())
        h2_dead = int((~(h2 > 0.0).any(dim=0)).sum().item())

    assert compute_tensor_fingerprint_from_state_dict(model.state_dict()) == fp_before
    return {
        "mode": EVAL_MODE_DETERMINISTIC, "parameter_count": param_count, "stage_10_seen_denominator": len(seen_s10),
        "distance_from_init_l2": dist_init_l2,
        "fc_in_gate_weight_l2": float(torch.linalg.vector_norm(w_gate, ord=2).item()),
        "w_gate_diag_l2": float(torch.linalg.vector_norm(w_diag, ord=2).item()),
        "w_gate_offdiag_l2": float(torch.linalg.vector_norm(w_offdiag, ord=2).item()),
        "w_gate_diag_mean": float(w_diag.mean().item()),
        "w_gate_offdiag_abs_mean": float(w_offdiag.abs().mean().item()),
        "rule_gate_gains": mask_gains,
        "ce_loss_stage_10": ce_s10, "mean_abs_logit_margin_stage_10": margin_s10,
        "h1_dead_relu_count": h1_dead, "h2_dead_relu_count": h2_dead,
    }


def train_ten_rule_l2init_der(
    model_kind: str, initial_model_state: Dict[str, torch.Tensor], init_params: Dict[str, torch.Tensor],
    tagged_schedule: List[Tuple[str, int, List[Dict[str, Any]]]], target_eval_ds: Dict[str, Any], target_sha: str,
    cfg: Dict[str, Any], seed: int, arm_tag: str, train_deadline: float, eval_ceiling: float,
    expected_start_fp: str, expected_init_fp: str, milestone_target_tag: Optional[str] = None,
    milestone_steps: Optional[List[int]] = None, teacher_by_rule: Optional[Dict[int, Dict[int, torch.Tensor]]] = None,
    expected_teacher_fp_by_rule: Optional[Dict[int, str]] = None,
) -> Tuple[nn.Module, float, Dict[str, Any], str, Dict[str, Any]]:
    """Trains SSRLInputGatedMLP or SSRLInstructionMLP with Adam, L2-Init, and optional per-rule DER++."""
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

    model = _instantiate_model(model_kind, float(cfg["gate_gain"]))
    exp_params = int(cfg["expected_gated_parameter_count"] if model_kind == "gated" else cfg["expected_bare_parameter_count"])
    assert sum(p.numel() for p in model.parameters()) == exp_params
    model.load_state_dict(initial_model_state)
    loaded_fp = compute_tensor_fingerprint_from_state_dict(model.state_dict())
    if loaded_fp != expected_start_fp:
        raise CheckpointVerificationError(f"Initial fingerprint mismatch for {arm_tag}: {loaded_fp} != {expected_start_fp}")

    optimizer = optim.Adam(model.parameters(), lr=cfg["learning_rate"], betas=tuple(cfg["betas"]), eps=cfg["eps"], weight_decay=cfg["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()
    l2_lambda, der_alpha = float(cfg["l2_init_lambda"]), float(cfg["der_alpha"])
    counters: Dict[str, Any] = {
        "total_optimizer_steps": 0, "total_presentations": 0, "rule_steps_by_tag": {}, "replay_updates_by_rule": {},
        "der_loss_steps": 0, "der_loss_sum": 0.0, "final_ce_loss": None, "final_l2_init_penalty": None, "final_der_mse_loss": None,
    }
    m_set = set(milestone_steps) if (milestone_target_tag and milestone_steps) else set()
    max_m = max(m_set) if m_set else -1
    milestone_records: Dict[str, Any] = {}
    t_start = time.monotonic()

    def _record_milestone(step_k: int) -> None:
        dl = time.monotonic() + eval_ceiling
        model.eval()
        res = compact_population_eval(model, target_eval_ds["eval_seen_examples"], f"{arm_tag}_step{step_k}", seed, step_k, target_sha, f"{milestone_target_tag}_seen")
        rec: Dict[str, Any] = {"target_updates": step_k, "total_steps": counters["total_optimizer_steps"], "metric": res["metric"]}
        if step_k in (0, max_m):
            rec["per_unit_outcomes"] = res["per_unit_outcomes"]
        milestone_records[str(step_k)] = rec
        if time.monotonic() > dl:
            raise DeadlineExceededError(f"Milestone {step_k} eval deadline exceeded for {arm_tag}")

    if 0 in m_set:
        _record_milestone(0)

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

        if milestone_target_tag and tag == milestone_target_tag and counters["rule_steps_by_tag"][tag] in m_set:
            _record_milestone(counters["rule_steps_by_tag"][tag])
            model.train()

    assert counters["total_optimizer_steps"] > 0 and counters["total_presentations"] > 0
    assert compute_tensor_fingerprint_from_state_dict(init_params) == expected_init_fp
    if teacher_by_rule is not None and expected_teacher_fp_by_rule is not None:
        for rid, td in teacher_by_rule.items():
            assert _teacher_dict_fingerprint(td) == expected_teacher_fp_by_rule[rid]
    counters["mean_der_mse_loss"] = counters["der_loss_sum"] / counters["der_loss_steps"] if counters["der_loss_steps"] > 0 else None
    return model, time.monotonic() - t_start, counters, loaded_fp, milestone_records


def run_isolated_transfer_probes(
    seed: int, m_gated_init_sd: Dict[str, torch.Tensor], init_g: Dict[str, torch.Tensor], fp_init_g: str,
    m_der_r10: SSRLInputGatedMLP, fp_der_r10: str, mb_der_r10: nn.Module, init_b: Dict[str, torch.Tensor],
    fp_init_b: str, fp_bare_r10: str, probe_ds: Dict[int, Dict[str, Any]], shas: Dict[str, str],
    cfg: Dict[str, Any], diagnostics: Dict[str, Any],
) -> Dict[str, Any]:
    """Runs 5 isolated 500-step transfer probes (PLAN.md 5.1) with zero state carry-over across probes."""
    t_ceil, e_ceil = float(cfg["wall_clock_ceiling_training_seconds_per_stage"]), float(cfg["wall_clock_ceiling_eval_seconds_per_stage"])
    n_psteps, bsz = int(cfg["transfer_probe_steps"]), int(cfg["batch_size"])
    milestones = list(cfg["transfer_probe_milestones"])
    der_r10_sd = copy.deepcopy(m_der_r10.state_dict())
    bare_r10_sd = copy.deepcopy(mb_der_r10.state_dict())

    per_probe: Dict[str, Any] = {}
    mean_delta_gated_vs_fresh: Dict[str, List[float]] = {str(k): [] for k in milestones}
    mean_delta_bare_vs_fresh: Dict[str, List[float]] = {str(k): [] for k in milestones}
    mean_delta_gated_vs_bare: Dict[str, List[float]] = {str(k): [] for k in milestones}

    for ps in cfg["transfer_probe_specs"]:
        pidx, rid = int(ps["probe_idx"]), int(ps["rule_id"])
        pds, psha = probe_ds[pidx], shas[ps["file"]]
        tag = f"probe_{pidx}"
        sched = [(tag, rid, b) for b in build_minibatch_schedule(pds["train_examples"], n_psteps, bsz, int(ps["sampler_seed"]))]
        arm_inputs = [
            ("fresh_gated", "gated", m_gated_init_sd, init_g, fp_init_g, fp_init_g),
            ("post10_gated", "gated", der_r10_sd, init_g, fp_der_r10, fp_init_g),
            ("post10_bare", "bare", bare_r10_sd, init_b, fp_bare_r10, fp_init_b),
        ]
        arms_out: Dict[str, Any] = {}
        for arm_name, mk, start_sd, iparams, sfp, ifp in arm_inputs:
            diagnostics["current_stage"] = f"seed_{seed}_probe_{pidx}_{arm_name}"
            pm, dt, pcnt, _, pmiles = train_ten_rule_l2init_der(mk, copy.deepcopy(start_sd), iparams, sched, pds, psha, cfg, seed, f"ig10_p{pidx}_{arm_name}_s{seed}", time.monotonic() + t_ceil, e_ceil, sfp, ifp, milestone_target_tag=tag, milestone_steps=milestones)
            unseen_res = compact_population_eval(pm, pds["eval_unseen_examples"], f"ig10_p{pidx}_{arm_name}_unseen_s{seed}", seed, n_psteps, psha, f"{tag}_unseen")
            arms_out[arm_name] = {"elapsed_train_seconds": dt, "counters": pcnt, "milestone_evaluations": pmiles, "final_unseen": unseen_res}

        # Assert zero state mutation on source checkpoints across probes
        assert compute_tensor_fingerprint_from_state_dict(m_der_r10.state_dict()) == fp_der_r10
        assert compute_tensor_fingerprint_from_state_dict(mb_der_r10.state_dict()) == fp_bare_r10

        p_deltas: Dict[str, Any] = {}
        for k in milestones:
            ks = str(k)
            fm = arms_out["fresh_gated"]["milestone_evaluations"][ks]["metric"]
            gm = arms_out["post10_gated"]["milestone_evaluations"][ks]["metric"]
            bm = arms_out["post10_bare"]["milestone_evaluations"][ks]["metric"]
            assert fm.denominator == gm.denominator == bm.denominator == 192
            dgf = ((gm.numerator - fm.numerator) / 192.0) * 100.0
            dbf = ((bm.numerator - fm.numerator) / 192.0) * 100.0
            dgb = ((gm.numerator - bm.numerator) / 192.0) * 100.0
            mean_delta_gated_vs_fresh[ks].append(dgf)
            mean_delta_bare_vs_fresh[ks].append(dbf)
            mean_delta_gated_vs_bare[ks].append(dgb)
            p_deltas[ks] = {"fresh_gated": fm.numerator, "post10_gated": gm.numerator, "post10_bare": bm.numerator, "delta_gated_minus_fresh_pp": dgf, "delta_bare_minus_fresh_pp": dbf, "delta_gated_minus_bare_pp": dgb}
        per_probe[tag] = {"rule_id": rid, "coords": ps["coords"], "arms": arms_out, "milestone_deltas": p_deltas}

    seed_mean_deltas = {
        str(k): {
            "mean_delta_gated_minus_fresh_pp": sum(mean_delta_gated_vs_fresh[str(k)]) / 5.0,
            "mean_delta_bare_minus_fresh_pp": sum(mean_delta_bare_vs_fresh[str(k)]) / 5.0,
            "mean_delta_gated_minus_bare_pp": sum(mean_delta_gated_vs_bare[str(k)]) / 5.0,
        }
        for k in milestones
    }
    return {"per_probe": per_probe, "seed_mean_transfer_deltas": seed_mean_deltas}


def run_ten_rule_stream_study(
    data_dir: str, destination: str, run_dir: str,
    sha256_overrides: Optional[Dict[str, str]] = None, config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Master procedural entry point for SSRL-16 10-Rule Continual Learning + Isolated Transfer Study."""
    cfg = copy.deepcopy(config or PROPOSED_TEN_RULE_STREAM_CONFIG)
    verify_ten_rule_stream_approval(cfg)
    assert get_input_gated_parameter_count() == cfg["expected_gated_parameter_count"] == 6658
    assert get_expected_parameter_count() == cfg["expected_bare_parameter_count"] == 6402
    assert len(cfg["stream_stages"]) == 10 and len(cfg["transfer_probe_specs"]) == 5
    abs_run_dir, abs_dest = reserve_run_directory(run_dir), os.path.abspath(destination)
    shas = {**DEFAULT_DATASET_SHA256, **(sha256_overrides or {})}

    def _load_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
        fpath = os.path.join(data_dir, spec["file"])
        ds = load_and_verify_dataset(
            fpath, expected_sha256=shas[spec["file"]],
            approved_config={**cfg, "study_id": spec["study"], "phase": spec["phase"], "seed": 101, "partition_seed": spec["pseed"], "rule_id": spec["rule_id"], "rule_indices": spec["coords"]},
        )
        if ds["metadata"].get("study_id") == "SSRL-16":
            assert ds == generate_pilot_dataset_dict(tuple(spec["coords"]), int(spec["rule_id"]), 101, partition_seed=int(spec["pseed"]), study_id=spec["study"], phase=spec["phase"])
        return ds

    stage_ds = {int(sp["stage"]): _load_spec(sp) for sp in cfg["stream_stages"]}
    comp_ds = {cs["key"]: _load_spec(cs) for cs in cfg["composition_specs"]}
    probe_ds = {int(ps["probe_idx"]): _load_spec(ps) for ps in cfg["transfer_probe_specs"]}

    probe_path = os.path.join(data_dir, "ssrl_probe_counterfactual_128.json")
    with open(probe_path, "rb") as f:
        if hashlib.sha256(f.read()).hexdigest() != shas["ssrl_probe_counterfactual_128.json"]:
            raise ValueError("Counterfactual probe SHA-256 mismatch")
    with open(probe_path, "r", encoding="utf-8") as f:
        dp = json.load(f)
    validate_counterfactual_probe_contract(dp, stage_ds[1], stage_ds[2])

    diagnostics: Dict[str, Any] = {
        "study_id": cfg["study_id"], "phase": cfg["phase"], "classification": cfg["classification"],
        "status": "IN_PROGRESS", "current_stage": "INITIALIZATION", "execution_mode": EVAL_MODE_DETERMINISTIC,
        "dataset_hashes": shas, "seeds_status": {s: "NOT_RUN" for s in cfg["seeds"]}, "seeds_data": {},
        "environment": {
            "python_version": sys.version.split()[0], "torch_version": torch.__version__, "device": "cpu",
            "step_budget_per_stream_rule": cfg["step_budget_per_stream_rule"], "transfer_probe_steps": cfg["transfer_probe_steps"],
            "l2_init_lambda": cfg["l2_init_lambda"], "der_alpha": cfg["der_alpha"], "replay_buffer_size_per_rule": cfg["replay_buffer_size_per_rule"],
        },
        "paired_ten_rule_comparisons": [], "run_directory": abs_run_dir, "destination": abs_dest,
    }
    diag_file = os.path.join(abs_run_dir, "failure_diagnostics.json")

    try:
        t_ceil, e_ceil = float(cfg["wall_clock_ceiling_training_seconds_per_stage"]), float(cfg["wall_clock_ceiling_eval_seconds_per_stage"])
        n_steps, bsz, rb, ra = int(cfg["step_budget_per_stream_rule"]), int(cfg["batch_size"]), cfg["replay_ratio_new_to_old"][0], cfg["replay_ratio_new_to_old"][1]
        raw_scheds = {sp["stage"]: build_minibatch_schedule(stage_ds[sp["stage"]]["train_examples"], n_steps, bsz, int(sp["sampler_seed"])) for sp in cfg["stream_stages"]}
        seq_scheds = {sp["stage"]: [(f"stage_{sp['stage']}", int(sp["rule_id"]), b) for b in raw_scheds[sp["stage"]]] for sp in cfg["stream_stages"]}
        bufs = {sp["rule_id"]: select_multi_rule_stratified_memory(stage_ds[sp["stage"]]["train_examples"], sp["coords"], cfg["replay_buffer_size_per_rule"], cfg["replay_batch_size"]) for sp in cfg["stream_stages"][:9]}
        der_scheds = {
            sp["stage"]: build_multi_rule_cyclic_replay_schedule(
                raw_scheds[sp["stage"]], f"stage_{sp['stage']}", int(sp["rule_id"]),
                {int(prev["rule_id"]): bufs[int(prev["rule_id"])] for prev in cfg["stream_stages"][: idx]},
                cfg["replay_batch_size"], rb, ra,
            )
            for idx, sp in enumerate(cfg["stream_stages"][1:], start=1)
        }
        diagnostics["schedule_hashes"] = {**{f"s{k}_seq": compute_schedule_hash(v) for k, v in seq_scheds.items()}, **{f"s{k}_der": compute_schedule_hash(v) for k, v in der_scheds.items()}}
        probe_x = torch.stack([torch.tensor(ex["features"] + ex["mask"], dtype=torch.float32) for ex in stage_ds[1]["eval_seen_examples"][:4]], dim=0)

        for seed in cfg["seeds"]:
            diagnostics["seeds_status"][seed] = "IN_PROGRESS"
            seed_rec: Dict[str, Any] = {"seed": seed}
            diagnostics["seeds_data"][str(seed)] = seed_rec
            torch.manual_seed(seed); m_g_init = _instantiate_model("gated", float(cfg["gate_gain"]))
            torch.manual_seed(seed); m_b_init = _instantiate_model("bare", float(cfg["gate_gain"]))
            m_g_init.eval(); m_b_init.eval()
            with torch.no_grad():
                assert torch.equal(m_g_init(probe_x), m_b_init(probe_x))
            init_g, fp_init_g = clone_frozen_init_params(m_g_init)
            init_b, fp_init_b = clone_frozen_init_params(m_b_init)

            # 1. Gated R1 + Freeze control
            diagnostics["current_stage"] = f"seed_{seed}_post_r1"
            m_r1, t_r1, cnt_r1, _, _ = train_ten_rule_l2init_der("gated", m_g_init.state_dict(), init_g, seq_scheds[1], stage_ds[10], shas[cfg["stream_stages"][9]["file"]], cfg, seed, f"ig10_r1_s{seed}", time.monotonic() + t_ceil, e_ceil, fp_init_g, fp_init_g)
            ev_r1 = compact_population_eval(m_r1, stage_ds[1]["eval_seen_examples"], f"ig10_r1_s{seed}", seed, n_steps, shas[cfg["stream_stages"][0]["file"]], "stage_1_seen")
            if cfg["enforce_post_r1_learnability_gate"]:
                _assert_learnability_gate(ev_r1["metric"], seed, "post_r1", cfg["gate_g_lc_threshold"])
            ckpt_r1 = capture_and_verify_checkpoint(m_r1.state_dict(), abs_run_dir, f"checkpoint_ig10_post_r1_s{seed}.pt")
            fp_r1 = ckpt_r1["tensor_fingerprint"]
            t_r1_logits, t_r1_fp, _ = capture_frozen_teacher_logits(m_r1, bufs[1], fp_r1)
            seed_rec["post_r1"] = {"elapsed_train_seconds": t_r1, "counters": cnt_r1, "checkpoint": ckpt_r1}
            diagnostics["current_stage"] = f"seed_{seed}_freeze_after_r1"
            ev_frz = compact_ten_rule_and_comp_battery_eval(m_r1, stage_ds, comp_ds, shas, cfg, dp, seed, 0, f"ig10_frz_s{seed}", time.monotonic() + e_ceil)
            seed_rec["freeze_after_r1"] = {"fingerprint": fp_r1, "eval": ev_frz}

            # 2. Gated Sequential No-Replay (Stages 2..10)
            m_seq, fp_seq, seq_teachers = m_r1, fp_r1, {1: t_r1_logits}
            for st in range(2, 11):
                diagnostics["current_stage"] = f"seed_{seed}_seq_s{st}"
                m_seq, _, cnt_s, _, _ = train_ten_rule_l2init_der("gated", m_seq.state_dict(), init_g, seq_scheds[st], stage_ds[10], shas[cfg["stream_stages"][9]["file"]], cfg, seed, f"ig10_s{st}_seq_s{seed}", time.monotonic() + t_ceil, e_ceil, fp_seq, fp_init_g)
                fp_seq = compute_tensor_fingerprint_from_state_dict(m_seq.state_dict())
                if st <= 9:
                    rid = int(cfg["stream_stages"][st - 1]["rule_id"])
                    seq_teachers[rid], _, _ = capture_frozen_teacher_logits(m_seq, bufs[rid], fp_seq)
            ev_seq = compact_ten_rule_and_comp_battery_eval(m_seq, stage_ds, comp_ds, shas, cfg, dp, seed, n_steps, f"ig10_seq10_s{seed}", time.monotonic() + e_ceil)
            ckpt_seq = capture_and_verify_checkpoint(m_seq.state_dict(), abs_run_dir, f"checkpoint_ig10_seq_r10_s{seed}.pt")
            seed_rec["seq_r10"] = {"counters": cnt_s, "checkpoint": ckpt_seq, "mechanism": compute_ten_rule_gate_snapshot(m_seq, init_g, stage_ds[10], cfg), "final_eval": ev_seq}

            # 3. Gated DER++ Stream (Stages 2..10, checkpoint at Stage 5 and Stage 10)
            m_der, fp_der, der_teachers, der_fps, der_times = m_r1, fp_r1, {1: t_r1_logits}, {1: t_r1_fp}, {"stage_1": t_r1}
            for st in range(2, 11):
                diagnostics["current_stage"] = f"seed_{seed}_der_s{st}"
                m_der, dt, cnt_d, _, _ = train_ten_rule_l2init_der("gated", m_der.state_dict(), init_g, der_scheds[st], stage_ds[10], shas[cfg["stream_stages"][9]["file"]], cfg, seed, f"ig10_s{st}_der_s{seed}", time.monotonic() + t_ceil, e_ceil, fp_der, fp_init_g, teacher_by_rule=dict(der_teachers), expected_teacher_fp_by_rule=dict(der_fps))
                fp_der = compute_tensor_fingerprint_from_state_dict(m_der.state_dict())
                der_times[f"stage_{st}"] = dt
                if st <= 9:
                    rid = int(cfg["stream_stages"][st - 1]["rule_id"])
                    der_teachers[rid], der_fps[rid], _ = capture_frozen_teacher_logits(m_der, bufs[rid], fp_der)
                if st == 5:
                    ev_d5 = compact_ten_rule_and_comp_battery_eval(m_der, stage_ds, comp_ds, shas, cfg, dp, seed, len(der_scheds[5]), f"ig10_d5_s{seed}", time.monotonic() + e_ceil)
                    ckpt_d5 = capture_and_verify_checkpoint(m_der.state_dict(), abs_run_dir, f"checkpoint_ig10_der_post_r5_s{seed}.pt")
                    seed_rec["der_post_r5"] = {"checkpoint": ckpt_d5, "mechanism": compute_ten_rule_gate_snapshot(m_der, init_g, stage_ds[10], cfg), "final_eval": ev_d5}

            ev_d10 = compact_ten_rule_and_comp_battery_eval(m_der, stage_ds, comp_ds, shas, cfg, dp, seed, len(der_scheds[10]), f"ig10_d10_s{seed}", time.monotonic() + e_ceil)
            ckpt_d10 = capture_and_verify_checkpoint(m_der.state_dict(), abs_run_dir, f"checkpoint_ig10_stream_der_r10_s{seed}.pt")
            seed_rec["stream_der_r10"] = {
                "phase_train_seconds": der_times, "counters": cnt_d, "checkpoint": ckpt_d10,
                "mechanism": compute_ten_rule_gate_snapshot(m_der, init_g, stage_ds[10], cfg), "final_eval": ev_d10,
                "buffer_logit_drifts": {f"rule_{rid}": evaluate_buffer_logit_drift(m_der, bufs[rid], der_teachers[rid]) for rid in der_teachers},
            }

            # 4. Matched Bare Control DER++ Stream (Stages 1..10)
            diagnostics["current_stage"] = f"seed_{seed}_bare_post_r1"
            mb_r1, tb1, _, _, _ = train_ten_rule_l2init_der("bare", m_b_init.state_dict(), init_b, seq_scheds[1], stage_ds[10], shas[cfg["stream_stages"][9]["file"]], cfg, seed, f"ig10_br1_s{seed}", time.monotonic() + t_ceil, e_ceil, fp_init_b, fp_init_b)
            if cfg["enforce_bare_post_r1_learnability_gate"]:
                _assert_learnability_gate(compact_population_eval(mb_r1, stage_ds[1]["eval_seen_examples"], f"ig10_br1_g_s{seed}", seed, n_steps, shas[cfg["stream_stages"][0]["file"]], "stage_1_seen")["metric"], seed, "bare_post_r1", cfg["gate_g_lc_threshold"])
            ev_br1 = compact_ten_rule_and_comp_battery_eval(mb_r1, stage_ds, comp_ds, shas, cfg, dp, seed, n_steps, f"ig10_br1_s{seed}", time.monotonic() + e_ceil)
            ckpt_br1 = capture_and_verify_checkpoint(mb_r1.state_dict(), abs_run_dir, f"checkpoint_ig10_bare_post_r1_s{seed}.pt")
            fp_br1 = ckpt_br1["tensor_fingerprint"]
            tb1_logits, tb1_fp, _ = capture_frozen_teacher_logits(mb_r1, bufs[1], fp_br1)
            seed_rec["bare_post_r1"] = {"elapsed_train_seconds": tb1, "checkpoint": ckpt_br1, "final_eval": ev_br1}

            mb, fp_b, b_teachers, b_fps, b_times = mb_r1, fp_br1, {1: tb1_logits}, {1: tb1_fp}, {"stage_1": tb1}
            for st in range(2, 11):
                diagnostics["current_stage"] = f"seed_{seed}_bare_der_s{st}"
                mb, dt, cnt_b, _, _ = train_ten_rule_l2init_der("bare", mb.state_dict(), init_b, der_scheds[st], stage_ds[10], shas[cfg["stream_stages"][9]["file"]], cfg, seed, f"ig10_bs{st}_der_s{seed}", time.monotonic() + t_ceil, e_ceil, fp_b, fp_init_b, teacher_by_rule=dict(b_teachers), expected_teacher_fp_by_rule=dict(b_fps))
                fp_b = compute_tensor_fingerprint_from_state_dict(mb.state_dict())
                b_times[f"stage_{st}"] = dt
                if st <= 9:
                    rid = int(cfg["stream_stages"][st - 1]["rule_id"])
                    b_teachers[rid], b_fps[rid], _ = capture_frozen_teacher_logits(mb, bufs[rid], fp_b)
            ev_b10 = compact_ten_rule_and_comp_battery_eval(mb, stage_ds, comp_ds, shas, cfg, dp, seed, len(der_scheds[10]), f"ig10_b10_s{seed}", time.monotonic() + e_ceil)
            ckpt_b10 = capture_and_verify_checkpoint(mb.state_dict(), abs_run_dir, f"checkpoint_ig10_bare_stream_der_r10_s{seed}.pt")
            seed_rec["bare_stream_der_r10"] = {
                "phase_train_seconds": b_times, "counters": cnt_b, "checkpoint": ckpt_b10,
                "mechanism": compute_mechanism_snapshot(mb, init_b, stage_ds[10]), "final_eval": ev_b10,
                "buffer_logit_drifts": {f"rule_{rid}": evaluate_buffer_logit_drift(mb, bufs[rid], b_teachers[rid]) for rid in b_teachers},
            }

            # 5. Isolated 5-Rule Forward Transfer Probes (PLAN.md 5.1)
            transfer_res = run_isolated_transfer_probes(seed, m_g_init.state_dict(), init_g, fp_init_g, m_der, fp_der, mb, init_b, fp_init_b, fp_b, probe_ds, shas, cfg, diagnostics)
            seed_rec["isolated_transfer_probes"] = transfer_res
            comp_rec: Dict[str, Any] = {
                "seed": seed,
                **{f"stream_der_stage_{st}_pct": ev_d10["metrics"][f"acc_stage_{st}_seen"].pct for st in range(1, 11)},
                **{f"bare_der_stage_{st}_pct": ev_b10["metrics"][f"acc_stage_{st}_seen"].pct for st in range(1, 11)},
                **{f"gated_minus_bare_stage_{st}_pp": ev_d10["metrics"][f"acc_stage_{st}_seen"].pct - ev_b10["metrics"][f"acc_stage_{st}_seen"].pct for st in range(1, 11)},
                "comp_6_seen_pct": ev_d10["metrics"]["acc_comp_6_seen"].pct, "comp_7_seen_pct": ev_d10["metrics"]["acc_comp_7_seen"].pct,
                "mean_transfer_delta_500_gated_pp": transfer_res["seed_mean_transfer_deltas"][str(cfg["transfer_probe_steps"])]["mean_delta_gated_minus_fresh_pp"],
                "mean_transfer_delta_500_bare_pp": transfer_res["seed_mean_transfer_deltas"][str(cfg["transfer_probe_steps"])]["mean_delta_bare_minus_fresh_pp"],
            }
            seed_rec["ten_rule_comparison"] = comp_rec
            diagnostics["paired_ten_rule_comparisons"].append(comp_rec)
            diagnostics["seeds_status"][seed] = "COMPLETED"

        comps = diagnostics["paired_ten_rule_comparisons"]
        summary_keys = ["comp_6_seen_pct", "comp_7_seen_pct", "mean_transfer_delta_500_gated_pp", "mean_transfer_delta_500_bare_pp", *(f"stream_der_stage_{st}_pct" for st in range(1, 11)), *(f"gated_minus_bare_stage_{st}_pp" for st in range(1, 11))]
        diagnostics["summary_statistics"] = {k: _summarize_scalar_list([c[k] for c in comps]) for k in summary_keys}
        diagnostics["status"], diagnostics["current_stage"] = "COMPLETED", "COMPLETED"
        payload = {"config": cfg, "diagnostics": diagnostics}
        atomic_write_compact_json(payload, abs_dest, refuse_overwrite=True)
        return payload

    except Exception as exc:
        diagnostics["status"], diagnostics["error_type"], diagnostics["error_message"] = "FAILED", type(exc).__name__, str(exc)
        for s in cfg["seeds"]:
            diagnostics["seeds_status"][s] = "FAILED" if diagnostics["seeds_status"][s] == "IN_PROGRESS" else ("COMPLETED" if diagnostics["seeds_status"][s] == "COMPLETED" else "NOT_RUN")
        try:
            atomic_write_compact_json({"diagnostics": diagnostics}, diag_file, refuse_overwrite=False)
        except Exception as write_err:
            print(f"[Error writing failure diagnostics: {write_err}]", file=sys.stderr)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="SSRL-16 10-Rule PLAN.md Capstone & Transfer Runner")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    run_ten_rule_stream_study(data_dir=args.data_dir, destination=args.destination, run_dir=args.run_dir)


if __name__ == "__main__":
    main()
