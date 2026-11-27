"""
research/lifelong_01/run_ssrl_five_rule_stream.py -- Procedural Runner for SSRL-15.

Evaluates SSRLInputGatedMLP (6,658 params) + L2-Init (lambda=0.01) + M=64/rule DER++ (alpha=0.5) on a
5-rule sequential stream R1 -> R2 -> R3 -> R4 -> R5 (where R1..R4 partition [0..15] disjointly and R5=[0,4,8,12]
overlaps all 4 prior rules by 1 bit each), plus zero-shot composition on unseen masks Comp-6 [1,5,9,13] and
Comp-7 [0,1,4,5], against standing control (freeze_after_r1), fresh_r5, seq_r5, and matched bare_stream_der_r5.
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
    SSRLInputGatedMLP, get_expected_parameter_count, get_input_gated_parameter_count,
)
from research.lifelong_01.ssrl_generator import generate_pilot_dataset_dict
from research.lifelong_01.ssrl_counterfactual_probe import validate_counterfactual_probe_contract, evaluate_counterfactual_probe
from research.lifelong_01.run_ssrl_joint_reference import (
    load_and_verify_dataset, reserve_run_directory, make_json_serializable,
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


SHA256_RULE_4_DEFAULT = "503a09c1f8c059681db038bd472dd7068cbe87f680d963b8e321a9bc652daa71"
SHA256_RULE_5_DEFAULT = "97b15b6f9a985acedac0fc084ef96f38f07a1563ac62e70e0cb779b4eb928ff8"
SHA256_COMP_6_DEFAULT = "960646d3d50b76385ac57b196b4de06dc3a54a80869ff3d5dcf9c32f77d60a10"
SHA256_COMP_7_DEFAULT = "237ae5912d3e32fe44977405ff80a2c53f43c2a63019f082f18de42419525ef4"

PROPOSED_FIVE_RULE_STREAM_CONFIG: Dict[str, Any] = {
    "study_id": "SSRL-15",
    "phase": "five_rule_overlapping_stream",
    "classification": "five_rule_overlapping_input_gated_der_m64_three_seed",
    "gate_approval_status": "APPROVED",
    "primary_model_class": "SSRLInputGatedMLP",
    "expected_gated_parameter_count": 6658,
    "bare_control_model_class": "SSRLInstructionMLP",
    "expected_bare_parameter_count": 6402,
    "gate_gain": 4.0,
    "seeds": [102, 103, 104],
    "sampler_seeds": {"1": 2026, "2": 2027, "3": 2028, "4": 2029, "5": 2030},
    "rules": {"1": [0, 1, 2, 3], "2": [4, 5, 6, 7], "3": [8, 9, 10, 11], "4": [12, 13, 14, 15], "5": [0, 4, 8, 12]},
    "composition_rules": {"6": [1, 5, 9, 13], "7": [0, 1, 4, 5]},
    "train_pool_size": 384,
    "batch_size": 16,
    "step_budget_per_rule": 2000,
    "replay_buffer_size_per_rule": 64,
    "replay_batch_size": 16,
    "replay_ratio_new_to_old": [4, 1],
    "milestone_r5_steps": [0, 25, 50, 100, 250, 500, 750, 1000, 1500, 2000],
    "learning_rate": 0.001,
    "betas": [0.9, 0.999],
    "eps": 1e-08,
    "weight_decay": 0.0,
    "l2_init_lambda": 0.01,
    "der_alpha": 0.5,
    "gate_g_lc_threshold": 0.90,
    "enforce_post_r1_learnability_gate": True,
    "enforce_bare_post_r1_learnability_gate": True,
    "enforce_fresh_r5_learnability_gate": True,
    "wall_clock_ceiling_training_seconds_per_stage": 60.0,
    "wall_clock_ceiling_eval_seconds_per_stage": 60.0,
}


def verify_five_rule_stream_approval(config: Dict[str, Any]) -> None:
    """AGENTS.md Section 2 / Section 8: Strictly blocks unapproved execution."""
    status = config.get("gate_approval_status")
    if status != "APPROVED":
        raise SpecificationApprovalError(
            f"SSRL-15 spec status is '{status}', not 'APPROVED'. Execution is forbidden before explicit authorization."
        )


def atomic_write_compact_json(payload: Any, destination: str, refuse_overwrite: bool = True) -> str:
    """Atomically serializes compact JSON keeping full raw per-unit vectors under tight byte caps."""
    abs_dest = os.path.abspath(destination)
    if refuse_overwrite and os.path.exists(abs_dest):
        raise FileExistsError(f"Destination exists, refusing overwrite: {abs_dest}")
    os.makedirs(os.path.dirname(abs_dest), exist_ok=True)
    tmp_path = f"{abs_dest}.tmp.{os.getpid()}"
    raw = json.dumps(make_json_serializable(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    with open(tmp_path, "wb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, abs_dest)
    return hashlib.sha256(raw).hexdigest()


def compact_five_rule_and_comp_battery_eval(
    model: nn.Module, datasets: Dict[str, Dict[str, Any]], hashes: Dict[str, str], dp: Dict[str, Any],
    seed: int, step: int, arm_tag: str, eval_deadline: float,
) -> Dict[str, Any]:
    """Evaluates R1..R5 plus zero-shot Comp-6 & Comp-7 (seen/unseen) and compact counterfactual probe."""
    if time.monotonic() > eval_deadline:
        raise DeadlineExceededError(f"Evaluation deadline exceeded before {arm_tag}")
    model.eval()
    assert not model.training
    t_start = time.monotonic()
    metrics, per_unit = {}, {}
    ordered_keys = [("rule_1", "1"), ("rule_2", "2"), ("rule_3", "3"), ("rule_4", "4"), ("rule_5", "5"), ("comp_6", "6"), ("comp_7", "7")]
    for prefix, rk in ordered_keys:
        ds, sha = datasets[rk], hashes[rk]
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
        "counterfactual_probe_r1_r2": {
            "overall_metrics": probe_full["overall_metrics"],
            "strata_metrics": probe_full["strata_metrics"],
            "pair_outcomes": pair_outcomes,
        },
    }


def compute_five_rule_gate_snapshot(
    model: SSRLInputGatedMLP, init_params: Dict[str, torch.Tensor], d5: Dict[str, Any], cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Records W_gate diag vs off-diag norms, active/distractor gains on R1..R5 + Comp6/Comp7, and R5 diagnostics."""
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count == get_input_gated_parameter_count() == 6658
    fp_before = compute_tensor_fingerprint_from_state_dict(model.state_dict())
    model.eval()
    assert not model.training

    all_masks: Dict[str, List[int]] = {
        **{f"rule_{k}": list(v) for k, v in cfg["rules"].items()},
        **{f"comp_{k}": list(v) for k, v in cfg["composition_rules"].items()},
    }
    with torch.no_grad():
        diffs = [(p.detach().to(torch.float32) - init_params[n].to(p.device, dtype=torch.float32)).reshape(-1) for n, p in model.named_parameters()]
        dist_init_l2 = float(torch.linalg.vector_norm(torch.cat(diffs, dim=0), ord=2).item())
        w_gate = model.fc_in_gate.weight.detach().to(torch.float32)
        diag_mask = torch.eye(16, dtype=torch.bool)
        w_diag_l2 = float(torch.linalg.vector_norm(w_gate[diag_mask], ord=2).item())
        w_offdiag_l2 = float(torch.linalg.vector_norm(w_gate[~diag_mask], ord=2).item())

        gates: Dict[str, torch.Tensor] = {}
        mask_gains: Dict[str, Dict[str, float]] = {}
        for name, active in all_masks.items():
            mv = torch.zeros((1, 16), dtype=torch.float32)
            mv[0, active] = 1.0
            g = model.compute_input_gate(mv).squeeze(0)
            gates[name] = g
            inactive = [idx for idx in range(16) if idx not in set(active)]
            mask_gains[name] = {
                "active_gain_mean": float(g[active].mean().item()),
                "all_distractor_gain_mean": float(g[inactive].mean().item()),
            }

        cosines: Dict[str, float] = {}
        for i in range(1, 5):
            cosines[f"cos_r{i}_r5"] = float(F.cosine_similarity(gates[f"rule_{i}"].unsqueeze(0), gates["rule_5"].unsqueeze(0), dim=1).item())
        for ck in ("comp_6", "comp_7"):
            cosines[f"cos_r1_{ck}"] = float(F.cosine_similarity(gates["rule_1"].unsqueeze(0), gates[ck].unsqueeze(0), dim=1).item())
            cosines[f"cos_r5_{ck}"] = float(F.cosine_similarity(gates["rule_5"].unsqueeze(0), gates[ck].unsqueeze(0), dim=1).item())

        seen_r5 = d5["eval_seen_examples"]
        assert len(seen_r5) == 192
        x5 = torch.tensor([ex["features"] + ex["mask"] for ex in seen_r5], dtype=torch.float32)
        y5 = torch.tensor([int(ex["target_class"]) for ex in seen_r5], dtype=torch.long)
        feat5, mask5 = x5[:, : model.feature_dim], x5[:, model.feature_dim :]
        u_gated = torch.cat([feat5 * model.compute_input_gate(mask5), mask5], dim=-1)
        h1 = model.relu1(model.fc1(u_gated))
        h2 = model.relu2(model.fc2(h1))
        logits = model.fc3(h2)
        assert torch.equal(logits, model(x5))
        ce_r5 = float(nn.CrossEntropyLoss()(logits, y5).item())
        margin_r5 = float(torch.abs(logits[:, 1] - logits[:, 0]).mean().item())
        h1_dead = int((~(h1 > 0.0).any(dim=0)).sum().item())
        h2_dead = int((~(h2 > 0.0).any(dim=0)).sum().item())

    assert compute_tensor_fingerprint_from_state_dict(model.state_dict()) == fp_before
    return {
        "mode": EVAL_MODE_DETERMINISTIC, "parameter_count": param_count, "rule_5_seen_denominator": len(seen_r5),
        "distance_from_init_l2": dist_init_l2,
        "fc_in_gate_weight_l2": float(torch.linalg.vector_norm(w_gate, ord=2).item()),
        "w_gate_diag_l2": w_diag_l2, "w_gate_offdiag_l2": w_offdiag_l2,
        "rule_gate_gains": mask_gains, "pairwise_gate_cosines": cosines,
        "ce_loss_r5": ce_r5, "mean_abs_logit_margin_r5": margin_r5,
        "h1_dead_relu_count": h1_dead, "h2_dead_relu_count": h2_dead,
    }


def train_five_rule_l2init_der(
    model_kind: str, initial_model_state: Dict[str, torch.Tensor], init_params: Dict[str, torch.Tensor],
    tagged_schedule: List[Tuple[str, int, List[Dict[str, Any]]]], d5: Dict[str, Any], sha5: str,
    cfg: Dict[str, Any], seed: int, arm_tag: str, train_deadline: float, eval_ceiling: float,
    expected_start_fp: str, expected_init_fp: str, evaluate_r5_milestones: bool = False,
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
    milestones_set = set(cfg["milestone_r5_steps"]) if evaluate_r5_milestones else set()
    milestone_records: Dict[str, Any] = {}
    t_start = time.monotonic()

    if 0 in milestones_set:
        t_eval_dl = time.monotonic() + eval_ceiling
        model.eval()
        milestone_records["0"] = {"rule_5_updates": 0, "total_steps": 0, **compact_population_eval(model, d5["eval_seen_examples"], f"{arm_tag}_r5step0", seed, 0, sha5, "rule_5_seen")}
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

        if tag == "rule_5" and counters["rule_steps_by_tag"].get("rule_5", 0) in milestones_set:
            r5_step = counters["rule_steps_by_tag"]["rule_5"]
            t_eval_dl = time.monotonic() + eval_ceiling
            model.eval()
            milestone_records[str(r5_step)] = {
                "rule_5_updates": r5_step, "total_steps": counters["total_optimizer_steps"],
                **compact_population_eval(model, d5["eval_seen_examples"], f"{arm_tag}_r5step{r5_step}", seed, r5_step, sha5, "rule_5_seen"),
            }
            if time.monotonic() > t_eval_dl:
                raise DeadlineExceededError(f"Milestone {r5_step} eval deadline exceeded for {arm_tag}")
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
    datasets: Dict[str, Dict[str, Any]], hashes: Dict[str, str], dp: Dict[str, Any], seed: int, step: int,
    arm_tag: str, ckpt_name: str, abs_run_dir: str, cfg: Dict[str, Any], t_eval_ceil: float,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    ev = compact_five_rule_and_comp_battery_eval(model, datasets, hashes, dp, seed, step, arm_tag, time.monotonic() + t_eval_ceil)
    mech = compute_five_rule_gate_snapshot(model, init_params, datasets["5"], cfg) if model_kind == "gated" else compute_mechanism_snapshot(model, init_params, datasets["5"])
    ckpt = capture_and_verify_checkpoint(model.state_dict(), abs_run_dir, ckpt_name)
    assert ckpt["tensor_fingerprint"] != expected_start_fp, f"Arm {ckpt_name} had zero parameter delta"
    return ev, mech, ckpt


def _build_five_rule_comparisons(
    seed: int, cfg: Dict[str, Any], miles_fresh: Dict[str, Any], miles_seq: Dict[str, Any],
    miles_der: Dict[str, Any], miles_bare: Dict[str, Any], ev_freeze_r1: Dict[str, Any],
    ev_seq_r5: Dict[str, Any], ev_der_r5: Dict[str, Any], ev_bare_r1: Dict[str, Any], ev_bare_r5: Dict[str, Any],
) -> Dict[str, Any]:
    milestone_deltas, steps_to_90 = {}, {"fresh_r5": None, "seq_r5": None, "stream_der_r5": None, "bare_stream_der_r5": None}
    thresh = cfg["gate_g_lc_threshold"]
    for step_k in cfg["milestone_r5_steps"]:
        ks = str(step_k)
        fm, sm, dm, bm = miles_fresh[ks]["metric"], miles_seq[ks]["metric"], miles_der[ks]["metric"], miles_bare[ks]["metric"]
        assert fm.denominator == sm.denominator == dm.denominator == bm.denominator == 192
        milestone_deltas[ks] = {
            "rule_5_updates": step_k, "fresh_r5_numerator": fm.numerator, "seq_r5_numerator": sm.numerator,
            "stream_der_r5_numerator": dm.numerator, "bare_stream_der_r5_numerator": bm.numerator,
            "delta_seq_minus_fresh_pp": ((sm.numerator - fm.numerator) / 192.0) * 100.0,
            "delta_stream_der_minus_fresh_pp": ((dm.numerator - fm.numerator) / 192.0) * 100.0,
            "delta_bare_der_minus_fresh_pp": ((bm.numerator - fm.numerator) / 192.0) * 100.0,
            "delta_gated_minus_bare_r5_pp": ((dm.numerator - bm.numerator) / 192.0) * 100.0,
        }
        for key, m in (("fresh_r5", fm), ("seq_r5", sm), ("stream_der_r5", dm), ("bare_stream_der_r5", bm)):
            if steps_to_90[key] is None and m.rate >= thresh:
                steps_to_90[key] = step_k

    def n192(ev: Dict[str, Any], k: str) -> int:
        m = ev["metrics"][k]
        assert m.denominator == 192
        return m.numerator

    out: Dict[str, Any] = {
        "seed": seed, "milestone_deltas": milestone_deltas, "steps_to_90pct_r5": steps_to_90,
        "final_transfer_delta_stream_der_pp": milestone_deltas[str(cfg["step_budget_per_rule"])]["delta_stream_der_minus_fresh_pp"],
        "r1_forgetting_total_seq_pp": ((n192(ev_freeze_r1, "acc_rule_1_seen") - n192(ev_seq_r5, "acc_rule_1_seen")) / 192.0) * 100.0,
        "r1_forgetting_total_stream_der_pp": ((n192(ev_freeze_r1, "acc_rule_1_seen") - n192(ev_der_r5, "acc_rule_1_seen")) / 192.0) * 100.0,
        "r1_forgetting_total_bare_der_pp": ((n192(ev_bare_r1, "acc_rule_1_seen") - n192(ev_bare_r5, "acc_rule_1_seen")) / 192.0) * 100.0,
    }
    for r in range(1, 6):
        out[f"gated_minus_bare_r{r}_pp"] = ((n192(ev_der_r5, f"acc_rule_{r}_seen") - n192(ev_bare_r5, f"acc_rule_{r}_seen")) / 192.0) * 100.0
    for ck in ("comp_6", "comp_7"):
        out[f"{ck}_seen_stream_der_pct"] = ev_der_r5["metrics"][f"acc_{ck}_seen"].pct
    return out


def run_five_rule_stream_study(
    dataset_paths: Dict[str, str], sha256_by_key: Dict[str, str], probe_dataset_path: str, sha256_probe: str,
    destination: str, run_dir: str, config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Master procedural entry point for SSRL-15 Five-Rule Overlapping Stream + Zero-Shot Composition Study."""
    cfg = copy.deepcopy(config or PROPOSED_FIVE_RULE_STREAM_CONFIG)
    verify_five_rule_stream_approval(cfg)
    assert get_input_gated_parameter_count() == cfg["expected_gated_parameter_count"] == 6658
    assert get_expected_parameter_count() == cfg["expected_bare_parameter_count"] == 6402
    assert len(set().union(*(cfg["rules"][str(r)] for r in range(1, 5)))) == 16
    assert set(cfg["rules"]["5"]) == {0, 4, 8, 12}
    abs_run_dir, abs_dest = reserve_run_directory(run_dir), os.path.abspath(destination)

    contract_meta = {
        "1": ("SSRL-01", "learnability_pilot", 42), "2": ("SSRL-02", "two_rule_diagnostic", 43),
        "3": ("SSRL-13", "three_rule_sequential_stream", 44), "4": ("SSRL-15", "five_rule_overlapping_stream", 45),
        "5": ("SSRL-15", "five_rule_overlapping_stream", 46), "6": ("SSRL-15", "five_rule_overlapping_stream", 47),
        "7": ("SSRL-15", "five_rule_overlapping_stream", 48),
    }
    all_coords = {**cfg["rules"], **cfg["composition_rules"]}
    datasets: Dict[str, Dict[str, Any]] = {}
    for k, (st, ph, pseed) in contract_meta.items():
        ds = load_and_verify_dataset(
            dataset_paths[k], expected_sha256=sha256_by_key[k],
            approved_config={**cfg, "study_id": st, "phase": ph, "seed": 101, "partition_seed": pseed, "rule_id": int(k), "rule_indices": all_coords[k]},
        )
        if ds["metadata"].get("study_id") == "SSRL-15":
            assert ds == generate_pilot_dataset_dict(tuple(all_coords[k]), int(k), 101, partition_seed=pseed, study_id=st, phase=ph)
        datasets[k] = ds

    with open(probe_dataset_path, "rb") as f:
        if hashlib.sha256(f.read()).hexdigest() != sha256_probe:
            raise ValueError("Counterfactual probe SHA-256 mismatch")
    with open(probe_dataset_path, "r", encoding="utf-8") as f:
        dp = json.load(f)
    validate_counterfactual_probe_contract(dp, datasets["1"], datasets["2"])

    diagnostics: Dict[str, Any] = {
        "study_id": cfg["study_id"], "phase": cfg["phase"], "classification": cfg["classification"],
        "status": "IN_PROGRESS", "current_stage": "INITIALIZATION", "execution_mode": EVAL_MODE_DETERMINISTIC,
        "dataset_hashes": {**sha256_by_key, "probe": sha256_probe},
        "environment": {
            "python_version": sys.version.split()[0], "torch_version": torch.__version__, "device": "cpu",
            "gate_gain": cfg["gate_gain"], "l2_init_lambda": cfg["l2_init_lambda"], "der_alpha": cfg["der_alpha"],
            "replay_buffer_size_per_rule": cfg["replay_buffer_size_per_rule"], "step_budget_per_rule": cfg["step_budget_per_rule"],
        },
        "seeds_status": {s: "NOT_RUN" for s in cfg["seeds"]}, "seeds_data": {},
        "paired_five_rule_comparisons": [], "run_directory": abs_run_dir, "destination": abs_dest,
    }
    diag_file = os.path.join(abs_run_dir, "failure_diagnostics.json")

    try:
        t_train_ceil, t_eval_ceil = float(cfg["wall_clock_ceiling_training_seconds_per_stage"]), float(cfg["wall_clock_ceiling_eval_seconds_per_stage"])
        n_steps, bsz = int(cfg["step_budget_per_rule"]), int(cfg["batch_size"])
        raw_scheds = {r: build_minibatch_schedule(datasets[str(r)]["train_examples"], n_steps, bsz, int(cfg["sampler_seeds"][str(r)])) for r in range(1, 6)}
        seq_scheds = {r: [(f"rule_{r}", r, b) for b in raw_scheds[r]] for r in range(1, 6)}
        bufs = {r: select_multi_rule_stratified_memory(datasets[str(r)]["train_examples"], cfg["rules"][str(r)], cfg["replay_buffer_size_per_rule"], cfg["replay_batch_size"]) for r in range(1, 5)}
        rb, ra = cfg["replay_ratio_new_to_old"]
        der_scheds = {r: build_multi_rule_cyclic_replay_schedule(raw_scheds[r], f"rule_{r}", r, {j: bufs[j] for j in range(1, r)}, cfg["replay_batch_size"], rb, ra) for r in range(2, 6)}
        diagnostics["buffer_ids_by_rule"] = {str(r): [int(x["example_id"]) for x in bufs[r]] for r in range(1, 5)}
        diagnostics["schedule_hashes"] = {**{f"r{r}_seq": compute_schedule_hash(seq_scheds[r]) for r in range(1, 6)}, **{f"r{r}_der": compute_schedule_hash(der_scheds[r]) for r in range(2, 6)}}
        probe_x = torch.stack([torch.tensor(ex["features"] + ex["mask"], dtype=torch.float32) for ex in datasets["1"]["eval_seen_examples"][:4]], dim=0)

        for seed in cfg["seeds"]:
            diagnostics["seeds_status"][seed] = "IN_PROGRESS"
            seed_rec: Dict[str, Any] = {"seed": seed}
            diagnostics["seeds_data"][str(seed)] = seed_rec
            torch.manual_seed(seed); m_gated_init = _instantiate_model("gated", float(cfg["gate_gain"]))
            torch.manual_seed(seed); m_bare_init = _instantiate_model("bare", float(cfg["gate_gain"]))
            m_gated_init.eval(); m_bare_init.eval()
            with torch.no_grad():
                assert torch.equal(m_gated_init(probe_x), m_bare_init(probe_x))
            init_g, fp_init_g = clone_frozen_init_params(m_gated_init)
            init_b, fp_init_b = clone_frozen_init_params(m_bare_init)
            seed_rec["fresh_init_fingerprint_gated"], seed_rec["fresh_init_fingerprint_bare"] = fp_init_g, fp_init_b

            # 1. Gated R1 + Freeze + Fresh R5
            diagnostics["current_stage"] = f"seed_{seed}_post_r1"
            m_r1, t_r1, cnt_r1, _, _ = train_five_rule_l2init_der("gated", m_gated_init.state_dict(), init_g, seq_scheds[1], datasets["5"], sha256_by_key["5"], cfg, seed, f"ig5_r1_s{seed}", time.monotonic() + t_train_ceil, t_eval_ceil, fp_init_g, fp_init_g)
            ev_r1 = compact_population_eval(m_r1, datasets["1"]["eval_seen_examples"], f"ig5_post_r1_s{seed}", seed, n_steps, sha256_by_key["1"], "rule_1_seen")
            if cfg["enforce_post_r1_learnability_gate"]:
                _assert_learnability_gate(ev_r1["metric"], seed, "post_r1", cfg["gate_g_lc_threshold"])
            ckpt_r1 = capture_and_verify_checkpoint(m_r1.state_dict(), abs_run_dir, f"checkpoint_ig5_post_r1_s{seed}.pt")
            fp_r1 = ckpt_r1["tensor_fingerprint"]
            t_r1_logits, t_r1_fp, _ = capture_frozen_teacher_logits(m_r1, bufs[1], fp_r1)
            seed_rec["post_r1"] = {"elapsed_train_seconds": t_r1, "counters": cnt_r1, "checkpoint": ckpt_r1, "post_r1_seen": ev_r1}

            diagnostics["current_stage"] = f"seed_{seed}_freeze_after_r1"
            ev_frz = compact_five_rule_and_comp_battery_eval(m_r1, datasets, sha256_by_key, dp, seed, 0, f"ig5_frz_s{seed}", time.monotonic() + t_eval_ceil)
            seed_rec["freeze_after_r1"] = {"fingerprint": fp_r1, "eval": ev_frz}

            diagnostics["current_stage"] = f"seed_{seed}_fresh_r5"
            m_fr5, t_fr5, cnt_fr5, _, miles_fr5 = train_five_rule_l2init_der("gated", m_gated_init.state_dict(), init_g, seq_scheds[5], datasets["5"], sha256_by_key["5"], cfg, seed, f"ig5_fr5_s{seed}", time.monotonic() + t_train_ceil, t_eval_ceil, fp_init_g, fp_init_g, evaluate_r5_milestones=True)
            if cfg["enforce_fresh_r5_learnability_gate"]:
                _assert_learnability_gate(miles_fr5[str(n_steps)]["metric"], seed, "fresh_r5", cfg["gate_g_lc_threshold"])
            ev_fr5, mech_fr5, ckpt_fr5 = _eval_mech_ckpt("gated", m_fr5, init_g, fp_init_g, datasets, sha256_by_key, dp, seed, n_steps, f"ig5_fr5_s{seed}", f"checkpoint_ig5_fresh_r5_s{seed}.pt", abs_run_dir, cfg, t_eval_ceil)
            seed_rec["fresh_r5"] = {"elapsed_train_seconds": t_fr5, "counters": cnt_fr5, "checkpoint": ckpt_fr5, "mechanism": mech_fr5, "milestone_evaluations": miles_fr5, "final_eval": ev_fr5}

            # 2. Gated Sequential No-Replay (R1 -> R2 -> R3 -> R4 -> R5)
            m_seq, fp_seq, seq_teachers, seq_times = m_r1, fp_r1, {1: t_r1_logits}, {}
            for r in range(2, 6):
                diagnostics["current_stage"] = f"seed_{seed}_seq_r{r}"
                m_seq, dt, cnt_s, _, miles_s = train_five_rule_l2init_der("gated", m_seq.state_dict(), init_g, seq_scheds[r], datasets["5"], sha256_by_key["5"], cfg, seed, f"ig5_sr{r}_s{seed}", time.monotonic() + t_train_ceil, t_eval_ceil, fp_seq, fp_init_g, evaluate_r5_milestones=(r == 5))
                fp_seq = compute_tensor_fingerprint_from_state_dict(m_seq.state_dict())
                seq_times[f"r{r}"] = dt
                if r <= 4:
                    seq_teachers[r], _, _ = capture_frozen_teacher_logits(m_seq, bufs[r], fp_seq)
            ev_sr5, mech_sr5, ckpt_sr5 = _eval_mech_ckpt("gated", m_seq, init_g, fp_r1, datasets, sha256_by_key, dp, seed, n_steps, f"ig5_sr5_s{seed}", f"checkpoint_ig5_seq_r5_s{seed}.pt", abs_run_dir, cfg, t_eval_ceil)
            seed_rec["seq_r5"] = {"phase_train_seconds": seq_times, "counters": cnt_s, "checkpoint": ckpt_sr5, "mechanism": mech_sr5, "milestone_evaluations": miles_s, "final_eval": ev_sr5, "buffer_logit_drifts": {f"r{j}": evaluate_buffer_logit_drift(m_seq, bufs[j], seq_teachers[j]) for j in range(1, 5)}}

            # 3. Gated DER++ Stream (R1 -> R2 -> R3 -> R4 -> R5)
            m_der, fp_der, der_teachers, der_fps, der_times = m_r1, fp_r1, {1: t_r1_logits}, {1: t_r1_fp}, {}
            for r in range(2, 5):
                diagnostics["current_stage"] = f"seed_{seed}_der_r{r}"
                m_der, dt, _, _, _ = train_five_rule_l2init_der("gated", m_der.state_dict(), init_g, der_scheds[r], datasets["5"], sha256_by_key["5"], cfg, seed, f"ig5_dr{r}_s{seed}", time.monotonic() + t_train_ceil, t_eval_ceil, fp_der, fp_init_g, teacher_by_rule=dict(der_teachers), expected_teacher_fp_by_rule=dict(der_fps))
                fp_der = compute_tensor_fingerprint_from_state_dict(m_der.state_dict())
                der_times[f"r{r}"] = dt
                der_teachers[r], der_fps[r], _ = capture_frozen_teacher_logits(m_der, bufs[r], fp_der)
            ev_dr4, mech_dr4, ckpt_dr4 = _eval_mech_ckpt("gated", m_der, init_g, fp_r1, datasets, sha256_by_key, dp, seed, len(der_scheds[4]), f"ig5_dr4_s{seed}", f"checkpoint_ig5_der_post_r4_s{seed}.pt", abs_run_dir, cfg, t_eval_ceil)
            seed_rec["der_post_r4"] = {"checkpoint": ckpt_dr4, "mechanism": mech_dr4, "final_eval": ev_dr4}

            diagnostics["current_stage"] = f"seed_{seed}_stream_der_r5"
            m_dr5, dt5, cnt_d5, _, miles_d5 = train_five_rule_l2init_der("gated", m_der.state_dict(), init_g, der_scheds[5], datasets["5"], sha256_by_key["5"], cfg, seed, f"ig5_dr5_s{seed}", time.monotonic() + t_train_ceil, t_eval_ceil, fp_der, fp_init_g, evaluate_r5_milestones=True, teacher_by_rule=dict(der_teachers), expected_teacher_fp_by_rule=dict(der_fps))
            der_times["r5"] = dt5
            ev_dr5, mech_dr5, ckpt_dr5 = _eval_mech_ckpt("gated", m_dr5, init_g, fp_der, datasets, sha256_by_key, dp, seed, len(der_scheds[5]), f"ig5_dr5_s{seed}", f"checkpoint_ig5_stream_der_r5_s{seed}.pt", abs_run_dir, cfg, t_eval_ceil)
            seed_rec["stream_der_r5"] = {"phase_train_seconds": der_times, "counters": cnt_d5, "checkpoint": ckpt_dr5, "mechanism": mech_dr5, "milestone_evaluations": miles_d5, "final_eval": ev_dr5, "buffer_logit_drifts": {f"r{j}": evaluate_buffer_logit_drift(m_dr5, bufs[j], der_teachers[j]) for j in range(1, 5)}}

            # 4. Matched Bare Control DER++ Stream (R1 -> R2 -> R3 -> R4 -> R5)
            diagnostics["current_stage"] = f"seed_{seed}_bare_post_r1"
            mb_r1, tb1, _, _, _ = train_five_rule_l2init_der("bare", m_bare_init.state_dict(), init_b, seq_scheds[1], datasets["5"], sha256_by_key["5"], cfg, seed, f"ig5_br1_s{seed}", time.monotonic() + t_train_ceil, t_eval_ceil, fp_init_b, fp_init_b)
            if cfg["enforce_bare_post_r1_learnability_gate"]:
                _assert_learnability_gate(compact_population_eval(mb_r1, datasets["1"]["eval_seen_examples"], f"ig5_br1_g_s{seed}", seed, n_steps, sha256_by_key["1"], "rule_1_seen")["metric"], seed, "bare_post_r1", cfg["gate_g_lc_threshold"])
            ev_br1, mech_br1, ckpt_br1 = _eval_mech_ckpt("bare", mb_r1, init_b, fp_init_b, datasets, sha256_by_key, dp, seed, n_steps, f"ig5_br1_s{seed}", f"checkpoint_ig5_bare_post_r1_s{seed}.pt", abs_run_dir, cfg, t_eval_ceil)
            fp_br1 = ckpt_br1["tensor_fingerprint"]
            tb1_logits, tb1_fp, _ = capture_frozen_teacher_logits(mb_r1, bufs[1], fp_br1)
            seed_rec["bare_post_r1"] = {"elapsed_train_seconds": tb1, "checkpoint": ckpt_br1, "mechanism": mech_br1, "final_eval": ev_br1}

            mb, fp_b, b_teachers, b_fps, b_times = mb_r1, fp_br1, {1: tb1_logits}, {1: tb1_fp}, {"r1": tb1}
            for r in range(2, 6):
                diagnostics["current_stage"] = f"seed_{seed}_bare_der_r{r}"
                mb, dt, cnt_b, _, miles_b = train_five_rule_l2init_der("bare", mb.state_dict(), init_b, der_scheds[r], datasets["5"], sha256_by_key["5"], cfg, seed, f"ig5_bdr{r}_s{seed}", time.monotonic() + t_train_ceil, t_eval_ceil, fp_b, fp_init_b, evaluate_r5_milestones=(r == 5), teacher_by_rule=dict(b_teachers), expected_teacher_fp_by_rule=dict(b_fps))
                fp_b = compute_tensor_fingerprint_from_state_dict(mb.state_dict())
                b_times[f"r{r}"] = dt
                if r <= 4:
                    b_teachers[r], b_fps[r], _ = capture_frozen_teacher_logits(mb, bufs[r], fp_b)
            ev_br5, mech_br5, ckpt_br5 = _eval_mech_ckpt("bare", mb, init_b, fp_br1, datasets, sha256_by_key, dp, seed, len(der_scheds[5]), f"ig5_br5_s{seed}", f"checkpoint_ig5_bare_stream_der_r5_s{seed}.pt", abs_run_dir, cfg, t_eval_ceil)
            seed_rec["bare_stream_der_r5"] = {"phase_train_seconds": b_times, "counters": cnt_b, "checkpoint": ckpt_br5, "mechanism": mech_br5, "milestone_evaluations": miles_b, "final_eval": ev_br5, "buffer_logit_drifts": {f"r{j}": evaluate_buffer_logit_drift(mb, bufs[j], b_teachers[j]) for j in range(1, 5)}}

            comp = _build_five_rule_comparisons(seed, cfg, miles_fr5, miles_s, miles_d5, miles_b, ev_frz, ev_sr5, ev_dr5, ev_br1, ev_br5)
            seed_rec["five_rule_comparison"] = comp
            diagnostics["paired_five_rule_comparisons"].append(comp)
            diagnostics["seeds_status"][seed] = "COMPLETED"

        comps = diagnostics["paired_five_rule_comparisons"]
        summary = {k: _summarize_scalar_list([c[k] for c in comps]) for k in ("r1_forgetting_total_seq_pp", "r1_forgetting_total_stream_der_pp", "r1_forgetting_total_bare_der_pp", "final_transfer_delta_stream_der_pp", "comp_6_seen_stream_der_pct", "comp_7_seen_stream_der_pct", *(f"gated_minus_bare_r{r}_pp" for r in range(1, 6)))}
        diagnostics["summary_statistics"], diagnostics["status"], diagnostics["current_stage"] = summary, "COMPLETED", "COMPLETED"
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
    parser = argparse.ArgumentParser(description="SSRL-15 5-Rule Overlapping Stream + Zero-Shot Composition Runner")
    for k, dft in [("dataset_1", None), ("dataset_2", None), ("dataset_3", None), ("dataset_4", None), ("dataset_5", None), ("comp_6", None), ("comp_7", None), ("probe_dataset", None), ("run_dir", None), ("destination", None), ("sha256_1", SHA256_RULE_1_DEFAULT), ("sha256_2", SHA256_RULE_2_DEFAULT), ("sha256_3", SHA256_RULE_3_DEFAULT), ("sha256_4", SHA256_RULE_4_DEFAULT), ("sha256_5", SHA256_RULE_5_DEFAULT), ("sha256_6", SHA256_COMP_6_DEFAULT), ("sha256_7", SHA256_COMP_7_DEFAULT), ("sha256_probe", SHA256_PROBE_DEFAULT)]:
        parser.add_argument(f"--{k}", required=(dft is None), default=dft)
    args = parser.parse_args()
    run_five_rule_stream_study(
        dataset_paths={"1": args.dataset_1, "2": args.dataset_2, "3": args.dataset_3, "4": args.dataset_4, "5": args.dataset_5, "6": args.comp_6, "7": args.comp_7},
        sha256_by_key={"1": args.sha256_1, "2": args.sha256_2, "3": args.sha256_3, "4": args.sha256_4, "5": args.sha256_5, "6": args.sha256_6, "7": args.sha256_7},
        probe_dataset_path=args.probe_dataset, sha256_probe=args.sha256_probe, destination=args.destination, run_dir=args.run_dir,
    )


if __name__ == "__main__":
    main()
