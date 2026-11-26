"""
research/lifelong_01/run_ssrl_three_rule_stream.py -- Procedural Runner for SSRL-13.

Evaluates L2-Init (lambda = 0.01 toward fresh theta_0) + DER++ (alpha = 0.5, M=64/rule) on a
3-rule sequential continual learning stream (Rule 1 -> Rule 2 -> Rule 3) across Seeds 102, 103, 104:
- FREEZE-AFTER-R1 (standing control after Rule 1)
- fresh_r3 (matched fresh L2-Init comparator on Rule 3, 1,000 updates)
- seq_r1_r2_r3 (sequential L2-Init without replay across R1 -> R2 -> R3)
- stream_der_m64 (sequential L2-Init + alternating M=64/rule DER++ 4:1 across R1 -> R2 -> R3)
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from research.lifelong_01.ssrl_checkpoints import capture_and_verify_checkpoint, compute_tensor_fingerprint_from_state_dict
from research.lifelong_01.ssrl_model import SSRLInstructionMLP, get_expected_parameter_count
from research.lifelong_01.ssrl_generator import generate_pilot_dataset_dict
from research.lifelong_01.ssrl_counterfactual_probe import validate_counterfactual_probe_contract, evaluate_counterfactual_probe
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


SHA256_RULE_1_DEFAULT = "00a21efad1dc07b826198d99b0b702aedddad57b6226f99301afbf16327fd0b7"
SHA256_RULE_2_DEFAULT = "27e813e73bf66f51e1bff07f7d2f19945e830afc6d5876b0cb4d11136cda4cd4"
SHA256_RULE_3_DEFAULT = "3a9b7da13fe90bc4da4d4da50ce40ce0cb4252489d9ab76f6fb375443c7dd449"
SHA256_PROBE_DEFAULT = "1c07d50fed772cfcff86e3bf30cc74a95cdc24d190ec767aa15365671fe06319"

PROPOSED_THREE_RULE_STREAM_CONFIG: Dict[str, Any] = {
    "study_id": "SSRL-13", "phase": "three_rule_sequential_stream",
    "classification": "multi_rule_l2init_der_m64_three_seed", "gate_approval_status": "APPROVED",
    "model_class": "SSRLInstructionMLP", "expected_parameter_count": 6402, "seeds": [102, 103, 104],
    "minibatch_sampler_seed_r1": 2026, "minibatch_sampler_seed_r2": 2027, "minibatch_sampler_seed_r3": 2028,
    "rule_1_id": 1, "rule_1_indices": [0, 1, 2, 3], "rule_2_id": 2, "rule_2_indices": [4, 5, 6, 7],
    "rule_3_id": 3, "rule_3_indices": [8, 9, 10, 11], "train_pool_size": 384, "batch_size": 16,
    "step_budget_r1": 2000, "step_budget_r2": 1000, "step_budget_r3": 1000,
    "replay_buffer_size_per_rule": 64, "replay_batch_size": 16, "replay_ratio_new_to_old": [4, 1],
    "milestone_r3_steps": [0, 25, 50, 100, 250, 500, 750, 1000],
    "learning_rate": 0.001, "betas": [0.9, 0.999], "eps": 1e-08, "weight_decay": 0.0,
    "l2_init_lambda": 0.01, "der_alpha": 0.5, "gate_g_lc_threshold": 0.90,
    "enforce_post_r1_learnability_gate": True, "enforce_fresh_r3_learnability_gate": True,
    "wall_clock_ceiling_training_seconds_per_stage": 60.0, "wall_clock_ceiling_eval_seconds_per_stage": 60.0,
}


def verify_three_rule_specification_approval(config: Dict[str, Any]) -> None:
    """AGENTS.md Section 2 / Section 8: Strictly blocks unapproved execution."""
    status = config.get("gate_approval_status")
    if status != "APPROVED":
        raise SpecificationApprovalError(f"SSRL-13 spec status is '{status}', not 'APPROVED'. Execution forbidden.")


def select_multi_rule_stratified_memory(
    train_examples: List[Dict[str, Any]], rule_indices: List[int], buffer_size: int = 64, replay_batch_size: int = 16,
) -> List[Dict[str, Any]]:
    """
    Selects a stratified pattern-diverse M-example buffer across any non-empty per-class pattern count
    (e.g., 6/6 on Rule 1 or 5/7 on Rule 2), packed into 8/8 class-balanced replay_batch_size chunks.
    """
    if buffer_size < 16 or buffer_size % replay_batch_size != 0 or replay_batch_size % 2 != 0:
        raise ValueError(f"Invalid buffer/batch configuration: {buffer_size}, {replay_batch_size}")
    patterns_by_class: Dict[int, Dict[Tuple[int, ...], List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for ex in train_examples:
        cls = int(ex["target_class"])
        patterns_by_class[cls][tuple(ex["features"][i] for i in rule_indices)].append(ex)

    per_class = buffer_size // 2
    rr_by_class: Dict[int, List[Dict[str, Any]]] = {}
    for cls in (0, 1):
        sorted_patterns = sorted(patterns_by_class[cls].keys())
        if len(sorted_patterns) == 0:
            raise ValueError(f"Class {cls} has zero active patterns")
        base_q, rem = divmod(per_class, len(sorted_patterns))
        per_pat, max_q = [], 0
        for p_idx, pat in enumerate(sorted_patterns):
            quota = base_q + (1 if p_idx < rem else 0)
            exs = sorted(patterns_by_class[cls][pat], key=lambda x: int(x["example_id"]))
            if len(exs) < quota:
                raise ValueError(f"Class {cls} pattern {pat} has {len(exs)} < {quota}")
            per_pat.append(exs[:quota])
            max_q = max(max_q, quota)
        rr = [p[r] for r in range(max_q) for p in per_pat if r < len(p)]
        assert len(rr) == per_class
        rr_by_class[cls] = rr

    half = replay_batch_size // 2
    selected: List[Dict[str, Any]] = []
    for c in range(buffer_size // replay_batch_size):
        c0 = sorted(rr_by_class[0][c * half : (c + 1) * half], key=lambda x: int(x["example_id"]))
        c1 = sorted(rr_by_class[1][c * half : (c + 1) * half], key=lambda x: int(x["example_id"]))
        selected.extend(c0 + c1)
    assert len(selected) == buffer_size and len({int(ex["example_id"]) for ex in selected}) == buffer_size
    assert sum(1 for ex in selected if int(ex["target_class"]) == 0) == per_class
    return selected


def build_multi_rule_cyclic_replay_schedule(
    schedule_new: List[List[Dict[str, Any]]], new_tag: str, new_rule_id: int,
    replay_buffers_by_rule: Dict[int, List[Dict[str, Any]]], replay_batch_size: int = 16,
    ratio_new: int = 4, ratio_replay: int = 1,
) -> List[Tuple[str, int, List[Dict[str, Any]]]]:
    """Interleaves new batches with cyclic chunks alternating across sorted prior rule IDs."""
    if ratio_new <= 0 or ratio_replay <= 0 or len(schedule_new) % ratio_new != 0 or not replay_buffers_by_rule:
        raise ValueError("Invalid multi-rule replay schedule parameters")
    prior_ids = sorted(int(rid) for rid in replay_buffers_by_rule.keys())
    chunks_by_rule = {}
    per_rule_idx = {rid: 0 for rid in prior_ids}
    for rid in prior_ids:
        buf = replay_buffers_by_rule[rid]
        if len(buf) == 0 or len(buf) % replay_batch_size != 0:
            raise ValueError(f"Rule {rid} buffer length ({len(buf)}) invalid for batch {replay_batch_size}")
        chunks_by_rule[rid] = [buf[i : i + replay_batch_size] for i in range(0, len(buf), replay_batch_size)]

    schedule: List[Tuple[str, int, List[Dict[str, Any]]]] = []
    b_idx, r_step = 0, 0
    for _ in range(len(schedule_new) // ratio_new):
        for _ in range(ratio_new):
            schedule.append((new_tag, new_rule_id, schedule_new[b_idx]))
            b_idx += 1
        for _ in range(ratio_replay):
            rid = prior_ids[r_step % len(prior_ids)]
            chunks = chunks_by_rule[rid]
            schedule.append(("replay", rid, chunks[per_rule_idx[rid] % len(chunks)]))
            per_rule_idx[rid] += 1
            r_step += 1
    return schedule


def compact_three_rule_battery_eval(
    model: SSRLInstructionMLP, d1: Dict[str, Any], d2: Dict[str, Any], d3: Dict[str, Any], dp: Dict[str, Any],
    seed: int, step: int, arm_tag: str, sha1: str, sha2: str, sha3: str, eval_deadline: float,
) -> Dict[str, Any]:
    """Evaluates all 6 split populations (R1/R2/R3 seen & unseen) plus R1/R2 counterfactual probe."""
    if time.monotonic() > eval_deadline:
        raise DeadlineExceededError(f"Evaluation deadline exceeded before {arm_tag}")
    model.eval()
    assert not model.training
    t_start = time.monotonic()
    splits = [
        ("rule_1_seen", d1["eval_seen_examples"], sha1, 192), ("rule_1_unseen", d1["eval_unseen_examples"], sha1, 64),
        ("rule_2_seen", d2["eval_seen_examples"], sha2, 192), ("rule_2_unseen", d2["eval_unseen_examples"], sha2, 64),
        ("rule_3_seen", d3["eval_seen_examples"], sha3, 192), ("rule_3_unseen", d3["eval_unseen_examples"], sha3, 64),
    ]
    metrics, per_unit = {}, {}
    for tag, exs, d_hash, exp_n in splits:
        assert len(exs) == exp_n, f"{tag} expected {exp_n}, got {len(exs)}"
        res = compact_population_eval(model, exs, arm_tag, seed, step, d_hash, tag)
        metrics[f"acc_{tag}"] = res["metric"]
        per_unit[tag] = res["per_unit_outcomes"]
    probe_full = evaluate_counterfactual_probe(model=model, probe_examples=dp["probe_examples"], device=torch.device("cpu"))
    compact_pairs = [{k: v for k, v in rec.items() if k != "features"} for rec in probe_full["pair_records"]]
    assert len(compact_pairs) == 128
    if time.monotonic() > eval_deadline:
        raise DeadlineExceededError(f"Evaluation deadline exceeded after {arm_tag}")
    return {
        "elapsed_eval_seconds": time.monotonic() - t_start, "metrics": metrics, "per_unit_populations": per_unit,
        "counterfactual_probe_r1_r2": {
            "overall_metrics": probe_full["overall_metrics"], "strata_metrics": probe_full["strata_metrics"], "pair_records": compact_pairs,
        },
    }


def train_multi_rule_l2init_der_schedule(
    initial_model_state: Dict[str, torch.Tensor], init_params: Dict[str, torch.Tensor],
    tagged_schedule: List[Tuple[str, int, List[Dict[str, Any]]]], d3: Dict[str, Any], sha3: str,
    config: Dict[str, Any], seed: int, arm_tag: str, train_deadline: float, eval_ceiling: float,
    expected_start_fp: str, expected_init_fp: str, evaluate_r3_milestones: bool = False,
    teacher_by_rule: Optional[Dict[int, Dict[int, torch.Tensor]]] = None,
    expected_teacher_fp_by_rule: Optional[Dict[int, str]] = None,
) -> Tuple[SSRLInstructionMLP, float, Dict[str, Any], str, Dict[str, Any]]:
    """Trains SSRLInstructionMLP with Adam, L2-Init toward init_params, and per-rule DER++ replay."""
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

    model = SSRLInstructionMLP(input_dim=32, hidden_dim=64, output_dim=2)
    assert get_expected_parameter_count() == config["expected_parameter_count"] == 6402
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


def _build_three_rule_comparisons(
    seed: int, cfg: Dict[str, Any], miles_fresh: Dict[str, Any], miles_seq: Dict[str, Any], miles_der: Dict[str, Any],
    ev_freeze_r1: Dict[str, Any], ev_seq_r2: Dict[str, Any], ev_seq_r3: Dict[str, Any],
    ev_der_r2: Dict[str, Any], ev_der_r3: Dict[str, Any],
) -> Dict[str, Any]:
    """Computes aligned R3 transfer deltas, steps-to-90%, cumulative R1 forgetting, and R2 forgetting."""
    milestone_deltas, steps_to_90 = {}, {"fresh_r3": None, "seq_r3": None, "stream_der_r3": None}
    thresh = cfg["gate_g_lc_threshold"]
    for step_k in cfg["milestone_r3_steps"]:
        k_str = str(step_k)
        fm, sm, dm = miles_fresh[k_str]["metric"], miles_seq[k_str]["metric"], miles_der[k_str]["metric"]
        assert fm.denominator == sm.denominator == dm.denominator == 192
        milestone_deltas[k_str] = {
            "rule_3_updates": step_k, "fresh_r3_numerator": fm.numerator, "fresh_r3_pct": fm.pct,
            "seq_r3_numerator": sm.numerator, "seq_r3_pct": sm.pct, "stream_der_r3_numerator": dm.numerator, "stream_der_r3_pct": dm.pct,
            "delta_seq_minus_fresh_pp": ((sm.numerator - fm.numerator) / 192.0) * 100.0,
            "delta_stream_der_minus_fresh_pp": ((dm.numerator - fm.numerator) / 192.0) * 100.0,
        }
        for key, m in (("fresh_r3", fm), ("seq_r3", sm), ("stream_der_r3", dm)):
            if steps_to_90[key] is None and m.rate >= thresh:
                steps_to_90[key] = step_k

    def n192(ev: Dict[str, Any], key: str) -> int:
        m = ev["metrics"][key]
        assert m.denominator == 192
        return m.numerator

    r1_base = n192(ev_freeze_r1, "acc_rule_1_seen")
    r1_seq_p2, r1_seq_p3 = n192(ev_seq_r2, "acc_rule_1_seen"), n192(ev_seq_r3, "acc_rule_1_seen")
    r2_seq_p2, r2_seq_p3 = n192(ev_seq_r2, "acc_rule_2_seen"), n192(ev_seq_r3, "acc_rule_2_seen")
    r1_der_p2, r1_der_p3 = n192(ev_der_r2, "acc_rule_1_seen"), n192(ev_der_r3, "acc_rule_1_seen")
    r2_der_p2, r2_der_p3 = n192(ev_der_r2, "acc_rule_2_seen"), n192(ev_der_r3, "acc_rule_2_seen")
    r3_seq, r3_der = n192(ev_seq_r3, "acc_rule_3_seen"), n192(ev_der_r3, "acc_rule_3_seen")
    final_k = str(cfg["step_budget_r3"])
    return {
        "seed": seed, "milestone_deltas": milestone_deltas, "steps_to_90pct_r3": steps_to_90,
        "final_transfer_delta_seq_pp": milestone_deltas[final_k]["delta_seq_minus_fresh_pp"],
        "final_transfer_delta_stream_der_pp": milestone_deltas[final_k]["delta_stream_der_minus_fresh_pp"],
        "r1_forgetting_total_seq_pp": ((r1_base - r1_seq_p3) / 192.0) * 100.0,
        "r1_forgetting_phase3_seq_pp": ((r1_seq_p2 - r1_seq_p3) / 192.0) * 100.0,
        "r2_forgetting_phase3_seq_pp": ((r2_seq_p2 - r2_seq_p3) / 192.0) * 100.0,
        "r1_forgetting_total_stream_der_pp": ((r1_base - r1_der_p3) / 192.0) * 100.0,
        "r1_forgetting_phase3_stream_der_pp": ((r1_der_p2 - r1_der_p3) / 192.0) * 100.0,
        "r2_forgetting_phase3_stream_der_pp": ((r2_der_p2 - r2_der_p3) / 192.0) * 100.0,
        "stream_minus_seq_r1_pp": ((r1_der_p3 - r1_seq_p3) / 192.0) * 100.0,
        "stream_minus_seq_r2_pp": ((r2_der_p3 - r2_seq_p3) / 192.0) * 100.0,
        "stream_minus_seq_r3_pp": ((r3_der - r3_seq) / 192.0) * 100.0,
    }


def _eval_and_ckpt(
    model: SSRLInstructionMLP, init_params: Dict[str, torch.Tensor], expected_start_fp: str,
    d1: Dict[str, Any], d2: Dict[str, Any], d3: Dict[str, Any], dp: Dict[str, Any],
    seed: int, step: int, arm_tag: str, ckpt_name: str, abs_run_dir: str,
    sha1: str, sha2: str, sha3: str, t_eval_ceil: float,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Evaluates 3-rule battery, snapshots mechanism, saves verified checkpoint, and asserts nonzero delta."""
    ev = compact_three_rule_battery_eval(model, d1, d2, d3, dp, seed, step, arm_tag, sha1, sha2, sha3, time.monotonic() + t_eval_ceil)
    mech = compute_mechanism_snapshot(model, init_params, d3)
    ckpt = capture_and_verify_checkpoint(model.state_dict(), abs_run_dir, ckpt_name)
    assert ckpt["tensor_fingerprint"] != expected_start_fp, f"Arm {ckpt_name} had zero parameter delta"
    return ev, mech, ckpt


def run_three_rule_stream_study(
    dataset_1_path: str, sha256_1: str, dataset_2_path: str, sha256_2: str,
    dataset_3_path: str, sha256_3: str, probe_dataset_path: str, sha256_probe: str,
    destination: str, run_dir: str, config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Master procedural entry point for SSRL-13 Three-Rule Sequential Continual Learning Study."""
    cfg = copy.deepcopy(config or PROPOSED_THREE_RULE_STREAM_CONFIG)
    verify_three_rule_specification_approval(cfg)
    assert get_expected_parameter_count() == cfg["expected_parameter_count"] == 6402
    assert len(set(cfg["rule_1_indices"]) | set(cfg["rule_2_indices"]) | set(cfg["rule_3_indices"])) == 12
    abs_run_dir = reserve_run_directory(run_dir)
    abs_dest = os.path.abspath(destination)

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
            "l2_init_lambda": cfg["l2_init_lambda"], "der_alpha": cfg["der_alpha"],
            "replay_buffer_size_per_rule": cfg["replay_buffer_size_per_rule"], "replay_batch_size": cfg["replay_batch_size"],
        },
        "seeds_status": {s: "NOT_RUN" for s in cfg["seeds"]}, "seeds_data": {},
        "paired_three_rule_comparisons": [], "run_directory": abs_run_dir, "destination": abs_dest,
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

        for seed in cfg["seeds"]:
            diagnostics["seeds_status"][seed] = "IN_PROGRESS"
            seed_rec: Dict[str, Any] = {"seed": seed}
            diagnostics["seeds_data"][str(seed)] = seed_rec
            torch.manual_seed(seed)
            model_init = SSRLInstructionMLP(input_dim=32, hidden_dim=64, output_dim=2)
            init_params, fp_init = clone_frozen_init_params(model_init)
            seed_rec["fresh_init_fingerprint"] = fp_init

            diagnostics["current_stage"] = f"seed_{seed}_phase_r1"
            m_r1, t_r1, cnt_r1, _, _ = train_multi_rule_l2init_der_schedule(
                model_init.state_dict(), init_params, s1, d3, sha256_3, cfg, seed, f"r3_post_r1_s{seed}", time.monotonic() + t_train_ceil, t_eval_ceil, fp_init, fp_init,
            )
            post_r1_eval = compact_population_eval(m_r1, d1["eval_seen_examples"], f"r3_post_r1_s{seed}", seed, cfg["step_budget_r1"], sha256_1, "rule_1_seen")
            if cfg["enforce_post_r1_learnability_gate"]:
                _assert_learnability_gate(post_r1_eval["metric"], seed, "post_r1", cfg["gate_g_lc_threshold"])
            ckpt_r1 = capture_and_verify_checkpoint(m_r1.state_dict(), abs_run_dir, f"checkpoint_r3_post_r1_s{seed}.pt")
            fp_r1 = ckpt_r1["tensor_fingerprint"]
            assert fp_r1 != fp_init
            t_r1_logits, t_r1_fp, t_r1_floats = capture_frozen_teacher_logits(m_r1, buf_r1, fp_r1)
            seed_rec["post_r1"] = {
                "elapsed_train_seconds": t_r1, "counters": cnt_r1, "checkpoint": ckpt_r1, "post_r1_seen": post_r1_eval,
                "teacher_logits_fp_r1": t_r1_fp, "teacher_logits_num_floats_r1": t_r1_floats, "teacher_logits_bytes_r1": t_r1_floats * 4,
            }

            diagnostics["current_stage"] = f"seed_{seed}_freeze_after_r1"
            ev_freeze_r1 = compact_three_rule_battery_eval(m_r1, d1, d2, d3, dp, seed, 0, f"r3_freeze_r1_s{seed}", sha256_1, sha256_2, sha256_3, time.monotonic() + t_eval_ceil)
            seed_rec["freeze_after_r1"] = {"fingerprint": fp_r1, "eval": ev_freeze_r1}

            diagnostics["current_stage"] = f"seed_{seed}_fresh_r3"
            m_fresh, t_fresh, cnt_fresh, _, miles_fresh = train_multi_rule_l2init_der_schedule(
                model_init.state_dict(), init_params, s3_seq, d3, sha256_3, cfg, seed, f"r3_fresh_s{seed}", time.monotonic() + t_train_ceil, t_eval_ceil, fp_init, fp_init, evaluate_r3_milestones=True,
            )
            if cfg["enforce_fresh_r3_learnability_gate"]:
                _assert_learnability_gate(miles_fresh[str(cfg["step_budget_r3"])]["metric"], seed, "fresh_r3", cfg["gate_g_lc_threshold"])
            ev_fresh, mech_fresh, ckpt_fresh = _eval_and_ckpt(m_fresh, init_params, fp_init, d1, d2, d3, dp, seed, len(s3_seq), f"r3_fresh_final_s{seed}", f"checkpoint_r3_fresh_r3_s{seed}.pt", abs_run_dir, sha256_1, sha256_2, sha256_3, t_eval_ceil)
            seed_rec["fresh_r3"] = {"elapsed_train_seconds": t_fresh, "counters": cnt_fresh, "checkpoint": ckpt_fresh, "mechanism": mech_fresh, "milestone_evaluations": miles_fresh, "final_eval": ev_fresh}

            diagnostics["current_stage"] = f"seed_{seed}_seq_post_r2"
            m_seq_r2, t_sr2, cnt_sr2, _, _ = train_multi_rule_l2init_der_schedule(m_r1.state_dict(), init_params, s2_seq, d3, sha256_3, cfg, seed, f"r3_seq_r2_s{seed}", time.monotonic() + t_train_ceil, t_eval_ceil, fp_r1, fp_init)
            ev_seq_r2, mech_seq_r2, ckpt_seq_r2 = _eval_and_ckpt(m_seq_r2, init_params, fp_r1, d1, d2, d3, dp, seed, len(s2_seq), f"r3_seq_r2_final_s{seed}", f"checkpoint_r3_seq_post_r2_s{seed}.pt", abs_run_dir, sha256_1, sha256_2, sha256_3, t_eval_ceil)
            fp_seq_r2 = ckpt_seq_r2["tensor_fingerprint"]
            t_seq_r2_logits, _, _ = capture_frozen_teacher_logits(m_seq_r2, buf_r2, fp_seq_r2)

            diagnostics["current_stage"] = f"seed_{seed}_seq_r3"
            m_seq_r3, t_sr3, cnt_sr3, _, miles_seq_r3 = train_multi_rule_l2init_der_schedule(m_seq_r2.state_dict(), init_params, s3_seq, d3, sha256_3, cfg, seed, f"r3_seq_r3_s{seed}", time.monotonic() + t_train_ceil, t_eval_ceil, fp_seq_r2, fp_init, evaluate_r3_milestones=True)
            ev_seq_r3, mech_seq_r3, ckpt_seq_r3 = _eval_and_ckpt(m_seq_r3, init_params, fp_seq_r2, d1, d2, d3, dp, seed, len(s3_seq), f"r3_seq_r3_final_s{seed}", f"checkpoint_r3_seq_r3_s{seed}.pt", abs_run_dir, sha256_1, sha256_2, sha256_3, t_eval_ceil)
            seed_rec["seq_post_r2"] = {"elapsed_train_seconds": t_sr2, "counters": cnt_sr2, "checkpoint": ckpt_seq_r2, "mechanism": mech_seq_r2, "final_eval": ev_seq_r2}
            seed_rec["seq_r3"] = {
                "elapsed_train_seconds": t_sr3, "counters": cnt_sr3, "checkpoint": ckpt_seq_r3, "mechanism": mech_seq_r3, "milestone_evaluations": miles_seq_r3, "final_eval": ev_seq_r3,
                "buffer_logit_drift_r1": evaluate_buffer_logit_drift(m_seq_r3, buf_r1, t_r1_logits),
                "buffer_logit_drift_r2": evaluate_buffer_logit_drift(m_seq_r3, buf_r2, t_seq_r2_logits),
            }

            diagnostics["current_stage"] = f"seed_{seed}_der_post_r2"
            m_der_r2, t_dr2, cnt_dr2, _, _ = train_multi_rule_l2init_der_schedule(m_r1.state_dict(), init_params, s2_der, d3, sha256_3, cfg, seed, f"r3_der_r2_s{seed}", time.monotonic() + t_train_ceil, t_eval_ceil, fp_r1, fp_init, teacher_by_rule={1: t_r1_logits}, expected_teacher_fp_by_rule={1: t_r1_fp})
            ev_der_r2, mech_der_r2, ckpt_der_r2 = _eval_and_ckpt(m_der_r2, init_params, fp_r1, d1, d2, d3, dp, seed, len(s2_der), f"r3_der_r2_final_s{seed}", f"checkpoint_r3_der_post_r2_s{seed}.pt", abs_run_dir, sha256_1, sha256_2, sha256_3, t_eval_ceil)
            fp_der_r2 = ckpt_der_r2["tensor_fingerprint"]
            t_der_r2_logits, t_der_r2_fp, t_der_r2_floats = capture_frozen_teacher_logits(m_der_r2, buf_r2, fp_der_r2)

            diagnostics["current_stage"] = f"seed_{seed}_stream_der_r3"
            m_der_r3, t_dr3, cnt_dr3, _, miles_der_r3 = train_multi_rule_l2init_der_schedule(m_der_r2.state_dict(), init_params, s3_der, d3, sha256_3, cfg, seed, f"r3_stream_der_s{seed}", time.monotonic() + t_train_ceil, t_eval_ceil, fp_der_r2, fp_init, evaluate_r3_milestones=True, teacher_by_rule={1: t_r1_logits, 2: t_der_r2_logits}, expected_teacher_fp_by_rule={1: t_r1_fp, 2: t_der_r2_fp})
            ev_der_r3, mech_der_r3, ckpt_der_r3 = _eval_and_ckpt(m_der_r3, init_params, fp_der_r2, d1, d2, d3, dp, seed, len(s3_der), f"r3_stream_der_final_s{seed}", f"checkpoint_r3_stream_der_r3_s{seed}.pt", abs_run_dir, sha256_1, sha256_2, sha256_3, t_eval_ceil)
            seed_rec["der_post_r2"] = {
                "elapsed_train_seconds": t_dr2, "counters": cnt_dr2, "checkpoint": ckpt_der_r2, "mechanism": mech_der_r2,
                "final_eval": ev_der_r2, "teacher_logits_fp_r2": t_der_r2_fp, "teacher_logits_num_floats_r2": t_der_r2_floats, "teacher_logits_bytes_r2": t_der_r2_floats * 4,
            }
            seed_rec["stream_der_r3"] = {
                "elapsed_train_seconds": t_dr3, "counters": cnt_dr3, "checkpoint": ckpt_der_r3, "mechanism": mech_der_r3, "milestone_evaluations": miles_der_r3, "final_eval": ev_der_r3,
                "buffer_logit_drift_r1": evaluate_buffer_logit_drift(m_der_r3, buf_r1, t_r1_logits),
                "buffer_logit_drift_r2": evaluate_buffer_logit_drift(m_der_r3, buf_r2, t_der_r2_logits),
            }

            comp = _build_three_rule_comparisons(seed, cfg, miles_fresh, miles_seq_r3, miles_der_r3, ev_freeze_r1, ev_seq_r2, ev_seq_r3, ev_der_r2, ev_der_r3)
            seed_rec["three_rule_comparison"] = comp
            diagnostics["paired_three_rule_comparisons"].append(comp)
            diagnostics["seeds_status"][seed] = "COMPLETED"

        comps = diagnostics["paired_three_rule_comparisons"]
        summary: Dict[str, Any] = {
            f"transfer_delta_stream_der_r3_step{k}_pp": _summarize_scalar_list([c["milestone_deltas"][str(k)]["delta_stream_der_minus_fresh_pp"] for c in comps])
            for k in cfg["milestone_r3_steps"]
        }
        for key in (
            "r1_forgetting_total_seq_pp", "r2_forgetting_phase3_seq_pp",
            "r1_forgetting_total_stream_der_pp", "r2_forgetting_phase3_stream_der_pp",
            "stream_minus_seq_r1_pp", "stream_minus_seq_r2_pp", "stream_minus_seq_r3_pp",
        ):
            summary[key] = _summarize_scalar_list([c[key] for c in comps])
        summary["buffer_logit_drift_mse_r1_stream_der"] = _summarize_scalar_list([diagnostics["seeds_data"][str(s)]["stream_der_r3"]["buffer_logit_drift_r1"]["mse_logit_drift"] for s in cfg["seeds"]])
        summary["buffer_logit_drift_mse_r2_stream_der"] = _summarize_scalar_list([diagnostics["seeds_data"][str(s)]["stream_der_r3"]["buffer_logit_drift_r2"]["mse_logit_drift"] for s in cfg["seeds"]])
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
    parser = argparse.ArgumentParser(description="SSRL-13 3-Rule Sequential Continual Learning Stream Runner")
    parser.add_argument("--dataset_1", required=True)
    parser.add_argument("--dataset_2", required=True)
    parser.add_argument("--dataset_3", required=True)
    parser.add_argument("--probe_dataset", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--sha256_1", default=SHA256_RULE_1_DEFAULT)
    parser.add_argument("--sha256_2", default=SHA256_RULE_2_DEFAULT)
    parser.add_argument("--sha256_3", default=SHA256_RULE_3_DEFAULT)
    parser.add_argument("--sha256_probe", default=SHA256_PROBE_DEFAULT)
    args = parser.parse_args()
    run_three_rule_stream_study(
        dataset_1_path=args.dataset_1, sha256_1=args.sha256_1,
        dataset_2_path=args.dataset_2, sha256_2=args.sha256_2,
        dataset_3_path=args.dataset_3, sha256_3=args.sha256_3,
        probe_dataset_path=args.probe_dataset, sha256_probe=args.sha256_probe,
        destination=args.destination, run_dir=args.run_dir,
    )


if __name__ == "__main__":
    main()
