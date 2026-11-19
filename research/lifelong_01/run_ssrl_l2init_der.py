"""
research/lifelong_01/run_ssrl_l2init_der.py -- Procedural Runner for SSRL-10.

Evaluates L2-Init (lambda = 0.01 anchored to fresh initialization theta_0) combined with
Dark Experience Replay (DER++, alpha = 0.5) on the M=16 pattern-diverse replay buffer across
Seeds 102, 103, 104 and four matched arms:
- FREEZE-AFTER-BASE (standing control: zero updates on verified post-A checkpoint)
- fresh_b (matched fresh comparator with L2-Init, 500 Task B updates)
- seq_b (sequential transfer from post-A with L2-Init and zeroed Adam moments, 500 Task B updates)
- diverse_der (sequential from post-A + L2-Init + M=16 pattern-diverse replay 4 B : 1 A with DER++, 625 total updates)
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
from research.lifelong_01.ssrl_model import SSRLInstructionMLP, get_expected_parameter_count
from research.lifelong_01.ssrl_counterfactual_probe import validate_counterfactual_probe_contract
from research.lifelong_01.run_ssrl_joint_reference import (
    load_and_verify_dataset, reserve_run_directory, atomic_write_json, make_json_serializable,
    DeadlineExceededError, SpecificationApprovalError, CheckpointVerificationError,
)
from research.lifelong_01.run_ssrl_two_rule import build_minibatch_schedule
from research.lifelong_01.run_ssrl_replay_baseline import build_replay_schedule
from research.lifelong_01.run_ssrl_diverse_replay import select_pattern_diverse_replay_memory
from research.lifelong_01.run_ssrl_paired_replication import compute_schedule_hash
from research.lifelong_01.run_ssrl_l2_init_plasticity import (
    LearnabilityGateError, _assert_learnability_gate, clone_frozen_init_params,
    compute_l2_init_penalty, compute_mechanism_snapshot, compact_population_eval,
    compact_battery_eval, build_seed_comparisons, _summarize_scalar_list,
)


PROPOSED_L2INIT_DER_CONFIG: Dict[str, Any] = {
    "study_id": "SSRL-10", "phase": "l2init_dark_experience_replay",
    "classification": "regularized_logit_replay_three_seed", "gate_approval_status": "APPROVED",
    "model_class": "SSRLInstructionMLP", "expected_parameter_count": 6402,
    "seeds": [102, 103, 104], "minibatch_sampler_seed_a": 2026, "minibatch_sampler_seed_b": 2027,
    "rule_a_id": 1, "rule_a_indices": [0, 1, 2, 3], "rule_b_id": 2, "rule_b_indices": [4, 5, 6, 7],
    "train_pool_size": 384, "batch_size": 16, "step_budget_a": 2000, "step_budget_b": 500,
    "replay_buffer_size": 16, "replay_ratio_b_to_a": [4, 1], "milestone_b_steps": [0, 25, 50, 100, 250, 500],
    "learning_rate": 0.001, "betas": [0.9, 0.999], "eps": 1e-08, "weight_decay": 0.0,
    "l2_init_lambda": 0.01, "der_alpha": 0.5, "gate_g_lc_threshold": 0.90,
    "enforce_post_a_learnability_gate": True, "enforce_fresh_b_learnability_gate": True,
    "wall_clock_ceiling_training_seconds_per_stage": 60.0, "wall_clock_ceiling_eval_seconds_per_stage": 60.0,
}

EVAL_MODE_DETERMINISTIC = "eval_deterministic_no_dropout"


def verify_l2init_der_specification_approval(config: Dict[str, Any]) -> None:
    """AGENTS.md Section 2 / Section 8: Strictly blocks unapproved execution."""
    status = config.get("gate_approval_status")
    if status != "APPROVED":
        raise SpecificationApprovalError(
            f"SSRL-10 L2-Init + DER++ specification status is '{status}', not 'APPROVED'. "
            "Execution is forbidden before explicit user authorization."
        )


def _teacher_dict_fingerprint(teacher_logits_by_id: Dict[int, torch.Tensor]) -> str:
    """Computes canonical SHA-256 tensor fingerprint over teacher buffer logits keyed by example_id."""
    ordered = {f"ex_{k}": teacher_logits_by_id[k] for k in sorted(teacher_logits_by_id.keys())}
    return compute_tensor_fingerprint_from_state_dict(ordered)


def capture_frozen_teacher_logits(
    model_a: SSRLInstructionMLP, buffer_examples: List[Dict[str, Any]], expected_fp: str,
) -> Tuple[Dict[int, torch.Tensor], str, int]:
    """Captures frozen post-A teacher logits for replay buffer examples under verified eval mode."""
    if len(buffer_examples) == 0:
        raise ValueError("Replay buffer must contain at least one example for DER teacher capture")
    fp_before = compute_tensor_fingerprint_from_state_dict(model_a.state_dict())
    if fp_before != expected_fp:
        raise CheckpointVerificationError(f"Post-A fingerprint mismatch before teacher capture: {fp_before} != {expected_fp}")

    was_training = model_a.training
    model_a.eval()
    assert not model_a.training, "Teacher logit capture requires eval mode"

    inputs = torch.stack([torch.tensor(ex["features"] + ex["mask"], dtype=torch.float32) for ex in buffer_examples], dim=0)
    with torch.no_grad():
        logits = model_a(inputs).detach().to(torch.float32).clone()
    if logits.shape != (len(buffer_examples), 2):
        raise ValueError(f"Expected teacher logits shape ({len(buffer_examples)}, 2), got {tuple(logits.shape)}")

    model_a.train(was_training)
    assert compute_tensor_fingerprint_from_state_dict(model_a.state_dict()) == expected_fp

    teacher_by_id: Dict[int, torch.Tensor] = {}
    for idx, ex in enumerate(buffer_examples):
        ex_id = int(ex["example_id"])
        if ex_id in teacher_by_id:
            raise ValueError(f"Duplicate example_id {ex_id} in DER replay buffer")
        teacher_by_id[ex_id] = logits[idx].clone()

    num_floats = sum(int(t.numel()) for t in teacher_by_id.values())
    assert num_floats == len(buffer_examples) * 2
    return teacher_by_id, _teacher_dict_fingerprint(teacher_by_id), num_floats


def compute_der_logit_loss(
    logits: torch.Tensor, batch: List[Dict[str, Any]], teacher_logits_by_id: Dict[int, torch.Tensor],
) -> torch.Tensor:
    """Computes DER++ MSE logit loss aligned strictly by example_id; returns 0.0 when matched."""
    if len(batch) == 0:
        raise ValueError("DER replay batch must not be empty")
    missing_ids = [int(ex["example_id"]) for ex in batch if int(ex["example_id"]) not in teacher_logits_by_id]
    if missing_ids:
        raise KeyError(f"Missing frozen teacher logits for replay example_ids: {missing_ids}")

    target_logits = torch.stack(
        [teacher_logits_by_id[int(ex["example_id"])].to(dtype=logits.dtype, device=logits.device) for ex in batch], dim=0
    )
    if logits.shape != target_logits.shape:
        raise ValueError(f"Logits shape {tuple(logits.shape)} != teacher target shape {tuple(target_logits.shape)}")
    return F.mse_loss(logits, target_logits)


def evaluate_buffer_logit_drift(
    model: SSRLInstructionMLP, buffer_examples: List[Dict[str, Any]], teacher_logits_by_id: Dict[int, torch.Tensor],
) -> Dict[str, Any]:
    """Evaluates MSE, max-abs, KL divergence, and argmax match count against frozen teacher buffer logits."""
    if len(buffer_examples) == 0:
        raise ValueError("Replay buffer must not be empty for logit drift evaluation")
    was_training = model.training
    model.eval()
    assert not model.training, "Logit drift evaluation requires eval mode"

    inputs = torch.stack([torch.tensor(ex["features"] + ex["mask"], dtype=torch.float32) for ex in buffer_examples], dim=0)
    with torch.no_grad():
        current_logits = model(inputs).detach().to(torch.float32)
    target_logits = torch.stack(
        [teacher_logits_by_id[int(ex["example_id"])].detach().to(torch.float32) for ex in buffer_examples], dim=0
    )
    assert current_logits.shape == target_logits.shape == (len(buffer_examples), 2)

    pred_match_num = int((current_logits.argmax(dim=-1) == target_logits.argmax(dim=-1)).sum().item())
    pred_match_den = len(buffer_examples)
    assert 0 <= pred_match_num <= pred_match_den

    result = {
        "execution_mode": EVAL_MODE_DETERMINISTIC,
        "buffer_size": pred_match_den,
        "num_logit_floats": int(current_logits.numel()),
        "mse_logit_drift": float(F.mse_loss(current_logits, target_logits).item()),
        "max_abs_logit_drift": float((current_logits - target_logits).abs().max().item()),
        "kl_teacher_to_current": float(F.kl_div(torch.log_softmax(current_logits, dim=-1), torch.softmax(target_logits, dim=-1), reduction="batchmean").item()),
        "argmax_match_numerator": pred_match_num,
        "argmax_match_denominator": pred_match_den,
    }
    model.train(was_training)
    return result


def train_l2init_der_schedule(
    initial_model_state: Dict[str, torch.Tensor], init_params: Dict[str, torch.Tensor],
    tagged_schedule: List[Tuple[str, int, List[Dict[str, Any]]]], db: Dict[str, Any], sha256_b: str,
    config: Dict[str, Any], seed: int, arm_tag: str, train_deadline: float, eval_ceiling: float,
    expected_start_fp: str, expected_init_fp: str, evaluate_b_milestones: bool = True,
    teacher_logits_by_id: Optional[Dict[int, torch.Tensor]] = None, expected_teacher_fp: Optional[str] = None,
) -> Tuple[SSRLInstructionMLP, float, Dict[str, Any], str, Dict[str, Any]]:
    """Trains SSRLInstructionMLP with Adam, L2-Init toward init_params, and optional DER++ replay loss."""
    if time.monotonic() > train_deadline:
        raise DeadlineExceededError(f"Training deadline exceeded before starting {arm_tag}")
    if compute_tensor_fingerprint_from_state_dict(init_params) != expected_init_fp:
        raise CheckpointVerificationError(f"Frozen init_params fingerprint mismatch before {arm_tag}")
    if teacher_logits_by_id is not None:
        if expected_teacher_fp is None:
            raise ValueError(f"expected_teacher_fp required for {arm_tag}")
        if _teacher_dict_fingerprint(teacher_logits_by_id) != expected_teacher_fp:
            raise CheckpointVerificationError(f"Teacher logit fingerprint mismatch before {arm_tag}")

    model = SSRLInstructionMLP(input_dim=32, hidden_dim=64, output_dim=2)
    assert get_expected_parameter_count() == config["expected_parameter_count"] == 6402
    model.load_state_dict(initial_model_state)
    loaded_fp = compute_tensor_fingerprint_from_state_dict(model.state_dict())
    if loaded_fp != expected_start_fp:
        raise CheckpointVerificationError(f"Initial fingerprint mismatch for {arm_tag}: {loaded_fp} != {expected_start_fp}")

    optimizer = optim.Adam(
        model.parameters(), lr=config["learning_rate"], betas=tuple(config["betas"]),
        eps=config["eps"], weight_decay=config["weight_decay"],
    )
    loss_fn = nn.CrossEntropyLoss()
    l2_lambda, der_alpha = float(config["l2_init_lambda"]), float(config["der_alpha"])
    counters: Dict[str, Any] = {
        "total_optimizer_steps": 0, "total_presentations": 0,
        "rule_a_updates": 0, "rule_a_presentations": 0, "rule_b_updates": 0, "rule_b_presentations": 0,
        "replay_a_updates": 0, "replay_a_presentations": 0, "der_loss_steps": 0, "der_loss_sum": 0.0,
        "final_ce_loss": None, "final_l2_init_penalty": None, "final_der_mse_loss": None,
    }
    milestones_set = set(config["milestone_b_steps"]) if evaluate_b_milestones else set()
    milestone_records: Dict[str, Any] = {}
    t_start = time.monotonic()

    if 0 in milestones_set:
        t_eval_dl = time.monotonic() + eval_ceiling
        model.eval()
        m_0 = compact_population_eval(model, db["eval_seen_examples"], f"{arm_tag}_bstep0", seed, 0, sha256_b, "rule_b_seen")
        milestone_records["0"] = {"rule_b_updates": 0, "total_steps": 0, **m_0}
        if time.monotonic() > t_eval_dl:
            raise DeadlineExceededError(f"Milestone 0 eval deadline exceeded for {arm_tag}")

    model.train()
    for tag, _, batch in tagged_schedule:
        if time.monotonic() > train_deadline:
            raise DeadlineExceededError(f"Training deadline exceeded at step {counters['total_optimizer_steps'] + 1} for {arm_tag}")
        if tag == "replay_a" and teacher_logits_by_id is None:
            raise ValueError(f"Received replay_a batch without teacher_logits_by_id in {arm_tag}")

        inputs = torch.stack([torch.tensor(ex["features"] + ex["mask"], dtype=torch.float32) for ex in batch], dim=0)
        targets = torch.tensor([int(ex["target_class"]) for ex in batch], dtype=torch.long)

        optimizer.zero_grad()
        logits = model(inputs)
        ce_loss = loss_fn(logits, targets)
        l2_penalty = compute_l2_init_penalty(model, init_params, l2_lambda)
        loss = ce_loss + l2_penalty
        if tag == "replay_a" and teacher_logits_by_id is not None:
            der_mse = compute_der_logit_loss(logits, batch, teacher_logits_by_id)
            loss = loss + (der_alpha * der_mse)
            counters["der_loss_steps"] += 1
            counters["der_loss_sum"] += float(der_mse.item())
            counters["final_der_mse_loss"] = float(der_mse.item())

        loss.backward()
        optimizer.step()

        counters["total_optimizer_steps"] += 1
        counters["total_presentations"] += len(batch)
        counters[f"{tag}_updates"] += 1
        counters[f"{tag}_presentations"] += len(batch)
        counters["final_ce_loss"] = float(ce_loss.item())
        counters["final_l2_init_penalty"] = float(l2_penalty.item())

        if tag == "rule_b" and counters["rule_b_updates"] in milestones_set:
            b_step = counters["rule_b_updates"]
            t_eval_dl = time.monotonic() + eval_ceiling
            model.eval()
            m_b = compact_population_eval(model, db["eval_seen_examples"], f"{arm_tag}_bstep{b_step}", seed, b_step, sha256_b, "rule_b_seen")
            milestone_records[str(b_step)] = {"rule_b_updates": b_step, "total_steps": counters["total_optimizer_steps"], **m_b}
            if time.monotonic() > t_eval_dl:
                raise DeadlineExceededError(f"Milestone {b_step} eval deadline exceeded for {arm_tag}")
            model.train()

    assert counters["total_optimizer_steps"] > 0 and counters["total_presentations"] > 0
    assert compute_tensor_fingerprint_from_state_dict(init_params) == expected_init_fp
    if teacher_logits_by_id is not None and expected_teacher_fp is not None:
        assert _teacher_dict_fingerprint(teacher_logits_by_id) == expected_teacher_fp
    counters["mean_der_mse_loss"] = counters["der_loss_sum"] / counters["der_loss_steps"] if counters["der_loss_steps"] > 0 else None
    return model, time.monotonic() - t_start, counters, loaded_fp, milestone_records


def _run_der_task_b_arm(
    arm_key: str, start_state: Dict[str, torch.Tensor], expected_start_fp: str,
    init_params: Dict[str, torch.Tensor], expected_init_fp: str,
    schedule: List[Tuple[str, int, List[Dict[str, Any]]]], da: Dict[str, Any], db: Dict[str, Any],
    dp: Dict[str, Any], sha256_a: str, sha256_b: str, cfg: Dict[str, Any], seed: int,
    abs_run_dir: str, t_train_ceil: float, t_eval_ceil: float, enforce_b_gate: bool = False,
    teacher_logits_by_id: Optional[Dict[int, torch.Tensor]] = None, expected_teacher_fp: Optional[str] = None,
    drift_eval_buffer: Optional[List[Dict[str, Any]]] = None, drift_teacher_by_id: Optional[Dict[int, torch.Tensor]] = None,
) -> Dict[str, Any]:
    """Trains a Task B L2-Init/DER arm, checks gates, snapshots mechanism + logit drift, and verifies checkpoint."""
    model, t_train, cnt, _, miles = train_l2init_der_schedule(
        start_state, init_params, schedule, db, sha256_b, cfg, seed, f"der_{arm_key}_s{seed}",
        time.monotonic() + t_train_ceil, t_eval_ceil, expected_start_fp, expected_init_fp,
        evaluate_b_milestones=True, teacher_logits_by_id=teacher_logits_by_id, expected_teacher_fp=expected_teacher_fp,
    )
    if enforce_b_gate:
        b_step_key = str(cfg["step_budget_b"])
        b_metric = (
            miles[b_step_key]["metric"] if b_step_key in miles
            else compact_population_eval(model, db["eval_seen_examples"], f"der_{arm_key}_gate_s{seed}", seed, cfg["step_budget_b"], sha256_b, "rule_b_seen")["metric"]
        )
        _assert_learnability_gate(b_metric, seed, arm_key, cfg["gate_g_lc_threshold"])

    final_eval = compact_battery_eval(model, da, db, dp, seed, len(schedule), f"der_{arm_key}_final_s{seed}", sha256_a, sha256_b, time.monotonic() + t_eval_ceil)
    mech = compute_mechanism_snapshot(model, init_params, db)
    ckpt = capture_and_verify_checkpoint(model.state_dict(), abs_run_dir, f"checkpoint_der_{arm_key}_s{seed}.pt")
    assert ckpt["tensor_fingerprint"] != expected_start_fp, f"Arm {arm_key} had zero parameter delta"

    arm_result: Dict[str, Any] = {
        "elapsed_train_seconds": t_train, "counters": cnt, "loaded_initial_fingerprint": expected_start_fp,
        "checkpoint": ckpt, "mechanism_snapshot": mech, "milestone_evaluations": miles, "final_eval": final_eval,
    }
    if drift_eval_buffer is not None and drift_teacher_by_id is not None:
        arm_result["buffer_logit_drift"] = evaluate_buffer_logit_drift(model, drift_eval_buffer, drift_teacher_by_id)
    return arm_result


def run_l2init_der_study(
    dataset_a_path: str, sha256_a: str, dataset_b_path: str, sha256_b: str,
    probe_dataset_path: str, sha256_probe: str, destination: str, run_dir: str,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Master procedural entry point for SSRL-10 L2-Init + Dark Experience Replay (DER++) Study."""
    cfg = copy.deepcopy(config or PROPOSED_L2INIT_DER_CONFIG)
    verify_l2init_der_specification_approval(cfg)
    assert get_expected_parameter_count() == cfg["expected_parameter_count"] == 6402
    abs_run_dir = reserve_run_directory(run_dir)
    abs_dest = os.path.abspath(destination)

    da = load_and_verify_dataset(
        dataset_a_path, expected_sha256=sha256_a,
        approved_config={**cfg, "study_id": "SSRL-01", "phase": "learnability_pilot", "seed": 101, "partition_seed": 42, "rule_id": cfg["rule_a_id"], "rule_indices": cfg["rule_a_indices"]},
    )
    db = load_and_verify_dataset(
        dataset_b_path, expected_sha256=sha256_b,
        approved_config={**cfg, "study_id": "SSRL-02", "phase": "two_rule_diagnostic", "seed": 101, "partition_seed": 43, "rule_id": cfg["rule_b_id"], "rule_indices": cfg["rule_b_indices"]},
    )
    with open(probe_dataset_path, "rb") as f:
        actual_probe_sha = hashlib.sha256(f.read()).hexdigest()
    if actual_probe_sha != sha256_probe:
        raise ValueError(f"Probe SHA mismatch: expected {sha256_probe}, got {actual_probe_sha}")
    with open(probe_dataset_path, "r", encoding="utf-8") as f:
        dp = json.load(f)
    validate_counterfactual_probe_contract(dp, da, db)

    diagnostics: Dict[str, Any] = {
        "study_id": cfg["study_id"], "phase": cfg["phase"], "classification": cfg["classification"],
        "status": "IN_PROGRESS", "current_stage": "INITIALIZATION", "execution_mode": EVAL_MODE_DETERMINISTIC,
        "dataset_hashes": {"sha256_a": sha256_a, "sha256_b": sha256_b, "sha256_probe": sha256_probe},
        "environment": {
            "python_version": sys.version.split()[0], "torch_version": torch.__version__, "device": "cpu",
            "l2_init_lambda": cfg["l2_init_lambda"], "der_alpha": cfg["der_alpha"],
        },
        "seeds_status": {s: "NOT_RUN" for s in cfg["seeds"]}, "seeds_data": {},
        "paired_transfer_comparisons": [], "run_directory": abs_run_dir, "destination": abs_dest,
    }
    diag_file = os.path.join(abs_run_dir, "failure_diagnostics.json")

    try:
        t_train_ceil = float(cfg["wall_clock_ceiling_training_seconds_per_stage"])
        t_eval_ceil = float(cfg["wall_clock_ceiling_eval_seconds_per_stage"])
        sched_a = build_minibatch_schedule(da["train_examples"], cfg["step_budget_a"], cfg["batch_size"], cfg["minibatch_sampler_seed_a"])
        sched_b = build_minibatch_schedule(db["train_examples"], cfg["step_budget_b"], cfg["batch_size"], cfg["minibatch_sampler_seed_b"])
        diverse_buf = select_pattern_diverse_replay_memory(da["train_examples"], cfg["rule_a_indices"], cfg["replay_buffer_size"])
        sched_diverse = build_replay_schedule(sched_b, diverse_buf, ratio_b=cfg["replay_ratio_b_to_a"][0], ratio_a=cfg["replay_ratio_b_to_a"][1])

        tagged_a = [("rule_a", 1, b) for b in sched_a]
        tagged_b = [("rule_b", 2, b) for b in sched_b]
        diagnostics["diverse_memory_example_ids"] = [int(ex["example_id"]) for ex in diverse_buf]
        diagnostics["schedule_hash_a"] = compute_schedule_hash(tagged_a)
        diagnostics["schedule_hash_b"] = compute_schedule_hash(tagged_b)
        diagnostics["schedule_hash_diverse"] = compute_schedule_hash(sched_diverse)

        for seed in cfg["seeds"]:
            diagnostics["seeds_status"][seed] = "IN_PROGRESS"
            seed_rec: Dict[str, Any] = {"seed": seed}
            diagnostics["seeds_data"][str(seed)] = seed_rec

            torch.manual_seed(seed)
            model_init = SSRLInstructionMLP(input_dim=32, hidden_dim=64, output_dim=2)
            init_params, fp_init = clone_frozen_init_params(model_init)
            seed_rec["fresh_init_fingerprint"] = fp_init
            seed_rec["mechanism_fresh_init"] = compute_mechanism_snapshot(model_init, init_params, db)

            diagnostics["current_stage"] = f"seed_{seed}_phase_a"
            model_a, t_train_a, cnt_a, _, _ = train_l2init_der_schedule(
                model_init.state_dict(), init_params, tagged_a, db, sha256_b, cfg, seed,
                f"der_phase_a_s{seed}", time.monotonic() + t_train_ceil, t_eval_ceil, fp_init, fp_init, evaluate_b_milestones=False,
            )
            post_a_eval_a = compact_population_eval(model_a, da["eval_seen_examples"], f"der_post_a_s{seed}", seed, cfg["step_budget_a"], sha256_a, "rule_a_seen")
            if cfg["enforce_post_a_learnability_gate"]:
                _assert_learnability_gate(post_a_eval_a["metric"], seed, "post_a", cfg["gate_g_lc_threshold"])
            ckpt_post_a = capture_and_verify_checkpoint(model_a.state_dict(), abs_run_dir, f"checkpoint_der_post_a_s{seed}.pt")
            fp_post_a = ckpt_post_a["tensor_fingerprint"]
            assert fp_post_a != fp_init, f"Seed {seed} phase_a had zero parameter delta"

            teacher_by_id, teacher_fp, teacher_num_floats = capture_frozen_teacher_logits(model_a, diverse_buf, fp_post_a)
            seed_rec["phase_a"] = {
                "elapsed_train_seconds": t_train_a, "counters": cnt_a, "checkpoint": ckpt_post_a,
                "post_a_rule_a_seen": post_a_eval_a, "mechanism_post_a": compute_mechanism_snapshot(model_a, init_params, db),
                "teacher_logits_fingerprint": teacher_fp, "teacher_logits_num_floats": teacher_num_floats,
                "teacher_logits_bytes_float32": teacher_num_floats * 4,
            }

            diagnostics["current_stage"] = f"seed_{seed}_freeze_after_base"
            eval_freeze = compact_battery_eval(model_a, da, db, dp, seed, 0, f"der_freeze_s{seed}", sha256_a, sha256_b, time.monotonic() + t_eval_ceil)
            assert compute_tensor_fingerprint_from_state_dict(model_a.state_dict()) == fp_post_a
            seed_rec["freeze_after_base"] = {"fingerprint": fp_post_a, "eval": eval_freeze}

            arm_specs = [
                ("fresh_b", model_init.state_dict(), fp_init, tagged_b, bool(cfg["enforce_fresh_b_learnability_gate"]), None, None, None, None),
                ("seq_b", model_a.state_dict(), fp_post_a, tagged_b, False, None, None, diverse_buf, teacher_by_id),
                ("diverse_der", model_a.state_dict(), fp_post_a, sched_diverse, False, teacher_by_id, teacher_fp, diverse_buf, teacher_by_id),
            ]
            for arm_key, start_sd, start_fp, arm_sched, arm_gate, arm_t_dict, arm_t_fp, drift_buf, drift_t_dict in arm_specs:
                diagnostics["current_stage"] = f"seed_{seed}_{arm_key}"
                seed_rec[arm_key] = _run_der_task_b_arm(
                    arm_key, start_sd, start_fp, init_params, fp_init, arm_sched, da, db, dp, sha256_a, sha256_b,
                    cfg, seed, abs_run_dir, t_train_ceil, t_eval_ceil, enforce_b_gate=arm_gate,
                    teacher_logits_by_id=arm_t_dict, expected_teacher_fp=arm_t_fp,
                    drift_eval_buffer=drift_buf, drift_teacher_by_id=drift_t_dict,
                )

            comp = build_seed_comparisons(
                seed, cfg, seed_rec["fresh_b"]["milestone_evaluations"], seed_rec["seq_b"]["milestone_evaluations"],
                seed_rec["diverse_der"]["milestone_evaluations"], eval_freeze, seed_rec["seq_b"]["final_eval"], seed_rec["diverse_der"]["final_eval"],
            )
            seed_rec["transfer_comparison"] = comp
            diagnostics["paired_transfer_comparisons"].append(comp)
            diagnostics["seeds_status"][seed] = "COMPLETED"

        comps = diagnostics["paired_transfer_comparisons"]
        summary: Dict[str, Any] = {
            f"transfer_delta_seq_step{k}_pp": _summarize_scalar_list([c["milestone_deltas"][str(k)]["delta_seq_minus_fresh_pp"] for c in comps])
            for k in cfg["milestone_b_steps"]
        }
        for k in cfg["milestone_b_steps"]:
            summary[f"transfer_delta_diverse_step{k}_pp"] = _summarize_scalar_list([c["milestone_deltas"][str(k)]["delta_diverse_minus_fresh_pp"] for c in comps])
        summary["forgetting_seq_b_pp"] = _summarize_scalar_list([c["forgetting_seq_b_pp"] for c in comps])
        summary["forgetting_diverse_replay_pp"] = _summarize_scalar_list([c["forgetting_diverse_replay_pp"] for c in comps])
        summary["diverse_minus_seq_rule_a_pp"] = _summarize_scalar_list([c["diverse_minus_seq_rule_a_pp"] for c in comps])
        summary["buffer_logit_drift_mse_seq_b"] = _summarize_scalar_list(
            [diagnostics["seeds_data"][str(s)]["seq_b"]["buffer_logit_drift"]["mse_logit_drift"] for s in cfg["seeds"]]
        )
        summary["buffer_logit_drift_mse_diverse_der"] = _summarize_scalar_list(
            [diagnostics["seeds_data"][str(s)]["diverse_der"]["buffer_logit_drift"]["mse_logit_drift"] for s in cfg["seeds"]]
        )
        diagnostics["summary_statistics"] = summary
        diagnostics["status"] = "COMPLETED"
        diagnostics["current_stage"] = "COMPLETED"

        payload = {"config": cfg, "diagnostics": diagnostics}
        atomic_write_json(make_json_serializable(payload), abs_dest, refuse_overwrite=True)
        return payload

    except Exception as exc:
        diagnostics["status"] = "FAILED"
        diagnostics["error_type"] = type(exc).__name__
        diagnostics["error_message"] = str(exc)
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
    parser = argparse.ArgumentParser(description="SSRL-10 L2-Init + Dark Experience Replay (DER++) Study Runner")
    parser.add_argument("--dataset_a", required=True)
    parser.add_argument("--dataset_b", required=True)
    parser.add_argument("--probe_dataset", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--sha256_a", default="00a21efad1dc07b826198d99b0b702aedddad57b6226f99301afbf16327fd0b7")
    parser.add_argument("--sha256_b", default="27e813e73bf66f51e1bff07f7d2f19945e830afc6d5876b0cb4d11136cda4cd4")
    parser.add_argument("--sha256_probe", default="1c07d50fed772cfcff86e3bf30cc74a95cdc24d190ec767aa15365671fe06319")
    args = parser.parse_args()
    run_l2init_der_study(
        dataset_a_path=args.dataset_a, sha256_a=args.sha256_a,
        dataset_b_path=args.dataset_b, sha256_b=args.sha256_b,
        probe_dataset_path=args.probe_dataset, sha256_probe=args.sha256_probe,
        destination=args.destination, run_dir=args.run_dir,
    )


if __name__ == "__main__":
    main()
