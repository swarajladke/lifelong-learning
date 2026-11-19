"""
research/lifelong_01/run_ssrl_l2init_der_m64.py -- Procedural Runner for SSRL-11.

Evaluates L2-Init (lambda = 0.01 anchored to fresh initialization theta_0) combined with
Dark Experience Replay (DER++, alpha = 0.5) using a scaled stratified M=64 pattern-diverse
replay buffer cycled in 16-example balanced chunks (4 B : 1 A cadence) across Seeds 102, 103, 104:
- FREEZE-AFTER-BASE (standing control: zero updates on verified post-A checkpoint)
- fresh_b (matched fresh comparator with L2-Init, 500 Task B updates)
- seq_b (sequential transfer from post-A with L2-Init and zeroed Adam moments, 500 Task B updates)
- diverse_der_m64 (sequential from post-A + L2-Init + cyclic 16-example chunks over M=64 DER++ buffer, 625 updates)
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

from research.lifelong_01.ssrl_checkpoints import capture_and_verify_checkpoint, compute_tensor_fingerprint_from_state_dict
from research.lifelong_01.ssrl_model import SSRLInstructionMLP, get_expected_parameter_count
from research.lifelong_01.ssrl_counterfactual_probe import validate_counterfactual_probe_contract
from research.lifelong_01.run_ssrl_joint_reference import (
    load_and_verify_dataset, reserve_run_directory, atomic_write_json, make_json_serializable,
    DeadlineExceededError, SpecificationApprovalError,
)
from research.lifelong_01.run_ssrl_two_rule import build_minibatch_schedule
from research.lifelong_01.run_ssrl_paired_replication import compute_schedule_hash
from research.lifelong_01.run_ssrl_l2_init_plasticity import (
    LearnabilityGateError, _assert_learnability_gate, clone_frozen_init_params,
    compute_mechanism_snapshot, compact_population_eval, compact_battery_eval,
    build_seed_comparisons, _summarize_scalar_list,
)
from research.lifelong_01.run_ssrl_l2init_der import (
    EVAL_MODE_DETERMINISTIC, capture_frozen_teacher_logits, evaluate_buffer_logit_drift, train_l2init_der_schedule,
)


PROPOSED_L2INIT_DER_M64_CONFIG: Dict[str, Any] = {
    "study_id": "SSRL-11", "phase": "l2init_dark_experience_replay_m64",
    "classification": "regularized_logit_replay_scaled_buffer_three_seed",
    "gate_approval_status": "APPROVED",
    "model_class": "SSRLInstructionMLP", "expected_parameter_count": 6402,
    "seeds": [102, 103, 104], "minibatch_sampler_seed_a": 2026, "minibatch_sampler_seed_b": 2027,
    "rule_a_id": 1, "rule_a_indices": [0, 1, 2, 3], "rule_b_id": 2, "rule_b_indices": [4, 5, 6, 7],
    "train_pool_size": 384, "batch_size": 16, "step_budget_a": 2000, "step_budget_b": 500,
    "replay_buffer_size": 64, "replay_batch_size": 16, "replay_ratio_b_to_a": [4, 1],
    "milestone_b_steps": [0, 25, 50, 100, 250, 500],
    "learning_rate": 0.001, "betas": [0.9, 0.999], "eps": 1e-08, "weight_decay": 0.0,
    "l2_init_lambda": 0.01, "der_alpha": 0.5, "gate_g_lc_threshold": 0.90,
    "enforce_post_a_learnability_gate": True, "enforce_fresh_b_learnability_gate": True,
    "wall_clock_ceiling_training_seconds_per_stage": 60.0, "wall_clock_ceiling_eval_seconds_per_stage": 60.0,
}


def verify_l2init_der_m64_specification_approval(config: Dict[str, Any]) -> None:
    """AGENTS.md Section 2 / Section 8: Strictly blocks unapproved execution."""
    status = config.get("gate_approval_status")
    if status != "APPROVED":
        raise SpecificationApprovalError(
            f"SSRL-11 L2-Init + DER++ (M=64) specification status is '{status}', not 'APPROVED'. "
            "Execution is forbidden before explicit user authorization."
        )


def select_stratified_pattern_diverse_memory(
    train_examples: List[Dict[str, Any]],
    rule_indices: List[int] = [0, 1, 2, 3],
    buffer_size: int = 64,
    replay_batch_size: int = 16,
) -> List[Dict[str, Any]]:
    """
    Deterministically selects a stratified pattern-diverse memory buffer of size buffer_size
    (default M=64) from train_examples packed into balanced replay_batch_size chunks:
    - 6 active patterns per class sorted canonically; examples sorted by ascending example_id.
    - Selects base_q = per_class // 6 lowest-ID examples per pattern plus 1 extra from first (per_class % 6) patterns.
    - Orders each class round-robin by within-pattern rank across all 6 patterns, then packs balanced chunks.
    """
    if buffer_size < 16 or buffer_size % replay_batch_size != 0 or replay_batch_size % 2 != 0:
        raise ValueError(
            f"Invalid buffer/batch configuration: buffer_size={buffer_size}, replay_batch_size={replay_batch_size}"
        )

    patterns_by_class: Dict[int, Dict[Tuple[int, ...], List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for ex in train_examples:
        cls = int(ex["target_class"])
        if cls not in (0, 1):
            raise ValueError(f"Unexpected target_class {cls}; expected 0 or 1")
        pat = tuple(ex["features"][i] for i in rule_indices)
        patterns_by_class[cls][pat].append(ex)

    if len(patterns_by_class[0]) != 6 or len(patterns_by_class[1]) != 6:
        raise ValueError(
            f"Expected 6 active patterns per class, found {len(patterns_by_class[0])} and {len(patterns_by_class[1])}"
        )

    per_class = buffer_size // 2
    base_q, rem = divmod(per_class, 6)
    rr_by_class: Dict[int, List[Dict[str, Any]]] = {}

    for cls in (0, 1):
        sorted_patterns = sorted(patterns_by_class[cls].keys())
        per_pat_selected: List[List[Dict[str, Any]]] = []
        max_quota = 0
        for p_idx, pat in enumerate(sorted_patterns):
            quota = base_q + (1 if p_idx < rem else 0)
            ex_sorted = sorted(patterns_by_class[cls][pat], key=lambda x: int(x["example_id"]))
            if len(ex_sorted) < quota:
                raise ValueError(f"Class {cls} pattern {pat} has {len(ex_sorted)} examples < quota {quota}")
            per_pat_selected.append(ex_sorted[:quota])
            max_quota = max(max_quota, quota)

        rr_list: List[Dict[str, Any]] = []
        for rank in range(max_quota):
            for p_list in per_pat_selected:
                if rank < len(p_list):
                    rr_list.append(p_list[rank])
        assert len(rr_list) == per_class
        rr_by_class[cls] = rr_list

    num_chunks = buffer_size // replay_batch_size
    half = replay_batch_size // 2
    selected: List[Dict[str, Any]] = []
    for chunk_idx in range(num_chunks):
        c0_half = sorted(rr_by_class[0][chunk_idx * half : (chunk_idx + 1) * half], key=lambda x: int(x["example_id"]))
        c1_half = sorted(rr_by_class[1][chunk_idx * half : (chunk_idx + 1) * half], key=lambda x: int(x["example_id"]))
        assert len(c0_half) == half and len(c1_half) == half
        selected.extend(c0_half + c1_half)

    assert len(selected) == buffer_size
    assert len({int(ex["example_id"]) for ex in selected}) == buffer_size
    assert sum(1 for ex in selected if int(ex["target_class"]) == 0) == per_class
    assert sum(1 for ex in selected if int(ex["target_class"]) == 1) == per_class
    return selected


def build_cyclic_replay_schedule(
    schedule_b: List[List[Dict[str, Any]]],
    replay_buffer: List[Dict[str, Any]],
    replay_batch_size: int = 16,
    ratio_b: int = 4,
    ratio_a: int = 1,
) -> List[Tuple[str, int, List[Dict[str, Any]]]]:
    """
    Interleaves Task B batches with deterministic replay_batch_size chunks cycling
    through replay_buffer at ratio_b : ratio_a cadence.
    """
    if ratio_b <= 0 or ratio_a <= 0:
        raise ValueError(f"Replay ratios must be positive, got ratio_b={ratio_b}, ratio_a={ratio_a}")
    if len(schedule_b) % ratio_b != 0:
        raise ValueError(f"schedule_b length ({len(schedule_b)}) not divisible by ratio_b ({ratio_b})")
    if len(replay_buffer) == 0 or len(replay_buffer) % replay_batch_size != 0:
        raise ValueError(f"replay_buffer length ({len(replay_buffer)}) must be nonzero multiple of {replay_batch_size}")

    chunks = [replay_buffer[i : i + replay_batch_size] for i in range(0, len(replay_buffer), replay_batch_size)]
    num_chunks = len(chunks)
    schedule: List[Tuple[str, int, List[Dict[str, Any]]]] = []
    b_idx, r_step = 0, 0
    for _ in range(len(schedule_b) // ratio_b):
        for _ in range(ratio_b):
            schedule.append(("rule_b", 2, schedule_b[b_idx]))
            b_idx += 1
        for _ in range(ratio_a):
            schedule.append(("replay_a", 1, chunks[r_step % num_chunks]))
            r_step += 1
    return schedule


def _run_derm64_task_b_arm(
    arm_key: str, start_state: Dict[str, torch.Tensor], expected_start_fp: str,
    init_params: Dict[str, torch.Tensor], expected_init_fp: str,
    schedule: List[Tuple[str, int, List[Dict[str, Any]]]], da: Dict[str, Any], db: Dict[str, Any],
    dp: Dict[str, Any], sha256_a: str, sha256_b: str, cfg: Dict[str, Any], seed: int,
    abs_run_dir: str, t_train_ceil: float, t_eval_ceil: float, enforce_b_gate: bool = False,
    teacher_logits_by_id: Optional[Dict[int, torch.Tensor]] = None, expected_teacher_fp: Optional[str] = None,
    drift_eval_buffer: Optional[List[Dict[str, Any]]] = None, drift_teacher_by_id: Optional[Dict[int, torch.Tensor]] = None,
) -> Dict[str, Any]:
    """Trains a Task B SSRL-11 arm, checks G-LC gate, evaluates full buffer drift, and verifies checkpoint."""
    model, t_train, cnt, _, miles = train_l2init_der_schedule(
        start_state, init_params, schedule, db, sha256_b, cfg, seed, f"derm64_{arm_key}_s{seed}",
        time.monotonic() + t_train_ceil, t_eval_ceil, expected_start_fp, expected_init_fp,
        evaluate_b_milestones=True, teacher_logits_by_id=teacher_logits_by_id, expected_teacher_fp=expected_teacher_fp,
    )
    if enforce_b_gate:
        b_step_key = str(cfg["step_budget_b"])
        b_metric = (
            miles[b_step_key]["metric"] if b_step_key in miles
            else compact_population_eval(model, db["eval_seen_examples"], f"derm64_{arm_key}_gate_s{seed}", seed, cfg["step_budget_b"], sha256_b, "rule_b_seen")["metric"]
        )
        _assert_learnability_gate(b_metric, seed, arm_key, cfg["gate_g_lc_threshold"])

    final_eval = compact_battery_eval(model, da, db, dp, seed, len(schedule), f"derm64_{arm_key}_final_s{seed}", sha256_a, sha256_b, time.monotonic() + t_eval_ceil)
    mech = compute_mechanism_snapshot(model, init_params, db)
    ckpt = capture_and_verify_checkpoint(model.state_dict(), abs_run_dir, f"checkpoint_derm64_{arm_key}_s{seed}.pt")
    assert ckpt["tensor_fingerprint"] != expected_start_fp, f"Arm {arm_key} had zero parameter delta"

    arm_result: Dict[str, Any] = {
        "elapsed_train_seconds": t_train, "counters": cnt, "loaded_initial_fingerprint": expected_start_fp,
        "checkpoint": ckpt, "mechanism_snapshot": mech, "milestone_evaluations": miles, "final_eval": final_eval,
    }
    if drift_eval_buffer is not None and drift_teacher_by_id is not None:
        arm_result["buffer_logit_drift"] = evaluate_buffer_logit_drift(model, drift_eval_buffer, drift_teacher_by_id)
    return arm_result


def run_l2init_der_m64_study(
    dataset_a_path: str, sha256_a: str, dataset_b_path: str, sha256_b: str,
    probe_dataset_path: str, sha256_probe: str, destination: str, run_dir: str,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Master procedural entry point for SSRL-11 L2-Init + DER++ (Scaled M=64 Replay Buffer) Study."""
    cfg = copy.deepcopy(config or PROPOSED_L2INIT_DER_M64_CONFIG)
    verify_l2init_der_m64_specification_approval(cfg)
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
            "replay_buffer_size": cfg["replay_buffer_size"], "replay_batch_size": cfg["replay_batch_size"],
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
        diverse_buf = select_stratified_pattern_diverse_memory(
            da["train_examples"], cfg["rule_a_indices"], cfg["replay_buffer_size"], cfg["replay_batch_size"],
        )
        sched_diverse = build_cyclic_replay_schedule(
            sched_b, diverse_buf, replay_batch_size=cfg["replay_batch_size"],
            ratio_b=cfg["replay_ratio_b_to_a"][0], ratio_a=cfg["replay_ratio_b_to_a"][1],
        )

        tagged_a = [("rule_a", 1, b) for b in sched_a]
        tagged_b = [("rule_b", 2, b) for b in sched_b]
        diagnostics["diverse_memory_example_ids"] = [int(ex["example_id"]) for ex in diverse_buf]
        diagnostics["schedule_hash_a"] = compute_schedule_hash(tagged_a)
        diagnostics["schedule_hash_b"] = compute_schedule_hash(tagged_b)
        diagnostics["schedule_hash_diverse_m64"] = compute_schedule_hash(sched_diverse)

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
                f"derm64_phase_a_s{seed}", time.monotonic() + t_train_ceil, t_eval_ceil, fp_init, fp_init, evaluate_b_milestones=False,
            )
            post_a_eval_a = compact_population_eval(model_a, da["eval_seen_examples"], f"derm64_post_a_s{seed}", seed, cfg["step_budget_a"], sha256_a, "rule_a_seen")
            if cfg["enforce_post_a_learnability_gate"]:
                _assert_learnability_gate(post_a_eval_a["metric"], seed, "post_a", cfg["gate_g_lc_threshold"])
            ckpt_post_a = capture_and_verify_checkpoint(model_a.state_dict(), abs_run_dir, f"checkpoint_derm64_post_a_s{seed}.pt")
            fp_post_a = ckpt_post_a["tensor_fingerprint"]
            assert fp_post_a != fp_init, f"Seed {seed} phase_a had zero parameter delta"

            teacher_by_id, teacher_fp, teacher_num_floats = capture_frozen_teacher_logits(model_a, diverse_buf, fp_post_a)
            assert teacher_num_floats == cfg["replay_buffer_size"] * 2
            seed_rec["phase_a"] = {
                "elapsed_train_seconds": t_train_a, "counters": cnt_a, "checkpoint": ckpt_post_a,
                "post_a_rule_a_seen": post_a_eval_a, "mechanism_post_a": compute_mechanism_snapshot(model_a, init_params, db),
                "teacher_logits_fingerprint": teacher_fp, "teacher_logits_num_floats": teacher_num_floats,
                "teacher_logits_bytes_float32": teacher_num_floats * 4,
            }

            diagnostics["current_stage"] = f"seed_{seed}_freeze_after_base"
            eval_freeze = compact_battery_eval(model_a, da, db, dp, seed, 0, f"derm64_freeze_s{seed}", sha256_a, sha256_b, time.monotonic() + t_eval_ceil)
            assert compute_tensor_fingerprint_from_state_dict(model_a.state_dict()) == fp_post_a
            seed_rec["freeze_after_base"] = {"fingerprint": fp_post_a, "eval": eval_freeze}

            arm_specs = [
                ("fresh_b", model_init.state_dict(), fp_init, tagged_b, bool(cfg["enforce_fresh_b_learnability_gate"]), None, None, None, None),
                ("seq_b", model_a.state_dict(), fp_post_a, tagged_b, False, None, None, diverse_buf, teacher_by_id),
                ("diverse_der_m64", model_a.state_dict(), fp_post_a, sched_diverse, False, teacher_by_id, teacher_fp, diverse_buf, teacher_by_id),
            ]
            for arm_key, start_sd, start_fp, arm_sched, arm_gate, arm_t_dict, arm_t_fp, drift_buf, drift_t_dict in arm_specs:
                diagnostics["current_stage"] = f"seed_{seed}_{arm_key}"
                seed_rec[arm_key] = _run_derm64_task_b_arm(
                    arm_key, start_sd, start_fp, init_params, fp_init, arm_sched, da, db, dp, sha256_a, sha256_b,
                    cfg, seed, abs_run_dir, t_train_ceil, t_eval_ceil, enforce_b_gate=arm_gate,
                    teacher_logits_by_id=arm_t_dict, expected_teacher_fp=arm_t_fp,
                    drift_eval_buffer=drift_buf, drift_teacher_by_id=drift_t_dict,
                )

            comp = build_seed_comparisons(
                seed, cfg, seed_rec["fresh_b"]["milestone_evaluations"], seed_rec["seq_b"]["milestone_evaluations"],
                seed_rec["diverse_der_m64"]["milestone_evaluations"], eval_freeze, seed_rec["seq_b"]["final_eval"], seed_rec["diverse_der_m64"]["final_eval"],
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
        summary["buffer_logit_drift_mse_diverse_der_m64"] = _summarize_scalar_list(
            [diagnostics["seeds_data"][str(s)]["diverse_der_m64"]["buffer_logit_drift"]["mse_logit_drift"] for s in cfg["seeds"]]
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
    parser = argparse.ArgumentParser(description="SSRL-11 L2-Init + DER++ (M=64 Buffer) Study Runner")
    parser.add_argument("--dataset_a", required=True)
    parser.add_argument("--dataset_b", required=True)
    parser.add_argument("--probe_dataset", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--sha256_a", default="00a21efad1dc07b826198d99b0b702aedddad57b6226f99301afbf16327fd0b7")
    parser.add_argument("--sha256_b", default="27e813e73bf66f51e1bff07f7d2f19945e830afc6d5876b0cb4d11136cda4cd4")
    parser.add_argument("--sha256_probe", default="1c07d50fed772cfcff86e3bf30cc74a95cdc24d190ec767aa15365671fe06319")
    args = parser.parse_args()
    run_l2init_der_m64_study(
        dataset_a_path=args.dataset_a, sha256_a=args.sha256_a,
        dataset_b_path=args.dataset_b, sha256_b=args.sha256_b,
        probe_dataset_path=args.probe_dataset, sha256_probe=args.sha256_probe,
        destination=args.destination, run_dir=args.run_dir,
    )


if __name__ == "__main__":
    main()
