"""
research/lifelong_01/run_ssrl_transfer_controls.py -- Procedural Runner for SSRL-07.
Sequential Transfer Controls on Seeds 102, 103, 104:
- FREEZE-AFTER-BASE (standing control: zero updates on verified post-A checkpoint)
- fresh_b (matched fresh baseline: reconstructed from initial seed, 500 Task B updates)
- seq_b (naive sequential transfer: starts from verified post-A checkpoint, 500 Task B updates)

Invariants:
- Evaluates seeds 102, 103, 104 sequentially.
- Fixed pinned datasets; fixed Task B sampler seed 2027.
- Evaluates Rule B seen accuracy at milestones [0, 25, 50, 100, 250, 500].
- Serializes full per-unit outcome records for all evaluation stages.
- Separate 60.0s deadline per training stage and 60.0s per evaluation stage.
- Immediate halt on any failure; persists diagnostics; records later stages as NOT_RUN.
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

from research.lifelong_01.ssrl_checkpoints import (
    capture_and_verify_checkpoint,
    compute_tensor_fingerprint_from_state_dict,
)
from research.lifelong_01.ssrl_model import SSRLInstructionMLP
from research.lifelong_01.ssrl_counterfactual_probe import (
    validate_counterfactual_probe_contract,
    evaluate_counterfactual_probe,
)
from research.lifelong_01.run_ssrl_joint_reference import (
    evaluate_population_records,
    load_and_verify_dataset,
    reserve_run_directory,
    atomic_write_json,
    make_json_serializable,
    DeadlineExceededError,
    SpecificationApprovalError,
    CheckpointVerificationError,
)
from research.lifelong_01.ssrl_types import (
    SSRLExampleRecord,
    compute_accuracy_measurement,
)
from research.lifelong_01.run_ssrl_two_rule import build_minibatch_schedule

PROPOSED_TRANSFER_CONTROLS_CONFIG: Dict[str, Any] = {
    "study_id": "SSRL-07",
    "phase": "sequential_transfer_controls",
    "classification": "sequential_transfer_controls_three_seed",
    "gate_approval_status": "APPROVED",
    "seeds": [102, 103, 104],
    "exploratory_reference_seed": 101,
    "minibatch_sampler_seed_b": 2027,
    "rule_a_id": 1,
    "rule_b_id": 2,
    "rule_a_indices": [0, 1, 2, 3],
    "rule_b_indices": [4, 5, 6, 7],
    "train_pool_size": 384,
    "batch_size": 16,
    "step_budget_b": 500,
    "milestone_steps": [0, 25, 50, 100, 250, 500],
    "learning_rate": 0.001,
    "betas": [0.9, 0.999],
    "eps": 1e-08,
    "weight_decay": 0.0,
    "gate_g_lc_threshold": 0.90,
    "wall_clock_ceiling_training_seconds_per_stage": 60.0,
    "wall_clock_ceiling_eval_seconds_per_stage": 60.0,
}


def verify_transfer_controls_specification_approval(config: Dict[str, Any]) -> None:
    """AGENTS.md Section 2 / Section 8: Strictly blocks unapproved execution."""
    status = config.get("gate_approval_status")
    if status != "APPROVED":
        raise SpecificationApprovalError(
            f"SSRL-07 transfer controls specification status is '{status}', not 'APPROVED'. "
            f"Execution is forbidden before explicit user authorization."
        )


def evaluate_battery(
    model: SSRLInstructionMLP,
    da: Dict[str, Any],
    db: Dict[str, Any],
    dp: Dict[str, Any],
    seed: int,
    step: int,
    arm_tag: str,
    sha256_a: str,
    sha256_b: str,
    eval_deadline: float,
) -> Dict[str, Any]:
    """Runs frozen evaluation across all 5 standard populations with deadline check."""
    if time.monotonic() > eval_deadline:
        raise DeadlineExceededError(f"Evaluation deadline exceeded before evaluating {arm_tag}")

    model.eval()
    device = torch.device("cpu")
    t_start = time.monotonic()

    eval_splits = [
        ("rule_a_seen", da["eval_seen_examples"], sha256_a),
        ("rule_a_unseen", da["eval_unseen_examples"], sha256_a),
        ("rule_b_seen", db["eval_seen_examples"], sha256_b),
        ("rule_b_unseen", db["eval_unseen_examples"], sha256_b),
    ]

    eval_records: Dict[str, Any] = {}
    metrics: Dict[str, Any] = {}

    for split_tag, examples, d_hash in eval_splits:
        recs = evaluate_population_records(
            model=model, examples=examples, arm=arm_tag,
            seed=seed, step=step, dataset_sha256=d_hash, device=device
        )
        eval_records[split_tag] = recs
        metrics[f"acc_{split_tag}"] = compute_accuracy_measurement(
            name=f"acc_{split_tag}", records=recs, input_set=d_hash, mode="eval_no_dropout"
        )

    probe_results = evaluate_counterfactual_probe(
        model=model, probe_examples=dp["probe_examples"], device=device
    )

    if time.monotonic() > eval_deadline:
        raise DeadlineExceededError(f"Evaluation deadline exceeded while completing {arm_tag}")

    return {
        "elapsed_eval_seconds": time.monotonic() - t_start,
        "metrics": metrics,
        "counterfactual_probe": probe_results,
        "eval_records": eval_records,
    }


def train_task_b_arm(
    initial_model_state: Dict[str, torch.Tensor],
    schedule_b: List[List[Dict[str, Any]]],
    db: Dict[str, Any],
    sha256_b: str,
    config: Dict[str, Any],
    seed: int,
    arm_tag: str,
    train_deadline: float,
    eval_ceiling: float,
    expected_fingerprint: Optional[str] = None,
) -> Tuple[SSRLInstructionMLP, float, Dict[str, int], str, Dict[int, Any]]:
    """
    Trains an arm on Task B minibatches with milestone evaluations.
    Evaluates Rule B seen accuracy at specified milestone steps.
    """
    if time.monotonic() > train_deadline:
        raise DeadlineExceededError(f"Training deadline exceeded before starting {arm_tag}")

    model = SSRLInstructionMLP(input_dim=32, hidden_dim=64, output_dim=2)
    model.load_state_dict(initial_model_state)
    actual_fp = compute_tensor_fingerprint_from_state_dict(model.state_dict())
    if expected_fingerprint is not None and actual_fp != expected_fingerprint:
        raise CheckpointVerificationError(
            f"Initial fingerprint mismatch for {arm_tag}: expected {expected_fingerprint}, got {actual_fp}"
        )

    device = torch.device("cpu")
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["learning_rate"], betas=tuple(config["betas"]),
        eps=config["eps"], weight_decay=config["weight_decay"],
    )
    loss_fn = nn.CrossEntropyLoss()
    counters = {"rule_b_updates": 0, "rule_b_presentations": 0, "total_optimizer_steps": 0}
    milestone_records: Dict[int, Any] = {}
    milestones_set = set(config["milestone_steps"])
    t_start = time.monotonic()

    # Step 0 evaluation if in milestones
    if 0 in milestones_set:
        t_eval_deadline = time.monotonic() + eval_ceiling
        model.eval()
        recs_0 = evaluate_population_records(
            model=model, examples=db["eval_seen_examples"], arm=f"{arm_tag}_step0",
            seed=seed, step=0, dataset_sha256=sha256_b, device=device
        )
        m_0 = compute_accuracy_measurement("acc_rule_b_seen", recs_0, sha256_b, "eval_no_dropout")
        milestone_records[0] = {
            "step": 0, "metric": m_0, "records": recs_0, "elapsed_seconds": time.monotonic() - t_start
        }
        if time.monotonic() > t_eval_deadline:
            raise DeadlineExceededError(f"Milestone 0 eval deadline exceeded for {arm_tag}")

    model.train()
    for step_idx, batch in enumerate(schedule_b):
        step_num = step_idx + 1
        if time.monotonic() > train_deadline:
            raise DeadlineExceededError(f"Training deadline exceeded at step {step_num} for {arm_tag}")

        inputs = [torch.tensor(ex["features"] + ex["mask"], dtype=torch.float32) for ex in batch]
        targets = [int(ex["target_class"]) for ex in batch]

        optimizer.zero_grad()
        out = model(torch.stack(inputs, dim=0))
        loss = loss_fn(out, torch.tensor(targets, dtype=torch.long))
        loss.backward()
        optimizer.step()

        counters["rule_b_updates"] += 1
        counters["rule_b_presentations"] += len(batch)
        counters["total_optimizer_steps"] += 1

        if step_num in milestones_set:
            t_eval_deadline = time.monotonic() + eval_ceiling
            model.eval()
            recs_m = evaluate_population_records(
                model=model, examples=db["eval_seen_examples"], arm=f"{arm_tag}_step{step_num}",
                seed=seed, step=step_num, dataset_sha256=sha256_b, device=device
            )
            m_m = compute_accuracy_measurement("acc_rule_b_seen", recs_m, sha256_b, "eval_no_dropout")
            milestone_records[step_num] = {
                "step": step_num, "metric": m_m, "records": recs_m, "elapsed_seconds": time.monotonic() - t_start
            }
            if time.monotonic() > t_eval_deadline:
                raise DeadlineExceededError(f"Milestone {step_num} eval deadline exceeded for {arm_tag}")
            model.train()

    t_train_total = time.monotonic() - t_start
    return model, t_train_total, counters, actual_fp, milestone_records


def run_transfer_controls_study(
    dataset_a_path: str,
    sha256_a: str,
    dataset_b_path: str,
    sha256_b: str,
    probe_dataset_path: str,
    sha256_probe: str,
    post_a_checkpoints_dir: str,
    destination: str,
    run_dir: str,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Procedural Master Entry Point for SSRL-07 Sequential Transfer Controls Study."""
    cfg = copy.deepcopy(config or PROPOSED_TRANSFER_CONTROLS_CONFIG)
    verify_transfer_controls_specification_approval(cfg)
    abs_run_dir = reserve_run_directory(run_dir)
    abs_dest = os.path.abspath(destination)

    # Validate dataset contracts and hashes
    config_a = {
        **cfg, "study_id": "SSRL-01", "phase": "learnability_pilot",
        "rule_id": cfg["rule_a_id"], "rule_indices": cfg["rule_a_indices"], "partition_seed": 42
    }
    config_b = {
        **cfg, "study_id": "SSRL-02", "phase": "two_rule_diagnostic",
        "rule_id": cfg["rule_b_id"], "rule_indices": cfg["rule_b_indices"], "partition_seed": 43
    }
    da = load_and_verify_dataset(dataset_a_path, expected_sha256=sha256_a, approved_config=config_a)
    db = load_and_verify_dataset(dataset_b_path, expected_sha256=sha256_b, approved_config=config_b)
    with open(probe_dataset_path, "r", encoding="utf-8") as f:
        dp = json.load(f)
    with open(probe_dataset_path, "rb") as f:
        actual_probe_sha = hashlib.sha256(f.read()).hexdigest()
    if actual_probe_sha != sha256_probe:
        raise ValueError(f"Probe SHA mismatch: expected {sha256_probe}, got {actual_probe_sha}")
    validate_counterfactual_probe_contract(dp, da, db)

    diagnostics: Dict[str, Any] = {
        "study_id": cfg["study_id"],
        "phase": cfg["phase"],
        "classification": cfg["classification"],
        "status": "IN_PROGRESS",
        "current_stage": "INITIALIZATION",
        "seeds_status": {s: "NOT_RUN" for s in cfg["seeds"]},
        "seeds_data": {},
        "paired_transfer_comparisons": [],
        "run_directory": abs_run_dir,
        "destination": abs_dest,
    }
    diag_file = os.path.join(abs_run_dir, "failure_diagnostics.json")

    try:
        t_ceiling_train = float(cfg["wall_clock_ceiling_training_seconds_per_stage"])
        t_ceiling_eval = float(cfg["wall_clock_ceiling_eval_seconds_per_stage"])

        train_examples_b = db["train_examples"]
        sched_b = build_minibatch_schedule(
            train_examples_b, step_budget=cfg["step_budget_b"],
            batch_size=cfg["batch_size"], seed=cfg["minibatch_sampler_seed_b"]
        )

        for seed in cfg["seeds"]:
            diagnostics["seeds_status"][seed] = "IN_PROGRESS"
            seed_record: Dict[str, Any] = {"seed": seed}
            diagnostics["seeds_data"][str(seed)] = seed_record

            # Load and verify post-A checkpoint
            ckpt_post_a_name = f"checkpoint_t1_post_a_s{seed}.pt"
            ckpt_post_a_path = os.path.join(post_a_checkpoints_dir, ckpt_post_a_name)
            if not os.path.isfile(ckpt_post_a_path):
                raise FileNotFoundError(f"Missing post-A checkpoint: {ckpt_post_a_path}")
            post_a_state = torch.load(ckpt_post_a_path, map_location="cpu", weights_only=True)
            fp_post_a = compute_tensor_fingerprint_from_state_dict(post_a_state)
            seed_record["post_a_checkpoint_loaded_fingerprint"] = fp_post_a

            # Reconstruct and verify fresh model initialization
            torch.manual_seed(seed)
            model_fresh_init = SSRLInstructionMLP(input_dim=32, hidden_dim=64, output_dim=2)
            fp_fresh_init = compute_tensor_fingerprint_from_state_dict(model_fresh_init.state_dict())
            seed_record["fresh_init_fingerprint"] = fp_fresh_init

            # 1. Arm 0: FREEZE-AFTER-BASE (Standing Control, AGENTS.md §13 Rule 1)
            diagnostics["current_stage"] = f"seed_{seed}_freeze_after_base_eval"
            model_freeze = SSRLInstructionMLP(input_dim=32, hidden_dim=64, output_dim=2)
            model_freeze.load_state_dict(post_a_state)
            t_deadline_eval = time.monotonic() + t_ceiling_eval
            eval_freeze = evaluate_battery(
                model=model_freeze, da=da, db=db, dp=dp, seed=seed, step=0,
                arm_tag=f"freeze_after_base_s{seed}", sha256_a=sha256_a, sha256_b=sha256_b,
                eval_deadline=t_deadline_eval
            )
            fp_freeze_post = compute_tensor_fingerprint_from_state_dict(model_freeze.state_dict())
            assert fp_freeze_post == fp_post_a, f"Model weights mutated in FREEZE arm for seed {seed}"
            seed_record["freeze_after_base"] = {
                "fingerprint": fp_freeze_post,
                "eval": eval_freeze,
            }

            # 2. Arm 1: fresh_b (Matched Fresh Comparator)
            diagnostics["current_stage"] = f"seed_{seed}_fresh_b_train"
            t_deadline_train = time.monotonic() + t_ceiling_train
            model_fresh, t_train_fresh, cnt_fresh, fp_fresh_start, miles_fresh = train_task_b_arm(
                initial_model_state=model_fresh_init.state_dict(),
                schedule_b=sched_b, db=db, sha256_b=sha256_b, config=cfg, seed=seed,
                arm_tag=f"fresh_b_s{seed}", train_deadline=t_deadline_train,
                eval_ceiling=t_ceiling_eval, expected_fingerprint=fp_fresh_init
            )
            diagnostics["current_stage"] = f"seed_{seed}_fresh_b_final_eval"
            t_deadline_eval = time.monotonic() + t_ceiling_eval
            eval_fresh_final = evaluate_battery(
                model=model_fresh, da=da, db=db, dp=dp, seed=seed, step=cfg["step_budget_b"],
                arm_tag=f"fresh_b_final_s{seed}", sha256_a=sha256_a, sha256_b=sha256_b,
                eval_deadline=t_deadline_eval
            )
            ckpt_fresh = capture_and_verify_checkpoint(
                model_fresh.state_dict(), abs_run_dir, f"checkpoint_fresh_b_s{seed}.pt"
            )
            seed_record["fresh_b"] = {
                "elapsed_train_seconds": t_train_fresh,
                "counters": cnt_fresh,
                "loaded_initial_fingerprint": fp_fresh_start,
                "checkpoint": ckpt_fresh,
                "milestone_evaluations": miles_fresh,
                "final_eval": eval_fresh_final,
            }

            # 3. Arm 2: seq_b (Naive Sequential Transfer)
            diagnostics["current_stage"] = f"seed_{seed}_seq_b_train"
            t_deadline_train = time.monotonic() + t_ceiling_train
            model_seq, t_train_seq, cnt_seq, fp_seq_start, miles_seq = train_task_b_arm(
                initial_model_state=post_a_state,
                schedule_b=sched_b, db=db, sha256_b=sha256_b, config=cfg, seed=seed,
                arm_tag=f"seq_b_s{seed}", train_deadline=t_deadline_train,
                eval_ceiling=t_ceiling_eval, expected_fingerprint=fp_post_a
            )
            diagnostics["current_stage"] = f"seed_{seed}_seq_b_final_eval"
            t_deadline_eval = time.monotonic() + t_ceiling_eval
            eval_seq_final = evaluate_battery(
                model=model_seq, da=da, db=db, dp=dp, seed=seed, step=cfg["step_budget_b"],
                arm_tag=f"seq_b_final_s{seed}", sha256_a=sha256_a, sha256_b=sha256_b,
                eval_deadline=t_deadline_eval
            )
            ckpt_seq = capture_and_verify_checkpoint(
                model_seq.state_dict(), abs_run_dir, f"checkpoint_seq_b_s{seed}.pt"
            )
            seed_record["seq_b"] = {
                "elapsed_train_seconds": t_train_seq,
                "counters": cnt_seq,
                "loaded_initial_fingerprint": fp_seq_start,
                "checkpoint": ckpt_seq,
                "milestone_evaluations": miles_seq,
                "final_eval": eval_seq_final,
            }

            # Milestone comparison and transfer deltas
            milestone_deltas: Dict[str, Any] = {}
            steps_to_threshold_seq: Optional[int] = None
            steps_to_threshold_fresh: Optional[int] = None

            for step_k in cfg["milestone_steps"]:
                acc_seq_num = miles_seq[step_k]["metric"].numerator
                acc_fresh_num = miles_fresh[step_k]["metric"].numerator
                acc_seq_pct = miles_seq[step_k]["metric"].pct
                acc_fresh_pct = miles_fresh[step_k]["metric"].pct
                delta_pp = ((acc_seq_num - acc_fresh_num) / 192.0) * 100.0

                milestone_deltas[str(step_k)] = {
                    "step": step_k,
                    "seq_b_numerator": acc_seq_num,
                    "seq_b_pct": acc_seq_pct,
                    "fresh_b_numerator": acc_fresh_num,
                    "fresh_b_pct": acc_fresh_pct,
                    "delta_transfer_pp": delta_pp,
                }
                if steps_to_threshold_seq is None and (acc_seq_num / 192.0) >= cfg["gate_g_lc_threshold"]:
                    steps_to_threshold_seq = step_k
                if steps_to_threshold_fresh is None and (acc_fresh_num / 192.0) >= cfg["gate_g_lc_threshold"]:
                    steps_to_threshold_fresh = step_k

            # Retention drop on Rule A
            acc_a_freeze_num = eval_freeze["metrics"]["acc_rule_a_seen"].numerator
            acc_a_seq_num = eval_seq_final["metrics"]["acc_rule_a_seen"].numerator
            forgetting_a_num = acc_a_freeze_num - acc_a_seq_num
            forgetting_a_pp = (forgetting_a_num / 192.0) * 100.0

            transfer_entry = {
                "seed": seed,
                "milestone_deltas": milestone_deltas,
                "steps_to_90pct_seq_b": steps_to_threshold_seq,
                "steps_to_90pct_fresh_b": steps_to_threshold_fresh,
                "final_transfer_delta_step500_pp": milestone_deltas[str(cfg["step_budget_b"])]["delta_transfer_pp"],
                "rule_a_freeze_numerator": acc_a_freeze_num,
                "rule_a_freeze_pct": eval_freeze["metrics"]["acc_rule_a_seen"].pct,
                "rule_a_seq_b_numerator": acc_a_seq_num,
                "rule_a_seq_b_pct": eval_seq_final["metrics"]["acc_rule_a_seen"].pct,
                "rule_a_forgetting_numerator": forgetting_a_num,
                "rule_a_forgetting_pp": forgetting_a_pp,
            }
            seed_record["transfer_comparison"] = transfer_entry
            diagnostics["paired_transfer_comparisons"].append(transfer_entry)
            diagnostics["seeds_status"][seed] = "COMPLETED"

        # Summary statistics across completed seeds
        def stats(arr: List[float]) -> Dict[str, Any]:
            if not arr:
                return {"per_seed": [], "mean": 0.0, "spread": 0.0, "min": 0.0, "max": 0.0}
            return {
                "per_seed": arr, "mean": sum(arr) / float(len(arr)),
                "spread": max(arr) - min(arr), "min": min(arr), "max": max(arr)
            }

        stats_dict: Dict[str, Any] = {}
        for step_k in cfg["milestone_steps"]:
            k_str = str(step_k)
            deltas_k = [
                entry["milestone_deltas"][k_str]["delta_transfer_pp"]
                for entry in diagnostics["paired_transfer_comparisons"]
                if k_str in entry["milestone_deltas"]
            ]
            stats_dict[f"transfer_delta_step{k_str}_pp"] = stats(deltas_k)

        ret_drops = [entry["rule_a_forgetting_pp"] for entry in diagnostics["paired_transfer_comparisons"]]
        stats_dict["naive_forgetting_rule_a_pp"] = stats(ret_drops)
        diagnostics["summary_statistics"] = stats_dict
        diagnostics["status"] = "COMPLETED"
        diagnostics["current_stage"] = "COMPLETED"

        results_payload = {"config": cfg, "diagnostics": diagnostics}
        atomic_write_json(make_json_serializable(results_payload), abs_dest, refuse_overwrite=True)
        return results_payload

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
        raise exc


def main() -> None:
    parser = argparse.ArgumentParser(description="SSRL-07 Sequential Transfer Controls Runner")
    parser.add_argument("--dataset_a", required=True, help="Path to Rule A dataset JSON")
    parser.add_argument("--dataset_b", required=True, help="Path to Rule B dataset JSON")
    parser.add_argument("--probe_dataset", required=True, help="Path to probe dataset JSON")
    parser.add_argument("--post_a_checkpoints_dir", required=True, help="Directory containing checkpoint_t1_post_a_s<seed>.pt")
    parser.add_argument("--run_dir", required=True, help="Unique run directory path")
    parser.add_argument("--destination", required=True, help="Path to output results.json")
    parser.add_argument("--sha256_a", default="00a21efad1dc07b826198d99b0b702aedddad57b6226f99301afbf16327fd0b7")
    parser.add_argument("--sha256_b", default="27e813e73bf66f51e1bff07f7d2f19945e830afc6d5876b0cb4d11136cda4cd4")
    parser.add_argument("--sha256_probe", default="1c07d50fed772cfcff86e3bf30cc74a95cdc24d190ec767aa15365671fe06319")
    args = parser.parse_args()

    run_transfer_controls_study(
        dataset_a_path=args.dataset_a,
        sha256_a=args.sha256_a,
        dataset_b_path=args.dataset_b,
        sha256_b=args.sha256_b,
        probe_dataset_path=args.probe_dataset,
        sha256_probe=args.sha256_probe,
        post_a_checkpoints_dir=args.post_a_checkpoints_dir,
        destination=args.destination,
        run_dir=args.run_dir,
    )


if __name__ == "__main__":
    main()
