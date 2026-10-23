"""
research/lifelong_01/run_ssrl_paired_replication.py -- Procedural Runner for SSRL-06.
Prospective Three-Seed Paired-Initialization Replication Study comparing Lowest-ID
Replay vs Pattern-Diverse Replay on Seeds 102, 103, 104.

Invariants:
- Evaluates seeds 102, 103, 104 sequentially.
- Fixed existing datasets; fixed A sampler seed 2026, fixed B sampler seed 2027.
- For each seed:
  1. Train Phase A for 2,000 steps; assert Gate G-LC (>= 90.0% = 173/192); capture checkpoint.
  2. Fork Arm 1 (Lowest-ID Replay): 625 steps (4 B : 1 A, M=16 lowest-ID buffer).
  3. Fork Arm 2 (Pattern-Diverse Replay): 625 steps (4 B : 1 A, M=16 diverse buffer) from
     identical starting post-A checkpoint and identical Rule B minibatch schedule.
- Separate 60.0s deadline per training stage and 60.0s per evaluation stage.
- Immediate halt on any failure; persists diagnostics; records later stages as NOT_RUN.
- Maximum budget: 9,750 updates and 156,000 presentations total across all 3 seeds.
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
from research.lifelong_01.run_ssrl_replay_baseline import (
    build_replay_schedule,
    select_replay_memory,
)
from research.lifelong_01.run_ssrl_diverse_replay import (
    compute_per_pattern_retention,
    select_pattern_diverse_replay_memory,
)
from research.lifelong_01.run_ssrl_joint_reference import evaluate_population_records

PROPOSED_PAIRED_REPLICATION_CONFIG: Dict[str, Any] = {
    "study_id": "SSRL-06",
    "phase": "paired_replication_prospective_three_seed",
    "classification": "prospective_three_seed_replication",
    "gate_approval_status": "APPROVED",
    "seeds": [102, 103, 104],
    "exploratory_reference_seed": 101,
    "minibatch_sampler_seed_a": 2026,
    "minibatch_sampler_seed_b": 2027,
    "rule_a_id": 1,
    "rule_b_id": 2,
    "rule_a_indices": [0, 1, 2, 3],
    "rule_b_indices": [4, 5, 6, 7],
    "train_pool_size": 384,
    "batch_size": 16,
    "step_budget_a_per_seed": 2000,
    "step_budget_b_per_arm": 500,
    "step_budget_replay_per_arm": 125,
    "step_budget_total_per_arm": 625,
    "step_budget_total_per_seed": 3250,
    "step_budget_total_all_seeds": 9750,
    "presentations_total_all_seeds": 156000,
    "replay_buffer_size": 16,
    "replay_ratio_b_to_a": [4, 1],
    "replay_cycles": 125,
    "gate_g_lc_threshold": 0.90,
    "learning_rate": 0.001,
    "betas": [0.9, 0.999],
    "eps": 1e-08,
    "weight_decay": 0.0,
    "wall_clock_ceiling_training_seconds_per_stage": 60.0,
    "wall_clock_ceiling_eval_seconds_per_stage": 60.0,
}


def verify_paired_replication_specification_approval(config: Dict[str, Any]) -> None:
    """AGENTS.md Section 2 / Section 8: Strictly blocks unapproved execution."""
    status = config.get("gate_approval_status")
    if status != "APPROVED":
        raise SpecificationApprovalError(
            f"SSRL-06 paired replication specification status is '{status}', not 'APPROVED'. "
            f"Execution is forbidden before explicit user authorization."
        )


def compute_schedule_hash(schedule: List[Tuple[str, int, List[Dict[str, Any]]]]) -> str:
    """Computes a deterministic hash of an interleaved minibatch schedule."""
    repr_str = "|".join(
        f"{tag}:{rule_id}:" + ",".join(str(ex["example_id"]) for ex in batch)
        for tag, rule_id, batch in schedule
    )
    return hashlib.sha256(repr_str.encode("utf-8")).hexdigest()


def evaluate_arm(
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
    """Runs frozen evaluation across all required populations with strict deadline check."""
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
    raw_a_seen: List[SSRLExampleRecord] = []

    for split_tag, examples, d_hash in eval_splits:
        recs = evaluate_population_records(
            model=model, examples=examples, arm=arm_tag,
            seed=seed, step=step, dataset_sha256=d_hash, device=device
        )
        eval_records[split_tag] = recs
        metrics[f"acc_{split_tag}"] = compute_accuracy_measurement(
            name=f"acc_{split_tag}", records=recs, input_set=d_hash, mode="eval_no_dropout"
        )
        if split_tag == "rule_a_seen":
            raw_a_seen = recs

    probe_results = evaluate_counterfactual_probe(
        model=model, probe_examples=dp["probe_examples"], device=device
    )

    if time.monotonic() > eval_deadline:
        raise DeadlineExceededError(f"Evaluation deadline exceeded while finishing {arm_tag}")

    per_pattern_a_seen = compute_per_pattern_retention(raw_a_seen, [0, 1, 2, 3])

    return {
        "elapsed_eval_seconds": time.monotonic() - t_start,
        "metrics": metrics,
        "per_pattern_rule_a_seen": per_pattern_a_seen,
        "counterfactual_probe": probe_results,
        "eval_records": eval_records,
    }


def train_single_arm(
    initial_model_state: Dict[str, torch.Tensor],
    schedule: List[Tuple[str, int, List[Dict[str, Any]]]],
    config: Dict[str, Any],
    train_deadline: float,
    expected_fingerprint: Optional[str] = None,
) -> Tuple[SSRLInstructionMLP, float, Dict[str, int], str]:
    """Trains a single model arm on an interleaved schedule with fresh Adam."""
    if time.monotonic() > train_deadline:
        raise DeadlineExceededError("Training deadline exceeded before starting arm training")

    model = SSRLInstructionMLP(input_dim=32, hidden_dim=64, output_dim=2)
    model.load_state_dict(initial_model_state)
    actual_fp = compute_tensor_fingerprint_from_state_dict(model.state_dict())
    if expected_fingerprint is not None and actual_fp != expected_fingerprint:
        raise CheckpointVerificationError(
            f"Initial model fingerprint mismatch before training: expected {expected_fingerprint}, got {actual_fp}"
        )
    model.train()

    optimizer = optim.Adam(
        model.parameters(),
        lr=config["learning_rate"], betas=tuple(config["betas"]),
        eps=config["eps"], weight_decay=config["weight_decay"],
    )
    loss_fn = nn.CrossEntropyLoss()
    counters = {
        "rule_a_updates": 0, "rule_a_presentations": 0,
        "rule_b_updates": 0, "rule_b_presentations": 0,
        "replay_a_updates": 0, "replay_a_presentations": 0,
        "total_optimizer_steps": 0, "total_presentations": 0,
    }
    t_start = time.monotonic()

    for step_idx, (tag, rule_id, batch) in enumerate(schedule):
        if time.monotonic() > train_deadline:
            raise DeadlineExceededError(f"Training deadline exceeded at update {step_idx}")

        inputs = [torch.tensor(ex["features"] + ex["mask"], dtype=torch.float32) for ex in batch]
        targets = [int(ex["target_class"]) for ex in batch]

        optimizer.zero_grad()
        out = model(torch.stack(inputs, dim=0))
        loss = loss_fn(out, torch.tensor(targets, dtype=torch.long))
        loss.backward()
        optimizer.step()

        batch_len = len(batch)
        if tag == "rule_a":
            counters["rule_a_updates"] += 1
            counters["rule_a_presentations"] += batch_len
        elif tag == "rule_b":
            counters["rule_b_updates"] += 1
            counters["rule_b_presentations"] += batch_len
        elif tag == "replay_a":
            counters["replay_a_updates"] += 1
            counters["replay_a_presentations"] += batch_len
        counters["total_optimizer_steps"] += 1
        counters["total_presentations"] += batch_len

    if time.monotonic() > train_deadline:
        raise DeadlineExceededError("Training deadline exceeded after completing schedule updates")

    return model, time.monotonic() - t_start, counters, actual_fp


def run_paired_replication_study(
    dataset_a_path: str,
    sha256_a: str,
    dataset_b_path: str,
    sha256_b: str,
    probe_dataset_path: str,
    sha256_probe: str,
    destination: str,
    run_dir: str,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Procedural Master Entry Point for SSRL-06 Paired Replication Study."""
    cfg = copy.deepcopy(config or PROPOSED_PAIRED_REPLICATION_CONFIG)
    verify_paired_replication_specification_approval(cfg)
    abs_run_dir = reserve_run_directory(run_dir)
    abs_dest = os.path.abspath(destination)

    # Validate dataset contracts and hashes
    config_a = {
        **cfg,
        "study_id": "SSRL-01",
        "phase": "learnability_pilot",
        "rule_id": cfg["rule_a_id"],
        "rule_indices": cfg["rule_a_indices"],
        "partition_seed": 42,
    }
    config_b = {
        **cfg,
        "study_id": "SSRL-02",
        "phase": "two_rule_diagnostic",
        "rule_id": cfg["rule_b_id"],
        "rule_indices": cfg["rule_b_indices"],
        "partition_seed": 43,
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
        "paired_differences": [],
        "run_directory": abs_run_dir,
        "destination": abs_dest,
    }
    diag_file = os.path.join(abs_run_dir, "failure_diagnostics.json")

    try:
        t_ceiling_train = float(cfg["wall_clock_ceiling_training_seconds_per_stage"])
        t_ceiling_eval = float(cfg["wall_clock_ceiling_eval_seconds_per_stage"])

        train_examples_a = da["train_examples"]
        train_examples_b = db["train_examples"]

        # Deterministic replay memories
        mem_lowest_id = select_replay_memory(train_examples_a, buffer_size=cfg["replay_buffer_size"])
        mem_diverse = select_pattern_diverse_replay_memory(train_examples_a, buffer_size=cfg["replay_buffer_size"])
        manifest_lowest_id = [int(ex["example_id"]) for ex in mem_lowest_id]
        manifest_diverse = [int(ex["example_id"]) for ex in mem_diverse]

        # Fixed Rule B minibatch schedule
        sched_b = build_minibatch_schedule(
            train_examples_b, step_budget=cfg["step_budget_b_per_arm"],
            batch_size=cfg["batch_size"], seed=cfg["minibatch_sampler_seed_b"]
        )
        sched_lowest_id = build_replay_schedule(sched_b, mem_lowest_id, cfg["replay_ratio_b_to_a"][0], cfg["replay_ratio_b_to_a"][1])
        sched_diverse = build_replay_schedule(sched_b, mem_diverse, cfg["replay_ratio_b_to_a"][0], cfg["replay_ratio_b_to_a"][1])

        hash_sched_lowest_id = compute_schedule_hash(sched_lowest_id)
        hash_sched_diverse = compute_schedule_hash(sched_diverse)

        for seed in cfg["seeds"]:
            diagnostics["current_stage"] = f"seed_{seed}_phase_a"
            diagnostics["seeds_status"][seed] = "IN_PROGRESS"
            seed_record: Dict[str, Any] = {
                "seed": seed,
                "memory_manifest_lowest_id": manifest_lowest_id,
                "memory_manifest_diverse": manifest_diverse,
                "schedule_hash_lowest_id": hash_sched_lowest_id,
                "schedule_hash_diverse": hash_sched_diverse,
            }
            diagnostics["seeds_data"][str(seed)] = seed_record

            # 1. Phase A Training
            torch.manual_seed(seed)
            model_a = SSRLInstructionMLP(input_dim=32, hidden_dim=64, output_dim=2)
            sched_a = build_minibatch_schedule(
                train_examples_a, step_budget=cfg["step_budget_a_per_seed"],
                batch_size=cfg["batch_size"], seed=cfg["minibatch_sampler_seed_a"]
            )
            tagged_sched_a = [("rule_a", 1, batch) for batch in sched_a]

            t_deadline_train_a = time.monotonic() + t_ceiling_train
            model_a, t_train_a, counters_a, fp_init_a = train_single_arm(
                model_a.state_dict(), tagged_sched_a, cfg, t_deadline_train_a
            )
            seed_record["elapsed_train_seconds_phase_a"] = t_train_a
            seed_record["counters_phase_a"] = counters_a
            seed_record["initial_fingerprint_phase_a"] = fp_init_a

            # Phase A Evaluation (Gate G-LC)
            diagnostics["current_stage"] = f"seed_{seed}_phase_a_eval"
            t_deadline_eval_a = time.monotonic() + t_ceiling_eval
            if time.monotonic() > t_deadline_eval_a:
                raise DeadlineExceededError(f"Evaluation deadline exceeded before Phase A evaluation for seed {seed}")
            recs_post_a = evaluate_population_records(
                model=model_a, examples=da["eval_seen_examples"], arm=f"post_a_s{seed}",
                seed=seed, step=cfg["step_budget_a_per_seed"], dataset_sha256=sha256_a, device=torch.device("cpu")
            )
            if time.monotonic() > t_deadline_eval_a:
                raise DeadlineExceededError(f"Evaluation deadline exceeded after Phase A evaluation for seed {seed}")
            m_post_a = compute_accuracy_measurement(
                name="post_a_seen_accuracy", records=recs_post_a, input_set=sha256_a, mode="eval_no_dropout"
            )
            seed_record["post_a_seen_accuracy"] = m_post_a
            seed_record["eval_records_post_a"] = recs_post_a

            if m_post_a.rate < cfg["gate_g_lc_threshold"]:
                raise AssertionError(
                    f"Seed {seed} failed Gate G-LC: accuracy {m_post_a.numerator}/{m_post_a.denominator} "
                    f"({m_post_a.pct:.2f}%) < threshold {cfg['gate_g_lc_threshold']*100:.1f}%."
                )

            # Checkpoint capture and reload equality verification
            ckpt_post_a = capture_and_verify_checkpoint(
                model_a.state_dict(), abs_run_dir, f"checkpoint_t1_post_a_s{seed}.pt"
            )
            seed_record["checkpoint_post_a"] = ckpt_post_a
            fp_post_a = ckpt_post_a["tensor_fingerprint"]

            # Run paired arms from verified identical checkpoint state
            arms_spec = [
                ("arm_lowest_id", sched_lowest_id, f"checkpoint_lowest_id_final_s{seed}.pt"),
                ("arm_diverse", sched_diverse, f"checkpoint_diverse_replay_final_s{seed}.pt"),
            ]
            arm_results: Dict[str, Any] = {}

            for arm_tag, sched_arm, ckpt_filename in arms_spec:
                diagnostics["current_stage"] = f"seed_{seed}_{arm_tag}_train"
                t_deadline_train = time.monotonic() + t_ceiling_train
                model_arm, t_train_arm, counters_arm, loaded_fp = train_single_arm(
                    model_a.state_dict(), sched_arm, cfg, t_deadline_train,
                    expected_fingerprint=fp_post_a
                )
                assert loaded_fp == fp_post_a, f"Loaded fingerprint mismatch for arm {arm_tag}"

                diagnostics["current_stage"] = f"seed_{seed}_{arm_tag}_eval"
                t_deadline_eval = time.monotonic() + t_ceiling_eval
                eval_res = evaluate_arm(
                    model_arm, da, db, dp, seed,
                    cfg["step_budget_a_per_seed"] + cfg["step_budget_total_per_arm"],
                    f"{arm_tag}_s{seed}", sha256_a, sha256_b, t_deadline_eval
                )
                ckpt_arm = capture_and_verify_checkpoint(
                    model_arm.state_dict(), abs_run_dir, ckpt_filename
                )

                seed_record[arm_tag] = {
                    "elapsed_train_seconds": t_train_arm,
                    "counters": counters_arm,
                    "loaded_initial_fingerprint": loaded_fp,
                    "checkpoint": ckpt_arm,
                    "eval": eval_res,
                }
                arm_results[arm_tag] = eval_res

            # Paired comparison metrics for seed
            acc_a_lid = arm_results["arm_lowest_id"]["metrics"]["acc_rule_a_seen"].numerator
            acc_a_div = arm_results["arm_diverse"]["metrics"]["acc_rule_a_seen"].numerator
            acc_b_lid = arm_results["arm_lowest_id"]["metrics"]["acc_rule_b_seen"].numerator
            acc_b_div = arm_results["arm_diverse"]["metrics"]["acc_rule_b_seen"].numerator

            post_a_num = m_post_a.numerator
            forg_lid = post_a_num - acc_a_lid
            forg_div = post_a_num - acc_a_div

            paired_entry = {
                "seed": seed,
                "post_a_seen_numerator": post_a_num,
                "post_a_seen_pct": m_post_a.pct,
                "lowest_id_seen_retention_numerator": acc_a_lid,
                "lowest_id_seen_retention_pct": arm_results["arm_lowest_id"]["metrics"]["acc_rule_a_seen"].pct,
                "lowest_id_seen_forgetting_numerator": forg_lid,
                "lowest_id_seen_forgetting_pp": (forg_lid / 192.0) * 100.0,
                "diverse_seen_retention_numerator": acc_a_div,
                "diverse_seen_retention_pct": arm_results["arm_diverse"]["metrics"]["acc_rule_a_seen"].pct,
                "diverse_seen_forgetting_numerator": forg_div,
                "diverse_seen_forgetting_pp": (forg_div / 192.0) * 100.0,
                "primary_delta_retention_pp": ((acc_a_div - acc_a_lid) / 192.0) * 100.0,
                "lowest_id_b_seen_numerator": acc_b_lid,
                "lowest_id_b_seen_pct": arm_results["arm_lowest_id"]["metrics"]["acc_rule_b_seen"].pct,
                "diverse_b_seen_numerator": acc_b_div,
                "diverse_b_seen_pct": arm_results["arm_diverse"]["metrics"]["acc_rule_b_seen"].pct,
                "secondary_delta_acquisition_pp": ((acc_b_div - acc_b_lid) / 192.0) * 100.0,
            }
            seed_record["paired_comparison"] = paired_entry
            diagnostics["paired_differences"].append(paired_entry)
            diagnostics["seeds_data"][str(seed)] = seed_record
            diagnostics["seeds_status"][seed] = "COMPLETED"

        # Summary statistics across completed seeds
        ret_deltas = [entry["primary_delta_retention_pp"] for entry in diagnostics["paired_differences"]]
        acq_deltas = [entry["secondary_delta_acquisition_pp"] for entry in diagnostics["paired_differences"]]

        diagnostics["summary_statistics"] = {
            "primary_retention_delta_pp": {
                "per_seed": ret_deltas, "mean": sum(ret_deltas) / float(len(ret_deltas)),
                "spread": max(ret_deltas) - min(ret_deltas), "min": min(ret_deltas), "max": max(ret_deltas),
            },
            "secondary_acquisition_delta_pp": {
                "per_seed": acq_deltas, "mean": sum(acq_deltas) / float(len(acq_deltas)),
                "spread": max(acq_deltas) - min(acq_deltas), "min": min(acq_deltas), "max": max(acq_deltas),
            }
        }
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
    parser = argparse.ArgumentParser(description="SSRL-06 Paired Replication Runner")
    parser.add_argument("--dataset_a", required=True, help="Path to Rule A dataset JSON")
    parser.add_argument("--dataset_b", required=True, help="Path to Rule B dataset JSON")
    parser.add_argument("--probe_dataset", required=True, help="Path to probe dataset JSON")
    parser.add_argument("--run_dir", required=True, help="Unique run directory path")
    parser.add_argument("--destination", required=True, help="Path to output results.json")
    parser.add_argument("--sha256_a", default="00a21efad1dc07b826198d99b0b702aedddad57b6226f99301afbf16327fd0b7")
    parser.add_argument("--sha256_b", default="27e813e73bf66f51e1bff07f7d2f19945e830afc6d5876b0cb4d11136cda4cd4")
    parser.add_argument("--sha256_probe", default="1c07d50fed772cfcff86e3bf30cc74a95cdc24d190ec767aa15365671fe06319")
    args = parser.parse_args()

    run_paired_replication_study(
        dataset_a_path=args.dataset_a,
        sha256_a=args.sha256_a,
        dataset_b_path=args.dataset_b,
        sha256_b=args.sha256_b,
        probe_dataset_path=args.probe_dataset,
        sha256_probe=args.sha256_probe,
        destination=args.destination,
        run_dir=args.run_dir,
    )


if __name__ == "__main__":
    main()
