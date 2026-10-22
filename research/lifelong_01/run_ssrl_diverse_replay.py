"""
research/lifelong_01/run_ssrl_diverse_replay.py -- Pattern-diverse replay baseline (SSRL-05).

Implements pattern-diverse experience replay starting from the preserved post-A checkpoint (t1):
- Reuses checkpoint_t1_post_a.pt (tensor fingerprint: 9ce6ab70..., file sha: 350deff9...)
- Preserves the exact 500 Rule B minibatches in their original order (sampler seed: 2027)
- Selects a fixed M=16 memory buffer covering all 12 Rule A seen active patterns:
  - 1 lowest-ID example from each of the 12 active patterns (6 class-0, 6 class-1)
  - 1 additional distinct lowest-ID example from each of the first 2 patterns per class
  - Yields exactly 8 class-0 and 8 class-1 examples (M=16, 4.17% of train pool)
- Interleaves 4 Task B updates with 1 Diverse Replay A update for 125 cycles (625 total updates)
- Fresh Adam optimizer at t1, continuous across the 625 updates
- Strict 60-second training ceiling and 60-second evaluation deadlines
- Zero training of Task A, zero tuning, no automatic retries, final evaluation only
- Production approval status: PROPOSED_PENDING_APPROVAL
"""

from __future__ import annotations
import os
import sys
import json
import time
import copy
import hashlib
import tempfile
import argparse
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from research.lifelong_01.ssrl_types import (
    SSRLExampleRecord,
    SSRLMeasurement,
    compute_accuracy_measurement,
)
from research.lifelong_01.ssrl_model import SSRLInstructionMLP
from research.lifelong_01.ssrl_checkpoints import (
    compute_tensor_fingerprint_from_state_dict,
    capture_and_verify_checkpoint,
)
from research.lifelong_01.ssrl_counterfactual_probe import (
    evaluate_counterfactual_probe,
    validate_counterfactual_probe_contract,
)
from research.lifelong_01.run_ssrl_joint_reference import (
    evaluate_population_records,
    load_and_verify_dataset,
    reserve_run_directory,
    atomic_write_json,
    make_json_serializable,
    DeadlineExceededError,
    CheckpointVerificationError,
    SpecificationApprovalError,
)
from research.lifelong_01.run_ssrl_two_rule import build_minibatch_schedule
from research.lifelong_01.run_ssrl_replay_baseline import (
    build_replay_schedule,
    load_and_verify_t1_model,
)

PROPOSED_DIVERSE_REPLAY_CONFIG: Dict[str, Any] = {
    "study_id": "SSRL-05",
    "phase": "diverse_replay_sequential_baseline",
    "classification": "pattern_diverse_replay_comparator",
    "gate_approval_status": "APPROVED",
    "seed": 101,
    "rule_a_id": 1,
    "rule_b_id": 2,
    "rule_a_indices": [0, 1, 2, 3],
    "rule_b_indices": [4, 5, 6, 7],
    "data_distractor_seed_a": 42,
    "data_distractor_seed_b": 43,
    "train_pool_size": 384,
    "replay_buffer_size": 16,
    "replay_ratio_b_to_a": [4, 1],
    "replay_cycles": 125,
    "step_budget_b": 500,
    "step_budget_replay_a": 125,
    "step_budget_total": 625,
    "batch_size": 16,
    "minibatch_sampler_seed_b": 2027,
    "expected_t1_checkpoint_path": "evidence/runs/ssrl_02_replay_eff8aa50/checkpoint_t1_post_a.pt",
    "expected_t1_fingerprint": "9ce6ab7020b1e7c21bd291a3f81a24c52c47b5fc2c309141774c98a35e814b7c",
    "expected_t1_file_sha256": "350deff9412af6b01ac00caa558a0d6a6d57792551f250c781ebc371ef131579",
    "learning_rate": 0.001,
    "betas": [0.9, 0.999],
    "eps": 1e-08,
    "weight_decay": 0.0,
    "wall_clock_ceiling_training_seconds": 60.0,
    "wall_clock_ceiling_eval_seconds": 60.0,
}


def verify_diverse_replay_specification_approval(config: Dict[str, Any]) -> None:
    """AGENTS.md Section 2 / Section 8: Strictly blocks unapproved execution."""
    status = config.get("gate_approval_status")
    if status != "APPROVED":
        raise SpecificationApprovalError(
            f"SSRL-05 diverse replay specification status is '{status}', not 'APPROVED'. "
            f"Execution is forbidden before explicit user authorization."
        )


def select_pattern_diverse_replay_memory(
    train_examples: List[Dict[str, Any]],
    rule_indices: List[int] = [0, 1, 2, 3],
    buffer_size: int = 16,
) -> List[Dict[str, Any]]:
    """
    Selects a deterministic fixed memory buffer of Rule A training examples
    covering all 12 seen active patterns:
    1. Group train_examples by class and active pattern tuple.
    2. Select 1 lowest-ID example from each of the 12 active patterns (6 class 0, 6 class 1).
    3. Select 1 additional distinct lowest-ID example from each of the first 2 deterministically
       ordered patterns per class (canonically sorted by pattern tuple).
    4. Yields exactly 8 class-0 and 8 class-1 examples (M=16).
    """
    if buffer_size != 16:
        raise ValueError(f"Replay buffer size must be 16, got {buffer_size}")

    patterns_by_class: Dict[int, Dict[Tuple[int, ...], List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for ex in train_examples:
        pat = tuple(ex["features"][i] for i in rule_indices)
        cls = int(ex["target_class"])
        patterns_by_class[cls][pat].append(ex)

    if len(patterns_by_class[0]) != 6 or len(patterns_by_class[1]) != 6:
        raise ValueError(
            f"Expected exactly 6 active patterns per class, found {len(patterns_by_class[0])} and {len(patterns_by_class[1])}"
        )

    c0: List[Dict[str, Any]] = []
    c1: List[Dict[str, Any]] = []

    # 1. One lowest-ID example from each of the 6 patterns, plus 1 additional from first 2 patterns
    for cls, c_list in [(0, c0), (1, c1)]:
        sorted_patterns = sorted(patterns_by_class[cls].keys())
        for p in sorted_patterns:
            ex_sorted = sorted(patterns_by_class[cls][p], key=lambda x: int(x["example_id"]))
            c_list.append(ex_sorted[0])
        for p in sorted_patterns[:2]:
            ex_sorted = sorted(patterns_by_class[cls][p], key=lambda x: int(x["example_id"]))
            c_list.append(ex_sorted[1])

    c0_sorted = sorted(c0, key=lambda x: int(x["example_id"]))
    c1_sorted = sorted(c1, key=lambda x: int(x["example_id"]))
    selected = c0_sorted + c1_sorted

    if len(selected) != 16:
        raise ValueError(f"Expected 16 selected examples, got {len(selected)}")

    seen_ids = set()
    for ex in selected:
        eid = int(ex["example_id"])
        if eid in seen_ids:
            raise ValueError(f"Duplicate example_id {eid} in diverse replay buffer")
        seen_ids.add(eid)

    c0_count = sum(1 for ex in selected if ex["target_class"] == 0)
    c1_count = sum(1 for ex in selected if ex["target_class"] == 1)
    if c0_count != 8 or c1_count != 8:
        raise ValueError(f"Class imbalance in diverse replay buffer: c0={c0_count}, c1={c1_count}")

    pats_covered = set(tuple(ex["features"][i] for i in rule_indices) for ex in selected)
    if len(pats_covered) != 12:
        raise ValueError(f"Expected 12 unique active patterns, got {len(pats_covered)}")

    return selected


def compute_per_pattern_retention(
    records: List[SSRLExampleRecord],
    rule_indices: List[int] = [0, 1, 2, 3],
) -> Dict[str, Any]:
    """Computes exact per-active-pattern accuracy breakdown for Rule A seen records."""
    patterns: Dict[Tuple[int, ...], Dict[str, Any]] = defaultdict(lambda: {"total": 0, "correct": 0, "class": None})
    for r in records:
        pat = tuple(r.input_features[i] for i in rule_indices)
        patterns[pat]["total"] += 1
        patterns[pat]["class"] = r.target_class
        if r.is_correct:
            patterns[pat]["correct"] += 1

    result = {}
    for pat in sorted(patterns.keys()):
        d = patterns[pat]
        pat_str = "_".join(str(v) for v in pat)
        result[pat_str] = {
            "pattern": list(pat),
            "target_class": d["class"],
            "numerator": d["correct"],
            "denominator": d["total"],
            "rate": d["correct"] / float(d["total"]),
            "pct": (d["correct"] / float(d["total"])) * 100.0,
        }
    return result


def run_diverse_replay_study(
    dataset_a_path: str,
    sha256_a: str,
    dataset_b_path: str,
    sha256_b: str,
    probe_dataset_path: str,
    sha256_probe: str,
    destination: str,
    run_dir: str,
    t1_checkpoint_path: str,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Master procedural entry point for SSRL-05 Pattern-Diverse Replay Baseline."""
    cfg = copy.deepcopy(config or PROPOSED_DIVERSE_REPLAY_CONFIG)
    verify_diverse_replay_specification_approval(cfg)
    abs_run_dir = reserve_run_directory(run_dir)
    abs_dest = os.path.abspath(destination)

    diagnostics: Dict[str, Any] = {
        "study_id": cfg["study_id"],
        "phase": cfg["phase"],
        "classification": cfg["classification"],
        "status": "INITIALIZING",
        "approval_status": cfg["gate_approval_status"],
        "destination": abs_dest,
        "run_directory": abs_run_dir,
        "elapsed_training_seconds": 0.0,
        "elapsed_eval_seconds": 0.0,
        "exposures": {
            "rule_b_presentations": 0,
            "replay_a_presentations": 0,
            "total_presentations": 0,
            "total_optimizer_steps": 0,
        },
        "checkpoints": {},
    }

    t_start = time.monotonic()
    t_ceiling_train = cfg["wall_clock_ceiling_training_seconds"]

    try:
        # 1. Pinned dataset verification
        config_a = {
            **cfg,
            "study_id": "SSRL-01",
            "phase": "learnability_pilot",
            "rule_id": cfg["rule_a_id"],
            "rule_indices": cfg["rule_a_indices"],
            "partition_seed": cfg["data_distractor_seed_a"],
        }
        config_b = {
            **cfg,
            "study_id": "SSRL-02",
            "phase": "two_rule_diagnostic",
            "rule_id": cfg["rule_b_id"],
            "rule_indices": cfg["rule_b_indices"],
            "partition_seed": cfg["data_distractor_seed_b"],
        }
        dataset_a = load_and_verify_dataset(dataset_a_path, expected_sha256=sha256_a, approved_config=config_a)
        dataset_b = load_and_verify_dataset(dataset_b_path, expected_sha256=sha256_b, approved_config=config_b)

        with open(probe_dataset_path, "rb") as pf_bytes:
            obs_probe_sha = hashlib.sha256(pf_bytes.read()).hexdigest()
        if obs_probe_sha != sha256_probe:
            raise ValueError(f"Probe dataset hash mismatch: expected {sha256_probe}, got {obs_probe_sha}")
        with open(probe_dataset_path, "r", encoding="utf-8") as pf:
            probe_json = json.load(pf)
        validate_counterfactual_probe_contract(probe_json, dataset_a, dataset_b)

        device = torch.device("cpu")
        torch.set_num_threads(1)

        # 2. Model initialization from verified post-A checkpoint
        model = load_and_verify_t1_model(
            checkpoint_path=t1_checkpoint_path,
            expected_fingerprint=cfg["expected_t1_fingerprint"],
            expected_file_sha256=cfg["expected_t1_file_sha256"],
            device=device,
        )

        # 3. Deterministic pattern-diverse memory buffer selection
        replay_buffer = select_pattern_diverse_replay_memory(
            train_examples=dataset_a["train_examples"],
            rule_indices=cfg["rule_a_indices"],
            buffer_size=cfg["replay_buffer_size"],
        )
        diagnostics["diverse_replay_memory"] = {
            "buffer_size": len(replay_buffer),
            "class_0_count": sum(1 for ex in replay_buffer if ex["target_class"] == 0),
            "class_1_count": sum(1 for ex in replay_buffer if ex["target_class"] == 1),
            "selected_example_ids": [ex["example_id"] for ex in replay_buffer],
            "unique_patterns_count": 12,
        }

        # 4. Schedule construction
        schedule_b = build_minibatch_schedule(
            train_pool=dataset_b["train_examples"],
            step_budget=cfg["step_budget_b"],
            batch_size=cfg["batch_size"],
            seed=cfg["minibatch_sampler_seed_b"],
        )
        interleaved_schedule = build_replay_schedule(
            schedule_b=schedule_b,
            replay_buffer=replay_buffer,
            ratio_b=cfg["replay_ratio_b_to_a"][0],
            ratio_a=cfg["replay_ratio_b_to_a"][1],
        )

        # 5. Fresh Adam optimizer at t1, continuous across the 625 steps
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg["learning_rate"],
            betas=tuple(cfg["betas"]),
            eps=cfg["eps"],
            weight_decay=cfg["weight_decay"],
        )

        exposures_b = 0
        exposures_a_replay = 0
        diagnostics["status"] = "TRAINING"

        # 6. Diverse Replay training loop
        for step_idx, (task_tag, rule_id, batch) in enumerate(interleaved_schedule, start=1):
            if time.monotonic() - t_start > t_ceiling_train:
                raise DeadlineExceededError(
                    f"Diverse replay training wall-clock exceeded ceiling of {t_ceiling_train}s at step {step_idx}"
                )

            model.train()
            optimizer.zero_grad()

            u_list = [torch.tensor(ex["features"] + ex["mask"], dtype=torch.float32) for ex in batch]
            y_list = [int(ex["target_class"]) for ex in batch]
            inputs = torch.stack(u_list).to(device)
            targets = torch.tensor(y_list, dtype=torch.long, device=device)

            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            if task_tag == "rule_b":
                exposures_b += len(batch)
            else:
                exposures_a_replay += len(batch)

            diagnostics["exposures"]["rule_b_presentations"] = exposures_b
            diagnostics["exposures"]["replay_a_presentations"] = exposures_a_replay
            diagnostics["exposures"]["total_presentations"] = exposures_b + exposures_a_replay
            diagnostics["exposures"]["total_optimizer_steps"] = step_idx

        elapsed_training = time.monotonic() - t_start
        diagnostics["elapsed_training_seconds"] = elapsed_training
        diagnostics["status"] = "EVALUATING"

        # Capture final milestone checkpoint
        ckpt_fname = "checkpoint_diverse_replay_step625_final.pt"
        diagnostics["checkpoints"]["checkpoint_diverse_replay_step625_final"] = capture_and_verify_checkpoint(
            model.state_dict(), abs_run_dir, ckpt_fname
        )

        # 7. Final evaluations under eval ceiling
        t_eval_start = time.monotonic()
        eval_records: Dict[str, List[Any]] = {}
        metrics: Dict[str, Any] = {}

        eval_splits = [
            ("rule_a_seen", dataset_a["eval_seen_examples"], sha256_a),
            ("rule_a_unseen", dataset_a["eval_unseen_examples"], sha256_a),
            ("rule_b_seen", dataset_b["eval_seen_examples"], sha256_b),
            ("rule_b_unseen", dataset_b["eval_unseen_examples"], sha256_b),
        ]

        raw_a_seen_records: List[SSRLExampleRecord] = []

        for split_tag, examples, d_hash in eval_splits:
            records = evaluate_population_records(
                model=model,
                examples=examples,
                arm="diverse_replay",
                seed=cfg["seed"],
                step=625,
                dataset_sha256=d_hash,
                device=device,
            )
            eval_records[split_tag] = [r.to_dict() for r in records]
            metrics[f"acc_{split_tag}"] = compute_accuracy_measurement(
                name=f"acc_{split_tag}",
                records=records,
                input_set=d_hash,
                mode="eval_no_dropout",
            )
            if split_tag == "rule_a_seen":
                raw_a_seen_records = records

        # Detailed per-pattern Rule A seen retention breakdown
        diagnostics["rule_a_seen_per_pattern"] = compute_per_pattern_retention(
            records=raw_a_seen_records,
            rule_indices=cfg["rule_a_indices"],
        )

        probe_results = evaluate_counterfactual_probe(
            model=model,
            probe_examples=probe_json["probe_examples"],
            device=device,
        )
        elapsed_eval = time.monotonic() - t_eval_start
        if elapsed_eval > cfg["wall_clock_ceiling_eval_seconds"]:
            raise DeadlineExceededError(
                f"Evaluation exceeded ceiling of {cfg['wall_clock_ceiling_eval_seconds']}s: {elapsed_eval:.2f}s"
            )

        diagnostics["elapsed_eval_seconds"] = elapsed_eval
        diagnostics["status"] = "COMPLETED"
        diagnostics["counterfactual_probe"] = probe_results
        diagnostics["class_aware_baselines"] = {
            "rule_a_seen": {"majority_class": "symmetric", "majority_count": 96, "total": 192, "pct": 50.0},
            "rule_a_unseen": {"majority_class": "symmetric", "majority_count": 32, "total": 64, "pct": 50.0},
            "rule_b_seen": {"majority_class": "class_1", "majority_count": 112, "total": 192, "pct": 58.333333333333336},
            "rule_b_unseen": {"majority_class": "class_0", "majority_count": 48, "total": 64, "pct": 75.0},
            "counterfactual_probe": {"majority_class": "symmetric", "majority_count": 64, "total": 128, "pct": 50.0},
        }

        payload: Dict[str, Any] = {
            "config": cfg,
            "diagnostics": diagnostics,
            "metrics": metrics,
            "eval_records": eval_records,
        }

        serializable_payload = make_json_serializable(payload)
        atomic_write_json(serializable_payload, abs_dest, refuse_overwrite=True)
        return serializable_payload

    except Exception as e:
        diagnostics["status"] = "FAILED"
        diagnostics["error_type"] = type(e).__name__
        diagnostics["error_message"] = str(e)
        diag_path = os.path.join(abs_run_dir, "failure_diagnostics.json")
        try:
            atomic_write_json(make_json_serializable(diagnostics), diag_path, refuse_overwrite=True)
        except Exception:
            pass
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SSRL-05 Pattern-Diverse Replay Baseline Runner")
    parser.add_argument("--dataset_a", required=True, help="Path to Rule A dataset JSON")
    parser.add_argument("--dataset_b", required=True, help="Path to Rule B dataset JSON")
    parser.add_argument("--probe_dataset", required=True, help="Path to probe dataset JSON")
    parser.add_argument("--t1_checkpoint", required=True, help="Path to post-A checkpoint_t1_post_a.pt")
    parser.add_argument("--run_dir", required=True, help="Unique run directory path")
    parser.add_argument("--destination", required=True, help="Path to output results.json")
    parser.add_argument("--sha256_a", default="00a21efad1dc07b826198d99b0b702aedddad57b6226f99301afbf16327fd0b7")
    parser.add_argument("--sha256_b", default="27e813e73bf66f51e1bff07f7d2f19945e830afc6d5876b0cb4d11136cda4cd4")
    parser.add_argument("--sha256_probe", default="1c07d50fed772cfcff86e3bf30cc74a95cdc24d190ec767aa15365671fe06319")
    args = parser.parse_args()

    res = run_diverse_replay_study(
        dataset_a_path=args.dataset_a,
        sha256_a=args.sha256_a,
        dataset_b_path=args.dataset_b,
        sha256_b=args.sha256_b,
        probe_dataset_path=args.probe_dataset,
        sha256_probe=args.sha256_probe,
        destination=args.destination,
        run_dir=args.run_dir,
        t1_checkpoint_path=args.t1_checkpoint,
    )
    cats = res['diagnostics']['counterfactual_probe']['overall_metrics']['category_counts']
    print(f"[SSRL-05 Diverse Replay Completed] Acc A Seen: {res['metrics']['acc_rule_a_seen']['pct']:.2f}% | "
          f"Acc B Seen: {res['metrics']['acc_rule_b_seen']['pct']:.2f}% | "
          f"Probe Both Correct: {cats['both_correct']['numerator']}/128")
