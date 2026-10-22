"""
research/lifelong_01/run_ssrl_replay_baseline.py -- Replay-assisted sequential learning baseline (SSRL-04).

Implements bounded experience replay starting from the preserved post-A checkpoint (t1):
- Reuses checkpoint_t1_post_a.pt (tensor fingerprint: 9ce6ab70..., file sha: 350deff9...)
- Preserves the exact 500 Rule B minibatches in their original order (sampler seed: 2027)
- Selects a fixed M=16 memory buffer (8 lowest-ID class-0 and 8 lowest-ID class-1 Rule A train examples)
- Interleaves 4 Task B updates with 1 Replay A update for 125 cycles (625 total updates)
- Fresh Adam optimizer at t1, continuous across the 625 updates
- Strict 60-second training ceiling and 60-second evaluation deadlines
- Zero training of Task A, zero tuning, no automatic retries, final evaluation only
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

PROPOSED_REPLAY_CONFIG: Dict[str, Any] = {
    "study_id": "SSRL-04",
    "phase": "replay_assisted_sequential_baseline",
    "classification": "bounded_replay_comparator",
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

APPROVED_REPLAY_CONFIG = PROPOSED_REPLAY_CONFIG


def verify_replay_specification_approval(config: Dict[str, Any]) -> None:
    """AGENTS.md Section 2 / Section 8: Strictly blocks unapproved execution."""
    status = config.get("gate_approval_status")
    if status != "APPROVED":
        raise SpecificationApprovalError(
            f"SSRL-04 replay specification status is '{status}', not 'APPROVED'. "
            f"Execution is forbidden before explicit user authorization."
        )


def select_replay_memory(
    train_examples: List[Dict[str, Any]],
    buffer_size: int = 16,
) -> List[Dict[str, Any]]:
    """
    Selects a deterministic fixed memory buffer of Rule A training examples:
    - Exactly 8 class-0 examples with lowest example_id
    - Exactly 8 class-1 examples with lowest example_id
    - Exclusively from train_examples; evaluation and probe sets are forbidden.
    """
    if buffer_size != 16:
        raise ValueError(f"Replay buffer size must be 16, got {buffer_size}")

    c0 = sorted([ex for ex in train_examples if int(ex["target_class"]) == 0], key=lambda x: int(x["example_id"]))[:8]
    c1 = sorted([ex for ex in train_examples if int(ex["target_class"]) == 1], key=lambda x: int(x["example_id"]))[:8]

    if len(c0) != 8 or len(c1) != 8:
        raise ValueError(f"Insufficient class-balanced examples in train pool: c0={len(c0)}, c1={len(c1)}")

    buffer_examples = c0 + c1
    seen_ids = set()
    for ex in buffer_examples:
        eid = int(ex["example_id"])
        if eid in seen_ids:
            raise ValueError(f"Duplicate example_id {eid} in replay buffer")
        seen_ids.add(eid)

    return buffer_examples


def build_replay_schedule(
    schedule_b: List[List[Dict[str, Any]]],
    replay_buffer: List[Dict[str, Any]],
    ratio_b: int = 4,
    ratio_a: int = 1,
) -> List[Tuple[str, int, List[Dict[str, Any]]]]:
    """
    Interleaves 500 Task B minibatches with Replay A minibatches:
    - 4 B minibatches followed by 1 Replay A minibatch
    - Preserves exact ordering of schedule_b
    - Yields 625 total batches (500 B, 125 Replay A)
    """
    if len(schedule_b) % ratio_b != 0:
        raise ValueError(f"schedule_b length ({len(schedule_b)}) not divisible by ratio_b ({ratio_b})")

    schedule: List[Tuple[str, int, List[Dict[str, Any]]]] = []
    b_idx = 0
    total_cycles = len(schedule_b) // ratio_b

    for _ in range(total_cycles):
        for _ in range(ratio_b):
            schedule.append(("rule_b", 2, schedule_b[b_idx]))
            b_idx += 1
        for _ in range(ratio_a):
            schedule.append(("replay_a", 1, replay_buffer))

    return schedule


def load_and_verify_t1_model(
    checkpoint_path: str,
    expected_fingerprint: str,
    expected_file_sha256: str,
    device: torch.device,
) -> SSRLInstructionMLP:
    """Loads post-A checkpoint and validates file hash and tensor fingerprint."""
    abs_path = os.path.abspath(checkpoint_path)
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"Initial post-A checkpoint not found: {abs_path}")

    with open(abs_path, "rb") as f:
        obs_sha = hashlib.sha256(f.read()).hexdigest()
    if obs_sha != expected_file_sha256:
        raise CheckpointVerificationError(
            f"Checkpoint file SHA-256 mismatch for {abs_path}: "
            f"expected {expected_file_sha256}, got {obs_sha}"
        )

    model = SSRLInstructionMLP(input_dim=32, hidden_dim=64, output_dim=2).to(device)
    loaded_sd = torch.load(abs_path, map_location=device, weights_only=True)
    obs_fp = compute_tensor_fingerprint_from_state_dict(loaded_sd)
    if obs_fp != expected_fingerprint:
        raise CheckpointVerificationError(
            f"Checkpoint tensor fingerprint mismatch for {abs_path}: "
            f"expected {expected_fingerprint}, got {obs_fp}"
        )

    model.load_state_dict(loaded_sd)
    return model


def run_replay_study(
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
    """Master procedural entry point for SSRL-04 Bounded Replay Sequential Baseline."""
    cfg = copy.deepcopy(config or PROPOSED_REPLAY_CONFIG)
    verify_replay_specification_approval(cfg)
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

        # 3. Deterministic memory buffer selection
        replay_buffer = select_replay_memory(
            train_examples=dataset_a["train_examples"],
            buffer_size=cfg["replay_buffer_size"],
        )
        diagnostics["replay_memory"] = {
            "buffer_size": len(replay_buffer),
            "class_0_count": sum(1 for ex in replay_buffer if ex["target_class"] == 0),
            "class_1_count": sum(1 for ex in replay_buffer if ex["target_class"] == 1),
            "selected_example_ids": [ex["example_id"] for ex in replay_buffer],
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

        # 6. Replay training loop
        for step_idx, (task_tag, rule_id, batch) in enumerate(interleaved_schedule, start=1):
            if time.monotonic() - t_start > t_ceiling_train:
                raise DeadlineExceededError(
                    f"Replay training wall-clock exceeded ceiling of {t_ceiling_train}s at step {step_idx}"
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
        ckpt_fname = "checkpoint_replay_step625_final.pt"
        diagnostics["checkpoints"]["checkpoint_replay_step625_final"] = capture_and_verify_checkpoint(
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

        for split_tag, examples, d_hash in eval_splits:
            records = evaluate_population_records(
                model=model,
                examples=examples,
                arm="replay_assisted",
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
    parser = argparse.ArgumentParser(description="SSRL-04 Replay-Assisted Sequential Baseline Runner")
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

    res = run_replay_study(
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
    print(f"[SSRL-04 Replay Completed] Acc A Seen: {res['metrics']['acc_rule_a_seen']['pct']:.2f}% | "
          f"Acc B Seen: {res['metrics']['acc_rule_b_seen']['pct']:.2f}% | "
          f"Probe Both Correct: {cats['both_correct']['numerator']}/128")
