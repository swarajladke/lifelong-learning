"""
research/lifelong_01/run_ssrl_joint_reference.py -- SSRL-03 Offline Joint-Training Learnability Reference Runner.

Scientific Classification:
- This is an OFFLINE MULTI-TASK LEARNABILITY REFERENCE, not a continual learning solution
  and not a guaranteed theoretical upper bound.
- Interleaves Rule A and Rule B presentations from initialization to establish an empirical
  reference for whether the fixed MLP architecture (32 -> 64 -> 64 -> 2) has sufficient representation
  capacity and gradient dynamics to simultaneously fit both parity rules under static instruction conditioning.

Design & Protocol Invariants:
1. Initialization: Loaded from verified checkpoint_t0_init.pt with tensor fingerprint verification.
2. Architecture: Strictly identical 6,402-parameter MLP (SSRLInstructionMLP).
3. Datasets: Pinned Rule A (00a21efad1...) and Rule B (27e813e7...).
4. Exposure Budget:
   - Exactly matches sequential study: 32,000 Rule A presentations (2,000 steps x 16),
     8,000 Rule B presentations (500 steps x 16). Total: 2,500 optimizer steps (40,000 presentations).
5. Sampling Schedule:
   - 4:1 cyclic interleaving: 4 successive Rule A batches followed by 1 Rule B batch per cycle,
     preserving the exact within-rule batch order from the original RNG streams.
6. Continuous Adam Optimization:
   - Disclosed difference: Adam optimizer runs continuously across all 2,500 steps without reset.
     Moment estimates (m_t, v_t) carry across task boundaries.
7. Evaluation Boundary:
   - Frozen evaluation sets and counterfactual probe evaluated at final checkpoint (step 2,500) only.
   - Strictly evaluation-only: probe outcomes are never used for loss calculation, early stopping, or tuning.
8. Artifacts & Checkpoints:
   - Five size-bounded checkpoints (< 100 KB each) with reload-and-verify equality.
   - Raw predictions, logits, exposure records, provenance, and failure diagnostics preserved.
"""

from __future__ import annotations
import sys
import os
import time
import json
import random
import argparse
import subprocess
import hashlib
import copy
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from research.lifelong_01.run_ssrl_learnability_pilot import SpecificationApprovalError
from research.lifelong_01.ssrl_types import SSRLExampleRecord, SSRLMeasurement, compute_accuracy_measurement
from research.lifelong_01.ssrl_model import SSRLInstructionMLP
from research.lifelong_01.ssrl_generator import load_and_verify_dataset, PilotDatasetContractError
from research.lifelong_01.ssrl_evaluator import atomic_write_json
from research.lifelong_01.ssrl_checkpoints import (
    capture_and_verify_checkpoint,
    compute_tensor_fingerprint_from_state_dict,
)
from research.lifelong_01.ssrl_counterfactual_probe import evaluate_counterfactual_probe


EXPECTED_T0_FINGERPRINT = "acb9143e56b062e595cfa413ff54a819d3f237af7692cc11a1d3abd3e06109c9"

PROPOSED_JOINT_REFERENCE_CONFIG: Dict[str, Any] = {
    "study_id": "SSRL-03",
    "directive": "LIFELONG-01-SSRL-03",
    "phase": "joint_learnability_reference",
    "classification": "offline_joint_training_reference",
    "seed": 101,
    "model_init_seed": 101,
    "expected_t0_fingerprint": EXPECTED_T0_FINGERPRINT,
    "data_distractor_seed_a": 42,
    "data_distractor_seed_b": 43,
    "minibatch_sampler_seed_a": 2026,
    "minibatch_sampler_seed_b": 2027,
    "rule_a_id": 1,
    "rule_a_indices": [0, 1, 2, 3],
    "rule_b_id": 2,
    "rule_b_indices": [4, 5, 6, 7],
    "input_dim": 32,
    "input_dim_features": 16,
    "active_k": 4,
    "hidden_dim": 64,
    "output_dim": 2,
    "step_budget_total": 2500,
    "step_budget_a": 2000,
    "step_budget_b": 500,
    "interleave_ratio_a_to_b": [4, 1],
    "batch_size": 16,
    "learning_rate": 1e-3,
    "betas": [0.9, 0.999],
    "eps": 1e-8,
    "weight_decay": 0.0,
    "adam_mode": "continuous_across_all_tasks_no_reset",
    "train_pool_size": 384,
    "eval_seen_size": 192,
    "eval_unseen_size": 64,
    "probe_size": 128,
    "gate_approval_status": "APPROVED",
    "wall_clock_ceiling_training_seconds": 180.0,
    "wall_clock_ceiling_probe_seconds": 60.0,
}

# Backward compatibility alias
APPROVED_JOINT_REFERENCE_CONFIG = PROPOSED_JOINT_REFERENCE_CONFIG


class DeadlineExceededError(Exception):
    """Raised when execution wall-clock exceeds pre-registered ceiling."""
    pass


class CheckpointVerificationError(Exception):
    """Raised when loaded checkpoint does not match expected tensor fingerprint."""
    pass


def verify_joint_specification_approval(config: Dict[str, Any]) -> None:
    """Verifies that the specification is approved before any compute."""
    if not isinstance(config, dict):
        raise SpecificationApprovalError("Config must be a dictionary")
    approval_status = config.get("gate_approval_status")
    if approval_status != "APPROVED":
        raise SpecificationApprovalError(
            f"Run specification approval status is '{approval_status}', not 'APPROVED'. "
            "Execution is forbidden before formal approval. Aborting prior to model construction."
        )


def reserve_run_directory(run_dir: str) -> str:
    """Exclusively creates run directory to prevent overwriting prior runs."""
    abs_dir = os.path.abspath(run_dir)
    os.makedirs(abs_dir, exist_ok=False)
    return abs_dir


def make_json_serializable(val: Any) -> Any:
    """Recursively converts dataclasses and objects with .to_dict() to JSON-compatible types."""
    if hasattr(val, "to_dict") and callable(val.to_dict):
        return make_json_serializable(val.to_dict())
    if isinstance(val, dict):
        return {str(k): make_json_serializable(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [make_json_serializable(v) for v in val]
    return val


def build_minibatch_schedule(
    train_pool: List[Dict[str, Any]],
    step_budget: int,
    batch_size: int,
    seed: int
) -> List[List[Dict[str, Any]]]:
    """Precomputes exact sequence of minibatches from dedicated RNG stream."""
    rng = random.Random(seed)
    schedule: List[List[Dict[str, Any]]] = []
    for _ in range(step_budget):
        batch = rng.sample(train_pool, batch_size)
        schedule.append(batch)
    return schedule


def interleave_minibatch_schedules(
    schedule_a: List[List[Dict[str, Any]]],
    schedule_b: List[List[Dict[str, Any]]],
    ratio_a: int = 4,
    ratio_b: int = 1
) -> List[Tuple[str, int, List[Dict[str, Any]]]]:
    """
    Interleaves two schedules cyclically with ratio_a batches from A followed by ratio_b batches from B.
    Preserves exact within-rule sequence order for both A and B minibatches.
    Returns list of (task_tag, task_rule_id, batch) tuples.
    """
    assert ratio_a > 0 and ratio_b > 0, "Ratios must be positive integers"
    total_a = len(schedule_a)
    total_b = len(schedule_b)
    assert total_a % ratio_a == 0, f"schedule_a length ({total_a}) must be divisible by ratio_a ({ratio_a})"
    assert total_b % ratio_b == 0, f"schedule_b length ({total_b}) must be divisible by ratio_b ({ratio_b})"
    cycles_a = total_a // ratio_a
    cycles_b = total_b // ratio_b
    assert cycles_a == cycles_b, f"Cycle count mismatch: A has {cycles_a} cycles, B has {cycles_b} cycles"

    interleaved: List[Tuple[str, int, List[Dict[str, Any]]]] = []
    ptr_a = 0
    ptr_b = 0

    for _ in range(cycles_a):
        for _ in range(ratio_a):
            interleaved.append(("rule_a", 1, schedule_a[ptr_a]))
            ptr_a += 1
        for _ in range(ratio_b):
            interleaved.append(("rule_b", 2, schedule_b[ptr_b]))
            ptr_b += 1

    assert ptr_a == total_a, f"Not all A batches consumed: {ptr_a}/{total_a}"
    assert ptr_b == total_b, f"Not all B batches consumed: {ptr_b}/{total_b}"
    assert len(interleaved) == total_a + total_b
    return interleaved


def load_and_verify_initial_model(
    checkpoint_path: str,
    expected_fingerprint: str,
    device: torch.device
) -> SSRLInstructionMLP:
    """Loads initial model from disk and asserts tensor fingerprint equality."""
    abs_path = os.path.abspath(checkpoint_path)
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"Initial checkpoint file not found: {abs_path}")

    model = SSRLInstructionMLP(
        input_dim=32,
        hidden_dim=64,
        output_dim=2
    ).to(device)

    loaded_sd = torch.load(abs_path, map_location=device, weights_only=True)
    obs_fp = compute_tensor_fingerprint_from_state_dict(loaded_sd)
    if obs_fp != expected_fingerprint:
        raise CheckpointVerificationError(
            f"Initial checkpoint at {abs_path} has tensor fingerprint '{obs_fp}', "
            f"which does not match expected fingerprint '{expected_fingerprint}'."
        )

    model.load_state_dict(loaded_sd)
    return model


def evaluate_population_records(
    model: nn.Module,
    examples: List[Dict[str, Any]],
    arm: str,
    seed: int,
    step: int,
    dataset_sha256: str,
    device: torch.device
) -> List[SSRLExampleRecord]:
    """
    Evaluates model over frozen examples under torch.no_grad() and model.eval().
    Guarantees zero mutation of weights, optimizer, or future batch sequences.
    """
    model.eval()
    records: List[SSRLExampleRecord] = []

    with torch.no_grad():
        for ex in examples:
            feat = ex["features"]
            mask = ex["mask"]
            u = torch.tensor(feat + mask, dtype=torch.float32, device=device).unsqueeze(0)
            logits = model(u)
            pred = int(torch.argmax(logits, dim=1).item())
            target = int(ex["target_class"])
            correct = (pred == target)
            records.append(SSRLExampleRecord(
                example_id=int(ex["example_id"]),
                rule_id=int(ex["rule_id"]),
                arm=arm,
                seed=seed,
                step=step,
                dataset_sha256=dataset_sha256,
                input_features=list(feat),
                selection_mask=list(mask),
                target_class=target,
                predicted_class=pred,
                is_correct=correct,
                is_held_out_pattern=bool(ex.get("is_held_out_pattern", False)),
            ))

    return records


def run_joint_reference_study(
    dataset_a_path: str,
    sha256_a: str,
    dataset_b_path: str,
    sha256_b: str,
    destination: str,
    run_dir: str,
    init_checkpoint_path: str,
    probe_dataset_path: str,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Master procedural entry point for SSRL-03 Offline Joint-Training Reference Study."""
    cfg = copy.deepcopy(config or PROPOSED_JOINT_REFERENCE_CONFIG)

    verify_joint_specification_approval(cfg)
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
        "elapsed_probe_seconds": 0.0,
        "exposures": {
            "rule_a_presentations": 0,
            "rule_b_presentations": 0,
            "total_presentations": 0,
            "total_optimizer_steps": 0
        },
        "checkpoints": {},
    }

    t_start = time.monotonic()
    t_ceiling_train = cfg["wall_clock_ceiling_training_seconds"]

    try:
        # 1. Pinned dataset verification with full contract validation
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

        device = torch.device("cpu")
        torch.set_num_threads(1)

        # 2. Model initialization from verified checkpoint
        model = load_and_verify_initial_model(
            checkpoint_path=init_checkpoint_path,
            expected_fingerprint=cfg["expected_t0_fingerprint"],
            device=device
        )

        # Capture initial milestone checkpoint
        ckpt_metadata: Dict[str, Any] = {}
        ckpt_metadata["checkpoint_joint_t0_init"] = capture_and_verify_checkpoint(
            model.state_dict(), abs_run_dir, "checkpoint_joint_t0_init.pt"
        )
        diagnostics["checkpoints"] = ckpt_metadata

        # 3. Precompute separate batch sequences from identical RNG streams
        train_pool_a = dataset_a["train_examples"]
        train_pool_b = dataset_b["train_examples"]

        schedule_a = build_minibatch_schedule(
            train_pool_a, cfg["step_budget_a"], cfg["batch_size"], cfg["minibatch_sampler_seed_a"]
        )
        schedule_b = build_minibatch_schedule(
            train_pool_b, cfg["step_budget_b"], cfg["batch_size"], cfg["minibatch_sampler_seed_b"]
        )

        # 4. Interleave schedules (4 A batches, 1 B batch per cycle)
        interleaved_schedule = interleave_minibatch_schedules(
            schedule_a, schedule_b,
            ratio_a=cfg["interleave_ratio_a_to_b"][0],
            ratio_b=cfg["interleave_ratio_a_to_b"][1]
        )

        # 5. Continuous Adam optimizer across all 2,500 steps
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg["learning_rate"],
            betas=tuple(cfg["betas"]),
            eps=cfg["eps"],
            weight_decay=cfg["weight_decay"]
        )

        checkpoint_steps = {
            500: "checkpoint_joint_step500.pt",
            1000: "checkpoint_joint_step1000.pt",
            2000: "checkpoint_joint_step2000.pt",
            2500: "checkpoint_joint_step2500_final.pt"
        }

        exposures_a = 0
        exposures_b = 0
        diagnostics["status"] = "TRAINING"

        # 6. Joint training loop
        for step_idx, (task_tag, rule_id, batch) in enumerate(interleaved_schedule, start=1):
            if time.monotonic() - t_start > t_ceiling_train:
                raise DeadlineExceededError(f"Training wall-clock exceeded ceiling of {t_ceiling_train}s at step {step_idx}")

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

            if task_tag == "rule_a":
                exposures_a += len(batch)
            else:
                exposures_b += len(batch)

            diagnostics["exposures"]["rule_a_presentations"] = exposures_a
            diagnostics["exposures"]["rule_b_presentations"] = exposures_b
            diagnostics["exposures"]["total_presentations"] = exposures_a + exposures_b
            diagnostics["exposures"]["total_optimizer_steps"] = step_idx

            # Milestone checkpoint capture
            if step_idx in checkpoint_steps:
                fname = checkpoint_steps[step_idx]
                ckpt_tag = fname.replace(".pt", "")
                ckpt_metadata[ckpt_tag] = capture_and_verify_checkpoint(
                    model.state_dict(), abs_run_dir, fname
                )
                diagnostics["checkpoints"] = ckpt_metadata

        elapsed_training = time.monotonic() - t_start
        diagnostics["elapsed_training_seconds"] = elapsed_training
        diagnostics["status"] = "EVALUATING"

        # 7. Final frozen evaluations (Step 2,500 only)
        eval_records: Dict[str, List[Any]] = {}
        metrics: Dict[str, Any] = {}

        eval_splits = [
            ("rule_a_seen", dataset_a["eval_seen_examples"], "eval_seen_examples", sha256_a),
            ("rule_a_unseen", dataset_a["eval_unseen_examples"], "eval_unseen_examples", sha256_a),
            ("rule_b_seen", dataset_b["eval_seen_examples"], "eval_seen_examples", sha256_b),
            ("rule_b_unseen", dataset_b["eval_unseen_examples"], "eval_unseen_examples", sha256_b),
        ]

        for split_tag, examples, split_key, d_hash in eval_splits:
            records = evaluate_population_records(
                model=model,
                examples=examples,
                arm="joint_reference",
                seed=cfg["seed"],
                step=2500,
                dataset_sha256=d_hash,
                device=device
            )
            eval_records[split_tag] = [r.to_dict() for r in records]
            acc_m = compute_accuracy_measurement(
                records=records,
                name=f"acc_{split_tag}",
                input_set=d_hash,
                mode="eval_no_dropout"
            )
            metrics[f"acc_{split_tag}"] = acc_m

        # 8. Evaluation-only counterfactual probe evaluation
        t_probe_start = time.monotonic()
        with open(probe_dataset_path, "r", encoding="utf-8") as pf:
            probe_json = json.load(pf)
        probe_results = evaluate_counterfactual_probe(
            model=model,
            probe_examples=probe_json["probe_examples"],
            device=device
        )
        elapsed_probe = time.monotonic() - t_probe_start
        if elapsed_probe > cfg["wall_clock_ceiling_probe_seconds"]:
            raise DeadlineExceededError(f"Probe evaluation exceeded ceiling of {cfg['wall_clock_ceiling_probe_seconds']}s")

        diagnostics["elapsed_probe_seconds"] = elapsed_probe
        diagnostics["status"] = "COMPLETED"
        diagnostics["counterfactual_probe"] = probe_results
        diagnostics["class_aware_baselines"] = {
            "rule_a_seen": {"majority_class": "symmetric", "majority_count": 96, "total": 192, "pct": 50.0},
            "rule_a_unseen": {"majority_class": "symmetric", "majority_count": 32, "total": 64, "pct": 50.0},
            "rule_b_seen": {"majority_class": "class_1", "majority_count": 112, "total": 192, "pct": 58.333333333333336},
            "rule_b_unseen": {"majority_class": "class_0", "majority_count": 48, "total": 64, "pct": 75.0},
            "counterfactual_probe": {"majority_class": "symmetric", "majority_count": 64, "total": 128, "pct": 50.0}
        }

        # 9. Assembly of complete results payload
        payload: Dict[str, Any] = {
            "config": cfg,
            "diagnostics": diagnostics,
            "metrics": metrics,
            "eval_records": eval_records
        }

        serializable_payload = make_json_serializable(payload)
        atomic_write_json(serializable_payload, abs_dest, refuse_overwrite=True)
        return serializable_payload

    except Exception as exc:
        diagnostics["status"] = "FAILED"
        diagnostics["error_type"] = type(exc).__name__
        diagnostics["error_message"] = str(exc)
        diagnostics["elapsed_total_seconds"] = time.monotonic() - t_start
        failure_diag_path = os.path.join(abs_run_dir, "failure_diagnostics.json")
        try:
            atomic_write_json(make_json_serializable(diagnostics), failure_diag_path, refuse_overwrite=True)
        except Exception as write_err:
            print(f"[SSRL-03 ERROR] Could not persist failure diagnostics to {failure_diag_path}: {write_err}", file=sys.stderr)
        raise


def main() -> None:
    """CLI entry point for SSRL-03 runner."""
    parser = argparse.ArgumentParser(description="SSRL-03 Joint-Training Reference Runner")
    parser.add_argument("--dataset_a", required=True, help="Path to Rule A dataset JSON")
    parser.add_argument("--sha256_a", required=True, help="Canonical SHA-256 for Rule A")
    parser.add_argument("--dataset_b", required=True, help="Path to Rule B dataset JSON")
    parser.add_argument("--sha256_b", required=True, help="Canonical SHA-256 for Rule B")
    parser.add_argument("--destination", required=True, help="Destination results JSON path")
    parser.add_argument("--run_dir", required=True, help="Run directory for checkpoints and artifacts")
    parser.add_argument("--init_checkpoint", required=True, help="Path to verified checkpoint_t0_init.pt")
    parser.add_argument("--probe_dataset", required=True, help="Path to counterfactual probe JSON")

    args = parser.parse_args()

    print(f"[SSRL-03] Starting Offline Joint-Training Reference Study...")
    try:
        run_joint_reference_study(
            dataset_a_path=args.dataset_a,
            sha256_a=args.sha256_a,
            dataset_b_path=args.dataset_b,
            sha256_b=args.sha256_b,
            destination=args.destination,
            run_dir=args.run_dir,
            init_checkpoint_path=args.init_checkpoint,
            probe_dataset_path=args.probe_dataset
        )
        print("[SSRL-03] Study completed successfully with exit code 0.")
        sys.exit(0)
    except Exception as e:
        print(f"[SSRL-03 ERROR] Run failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
