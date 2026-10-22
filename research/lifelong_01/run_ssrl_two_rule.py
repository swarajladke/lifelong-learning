"""
research/lifelong_01/run_ssrl_two_rule.py -- Master runner for SSRL-02 Two-Rule Study.

Implements the bounded two-rule exploratory sequential parity learning protocol:
- Arms:
  1. seq_ab: Trains 2,000 steps on Rule A, resets optimizer, adapts 500 steps on Rule B.
  2. fresh_b: Trains 500 steps on Rule B from shared initial model state theta_0.
  3. freeze_after_a: Preserves post-A checkpoint theta_A without Rule B updates.
- Identical Rule B minibatches shared between seq_ab and fresh_b.
- Disjoint RNG streams for model init, Rule A batches, and Rule B batches.
- Intermediate evaluation checkpoints during Rule B adaptation (steps 0, 100, 250, 500).
- Pure evaluation under torch.no_grad() and model.eval() with zero state leakage.
- Atomic artifact writing and exhaustive provenance tracking.

Guarded by main: this module does NOT execute compute upon import.
Designed for remote execution on Kaggle.
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
from research.lifelong_01.ssrl_evaluator import (
    evaluate_gate_g_lc,
    compute_forgetting_from_records,
    compute_transfer_from_records,
    atomic_write_json
)


APPROVED_TWO_RULE_CONFIG: Dict[str, Any] = {
    "study_id": "SSRL-02",
    "directive": "LIFELONG-01-SSRL-02",
    "phase": "two_rule_diagnostic",
    "seed": 101,
    "model_init_seed": 101,
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
    "step_budget_a": 2000,
    "step_budget_b": 500,
    "transfer_eval_steps": [0, 100, 250, 500],
    "batch_size": 16,
    "learning_rate": 1e-3,
    "betas": [0.9, 0.999],
    "eps": 1e-8,
    "weight_decay": 0.0,
    "train_pool_size": 384,
    "eval_seen_size": 192,
    "eval_unseen_size": 64,
    "gate_g_lc_threshold": 0.90,
    "gate_approval_status": "APPROVED",
    "wall_clock_ceiling_training_seconds": 180.0,
    "wall_clock_ceiling_eval_seconds": 60.0,
}


class DeadlineExceededError(Exception):
    """Raised when execution wall-clock exceeds pre-registered ceiling."""
    pass


def verify_run_specification_approval(config: Dict[str, Any]) -> None:
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
            is_correct = (pred == target)

            rec = SSRLExampleRecord(
                example_id=int(ex["example_id"]),
                rule_id=int(ex["rule_id"]),
                arm=arm,
                seed=seed,
                step=step,
                dataset_sha256=dataset_sha256,
                input_features=feat,
                selection_mask=mask,
                target_class=target,
                predicted_class=pred,
                is_correct=is_correct,
                is_held_out_pattern=bool(ex["is_held_out_pattern"]),
            )
            records.append(rec)

    assert len(records) == len(examples), f"Expected {len(examples)} records, got {len(records)}"
    return records


def run_two_rule_study(
    config: Dict[str, Any],
    dataset_a_path: str,
    expected_sha_a: str,
    dataset_b_path: str,
    expected_sha_b: str,
    destination: str,
    run_dir: str,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """Master orchestrator for SSRL-02 study."""
    if os.path.exists(destination):
        raise FileExistsError(f"Destination file already exists: {destination}")

    abs_run_dir = reserve_run_directory(run_dir)
    diagnostics: Dict[str, Any] = {
        "study_id": config["study_id"],
        "phase": config["phase"],
        "status": "INITIALIZED",
        "run_directory": abs_run_dir,
        "destination": destination,
    }

    try:
        # 1. Approval Guard
        verify_run_specification_approval(config)
        diagnostics["approval_status"] = "APPROVED"

        # 2. Dataset Assertions
        config_a = {
            **config,
            "study_id": "SSRL-01",
            "phase": "learnability_pilot",
            "rule_id": config["rule_a_id"],
            "rule_indices": config["rule_a_indices"],
            "partition_seed": config["data_distractor_seed_a"],
        }
        config_b = {
            **config,
            "study_id": "SSRL-02",
            "phase": "two_rule_diagnostic",
            "rule_id": config["rule_b_id"],
            "rule_indices": config["rule_b_indices"],
            "partition_seed": config["data_distractor_seed_b"],
        }

        dataset_a = load_and_verify_dataset(dataset_a_path, expected_sha256=expected_sha_a, approved_config=config_a)
        dataset_b = load_and_verify_dataset(dataset_b_path, expected_sha256=expected_sha_b, approved_config=config_b)
        diagnostics["status"] = "DATASETS_VERIFIED"

        if device is None:
            device = torch.device("cpu")

        # 3. Model Initialization (theta_0)
        torch.manual_seed(config["model_init_seed"])
        initial_model = SSRLInstructionMLP(
            input_dim=config["input_dim"],
            hidden_dim=config["hidden_dim"],
            output_dim=config["output_dim"]
        ).to(device)
        theta_0 = copy.deepcopy(initial_model.state_dict())
        criterion = nn.CrossEntropyLoss()

        # 4. Precompute Minibatch Schedules
        batches_a = build_minibatch_schedule(dataset_a["train_examples"], config["step_budget_a"], config["batch_size"], config["minibatch_sampler_seed_a"])
        batches_b = build_minibatch_schedule(dataset_b["train_examples"], config["step_budget_b"], config["batch_size"], config["minibatch_sampler_seed_b"])

        eval_records_all: Dict[str, Any] = {}
        t_start_training = time.monotonic()

        # Checkpoint t0 (Untrained baseline)
        model = SSRLInstructionMLP(32, 64, 2).to(device)
        model.load_state_dict(theta_0)
        eval_records_all["t0_rule_a_seen"] = evaluate_population_records(model, dataset_a["eval_seen_examples"], "init", config["seed"], 0, expected_sha_a, device)
        eval_records_all["t0_rule_a_unseen"] = evaluate_population_records(model, dataset_a["eval_unseen_examples"], "init", config["seed"], 0, expected_sha_a, device)
        eval_records_all["t0_rule_b_seen"] = evaluate_population_records(model, dataset_b["eval_seen_examples"], "init", config["seed"], 0, expected_sha_b, device)
        eval_records_all["t0_rule_b_unseen"] = evaluate_population_records(model, dataset_b["eval_unseen_examples"], "init", config["seed"], 0, expected_sha_b, device)

        # =====================================================================
        # ARM 1: Sequential (seq_ab)
        # =====================================================================
        model.load_state_dict(theta_0)
        optimizer_a = optim.Adam(model.parameters(), lr=config["learning_rate"], betas=config["betas"], eps=config["eps"], weight_decay=config["weight_decay"])

        # Train Task A (2000 steps)
        model.train()
        for step_idx, batch in enumerate(batches_a, start=1):
            x_m = torch.tensor([ex["features"] + ex["mask"] for ex in batch], dtype=torch.float32, device=device)
            targets = torch.tensor([ex["target_class"] for ex in batch], dtype=torch.long, device=device)
            optimizer_a.zero_grad()
            loss = criterion(model(x_m), targets)
            loss.backward()
            optimizer_a.step()

        # Checkpoint t1 (Post-Rule A)
        eval_records_all["t1_seq_rule_a_seen"] = evaluate_population_records(model, dataset_a["eval_seen_examples"], "seq_ab", config["seed"], 2000, expected_sha_a, device)
        eval_records_all["t1_seq_rule_a_unseen"] = evaluate_population_records(model, dataset_a["eval_unseen_examples"], "seq_ab", config["seed"], 2000, expected_sha_a, device)
        eval_records_all["t1_seq_rule_b_seen"] = evaluate_population_records(model, dataset_b["eval_seen_examples"], "seq_ab", config["seed"], 2000, expected_sha_b, device)
        eval_records_all["t1_seq_rule_b_unseen"] = evaluate_population_records(model, dataset_b["eval_unseen_examples"], "seq_ab", config["seed"], 2000, expected_sha_b, device)

        theta_a = copy.deepcopy(model.state_dict())

        # Gate G-LC Verification on Task A Seen
        m_t1_a_seen = compute_accuracy_measurement("rule_a_t1_seen", eval_records_all["t1_seq_rule_a_seen"], expected_sha_a)
        gate_result = evaluate_gate_g_lc(m_t1_a_seen, config["gate_g_lc_threshold"], "APPROVED")
        diagnostics["gate_g_lc_rule_a"] = gate_result
        if not gate_result["passed"]:
            raise RuntimeError(f"Gate G-LC FAILED on Task A: observed={gate_result['observed_pct']:.2f}% < {gate_result['reference_threshold']*100:.1f}%")

        # Boundary Reset: Construct fresh Adam optimizer for Task B
        optimizer_b_seq = optim.Adam(model.parameters(), lr=config["learning_rate"], betas=config["betas"], eps=config["eps"], weight_decay=config["weight_decay"])

        # Train Task B (500 steps) with intermediate evaluations
        model.train()
        for step_idx, batch in enumerate(batches_b, start=1):
            x_m = torch.tensor([ex["features"] + ex["mask"] for ex in batch], dtype=torch.float32, device=device)
            targets = torch.tensor([ex["target_class"] for ex in batch], dtype=torch.long, device=device)
            optimizer_b_seq.zero_grad()
            loss = criterion(model(x_m), targets)
            loss.backward()
            optimizer_b_seq.step()

            if step_idx in (100, 250, 500):
                eval_records_all[f"t_step{step_idx}_seq_rule_b_seen"] = evaluate_population_records(model, dataset_b["eval_seen_examples"], "seq_ab", config["seed"], 2000 + step_idx, expected_sha_b, device)
                eval_records_all[f"t_step{step_idx}_seq_rule_b_unseen"] = evaluate_population_records(model, dataset_b["eval_unseen_examples"], "seq_ab", config["seed"], 2000 + step_idx, expected_sha_b, device)
                model.train()

        # Checkpoint t2 (Post-Rule B for seq_ab)
        eval_records_all["t2_seq_rule_a_seen"] = evaluate_population_records(model, dataset_a["eval_seen_examples"], "seq_ab", config["seed"], 2500, expected_sha_a, device)
        eval_records_all["t2_seq_rule_a_unseen"] = evaluate_population_records(model, dataset_a["eval_unseen_examples"], "seq_ab", config["seed"], 2500, expected_sha_a, device)

        # =====================================================================
        # ARM 2: Fresh Control (fresh_b)
        # =====================================================================
        model_fresh = SSRLInstructionMLP(32, 64, 2).to(device)
        model_fresh.load_state_dict(theta_0)
        optimizer_fresh = optim.Adam(model_fresh.parameters(), lr=config["learning_rate"], betas=config["betas"], eps=config["eps"], weight_decay=config["weight_decay"])

        model_fresh.train()
        for step_idx, batch in enumerate(batches_b, start=1):
            x_m = torch.tensor([ex["features"] + ex["mask"] for ex in batch], dtype=torch.float32, device=device)
            targets = torch.tensor([ex["target_class"] for ex in batch], dtype=torch.long, device=device)
            optimizer_fresh.zero_grad()
            loss = criterion(model_fresh(x_m), targets)
            loss.backward()
            optimizer_fresh.step()

            if step_idx in (100, 250, 500):
                eval_records_all[f"t_step{step_idx}_fresh_rule_b_seen"] = evaluate_population_records(model_fresh, dataset_b["eval_seen_examples"], "fresh_b", config["seed"], step_idx, expected_sha_b, device)
                eval_records_all[f"t_step{step_idx}_fresh_rule_b_unseen"] = evaluate_population_records(model_fresh, dataset_b["eval_unseen_examples"], "fresh_b", config["seed"], step_idx, expected_sha_b, device)
                model_fresh.train()

        # =====================================================================
        # ARM 3: Freeze After A (freeze_after_a)
        # =====================================================================
        model_frozen = SSRLInstructionMLP(32, 64, 2).to(device)
        model_frozen.load_state_dict(theta_a)
        eval_records_all["t2_frozen_rule_a_seen"] = evaluate_population_records(model_frozen, dataset_a["eval_seen_examples"], "freeze_after_a", config["seed"], 2000, expected_sha_a, device)
        eval_records_all["t2_frozen_rule_a_unseen"] = evaluate_population_records(model_frozen, dataset_a["eval_unseen_examples"], "freeze_after_a", config["seed"], 2000, expected_sha_a, device)
        eval_records_all["t2_frozen_rule_b_seen"] = evaluate_population_records(model_frozen, dataset_b["eval_seen_examples"], "freeze_after_a", config["seed"], 2000, expected_sha_b, device)

        elapsed_training = time.monotonic() - t_start_training
        if elapsed_training > config["wall_clock_ceiling_training_seconds"]:
            raise DeadlineExceededError(f"Training time {elapsed_training:.2f}s exceeded ceiling {config['wall_clock_ceiling_training_seconds']}s")

        # 5. Compute Decomposed Metrics
        metrics: Dict[str, Any] = {}
        # Standing Control Arm: Freeze-After-A accuracy on Rule A at t2 (AGENTS.md Rule 13.1)
        metrics["acc_t2_frozen_rule_a_seen"] = compute_accuracy_measurement("frozen_rule_a_seen", eval_records_all["t2_frozen_rule_a_seen"], expected_sha_a).to_dict()
        metrics["acc_t2_frozen_rule_a_unseen"] = compute_accuracy_measurement("frozen_rule_a_unseen", eval_records_all["t2_frozen_rule_a_unseen"], expected_sha_a).to_dict()

        # Baseline accuracies at t0, t1, t2
        metrics["acc_t0_rule_a_seen"] = compute_accuracy_measurement("rule_a_t0_seen", eval_records_all["t0_rule_a_seen"], expected_sha_a).to_dict()
        metrics["acc_t0_rule_b_seen"] = compute_accuracy_measurement("rule_b_t0_seen", eval_records_all["t0_rule_b_seen"], expected_sha_b).to_dict()
        metrics["acc_t1_seq_rule_a_seen"] = m_t1_a_seen.to_dict()
        metrics["acc_t1_seq_rule_a_unseen"] = compute_accuracy_measurement("rule_a_t1_unseen", eval_records_all["t1_seq_rule_a_unseen"], expected_sha_a).to_dict()
        metrics["acc_t2_seq_rule_a_seen"] = compute_accuracy_measurement("rule_a_t2_seen", eval_records_all["t2_seq_rule_a_seen"], expected_sha_a).to_dict()
        metrics["acc_t2_seq_rule_a_unseen"] = compute_accuracy_measurement("rule_a_t2_unseen", eval_records_all["t2_seq_rule_a_unseen"], expected_sha_a).to_dict()

        # Forgetting on Rule A (seen and unseen)
        metrics["forgetting_rule_a_seen"] = compute_forgetting_from_records(eval_records_all["t1_seq_rule_a_seen"], eval_records_all["t2_seq_rule_a_seen"], "forgetting_rule_a_seen")
        metrics["forgetting_rule_a_unseen"] = compute_forgetting_from_records(eval_records_all["t1_seq_rule_a_unseen"], eval_records_all["t2_seq_rule_a_unseen"], "forgetting_rule_a_unseen")

        # Zero-shot Transfer delta on Rule B at step 0
        metrics["transfer_delta_step0_seen"] = compute_transfer_from_records(eval_records_all["t1_seq_rule_b_seen"], eval_records_all["t0_rule_b_seen"], "transfer_step0_seen")
        metrics["transfer_delta_step0_unseen"] = compute_transfer_from_records(eval_records_all["t1_seq_rule_b_unseen"], eval_records_all["t0_rule_b_unseen"], "transfer_step0_unseen")

        # Transfer deltas on Rule B at steps 100, 250, 500
        for s in (100, 250, 500):
            metrics[f"transfer_delta_step{s}_seen"] = compute_transfer_from_records(eval_records_all[f"t_step{s}_seq_rule_b_seen"], eval_records_all[f"t_step{s}_fresh_rule_b_seen"], f"transfer_step{s}_seen")
            metrics[f"transfer_delta_step{s}_unseen"] = compute_transfer_from_records(eval_records_all[f"t_step{s}_seq_rule_b_unseen"], eval_records_all[f"t_step{s}_fresh_rule_b_unseen"], f"transfer_step{s}_unseen")

        diagnostics["metrics"] = metrics
        diagnostics["status"] = "COMPLETED_SUCCESS"
        diagnostics["elapsed_training_seconds"] = elapsed_training
        diagnostics["exposures"] = {
            "seq_ab": {"steps": 2500, "samples": 40000},
            "fresh_b": {"steps": 500, "samples": 8000},
            "freeze_after_a": {"steps": 2000, "samples": 32000},
        }

        # Serialize complete output
        serialized_records = {k: [r.to_dict() for r in v] for k, v in eval_records_all.items()}
        final_output = {
            "diagnostics": diagnostics,
            "config": config,
            "metrics": metrics,
            "eval_records": serialized_records,
        }
        serializable_output = make_json_serializable(final_output)
        atomic_write_json(serializable_output, destination, refuse_overwrite=True)
        return serializable_output

    except Exception as e:
        diagnostics["status"] = "FAILED"
        diagnostics["error_type"] = type(e).__name__
        diagnostics["error_message"] = str(e)
        failure_path = os.path.join(abs_run_dir, "failure_diagnostics.json")
        try:
            serializable_diag = make_json_serializable(diagnostics)
            atomic_write_json(serializable_diag, failure_path, refuse_overwrite=False)
        except Exception:
            pass
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SSRL-02 Two-Rule Sequential Parity Learning Runner")
    parser.add_argument("--dataset_a", required=True, help="Path to Rule A dataset JSON")
    parser.add_argument("--sha256_a", required=True, help="Expected SHA-256 for Rule A dataset")
    parser.add_argument("--dataset_b", required=True, help="Path to Rule B dataset JSON")
    parser.add_argument("--sha256_b", required=True, help="Expected SHA-256 for Rule B dataset")
    parser.add_argument("--destination", required=True, help="Path to destination results JSON")
    parser.add_argument("--run_dir", required=True, help="Unique path to reserve for run directory")
    args = parser.parse_args()

    run_two_rule_study(
        config=APPROVED_TWO_RULE_CONFIG,
        dataset_a_path=args.dataset_a,
        expected_sha_a=args.sha256_a,
        dataset_b_path=args.dataset_b,
        expected_sha_b=args.sha256_b,
        destination=args.destination,
        run_dir=args.run_dir,
    )
