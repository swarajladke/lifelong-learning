"""
research/lifelong_01/run_ssrl_learnability_pilot.py -- Master runner for SSRL-01 Pilot.

Scoped execution for:
- Task 1 (rule coordinates [0, 1, 2, 3]).
- Seed 101.
- Arm: fresh_control.
- 2,000 Adam steps, lr=1e-3, batch_size=16.
- Separate seen-pattern and unseen-pattern evaluation.

Guards:
1. Reserves unique run directory by exclusive creation (os.makedirs(..., exist_ok=False))
   before beginning work. Refuses existing directories to preserve prior artifacts.
2. Verifies dataset SHA-256 byte hash and exhaustive semantic contract before model construction.
3. Enforces versioned specification approval before calling any model constructor.
   Aborts execution if approval status is pending, missing, or mismatched.
   Refuses to compute passing gates or print SUCCESS from proposed thresholds.
4. Applies intended deterministic settings (seeds, deterministic algorithms,
   CUBLAS_WORKSPACE_CONFIG) BEFORE capturing effective environment provenance.
   Records device identity, effective backend flags, file hashes, and initial/final
   model fingerprints during neural runs (never fabricating hashes in model-free tests).
5. Checks monotonic deadlines strictly BEFORE and AFTER completed operations.
6. Persists partial evaluation records incrementally into diagnostics immediately after
   each prediction completes; updates counts at the same point; on evaluation failure,
   preserves all completed records, identifies the incomplete population, and halts without
   computing a passing verdict on partial populations.
7. Error handling writes diagnostics atomically to a unique failure file; never overwrites.
8. Guarded by main: imports construct nothing.
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
import traceback
from typing import Dict, Any, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from research.lifelong_01.ssrl_types import SSRLExampleRecord, SSRLMeasurement
from research.lifelong_01.ssrl_model import SSRLInstructionMLP
from research.lifelong_01.ssrl_generator import load_and_verify_dataset, PilotDatasetContractError
from research.lifelong_01.ssrl_evaluator import (
    evaluate_gate_g_lc,
    compute_accuracy_measurement,
    atomic_write_json
)


APPROVED_PILOT_CONFIG: Dict[str, Any] = {
    "study_id": "SSRL-01",
    "directive": "LIFELONG-01-PILOT",
    "phase": "learnability_pilot",
    "arm": "fresh_control",
    "seed": 101,
    "rule_id": 1,
    "rule_indices": [0, 1, 2, 3],
    "input_dim": 32,
    "input_dim_features": 16,
    "active_k": 4,
    "hidden_dim": 64,
    "output_dim": 2,
    "step_budget": 2000,
    "batch_size": 16,
    "learning_rate": 1e-3,
    "betas": [0.9, 0.999],
    "eps": 1e-8,
    "weight_decay": 0.0,
    "train_pool_size": 384,
    "eval_seen_size": 192,
    "eval_unseen_size": 64,
    "proposed_gate_threshold": 0.90,
    "gate_approval_status": "APPROVED",
    "wall_clock_ceiling_training_seconds": 120.0,
    "wall_clock_ceiling_eval_seconds": 60.0,
}


class SpecificationApprovalError(Exception):
    """Raised when an unapproved or mismatched run specification is submitted for execution."""
    pass


def verify_run_specification_approval(config: Dict[str, Any]) -> None:
    """
    Checks that the run specification is explicitly marked as APPROVED.
    If the approval state is pending, missing, or mismatched, raises
    SpecificationApprovalError before calling any model constructor or
    allocating compute buffers.
    """
    if not isinstance(config, dict):
        raise SpecificationApprovalError("Run configuration must be a dictionary")

    approval_status = config.get("gate_approval_status")
    if approval_status is None:
        raise SpecificationApprovalError("Run configuration is missing required 'gate_approval_status'")

    if approval_status != "APPROVED":
        raise SpecificationApprovalError(
            f"Run specification approval status is '{approval_status}', not 'APPROVED'. "
            "Execution is forbidden before formal approval. Aborting prior to model construction."
        )


def reserve_run_directory(run_dir: str) -> str:
    """
    Reserves a unique run directory by exclusive creation before beginning work.
    Raises FileExistsError if the directory already exists, protecting existing contents.
    """
    run_dir = os.path.abspath(run_dir)
    if os.path.exists(run_dir):
        raise FileExistsError(f"Run directory already exists: {run_dir}. Supply a unique run directory.")
    os.makedirs(run_dir, exist_ok=False)
    return run_dir


def get_file_sha256(filepath: str) -> str:
    """Computes SHA-256 hash of a file on disk."""
    if not os.path.exists(filepath):
        return "FILE_NOT_FOUND"
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def configure_determinism(seed: int, device_name: str = "cpu") -> None:
    """
    Applies intended deterministic settings before capturing effective values.
    
    Launch Requirement:
    When running with CUDA and deterministic algorithms enabled, PyTorch requires
    the environment variable CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8.
    """
    if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    torch.manual_seed(seed)

    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

    if torch.cuda.is_available() and device_name == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.synchronize()


def compute_model_fingerprint(model: nn.Module) -> str:
    """
    Computes a deterministic SHA-256 fingerprint over all parameter tensors of a model.
    Only called during an authorized neural run when an actual model exists.
    """
    hasher = hashlib.sha256()
    for name, p in sorted(model.named_parameters()):
        hasher.update(name.encode("utf-8"))
        hasher.update(p.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


def capture_environment_provenance(
    device: Optional[torch.device] = None,
    dataset_sha256: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    initial_model_fingerprint: Optional[str] = None,
    final_model_fingerprint: Optional[str] = None
) -> Dict[str, Any]:
    """Captures complete execution environment provenance after applying deterministic settings."""
    commit_sha = "UNKNOWN"
    is_dirty = True
    try:
        commit_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        diff_out = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL).decode().strip()
        is_dirty = len(diff_out) > 0
    except Exception:
        pass

    base_dir = os.path.dirname(os.path.abspath(__file__))
    source_files = [
        "run_ssrl_learnability_pilot.py",
        "ssrl_generator.py",
        "ssrl_evaluator.py",
        "ssrl_model.py",
        "ssrl_types.py",
    ]
    source_hashes = {sf: get_file_sha256(os.path.join(base_dir, sf)) for sf in source_files}

    cfg = config if config is not None else APPROVED_PILOT_CONFIG
    config_bytes = json.dumps(cfg, sort_keys=True).encode("utf-8")
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()

    cuda_available = torch.cuda.is_available()
    cuda_deterministic = torch.backends.cudnn.deterministic if cuda_available else None
    cuda_benchmark = torch.backends.cudnn.benchmark if cuda_available else None

    device_info = {
        "type": device.type if device is not None else "cpu",
        "str": str(device) if device is not None else "cpu",
        "cuda_device_name": torch.cuda.get_device_name(device) if (device is not None and device.type == "cuda" and cuda_available) else None,
    }

    return {
        "git_commit": commit_sha,
        "git_dirty": is_dirty,
        "config_sha256": config_sha256,
        "dataset_sha256": dataset_sha256,
        "source_file_hashes": source_hashes,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "device_identity": device_info,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda if cuda_available else None,
        "cudnn_version": torch.backends.cudnn.version() if cuda_available else None,
        "cudnn_deterministic": cuda_deterministic,
        "cudnn_benchmark": cuda_benchmark,
        "initial_model_fingerprint": initial_model_fingerprint,
        "final_model_fingerprint": final_model_fingerprint,
    }


def evaluate_model_on_split(
    model: nn.Module,
    examples: List[Dict[str, Any]],
    dataset_sha256: str,
    arm: str,
    seed: int,
    step: int,
    device: torch.device,
    deadline_monotonic: float,
    pop_name: str,
    diagnostics: Dict[str, Any]
) -> List[SSRLExampleRecord]:
    """
    Performs genuine inference evaluation over a list of dataset example dictionaries.
    - Checks monotonic deadlines strictly BEFORE and AFTER completed operations.
    - Appends serialized per-example records into diagnostics immediately after each prediction completes.
    - Updates record counts at the same point.
    - Marks incomplete_population during evaluation; cleared only upon full split completion.
    """
    model.eval()
    records: List[SSRLExampleRecord] = []

    if "eval_records_by_population" not in diagnostics:
        diagnostics["eval_records_by_population"] = {}
    if pop_name not in diagnostics["eval_records_by_population"]:
        diagnostics["eval_records_by_population"][pop_name] = []

    diagnostics["active_eval_population"] = pop_name
    diagnostics["incomplete_population"] = pop_name

    with torch.no_grad():
        for ex in examples:
            # Pre-operation deadline check
            if time.monotonic() > deadline_monotonic:
                raise TimeoutError(f"Evaluation of {pop_name} exceeded monotonic deadline BEFORE inference on example {ex.get('example_id')}")

            feat = ex["features"]
            mask = ex["mask"]
            u = torch.tensor(feat + mask, dtype=torch.float32, device=device).unsqueeze(0)
            logits = model(u)

            # Post-inference deadline check
            if time.monotonic() > deadline_monotonic:
                raise TimeoutError(f"Evaluation of {pop_name} exceeded monotonic deadline AFTER inference on example {ex.get('example_id')}")

            if not torch.isfinite(logits).all():
                raise FloatingPointError(f"Non-finite output detected during evaluation on example {ex['example_id']}")

            pred_class = int(torch.argmax(logits, dim=1).item())
            target_class = int(ex["target_class"])
            is_correct = (pred_class == target_class)

            rec = SSRLExampleRecord(
                example_id=int(ex["example_id"]),
                rule_id=int(ex["rule_id"]),
                arm=arm,
                seed=seed,
                step=step,
                dataset_sha256=dataset_sha256,
                input_features=feat,
                selection_mask=mask,
                target_class=target_class,
                predicted_class=pred_class,
                is_correct=is_correct,
                is_held_out_pattern=bool(ex["is_held_out_pattern"]),
            )
            records.append(rec)

            # Append serialized record into diagnostics immediately and update counts
            diagnostics["eval_records_by_population"][pop_name].append(rec.to_dict())
            diagnostics["eval_records_count"] = diagnostics.get("eval_records_count", 0) + 1

            # Post-record deadline check
            if time.monotonic() > deadline_monotonic:
                raise TimeoutError(f"Evaluation of {pop_name} exceeded monotonic deadline AFTER recording example {ex.get('example_id')}")

    assert len(records) == len(examples), f"Evaluation count mismatch for {pop_name}: {len(records)} != {len(examples)}"
    diagnostics["incomplete_population"] = None
    return records


def run_learnability_pilot(
    dataset_path: str,
    expected_dataset_sha256: str,
    output_destination: Optional[str] = None,
    device_name: str = "cpu",
    config: Optional[Dict[str, Any]] = None,
    output_artifact_path: Optional[str] = None
) -> int:
    """
    Executes the single-rule learnability pilot under strict verification guards.
    Reserves a unique run path and refuses existing destination without modifying it.
    """
    dest = output_destination if output_destination is not None else output_artifact_path
    if dest is None:
        raise ValueError("Must supply output_destination or output_artifact_path")

    cfg = config if config is not None else APPROVED_PILOT_CONFIG

    # 1. Exclusive Reservation of Output Destination
    output_destination_abs = os.path.abspath(dest)
    if output_destination_abs.endswith(".json"):
        artifact_path = output_destination_abs
        target_dir = os.path.dirname(artifact_path)
        if os.path.exists(artifact_path):
            print(f"[Pilot] ERROR: Run destination file already exists: {artifact_path}", file=sys.stderr)
            print("[Pilot] Refusing run to protect existing artifacts. Supply a unique destination.", file=sys.stderr)
            return 1
        os.makedirs(target_dir, exist_ok=True)
    else:
        try:
            run_dir = reserve_run_directory(output_destination_abs)
        except FileExistsError as e:
            print(f"[Pilot] ERROR: {e}", file=sys.stderr)
            print("[Pilot] Refusing run to protect existing artifacts. Supply a unique run directory.", file=sys.stderr)
            return 1
        artifact_path = os.path.join(run_dir, "results.json")

    start_monotonic = time.monotonic()
    diagnostics: Dict[str, Any] = {
        "status": "INITIALIZING",
        "config": cfg,
        "dataset_path": dataset_path,
        "expected_dataset_sha256": expected_dataset_sha256,
        "completed_training_steps": 0,
        "total_examples_presented": 0,
        "eval_records_count": 0,
        "eval_records_by_population": {},
        "incomplete_population": None,
    }

    try:
        # 2. Configure Determinism and Capture Provenance BEFORE Dataset/Model Work
        seed = cfg["seed"]
        configure_determinism(seed=seed, device_name=device_name)
        device = torch.device(device_name)

        diagnostics["provenance"] = capture_environment_provenance(
            device=device,
            dataset_sha256=expected_dataset_sha256,
            config=cfg,
            initial_model_fingerprint=None,
            final_model_fingerprint=None
        )

        # 3. Dataset Verification Before Any Neural Model Construction
        print(f"[Pilot] Verifying dataset at {dataset_path} against SHA-256 and Pilot Contract...")
        dataset = load_and_verify_dataset(
            filepath=dataset_path,
            expected_sha256=expected_dataset_sha256,
            approved_config=cfg
        )
        diagnostics["dataset_metadata"] = dataset["metadata"]
        diagnostics["status"] = "DATASET_AND_CONTRACT_VERIFIED"
        print(f"[Pilot] Dataset and Contract verified. Hash: {expected_dataset_sha256[:16]}...")

        # 4. Enforce Specification Approval BEFORE Model Construction
        print(f"[Pilot] Verifying specification approval status...")
        verify_run_specification_approval(cfg)
        print(f"[Pilot] Specification approval verified: APPROVED.")

        # 5. Model Construction
        model = SSRLInstructionMLP(
            input_dim=cfg["input_dim"],
            hidden_dim=cfg["hidden_dim"],
            output_dim=cfg["output_dim"]
        ).to(device)

        param_count = sum(p.numel() for p in model.parameters())
        assert param_count == 6402, f"Parameter count mismatch: {param_count} != 6402"
        diagnostics["param_count"] = param_count

        initial_fingerprint = compute_model_fingerprint(model)
        diagnostics["provenance"]["initial_model_fingerprint"] = initial_fingerprint

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg["learning_rate"],
            betas=tuple(cfg["betas"]),
            eps=cfg["eps"],
            weight_decay=cfg["weight_decay"]
        )

        train_pool = dataset["train_examples"]
        step_budget = cfg["step_budget"]
        batch_size = cfg["batch_size"]
        training_ceiling = cfg["wall_clock_ceiling_training_seconds"]

        # 6. Training Loop with Pre/Post-Operation Monotonic Deadline Checks
        diagnostics["status"] = "TRAINING"
        print(f"[Pilot] Starting training: {step_budget} steps, batch size {batch_size}...")
        train_start_monotonic = time.monotonic()
        train_deadline = train_start_monotonic + training_ceiling

        model.train()
        for step in range(1, step_budget + 1):
            if time.monotonic() > train_deadline:
                raise TimeoutError(f"Training exceeded monotonic ceiling of {training_ceiling:.1f}s BEFORE step {step}")

            batch_examples = random.sample(train_pool, batch_size)
            batch_inputs = [ex["features"] + ex["mask"] for ex in batch_examples]
            batch_targets = [ex["target_class"] for ex in batch_examples]

            u_batch = torch.tensor(batch_inputs, dtype=torch.float32, device=device)
            y_batch = torch.tensor(batch_targets, dtype=torch.long, device=device)

            optimizer.zero_grad()
            logits = model(u_batch)
            loss = criterion(logits, y_batch)

            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss ({loss.item()}) detected at training step {step}")

            loss.backward()
            optimizer.step()

            if device.type == "cuda":
                torch.cuda.synchronize()

            if time.monotonic() > train_deadline:
                raise TimeoutError(f"Training exceeded monotonic ceiling of {training_ceiling:.1f}s AFTER completing step {step}")

            diagnostics["completed_training_steps"] = step
            diagnostics["total_examples_presented"] += batch_size

        elapsed_training = time.monotonic() - train_start_monotonic
        diagnostics["elapsed_training_seconds"] = elapsed_training
        print(f"[Pilot] Training complete in {elapsed_training:.2f}s. Completed {step_budget} steps.")

        final_fingerprint = compute_model_fingerprint(model)
        diagnostics["provenance"]["final_model_fingerprint"] = final_fingerprint

        # 7. Incremental Evaluation Loop with Pre/Post-Operation Deadlines and Partial Record Tracking
        diagnostics["status"] = "EVALUATING"
        eval_start_monotonic = time.monotonic()
        eval_ceiling = cfg["wall_clock_ceiling_eval_seconds"]
        eval_deadline = eval_start_monotonic + eval_ceiling

        eval_seen_examples = dataset["eval_seen_examples"]
        eval_unseen_examples = dataset["eval_unseen_examples"]

        rec_seen = evaluate_model_on_split(
            model=model,
            examples=eval_seen_examples,
            dataset_sha256=expected_dataset_sha256,
            arm=cfg["arm"],
            seed=seed,
            step=step_budget,
            device=device,
            deadline_monotonic=eval_deadline,
            pop_name="eval_seen",
            diagnostics=diagnostics
        )

        rec_unseen = evaluate_model_on_split(
            model=model,
            examples=eval_unseen_examples,
            dataset_sha256=expected_dataset_sha256,
            arm=cfg["arm"],
            seed=seed,
            step=step_budget,
            device=device,
            deadline_monotonic=eval_deadline,
            pop_name="eval_unseen",
            diagnostics=diagnostics
        )

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed_eval = time.monotonic() - eval_start_monotonic
        diagnostics["elapsed_eval_seconds"] = elapsed_eval

        # 8. Check Population Completeness Before Computing Gate
        exp_seen_n = cfg["eval_seen_size"]
        exp_unseen_n = cfg["eval_unseen_size"]
        actual_seen_n = len(diagnostics["eval_records_by_population"].get("eval_seen", []))
        actual_unseen_n = len(diagnostics["eval_records_by_population"].get("eval_unseen", []))

        if actual_seen_n != exp_seen_n:
            diagnostics["incomplete_population"] = "eval_seen"
            raise RuntimeError(f"Cannot compute verdict: eval_seen incomplete ({actual_seen_n}/{exp_seen_n})")
        if actual_unseen_n != exp_unseen_n:
            diagnostics["incomplete_population"] = "eval_unseen"
            raise RuntimeError(f"Cannot compute verdict: eval_unseen incomplete ({actual_unseen_n}/{exp_unseen_n})")

        # 9. Metric Derivation and Gate Evaluation
        m_seen = compute_accuracy_measurement("t1_seen_active_patterns", rec_seen, input_set="eval_seen")
        m_unseen = compute_accuracy_measurement("t1_held_out_active_patterns", rec_unseen, input_set="eval_unseen")

        gate_res = evaluate_gate_g_lc(
            seen_measurement=m_seen,
            threshold=cfg["proposed_gate_threshold"],
            approval_status=cfg.get("gate_approval_status", "PROPOSED_PENDING_APPROVAL")
        )

        diagnostics["measurements"] = {
            "seen_patterns": m_seen.to_dict(),
            "unseen_patterns": m_unseen.to_dict(),
        }
        diagnostics["gate_evaluation"] = gate_res
        diagnostics["total_wall_clock_seconds"] = time.monotonic() - start_monotonic

        if not gate_res["passed"]:
            diagnostics["status"] = "FAILED_GATE" if cfg.get("gate_approval_status") == "APPROVED" else "PROPOSED_THRESHOLD_UNAPPROVED"
            atomic_write_json(diagnostics, artifact_path, refuse_overwrite=True)
            print(f"\n[Pilot] GATE NOT PASSED (Status: {gate_res['verdict']}): {gate_res['numerator']}/{gate_res['denominator']} ({gate_res['observed_pct']:.2f}%) vs {gate_res['reference_threshold']*100:.1f}%")
            return 1

        diagnostics["status"] = "COMPLETED_GATE_PASSED"
        atomic_write_json(diagnostics, artifact_path, refuse_overwrite=True)
        print(f"\n[Pilot] SUCCESS: Gate G-LC PASSED ({gate_res['numerator']}/{gate_res['denominator']} = {gate_res['observed_pct']:.2f}%).")
        print(f"[Pilot] Held-out pattern generalization: {m_unseen.numerator}/{m_unseen.denominator} ({m_unseen.pct:.2f}%).")
        return 0

    except Exception as e:
        diagnostics["status"] = "ERROR"
        diagnostics["error_type"] = type(e).__name__
        diagnostics["error_message"] = str(e)
        diagnostics["traceback"] = traceback.format_exc()
        diagnostics["total_wall_clock_seconds"] = time.monotonic() - start_monotonic

        error_path = artifact_path if not os.path.exists(artifact_path) else artifact_path + ".failure_diag.json"
        try:
            atomic_write_json(diagnostics, error_path, refuse_overwrite=True)
            print(f"[Pilot] Diagnostics persisted to {error_path} prior to error exit.")
        except Exception as write_err:
            print(f"[Pilot] CRITICAL: Failed to persist failure diagnostics: {write_err}", file=sys.stderr)

        print(f"[Pilot] Halting with nonzero exit due to {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SSRL-01 Learnability Pilot Runner")
    parser.add_argument("--dataset", required=True, help="Path to verified pilot dataset JSON")
    parser.add_argument("--sha256", required=True, help="Expected SHA-256 hash of dataset file")
    parser.add_argument("--output", required=True, help="Unique path to output results JSON or run directory")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Execution device")
    args = parser.parse_args()

    exit_code = run_learnability_pilot(
        dataset_path=args.dataset,
        expected_dataset_sha256=args.sha256,
        output_destination=args.output,
        device_name=args.device
    )
    sys.exit(exit_code)
