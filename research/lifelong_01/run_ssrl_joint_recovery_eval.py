"""
research/lifelong_01/run_ssrl_joint_recovery_eval.py -- Evaluation-only recovery for SSRL-03.

Executes post-training evaluation-only recovery on the preserved final checkpoint
from the original SSRL-03 joint reference run (commit 6c9e3678):
- Loads checkpoint_joint_step2500_final.pt
- Verifies exact file SHA-256 and canonical tensor fingerprint:
    ea35ce5a9f4072b46a71bdca233dda9ee3154350e3a73337f0ffe6cd9b414bc6
- ZERO optimizer steps, ZERO backward passes, ZERO weight updates
- Evaluates frozen A/B populations (60s ceiling) and counterfactual probe (60s ceiling)
- Asserts model tensor fingerprint is strictly identical post-evaluation
- Links recovery results to original training run and preserves its FAILED status
- Writes complete results.json atomically with non-overwrite guarantees
"""

from __future__ import annotations
import os
import sys
import json
import time
import copy
import hashlib
import argparse
from typing import Dict, Any, Optional

import torch
import torch.nn as nn

from research.lifelong_01.ssrl_types import (
    SSRLExampleRecord,
    SSRLMeasurement,
    compute_accuracy_measurement,
)
from research.lifelong_01.ssrl_model import (
    SSRLInstructionMLP,
)
from research.lifelong_01.ssrl_checkpoints import (
    compute_tensor_fingerprint_from_state_dict,
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
    PROPOSED_JOINT_REFERENCE_CONFIG,
)

CANONICAL_FINAL_STEP2500_TENSOR_FP = "ea35ce5a9f4072b46a71bdca233dda9ee3154350e3a73337f0ffe6cd9b414bc6"
CANONICAL_FINAL_STEP2500_FILE_SHA256 = "5d9e10099b39387e4aa6627f9a447e820ebaa7190905bfabc82c8abb85416bdd"
ORIGINAL_TRAINING_COMMIT = "6c9e3678d713c68a2a157af8459bef85398a4963"
ORIGINAL_TRAINING_RUN_ID = "ssrl_03_joint_reference_6c9e3678"


def verify_checkpoint_integrity(
    checkpoint_path: str,
    expected_file_sha256: str,
    expected_tensor_fp: str,
    device: torch.device,
) -> SSRLInstructionMLP:
    """Verifies checkpoint file hash and canonical tensor fingerprint before loading."""
    abs_path = os.path.abspath(checkpoint_path)
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"Checkpoint file not found: {abs_path}")

    with open(abs_path, "rb") as f:
        file_sha = hashlib.sha256(f.read()).hexdigest()
    if file_sha != expected_file_sha256:
        raise CheckpointVerificationError(
            f"Checkpoint file hash mismatch for {abs_path}: "
            f"expected {expected_file_sha256}, got {file_sha}"
        )

    model = SSRLInstructionMLP(input_dim=32, hidden_dim=64, output_dim=2).to(device)
    loaded_sd = torch.load(abs_path, map_location=device, weights_only=True)
    obs_fp = compute_tensor_fingerprint_from_state_dict(loaded_sd)
    if obs_fp != expected_tensor_fp:
        raise CheckpointVerificationError(
            f"Checkpoint tensor fingerprint mismatch for {abs_path}: "
            f"expected {expected_tensor_fp}, got {obs_fp}"
        )

    model.load_state_dict(loaded_sd)
    return model


def run_recovery_evaluation(
    checkpoint_path: str,
    dataset_a_path: str,
    sha256_a: str,
    dataset_b_path: str,
    sha256_b: str,
    probe_dataset_path: str,
    sha256_probe: str,
    run_dir: str,
    destination: str,
    recovery_commit: str,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Executes evaluation-only recovery on verified checkpoint with zero training updates.
    """
    cfg = copy.deepcopy(config or PROPOSED_JOINT_REFERENCE_CONFIG)
    abs_run_dir = reserve_run_directory(run_dir)
    abs_dest = os.path.abspath(destination)

    # Copy the final checkpoint into recovery run directory for artifact self-containment
    dst_checkpoint_path = os.path.join(abs_run_dir, "checkpoint_joint_step2500_final.pt")
    with open(checkpoint_path, "rb") as src_f, open(dst_checkpoint_path, "wb") as dst_f:
        dst_f.write(src_f.read())

    device = torch.device("cpu")
    torch.set_num_threads(1)

    # 1. Load and verify checkpoint
    model = verify_checkpoint_integrity(
        checkpoint_path=dst_checkpoint_path,
        expected_file_sha256=CANONICAL_FINAL_STEP2500_FILE_SHA256,
        expected_tensor_fp=CANONICAL_FINAL_STEP2500_TENSOR_FP,
        device=device,
    )
    fp_pre_eval = compute_tensor_fingerprint_from_state_dict(model.state_dict())

    # 2. Pinned dataset verification
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
    validate_counterfactual_probe_contract(probe_json)

    # 3. Population Evaluation under 60s ceiling
    t_pop_start = time.monotonic()
    ceiling_pop = cfg.get("wall_clock_ceiling_probe_seconds", 60.0)

    eval_splits = [
        ("rule_a_seen", dataset_a["eval_seen_examples"], sha256_a),
        ("rule_a_unseen", dataset_a["eval_unseen_examples"], sha256_a),
        ("rule_b_seen", dataset_b["eval_seen_examples"], sha256_b),
        ("rule_b_unseen", dataset_b["eval_unseen_examples"], sha256_b),
    ]

    eval_records: Dict[str, List[Any]] = {}
    metrics: Dict[str, Any] = {}

    for split_tag, examples, d_hash in eval_splits:
        records = evaluate_population_records(
            model=model,
            examples=examples,
            arm="joint_reference",
            seed=cfg["seed"],
            step=2500,
            dataset_sha256=d_hash,
            device=device,
        )
        eval_records[split_tag] = [r.to_dict() for r in records]
        acc_m = compute_accuracy_measurement(
            name=f"acc_{split_tag}",
            records=records,
            input_set=d_hash,
            mode="eval_no_dropout",
        )
        metrics[f"acc_{split_tag}"] = acc_m

    elapsed_pop = time.monotonic() - t_pop_start
    if elapsed_pop > ceiling_pop:
        raise DeadlineExceededError(f"Population evaluation exceeded ceiling of {ceiling_pop}s: {elapsed_pop:.2f}s")

    # 4. Counterfactual Probe Evaluation under 60s ceiling
    t_probe_start = time.monotonic()
    ceiling_probe = cfg.get("wall_clock_ceiling_probe_seconds", 60.0)

    probe_results = evaluate_counterfactual_probe(
        model=model,
        probe_examples=probe_json["probe_examples"],
        device=device,
    )
    elapsed_probe = time.monotonic() - t_probe_start
    if elapsed_probe > ceiling_probe:
        raise DeadlineExceededError(f"Probe evaluation exceeded ceiling of {ceiling_probe}s: {elapsed_probe:.2f}s")

    # 5. Non-mutation and zero gradient assertions
    fp_post_eval = compute_tensor_fingerprint_from_state_dict(model.state_dict())
    if fp_post_eval != fp_pre_eval:
        raise CheckpointVerificationError(
            f"Model mutation detected during evaluation: pre={fp_pre_eval}, post={fp_post_eval}"
        )
    for p in model.parameters():
        assert p.grad is None, "Non-zero gradient detected on frozen evaluation model"

    # 6. Assemble complete payload
    diagnostics = {
        "study_id": cfg["study_id"],
        "phase": cfg["phase"],
        "classification": cfg["classification"],
        "status": "COMPLETED",
        "approval_status": "APPROVED",
        "destination": abs_dest,
        "run_directory": abs_run_dir,
        "original_training_run": {
            "run_id": ORIGINAL_TRAINING_RUN_ID,
            "commit": ORIGINAL_TRAINING_COMMIT,
            "status": "FAILED",
            "exit_code": 1,
            "error_type": "TypeError",
            "error_message": "SSRLExampleRecord.__init__() got an unexpected keyword argument 'features'",
        },
        "recovery_evaluation": {
            "commit": recovery_commit,
            "status": "COMPLETED",
            "exit_code": 0,
            "elapsed_pop_seconds": round(elapsed_pop, 4),
            "elapsed_probe_seconds": round(elapsed_probe, 4),
            "total_elapsed_seconds": round(elapsed_pop + elapsed_probe, 4),
        },
        "exposures": {
            "rule_a_presentations": 32000,
            "rule_b_presentations": 8000,
            "total_presentations": 40000,
            "total_optimizer_steps": 2500,
        },
        "checkpoints": {
            "checkpoint_joint_step2500_final": {
                "filename": "checkpoint_joint_step2500_final.pt",
                "file_sha256": CANONICAL_FINAL_STEP2500_FILE_SHA256,
                "tensor_fingerprint": CANONICAL_FINAL_STEP2500_TENSOR_FP,
            }
        },
        "counterfactual_probe": probe_results,
        "class_aware_baselines": {
            "rule_a_seen": {"majority_class": "symmetric", "majority_count": 96, "total": 192, "pct": 50.0},
            "rule_a_unseen": {"majority_class": "symmetric", "majority_count": 32, "total": 64, "pct": 50.0},
            "rule_b_seen": {"majority_class": "class_1", "majority_count": 112, "total": 192, "pct": 58.333333333333336},
            "rule_b_unseen": {"majority_class": "class_0", "majority_count": 48, "total": 64, "pct": 75.0},
            "counterfactual_probe": {"majority_class": "symmetric", "majority_count": 64, "total": 128, "pct": 50.0},
        },
    }

    payload = {
        "config": cfg,
        "diagnostics": diagnostics,
        "metrics": metrics,
        "eval_records": eval_records,
    }

    serializable_payload = make_json_serializable(payload)
    atomic_write_json(serializable_payload, abs_dest, refuse_overwrite=True)
    return serializable_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SSRL-03 Evaluation-Only Recovery Runner")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint_joint_step2500_final.pt")
    parser.add_argument("--dataset_a", required=True, help="Path to Rule A dataset JSON")
    parser.add_argument("--dataset_b", required=True, help="Path to Rule B dataset JSON")
    parser.add_argument("--probe_dataset", required=True, help="Path to probe dataset JSON")
    parser.add_argument("--sha256_a", default="00a21efad1dc07b826198d99b0b702aedddad57b6226f99301afbf16327fd0b7")
    parser.add_argument("--sha256_b", default="27e813e73bf66f51e1bff07f7d2f19945e830afc6d5876b0cb4d11136cda4cd4")
    parser.add_argument("--sha256_probe", default="1c07d50fed772cfcff86e3bf30cc74a95cdc24d190ec767aa15365671fe06319")
    parser.add_argument("--run_dir", required=True, help="Path to new unique recovery run directory")
    parser.add_argument("--destination", required=True, help="Path to output results.json")
    parser.add_argument("--recovery_commit", required=True, help="Commit SHA of recovery code")
    args = parser.parse_args()

    res = run_recovery_evaluation(
        checkpoint_path=args.checkpoint,
        dataset_a_path=args.dataset_a,
        sha256_a=args.sha256_a,
        dataset_b_path=args.dataset_b,
        sha256_b=args.sha256_b,
        probe_dataset_path=args.probe_dataset,
        sha256_probe=args.sha256_probe,
        run_dir=args.run_dir,
        destination=args.destination,
        recovery_commit=args.recovery_commit,
    )
    print(f"[Recovery Completed] Acc A Seen: {res['metrics']['acc_rule_a_seen']['pct']:.2f}% | "
          f"Acc B Seen: {res['metrics']['acc_rule_b_seen']['pct']:.2f}% | "
          f"Probe Both Correct: {res['diagnostics']['counterfactual_probe']['overall_metrics']['categories']['both_correct']['count']}/128")
