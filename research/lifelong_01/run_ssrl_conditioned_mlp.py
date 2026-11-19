"""
research/lifelong_01/run_ssrl_conditioned_mlp.py -- Procedural Runner for SSRL-09.

Evaluates SSRLConditionedMLP (LayerNorm + learned mask-conditioned gating) on
Seeds 102, 103, 104 across four matched arms:
- FREEZE-AFTER-BASE (standing control: zero updates on verified post-A checkpoint)
- fresh_b (matched fresh comparator, 500 Task B updates)
- seq_b (naive sequential transfer from post-A with zeroed Adam moments, 500 Task B updates)
- diverse_replay (sequential + M=16 pattern-diverse replay 4 B : 1 A, 625 total updates)
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

from research.lifelong_01.ssrl_checkpoints import (
    capture_and_verify_checkpoint,
    compute_tensor_fingerprint_from_state_dict,
)
from research.lifelong_01.ssrl_model import (
    SSRLConditionedMLP,
    get_conditioned_parameter_count,
)
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
from research.lifelong_01.ssrl_types import compute_accuracy_measurement
from research.lifelong_01.run_ssrl_two_rule import build_minibatch_schedule
from research.lifelong_01.run_ssrl_replay_baseline import build_replay_schedule
from research.lifelong_01.run_ssrl_diverse_replay import select_pattern_diverse_replay_memory
from research.lifelong_01.run_ssrl_paired_replication import compute_schedule_hash
from research.lifelong_01.run_ssrl_l2_init_plasticity import (
    LearnabilityGateError,
    _assert_learnability_gate,
    build_seed_comparisons,
    _summarize_scalar_list,
)


PROPOSED_CONDITIONED_MLP_CONFIG: Dict[str, Any] = {
    "study_id": "SSRL-09",
    "phase": "conditioned_mlp_layernorm_gating",
    "classification": "architectural_conditioning_three_seed",
    "gate_approval_status": "APPROVED",
    "model_class": "SSRLConditionedMLP",
    "expected_parameter_count": 6722,
    "seeds": [102, 103, 104],
    "minibatch_sampler_seed_a": 2026,
    "minibatch_sampler_seed_b": 2027,
    "rule_a_id": 1,
    "rule_a_indices": [0, 1, 2, 3],
    "rule_b_id": 2,
    "rule_b_indices": [4, 5, 6, 7],
    "train_pool_size": 384,
    "batch_size": 16,
    "step_budget_a": 2000,
    "step_budget_b": 500,
    "replay_buffer_size": 16,
    "replay_ratio_b_to_a": [4, 1],
    "milestone_b_steps": [0, 25, 50, 100, 250, 500],
    "learning_rate": 0.001,
    "betas": [0.9, 0.999],
    "eps": 1e-08,
    "weight_decay": 0.0,
    "gate_g_lc_threshold": 0.90,
    "enforce_post_a_learnability_gate": True,
    "enforce_fresh_b_learnability_gate": True,
    "wall_clock_ceiling_training_seconds_per_stage": 60.0,
    "wall_clock_ceiling_eval_seconds_per_stage": 60.0,
}

EVAL_MODE_CONDITIONED = "eval_layernorm_no_dropout"


def verify_conditioned_mlp_specification_approval(config: Dict[str, Any]) -> None:
    """AGENTS.md Section 2 / Section 8: Strictly blocks unapproved execution."""
    status = config.get("gate_approval_status")
    if status != "APPROVED":
        raise SpecificationApprovalError(
            f"SSRL-09 conditioned MLP specification status is '{status}', not 'APPROVED'. "
            "Execution is forbidden before explicit user authorization."
        )


def compute_conditioned_mechanism_snapshot(
    model: SSRLConditionedMLP, db: Dict[str, Any]
) -> Dict[str, Any]:
    """Records parameter count, weight norms, Rule A/B mask-gate separation, and Rule B diagnostics."""
    param_count = sum(p.numel() for p in model.parameters())
    expected_count = get_conditioned_parameter_count()
    assert param_count == expected_count == 6722, f"Expected 6722 params, got {param_count}"

    fp_before = compute_tensor_fingerprint_from_state_dict(model.state_dict())
    model.eval()
    assert not model.training, "Mechanism snapshot requires eval mode"

    with torch.no_grad():
        all_p = torch.cat([p.detach().to(torch.float32).reshape(-1) for p in model.parameters()], dim=0)
        w_feat = model.fc1_feat.weight.detach().to(torch.float32)
        w_gate = model.fc1_gate.weight.detach().to(torch.float32)

        mask_a = torch.tensor([[1.0] * 4 + [0.0] * 12], dtype=torch.float32)
        mask_b = torch.tensor([[0.0] * 4 + [1.0] * 4 + [0.0] * 8], dtype=torch.float32)
        g_a = torch.sigmoid(model.fc1_gate(mask_a))
        g_b = torch.sigmoid(model.fc1_gate(mask_b))
        gate_cos = float(F.cosine_similarity(g_a, g_b, dim=1).item())
        gate_l2_dist = float(torch.linalg.vector_norm(g_a - g_b, ord=2).item())

        seen_b = db["eval_seen_examples"]
        assert len(seen_b) == 192, f"Expected 192 Rule B seen examples, got {len(seen_b)}"
        x_b = torch.tensor([ex["features"] + ex["mask"] for ex in seen_b], dtype=torch.float32)
        y_b = torch.tensor([int(ex["target_class"]) for ex in seen_b], dtype=torch.long)

        feat_b, m_b = x_b[:, : model.feature_dim], x_b[:, model.feature_dim :]
        h1 = torch.sigmoid(model.fc1_gate(m_b)) * model.relu1(model.ln1(model.fc1_feat(feat_b)))
        h2 = model.relu2(model.ln2(model.fc2(h1)))
        logits = model.fc3(h2)
        assert torch.equal(logits, model(x_b)), "Conditioned forward diagnostic mismatch"

        ce_loss_b = float(nn.CrossEntropyLoss()(logits, y_b).item())
        abs_margins_b = torch.abs(logits[:, 1] - logits[:, 0])
        h1_dead = int((~(h1 > 0.0).any(dim=0)).sum().item())
        h2_dead = int((~(h2 > 0.0).any(dim=0)).sum().item())

    fp_after = compute_tensor_fingerprint_from_state_dict(model.state_dict())
    assert fp_after == fp_before, "Conditioned mechanism snapshot mutated parameters"
    return {
        "mode": EVAL_MODE_CONDITIONED,
        "parameter_count": param_count,
        "hidden_dim": 64,
        "rule_b_seen_denominator": len(seen_b),
        "global_l2": float(torch.linalg.vector_norm(all_p, ord=2).item()),
        "fc1_feat_weight_l2": float(torch.linalg.vector_norm(w_feat, ord=2).item()),
        "fc1_gate_weight_l2": float(torch.linalg.vector_norm(w_gate, ord=2).item()),
        "fc2_weight_l2": float(torch.linalg.vector_norm(model.fc2.weight.detach().to(torch.float32), ord=2).item()),
        "fc3_weight_l2": float(torch.linalg.vector_norm(model.fc3.weight.detach().to(torch.float32), ord=2).item()),
        "fc1_feat_rule_a_cols_l2": float(torch.linalg.vector_norm(w_feat[:, 0:4], ord=2).item()),
        "fc1_feat_rule_b_cols_l2": float(torch.linalg.vector_norm(w_feat[:, 4:8], ord=2).item()),
        "gate_a_mean": float(g_a.mean().item()),
        "gate_b_mean": float(g_b.mean().item()),
        "gate_ab_cosine_similarity": gate_cos,
        "gate_ab_l2_distance": gate_l2_dist,
        "mean_abs_logit_margin_b": float(abs_margins_b.mean().item()),
        "cross_entropy_loss_b": ce_loss_b,
        "h1_dead_relu_count": h1_dead,
        "h2_dead_relu_count": h2_dead,
    }


def compact_population_eval_conditioned(
    model: SSRLConditionedMLP, examples: List[Dict[str, Any]], arm_tag: str,
    seed: int, step: int, dataset_sha256: str, split_name: str,
) -> Dict[str, Any]:
    """Evaluates population under eval_layernorm_no_dropout with compact per-unit outcomes."""
    recs = evaluate_population_records(
        model=model, examples=examples, arm=arm_tag, seed=seed, step=step,
        dataset_sha256=dataset_sha256, device=torch.device("cpu"),
    )
    metric = compute_accuracy_measurement(
        name=f"acc_{split_name}", records=recs, input_set=dataset_sha256, mode=EVAL_MODE_CONDITIONED
    )
    return {
        "metric": metric,
        "per_unit_outcomes": {
            "example_ids": [r.example_id for r in recs],
            "is_correct": [r.is_correct for r in recs],
            "predicted_class": [r.predicted_class for r in recs],
            "target_class": [r.target_class for r in recs],
        },
    }


def compact_battery_eval_conditioned(
    model: SSRLConditionedMLP, da: Dict[str, Any], db: Dict[str, Any], dp: Dict[str, Any],
    seed: int, step: int, arm_tag: str, sha256_a: str, sha256_b: str, eval_deadline: float,
) -> Dict[str, Any]:
    """Runs compact 5-population battery evaluation under monotonic eval deadline."""
    if time.monotonic() > eval_deadline:
        raise DeadlineExceededError(f"Evaluation deadline exceeded before evaluating {arm_tag}")
    model.eval()
    assert not model.training
    t_start = time.monotonic()
    splits = [
        ("rule_a_seen", da["eval_seen_examples"], sha256_a, 192),
        ("rule_a_unseen", da["eval_unseen_examples"], sha256_a, 64),
        ("rule_b_seen", db["eval_seen_examples"], sha256_b, 192),
        ("rule_b_unseen", db["eval_unseen_examples"], sha256_b, 64),
    ]
    metrics: Dict[str, Any] = {}
    per_unit_populations: Dict[str, Any] = {}
    for split_tag, examples, d_hash, expected_n in splits:
        assert len(examples) == expected_n, f"{split_tag} expected {expected_n}, got {len(examples)}"
        res = compact_population_eval_conditioned(model, examples, arm_tag, seed, step, d_hash, split_tag)
        metrics[f"acc_{split_tag}"] = res["metric"]
        per_unit_populations[split_tag] = res["per_unit_outcomes"]

    probe_full = evaluate_counterfactual_probe(model=model, probe_examples=dp["probe_examples"], device=torch.device("cpu"))
    compact_pairs = [{k: v for k, v in rec.items() if k != "features"} for rec in probe_full["pair_records"]]
    assert len(compact_pairs) == 128, f"Expected 128 probe pairs, got {len(compact_pairs)}"
    if time.monotonic() > eval_deadline:
        raise DeadlineExceededError(f"Evaluation deadline exceeded after evaluating {arm_tag}")
    return {
        "elapsed_eval_seconds": time.monotonic() - t_start,
        "mode": EVAL_MODE_CONDITIONED,
        "metrics": metrics,
        "per_unit_populations": per_unit_populations,
        "counterfactual_probe": {
            "overall_metrics": probe_full["overall_metrics"],
            "strata_metrics": probe_full["strata_metrics"],
            "pair_records": compact_pairs,
        },
    }


def train_conditioned_schedule(
    initial_model_state: Dict[str, torch.Tensor], tagged_schedule: List[Tuple[str, int, List[Dict[str, Any]]]],
    db: Dict[str, Any], sha256_b: str, config: Dict[str, Any], seed: int, arm_tag: str,
    train_deadline: float, eval_ceiling: float, expected_initial_fingerprint: str,
    evaluate_b_milestones: bool = True,
) -> Tuple[SSRLConditionedMLP, float, Dict[str, int], str, Dict[str, Any]]:
    """Trains SSRLConditionedMLP on a tagged schedule with zeroed Adam moments and CE loss."""
    if time.monotonic() > train_deadline:
        raise DeadlineExceededError(f"Training deadline exceeded before starting {arm_tag}")

    model = SSRLConditionedMLP(feature_dim=16, mask_dim=16, hidden_dim=64, output_dim=2)
    model.load_state_dict(initial_model_state)
    loaded_fp = compute_tensor_fingerprint_from_state_dict(model.state_dict())
    if loaded_fp != expected_initial_fingerprint:
        raise CheckpointVerificationError(
            f"Initial fingerprint mismatch for {arm_tag}: expected {expected_initial_fingerprint}, got {loaded_fp}"
        )

    optimizer = optim.Adam(
        model.parameters(),
        lr=config["learning_rate"], betas=tuple(config["betas"]),
        eps=config["eps"], weight_decay=config["weight_decay"],
    )
    loss_fn = nn.CrossEntropyLoss()
    counters = {
        "total_optimizer_steps": 0, "total_presentations": 0,
        "rule_a_updates": 0, "rule_a_presentations": 0,
        "rule_b_updates": 0, "rule_b_presentations": 0,
        "replay_a_updates": 0, "replay_a_presentations": 0,
    }
    milestones_set = set(config["milestone_b_steps"]) if evaluate_b_milestones else set()
    milestone_records: Dict[str, Any] = {}
    t_start = time.monotonic()

    if 0 in milestones_set:
        t_eval_dl = time.monotonic() + eval_ceiling
        model.eval()
        m_0 = compact_population_eval_conditioned(
            model, db["eval_seen_examples"], f"{arm_tag}_bstep0", seed, 0, sha256_b, "rule_b_seen"
        )
        milestone_records["0"] = {"rule_b_updates": 0, "total_steps": 0, **m_0}
        if time.monotonic() > t_eval_dl:
            raise DeadlineExceededError(f"Milestone 0 eval deadline exceeded for {arm_tag}")

    model.train()
    for tag, _, batch in tagged_schedule:
        if time.monotonic() > train_deadline:
            raise DeadlineExceededError(f"Training deadline exceeded at step {counters['total_optimizer_steps'] + 1} for {arm_tag}")

        inputs = torch.stack([torch.tensor(ex["features"] + ex["mask"], dtype=torch.float32) for ex in batch], dim=0)
        targets = torch.tensor([int(ex["target_class"]) for ex in batch], dtype=torch.long)

        optimizer.zero_grad()
        loss = loss_fn(model(inputs), targets)
        loss.backward()
        optimizer.step()

        counters["total_optimizer_steps"] += 1
        counters["total_presentations"] += len(batch)
        counters[f"{tag}_updates"] += 1
        counters[f"{tag}_presentations"] += len(batch)

        if tag == "rule_b" and counters["rule_b_updates"] in milestones_set:
            b_step = counters["rule_b_updates"]
            t_eval_dl = time.monotonic() + eval_ceiling
            model.eval()
            m_b = compact_population_eval_conditioned(
                model, db["eval_seen_examples"], f"{arm_tag}_bstep{b_step}", seed, b_step, sha256_b, "rule_b_seen"
            )
            milestone_records[str(b_step)] = {
                "rule_b_updates": b_step, "total_steps": counters["total_optimizer_steps"], **m_b
            }
            if time.monotonic() > t_eval_dl:
                raise DeadlineExceededError(f"Milestone {b_step} eval deadline exceeded for {arm_tag}")
            model.train()

    assert counters["total_optimizer_steps"] > 0 and counters["total_presentations"] > 0
    return model, time.monotonic() - t_start, counters, loaded_fp, milestone_records


def _run_conditioned_task_b_arm(
    arm_key: str, start_state: Dict[str, torch.Tensor], expected_start_fp: str,
    schedule: List[Tuple[str, int, List[Dict[str, Any]]]], da: Dict[str, Any], db: Dict[str, Any],
    dp: Dict[str, Any], sha256_a: str, sha256_b: str, cfg: Dict[str, Any], seed: int,
    abs_run_dir: str, t_train_ceil: float, t_eval_ceil: float, enforce_b_gate: bool = False,
) -> Dict[str, Any]:
    """Trains a Task B conditioned arm, enforces optional G-LC gate, snapshots mechanism, saves checkpoint."""
    model, t_train, cnt, _, miles = train_conditioned_schedule(
        start_state, schedule, db, sha256_b, cfg, seed,
        f"cond_{arm_key}_s{seed}", time.monotonic() + t_train_ceil, t_eval_ceil, expected_start_fp, evaluate_b_milestones=True,
    )
    if enforce_b_gate:
        b_step_key = str(cfg["step_budget_b"])
        b_metric = (
            miles[b_step_key]["metric"] if b_step_key in miles
            else compact_population_eval_conditioned(
                model, db["eval_seen_examples"], f"cond_{arm_key}_gate_s{seed}", seed, cfg["step_budget_b"], sha256_b, "rule_b_seen"
            )["metric"]
        )
        _assert_learnability_gate(b_metric, seed, arm_key, cfg["gate_g_lc_threshold"])
    final_eval = compact_battery_eval_conditioned(
        model, da, db, dp, seed, len(schedule), f"cond_{arm_key}_final_s{seed}", sha256_a, sha256_b, time.monotonic() + t_eval_ceil
    )
    mech = compute_conditioned_mechanism_snapshot(model, db)
    ckpt = capture_and_verify_checkpoint(model.state_dict(), abs_run_dir, f"checkpoint_cond_{arm_key}_s{seed}.pt")
    return {
        "elapsed_train_seconds": t_train, "counters": cnt, "loaded_initial_fingerprint": expected_start_fp,
        "checkpoint": ckpt, "mechanism_snapshot": mech, "milestone_evaluations": miles, "final_eval": final_eval,
    }


def run_conditioned_mlp_study(
    dataset_a_path: str, sha256_a: str, dataset_b_path: str, sha256_b: str,
    probe_dataset_path: str, sha256_probe: str, destination: str, run_dir: str,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Master procedural entry point for SSRL-09 Conditioned MLP Study."""
    cfg = copy.deepcopy(config or PROPOSED_CONDITIONED_MLP_CONFIG)
    verify_conditioned_mlp_specification_approval(cfg)
    assert get_conditioned_parameter_count() == cfg["expected_parameter_count"] == 6722
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
        "status": "IN_PROGRESS", "current_stage": "INITIALIZATION",
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
        diagnostics["schedule_hash_diverse"] = compute_schedule_hash(sched_diverse)

        for seed in cfg["seeds"]:
            diagnostics["seeds_status"][seed] = "IN_PROGRESS"
            seed_rec: Dict[str, Any] = {"seed": seed}
            diagnostics["seeds_data"][str(seed)] = seed_rec

            torch.manual_seed(seed)
            model_init = SSRLConditionedMLP(feature_dim=16, mask_dim=16, hidden_dim=64, output_dim=2)
            fp_init = compute_tensor_fingerprint_from_state_dict(model_init.state_dict())
            seed_rec["fresh_init_fingerprint"] = fp_init
            seed_rec["mechanism_fresh_init"] = compute_conditioned_mechanism_snapshot(model_init, db)

            diagnostics["current_stage"] = f"seed_{seed}_phase_a"
            model_a, t_train_a, cnt_a, _, _ = train_conditioned_schedule(
                model_init.state_dict(), tagged_a, db, sha256_b, cfg, seed,
                f"cond_phase_a_s{seed}", time.monotonic() + t_train_ceil, t_eval_ceil, fp_init, evaluate_b_milestones=False,
            )
            post_a_eval_a = compact_population_eval_conditioned(model_a, da["eval_seen_examples"], f"cond_post_a_s{seed}", seed, cfg["step_budget_a"], sha256_a, "rule_a_seen")
            if cfg["enforce_post_a_learnability_gate"]:
                _assert_learnability_gate(post_a_eval_a["metric"], seed, "post_a", cfg["gate_g_lc_threshold"])
            ckpt_post_a = capture_and_verify_checkpoint(model_a.state_dict(), abs_run_dir, f"checkpoint_cond_post_a_s{seed}.pt")
            fp_post_a = ckpt_post_a["tensor_fingerprint"]
            seed_rec["phase_a"] = {
                "elapsed_train_seconds": t_train_a, "counters": cnt_a, "checkpoint": ckpt_post_a,
                "post_a_rule_a_seen": post_a_eval_a, "mechanism_post_a": compute_conditioned_mechanism_snapshot(model_a, db),
            }

            diagnostics["current_stage"] = f"seed_{seed}_freeze_after_base"
            eval_freeze = compact_battery_eval_conditioned(model_a, da, db, dp, seed, 0, f"cond_freeze_s{seed}", sha256_a, sha256_b, time.monotonic() + t_eval_ceil)
            assert compute_tensor_fingerprint_from_state_dict(model_a.state_dict()) == fp_post_a
            seed_rec["freeze_after_base"] = {"fingerprint": fp_post_a, "eval": eval_freeze}

            arm_specs = [
                ("fresh_b", model_init.state_dict(), fp_init, tagged_b, bool(cfg["enforce_fresh_b_learnability_gate"])),
                ("seq_b", model_a.state_dict(), fp_post_a, tagged_b, False),
                ("diverse_replay", model_a.state_dict(), fp_post_a, sched_diverse, False),
            ]
            for arm_key, start_sd, start_fp, arm_sched, arm_gate in arm_specs:
                diagnostics["current_stage"] = f"seed_{seed}_{arm_key}"
                seed_rec[arm_key] = _run_conditioned_task_b_arm(
                    arm_key, start_sd, start_fp, arm_sched, da, db, dp, sha256_a, sha256_b,
                    cfg, seed, abs_run_dir, t_train_ceil, t_eval_ceil, enforce_b_gate=arm_gate,
                )

            comp = build_seed_comparisons(
                seed, cfg,
                seed_rec["fresh_b"]["milestone_evaluations"],
                seed_rec["seq_b"]["milestone_evaluations"],
                seed_rec["diverse_replay"]["milestone_evaluations"],
                eval_freeze, seed_rec["seq_b"]["final_eval"], seed_rec["diverse_replay"]["final_eval"],
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
    parser = argparse.ArgumentParser(description="SSRL-09 Conditioned MLP Study Runner")
    parser.add_argument("--dataset_a", required=True)
    parser.add_argument("--dataset_b", required=True)
    parser.add_argument("--probe_dataset", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--sha256_a", default="00a21efad1dc07b826198d99b0b702aedddad57b6226f99301afbf16327fd0b7")
    parser.add_argument("--sha256_b", default="27e813e73bf66f51e1bff07f7d2f19945e830afc6d5876b0cb4d11136cda4cd4")
    parser.add_argument("--sha256_probe", default="1c07d50fed772cfcff86e3bf30cc74a95cdc24d190ec767aa15365671fe06319")
    args = parser.parse_args()
    run_conditioned_mlp_study(
        dataset_a_path=args.dataset_a, sha256_a=args.sha256_a,
        dataset_b_path=args.dataset_b, sha256_b=args.sha256_b,
        probe_dataset_path=args.probe_dataset, sha256_probe=args.sha256_probe,
        destination=args.destination, run_dir=args.run_dir,
    )


if __name__ == "__main__":
    main()
