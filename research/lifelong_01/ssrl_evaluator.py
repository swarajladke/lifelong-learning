"""
research/lifelong_01/ssrl_evaluator.py -- Evaluation and atomic persistence engine for SSRL-01.

Provides:
- Exact evaluation logic deriving denominations strictly from per-example records.
- Gate G-LC learnability verification against versioned configuration threshold.
- Seed-level transfer score aggregation requiring strict set equality (no partial intersections).
- Exact two-sided paired sign-flip permutation test (enumerating all 2^N configurations).
- Atomic failure-safe serialization before failure exits.
"""

from __future__ import annotations
import json
import os
import tempfile
import math
from typing import List, Dict, Any, Tuple
from research.lifelong_01.ssrl_types import SSRLExampleRecord, SSRLMeasurement, compute_accuracy_measurement


def evaluate_records_by_pattern_split(
    records: List[SSRLExampleRecord],
    task_name: str
) -> Dict[str, SSRLMeasurement]:
    """
    Partitions evaluation records into seen-pattern fitting and unseen-pattern generalization,
    returning explicit SSRLMeasurement objects for each.
    """
    seen_records = [r for r in records if not r.is_held_out_pattern]
    unseen_records = [r for r in records if r.is_held_out_pattern]

    m_seen = compute_accuracy_measurement(
        name=f"{task_name}_seen_pattern_acc",
        records=seen_records,
        input_set=f"{task_name}_seen",
        mode="eval_no_dropout"
    )
    m_unseen = compute_accuracy_measurement(
        name=f"{task_name}_unseen_pattern_acc",
        records=unseen_records,
        input_set=f"{task_name}_unseen",
        mode="eval_no_dropout"
    )

    return {
        "seen_patterns": m_seen,
        "unseen_patterns": m_unseen,
    }


def evaluate_gate_g_lc(
    seen_measurement: SSRLMeasurement,
    threshold: float,
    approval_status: str = "PROPOSED_PENDING_APPROVAL"
) -> Dict[str, Any]:
    """
    Evaluates Gate G-LC: seen active-pattern accuracy >= threshold.
    The threshold is passed explicitly from a versioned configuration dictionary,
    never inferred from an unexplained default.
    Enforces that unapproved/proposed thresholds cannot return a passing gate.
    """
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"Invalid gate threshold {threshold}, must be between 0.0 and 1.0")

    observed = seen_measurement.rate
    meets_threshold = observed >= threshold
    deviation = observed - threshold

    is_approved = (approval_status == "APPROVED")
    passed = meets_threshold and is_approved

    if not is_approved:
        verdict = "PROPOSED_THRESHOLD_UNAPPROVED"
    elif passed:
        verdict = "PASSED"
    else:
        verdict = "FAILED"

    return {
        "gate_name": "Gate G-LC (Learnability Control)",
        "observed_rate": observed,
        "observed_pct": seen_measurement.pct,
        "reference_threshold": threshold,
        "approval_status": approval_status,
        "numerator": seen_measurement.numerator,
        "denominator": seen_measurement.denominator,
        "meets_threshold": meets_threshold,
        "passed": passed,
        "deviation": deviation,
        "verdict": verdict,
    }


def compute_seed_transfer_score(
    seq_records_by_rule: Dict[int, List[SSRLExampleRecord]],
    fresh_records_by_rule: Dict[int, List[SSRLExampleRecord]],
    declared_transfer_rules: List[int]
) -> float:
    """
    Computes summary transfer delta for one seed across all evaluated transfer rules.
    Delta = mean_{j} ( Acc_seq(rule_j) - Acc_fresh(rule_j) ).
    Strict Guard: Requires that BOTH dictionaries match the declared transfer rules exactly.
    Partial intersections or missing rules are rejected immediately.
    """
    declared_set = set(declared_transfer_rules)
    seq_set = set(seq_records_by_rule.keys())
    fresh_set = set(fresh_records_by_rule.keys())

    if seq_set != declared_set:
        raise ValueError(
            f"Sequential evaluation rules do not match declared transfer rules!\n"
            f"Declared: {sorted(declared_set)}\nFound: {sorted(seq_set)}"
        )
    if fresh_set != declared_set:
        raise ValueError(
            f"Fresh control evaluation rules do not match declared transfer rules!\n"
            f"Declared: {sorted(declared_set)}\nFound: {sorted(fresh_set)}"
        )

    diffs = []
    for r_id in declared_transfer_rules:
        r_seq = seq_records_by_rule[r_id]
        r_fresh = fresh_records_by_rule[r_id]
        if len(r_seq) == 0:
            raise ValueError(f"Sequential records for rule {r_id} are empty")
        if len(r_fresh) == 0:
            raise ValueError(f"Fresh records for rule {r_id} are empty")

        # Verify aligned evaluation identities and targets
        if len(r_seq) != len(r_fresh):
            raise ValueError(f"Population length mismatch for rule {r_id}: seq={len(r_seq)}, fresh={len(r_fresh)}")
        for r_s, r_f in zip(r_seq, r_fresh):
            if r_s.example_id != r_f.example_id:
                raise ValueError(f"Misaligned example IDs for rule {r_id}: seq={r_s.example_id}, fresh={r_f.example_id}")
            if r_s.target_class != r_f.target_class:
                raise ValueError(f"Target class mismatch for example {r_s.example_id}: seq={r_s.target_class}, fresh={r_f.target_class}")

        acc_s = sum(1 for r in r_seq if r.is_correct) / float(len(r_seq))
        acc_f = sum(1 for r in r_fresh if r.is_correct) / float(len(r_fresh))
        diffs.append(acc_s - acc_f)

    return sum(diffs) / float(len(diffs))


def compute_exact_sign_flip_test(paired_diffs: List[float], eps: float = 1e-12) -> Dict[str, Any]:
    """
    Computes exact two-sided sign-flip permutation test across 2^N configurations.
    Assumption: paired differences are sign-exchangeable under H0.
    """
    n = len(paired_diffs)
    if n == 0:
        raise ValueError("Cannot perform permutation test on empty paired differences list")

    obs_mean = sum(paired_diffs) / float(n)
    obs_t = abs(obs_mean)
    total_perms = 1 << n
    count_extreme = 0

    for mask in range(total_perms):
        perm_sum = 0.0
        for i in range(n):
            sign = 1.0 if ((mask >> i) & 1) == 0 else -1.0
            perm_sum += sign * paired_diffs[i]
        perm_t = abs(perm_sum / float(n))
        if perm_t >= (obs_t - eps):
            count_extreme += 1

    p_value = count_extreme / float(total_perms)
    resolution_floor = 2.0 / float(total_perms)

    return {
        "n_seeds": n,
        "observed_mean_diff": obs_mean,
        "total_permutations": total_perms,
        "extreme_count": count_extreme,
        "exact_p_value": p_value,
        "resolution_floor": resolution_floor,
        "assumption": "Sign-exchangeability of seed-level paired differences under H0",
    }


def atomic_write_json(data: Dict[str, Any], target_path: str, refuse_overwrite: bool = True) -> None:
    """
    Writes JSON data atomically using temporary file rename to prevent
    corrupted or partially-written artifacts upon unexpected failure.
    If refuse_overwrite is True and target_path already exists, raises FileExistsError.
    """
    target_path = os.path.abspath(target_path)
    target_dir = os.path.dirname(target_path)
    os.makedirs(target_dir, exist_ok=True)

    if refuse_overwrite and os.path.exists(target_path):
        raise FileExistsError(f"Target artifact already exists and refuse_overwrite=True: {target_path}")

    temp_fd, temp_path = tempfile.mkstemp(dir=target_dir, prefix="tmp_atomic_", suffix=".json")
    try:
        with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        os.replace(temp_path, target_path)
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise
