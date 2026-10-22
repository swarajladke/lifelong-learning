"""
research/lifelong_01/ssrl_counterfactual_probe.py -- Counterfactual Mask Probe for SSRL-02.

Specification:
1. Exactly 128 unique feature vectors x in {-1, +1}^16, each evaluated under BOTH:
   - Mask A: coords [0, 1, 2, 3] (Rule 1)
   - Mask B: coords [4, 5, 6, 7] (Rule 2)
2. Opposing parity requirement: Parity_A(x) != Parity_B(x) for all 128 vectors.
3. Exact balanced orientation:
   - Exactly 64 vectors with (target_a=1, target_b=0)
   - Exactly 64 vectors with (target_a=0, target_b=1)
4. Stratified coverage across seen and held-out active subcubes:
   - Stratum 1 (seen_seen): 64 vectors (32 per orientation)
   - Stratum 2 (seen_unseen): 32 vectors (16 per orientation)
   - Stratum 3 (unseen_seen): 32 vectors (16 per orientation)
   - Stratum 4 (unseen_unseen): 0 vectors (explicitly excluded)
5. Strict training pool exclusion:
   x not in train_examples(Rule A) and x not in train_examples(Rule B).
6. Deterministic sampling with seed 3030. Never relaxes quotas silently.
7. Evaluates both masks and partitions outcomes into 4 descriptive categories:
   - both_correct
   - both_incorrect_with_switched_predictions
   - unchanged_prediction_matching_a_target
   - unchanged_prediction_matching_b_target
"""

from __future__ import annotations
import json
import random
from typing import Dict, List, Tuple, Any, Optional
import torch
import torch.nn as nn


PROBE_SAMPLE_SIZE: int = 128
PROBE_SEED_DEFAULT: int = 3030

RULE_A_INDICES: List[int] = [0, 1, 2, 3]
RULE_B_INDICES: List[int] = [4, 5, 6, 7]
DISTRACTOR_INDICES: List[int] = [8, 9, 10, 11, 12, 13, 14, 15]

STRATA_QUOTAS: Dict[str, int] = {
    "seen_seen": 64,
    "seen_unseen": 32,
    "unseen_seen": 32,
    "unseen_unseen": 0,
}

ORIENTATION_QUOTAS: Dict[Tuple[int, int], int] = {
    (1, 0): 64,
    (0, 1): 64,
}

# Per-stratum orientation breakdown: exactly half (1,0) and half (0,1)
STRATA_ORIENTATION_QUOTAS: Dict[str, Dict[Tuple[int, int], int]] = {
    "seen_seen": {(1, 0): 32, (0, 1): 32},
    "seen_unseen": {(1, 0): 16, (0, 1): 16},
    "unseen_seen": {(1, 0): 16, (0, 1): 16},
    "unseen_unseen": {(1, 0): 0, (0, 1): 0},
}


def compute_parity(bits: List[int]) -> int:
    """Computes multiplicative parity label: 1 if even count of -1s, 0 if odd."""
    prod = 1
    for b in bits:
        prod *= b
    return 1 if prod == 1 else 0


def extract_active_patterns(dataset: Dict[str, Any], rule_indices: List[int]) -> Tuple[set, set]:
    """Extracts seen and unseen active patterns from a pinned dataset."""
    seen_patterns = {
        tuple(e["features"][i] for i in rule_indices)
        for e in dataset["eval_seen_examples"]
    }
    unseen_patterns = {
        tuple(e["features"][i] for i in rule_indices)
        for e in dataset["eval_unseen_examples"]
    }
    return seen_patterns, unseen_patterns


def generate_counterfactual_probe_dataset(
    dataset_a: Dict[str, Any],
    dataset_b: Dict[str, Any],
    seed: int = PROBE_SEED_DEFAULT,
) -> Dict[str, Any]:
    """
    Generates the counterfactual mask probe dataset according to the pinned specification.
    Raises RuntimeError if quotas cannot be satisfied without relaxing constraints.
    """
    rng = random.Random(seed)

    # 1. Extract active pattern partitions
    seen_a, unseen_a = extract_active_patterns(dataset_a, RULE_A_INDICES)
    seen_b, unseen_b = extract_active_patterns(dataset_b, RULE_B_INDICES)

    # 2. Extract forbidden training vectors from both training pools
    forbidden_xs = {
        tuple(e["features"]) for e in dataset_a["train_examples"]
    } | {
        tuple(e["features"]) for e in dataset_b["train_examples"]
    }

    # Group active patterns by parity
    pats_a_by_parity: Dict[str, Dict[int, List[Tuple[int, ...]]]] = {
        "seen": {0: [p for p in seen_a if compute_parity(list(p)) == 0],
                 1: [p for p in seen_a if compute_parity(list(p)) == 1]},
        "unseen": {0: [p for p in unseen_a if compute_parity(list(p)) == 0],
                   1: [p for p in unseen_a if compute_parity(list(p)) == 1]},
    }
    pats_b_by_parity: Dict[str, Dict[int, List[Tuple[int, ...]]]] = {
        "seen": {0: [p for p in seen_b if compute_parity(list(p)) == 0],
                 1: [p for p in seen_b if compute_parity(list(p)) == 1]},
        "unseen": {0: [p for p in unseen_b if compute_parity(list(p)) == 0],
                   1: [p for p in unseen_b if compute_parity(list(p)) == 1]},
    }

    selected_xs: set = set()
    probe_examples: List[Dict[str, Any]] = []
    probe_id_counter = 0

    strata_definitions = [
        ("seen_seen", "seen", "seen"),
        ("seen_unseen", "seen", "unseen"),
        ("unseen_seen", "unseen", "seen"),
    ]

    for stratum_name, a_split, b_split in strata_definitions:
        quotas = STRATA_ORIENTATION_QUOTAS[stratum_name]
        for (target_a, target_b), quota in quotas.items():
            if quota == 0:
                continue

            available_a = pats_a_by_parity[a_split][target_a]
            available_b = pats_b_by_parity[b_split][target_b]

            if not available_a or not available_b:
                raise RuntimeError(
                    f"Quota infeasibility: stratum {stratum_name} orientation ({target_a}, {target_b}) "
                    f"has empty pattern pool (A: {len(available_a)}, B: {len(available_b)})."
                )

            collected_for_cell = 0
            # Generate all active pairs
            active_pairs = [(pa, pb) for pa in available_a for pb in available_b]
            rng.shuffle(active_pairs)

            # Distractor combinations
            max_attempts = 100000
            attempts = 0
            while collected_for_cell < quota and attempts < max_attempts:
                attempts += 1
                pa, pb = active_pairs[attempts % len(active_pairs)]
                distractors = [rng.choice([-1, 1]) for _ in range(len(DISTRACTOR_INDICES))]
                candidate_x = list(pa) + list(pb) + distractors
                cand_tuple = tuple(candidate_x)

                if cand_tuple in forbidden_xs or cand_tuple in selected_xs:
                    continue

                # Verify properties
                assert compute_parity(list(pa)) == target_a
                assert compute_parity(list(pb)) == target_b
                assert target_a != target_b

                selected_xs.add(cand_tuple)
                is_held_out_a = (pa in unseen_a)
                is_held_out_b = (pb in unseen_b)

                probe_examples.append({
                    "probe_id": probe_id_counter,
                    "features": candidate_x,
                    "target_class_a": target_a,
                    "target_class_b": target_b,
                    "stratum": stratum_name,
                    "active_pattern_a": list(pa),
                    "active_pattern_b": list(pb),
                    "is_held_out_a": is_held_out_a,
                    "is_held_out_b": is_held_out_b,
                })
                probe_id_counter += 1
                collected_for_cell += 1

            if collected_for_cell < quota:
                raise RuntimeError(
                    f"Probe quota cannot be satisfied without relaxing constraints for "
                    f"stratum '{stratum_name}' orientation ({target_a}, {target_b}): "
                    f"requested {quota}, collected {collected_for_cell}."
                )

    # Sort deterministically by probe_id
    probe_examples.sort(key=lambda e: e["probe_id"])

    # Compile dataset object
    dataset = {
        "metadata": {
            "study_id": "SSRL-02",
            "phase": "counterfactual_mask_diagnostic",
            "sample_size": len(probe_examples),
            "probe_seed": seed,
            "rule_a_indices": RULE_A_INDICES,
            "rule_b_indices": RULE_B_INDICES,
            "distractor_indices": DISTRACTOR_INDICES,
            "strata_quotas": STRATA_QUOTAS,
            "orientation_quotas": {"target_a_1_b_0": 64, "target_a_0_b_1": 64},
            "rule_a_dataset_sha256": dataset_a.get("metadata", {}).get("dataset_sha256"),
            "rule_b_dataset_sha256": dataset_b.get("metadata", {}).get("dataset_sha256"),
        },
        "probe_examples": probe_examples,
    }

    # Contract verification before return
    validate_counterfactual_probe_contract(dataset, dataset_a, dataset_b)
    return dataset


def validate_counterfactual_probe_contract(
    probe_dataset: Dict[str, Any],
    dataset_a: Dict[str, Any],
    dataset_b: Dict[str, Any],
) -> bool:
    """
    Validates that a counterfactual probe dataset satisfies all contract rules.
    Raises AssertionError/ValueError on any violation. Never allows relaxed quotas.
    """
    meta = probe_dataset.get("metadata", {})
    examples = probe_dataset.get("probe_examples", [])

    if len(examples) != PROBE_SAMPLE_SIZE:
        raise ValueError(f"Expected exactly {PROBE_SAMPLE_SIZE} probe examples, found {len(examples)}.")

    seen_a, unseen_a = extract_active_patterns(dataset_a, RULE_A_INDICES)
    seen_b, unseen_b = extract_active_patterns(dataset_b, RULE_B_INDICES)

    forbidden_xs = {
        tuple(e["features"]) for e in dataset_a["train_examples"]
    } | {
        tuple(e["features"]) for e in dataset_b["train_examples"]
    }

    seen_seen_count = 0
    seen_unseen_count = 0
    unseen_seen_count = 0
    unseen_unseen_count = 0

    count_a1_b0 = 0
    count_a0_b1 = 0

    unique_xs = set()

    for e in examples:
        x = e["features"]
        if len(x) != 16:
            raise ValueError(f"Probe example {e['probe_id']} has feature length {len(x)}, expected 16.")
        x_tup = tuple(x)
        if x_tup in unique_xs:
            raise ValueError(f"Duplicate feature vector detected in probe: probe_id={e['probe_id']}")
        unique_xs.add(x_tup)

        if x_tup in forbidden_xs:
            raise ValueError(f"Probe vector {e['probe_id']} intersects with training pools.")

        pa = [x[i] for i in RULE_A_INDICES]
        pb = [x[i] for i in RULE_B_INDICES]

        parity_a = compute_parity(pa)
        parity_b = compute_parity(pb)

        if parity_a == parity_b:
            raise ValueError(f"Probe vector {e['probe_id']} violates opposing parity constraint.")

        if e["target_class_a"] != parity_a or e["target_class_b"] != parity_b:
            raise ValueError(f"Probe vector {e['probe_id']} label fields do not match recomputed parity.")

        if (parity_a, parity_b) == (1, 0):
            count_a1_b0 += 1
        elif (parity_a, parity_b) == (0, 1):
            count_a0_b1 += 1

        is_a_unseen = (tuple(pa) in unseen_a)
        is_b_unseen = (tuple(pb) in unseen_b)

        if not is_a_unseen and not is_b_unseen:
            seen_seen_count += 1
            if e["stratum"] != "seen_seen":
                raise ValueError(f"Stratum label mismatch for {e['probe_id']}: expected seen_seen")
        elif not is_a_unseen and is_b_unseen:
            seen_unseen_count += 1
            if e["stratum"] != "seen_unseen":
                raise ValueError(f"Stratum label mismatch for {e['probe_id']}: expected seen_unseen")
        elif is_a_unseen and not is_b_unseen:
            unseen_seen_count += 1
            if e["stratum"] != "unseen_seen":
                raise ValueError(f"Stratum label mismatch for {e['probe_id']}: expected unseen_seen")
        else:
            unseen_unseen_count += 1

    if seen_seen_count != STRATA_QUOTAS["seen_seen"]:
        raise ValueError(f"Stratum 'seen_seen' count {seen_seen_count} != expected {STRATA_QUOTAS['seen_seen']}.")
    if seen_unseen_count != STRATA_QUOTAS["seen_unseen"]:
        raise ValueError(f"Stratum 'seen_unseen' count {seen_unseen_count} != expected {STRATA_QUOTAS['seen_unseen']}.")
    if unseen_seen_count != STRATA_QUOTAS["unseen_seen"]:
        raise ValueError(f"Stratum 'unseen_seen' count {unseen_seen_count} != expected {STRATA_QUOTAS['unseen_seen']}.")
    if unseen_unseen_count != STRATA_QUOTAS["unseen_unseen"]:
        raise ValueError(f"Stratum 'unseen_unseen' count {unseen_unseen_count} must be strictly 0.")

    if count_a1_b0 != 64 or count_a0_b1 != 64:
        raise ValueError(f"Orientation balance failure: (1,0)={count_a1_b0}, (0,1)={count_a0_b1}, expected (64, 64).")

    return True


def evaluate_counterfactual_probe(
    model: nn.Module,
    probe_examples: List[Dict[str, Any]],
    device: torch.device,
) -> Dict[str, Any]:
    """
    Evaluates model on probe pairs under both masks.
    Partitions outcomes into 4 descriptive categories.
    Saves per-pair targets, predictions, and logits.
    
    Warning: Output switching alone does NOT imply successful instruction following.
    Joint correctness on opposing pairs provides stronger evidence, but remains
    limited to the evaluated sample. Do not infer mask ignorance or a specific routing
    mechanism solely from argmax classification outcomes.
    """
    mask_a = [1 if i in RULE_A_INDICES else 0 for i in range(16)]
    mask_b = [1 if i in RULE_B_INDICES else 0 for i in range(16)]

    model.eval()

    pair_records: List[Dict[str, Any]] = []

    # Counters overall
    cat_counts_overall = {
        "both_correct": 0,
        "both_incorrect_with_switched_predictions": 0,
        "unchanged_prediction_matching_a_target": 0,
        "unchanged_prediction_matching_b_target": 0,
    }

    # Per-stratum counters
    cat_counts_by_stratum: Dict[str, Dict[str, int]] = {
        "seen_seen": {k: 0 for k in cat_counts_overall},
        "seen_unseen": {k: 0 for k in cat_counts_overall},
        "unseen_seen": {k: 0 for k in cat_counts_overall},
    }

    correct_mask_a_overall = 0
    correct_mask_b_overall = 0

    correct_mask_a_by_stratum = {"seen_seen": 0, "seen_unseen": 0, "unseen_seen": 0}
    correct_mask_b_by_stratum = {"seen_seen": 0, "seen_unseen": 0, "unseen_seen": 0}
    total_by_stratum = {"seen_seen": 0, "seen_unseen": 0, "unseen_seen": 0}

    with torch.no_grad():
        for ex in probe_examples:
            x = ex["features"]
            y_a = ex["target_class_a"]
            y_b = ex["target_class_b"]
            stratum = ex["stratum"]
            total_by_stratum[stratum] += 1

            # Input with Mask A
            inp_a = torch.tensor(x + mask_a, dtype=torch.float32, device=device).unsqueeze(0)
            logits_a = model(inp_a)[0]
            pred_a = int(torch.argmax(logits_a).item())

            # Input with Mask B
            inp_b = torch.tensor(x + mask_b, dtype=torch.float32, device=device).unsqueeze(0)
            logits_b = model(inp_b)[0]
            pred_b = int(torch.argmax(logits_b).item())

            correct_a = (pred_a == y_a)
            correct_b = (pred_b == y_b)

            if correct_a:
                correct_mask_a_overall += 1
                correct_mask_a_by_stratum[stratum] += 1
            if correct_b:
                correct_mask_b_overall += 1
                correct_mask_b_by_stratum[stratum] += 1

            # Categorize into 4 mutually exclusive categories
            is_switched = (pred_a != pred_b)
            if correct_a and correct_b:
                cat = "both_correct"
            elif (not correct_a) and (not correct_b) and is_switched:
                cat = "both_incorrect_with_switched_predictions"
            elif pred_a == pred_b and pred_a == y_a:
                cat = "unchanged_prediction_matching_a_target"
            elif pred_a == pred_b and pred_a == y_b:
                cat = "unchanged_prediction_matching_b_target"
            else:
                raise RuntimeError(
                    f"Category partition defect: pred_a={pred_a}, pred_b={pred_b}, y_a={y_a}, y_b={y_b}"
                )

            cat_counts_overall[cat] += 1
            cat_counts_by_stratum[stratum][cat] += 1

            pair_records.append({
                "probe_id": ex["probe_id"],
                "features": x,
                "stratum": stratum,
                "target_class_a": y_a,
                "target_class_b": y_b,
                "predicted_class_a": pred_a,
                "predicted_class_b": pred_b,
                "logits_a": [float(logits_a[0].item()), float(logits_a[1].item())],
                "logits_b": [float(logits_b[0].item()), float(logits_b[1].item())],
                "outcome_category": cat,
                "is_jointly_correct": (cat == "both_correct"),
                "is_switched": is_switched,
            })

    total_pairs = len(probe_examples)
    assert sum(cat_counts_overall.values()) == total_pairs, "Outcome counts do not sum to total pairs."

    def compile_metrics_dict(tot: int, ca: int, cb: int, cats: Dict[str, int]) -> Dict[str, Any]:
        if tot == 0:
            return {}
        switched = cats["both_correct"] + cats["both_incorrect_with_switched_predictions"]
        joint = cats["both_correct"]
        return {
            "total_pairs": tot,
            "accuracy_mask_a": {"numerator": ca, "denominator": tot, "pct": ca / tot * 100.0},
            "accuracy_mask_b": {"numerator": cb, "denominator": tot, "pct": cb / tot * 100.0},
            "switch_rate": {"numerator": switched, "denominator": tot, "pct": switched / tot * 100.0},
            "joint_correctness_rate": {"numerator": joint, "denominator": tot, "pct": joint / tot * 100.0},
            "category_counts": {k: {"numerator": v, "denominator": tot, "pct": v / tot * 100.0} for k, v in cats.items()},
        }

    overall_metrics = compile_metrics_dict(
        total_pairs,
        correct_mask_a_overall,
        correct_mask_b_overall,
        cat_counts_overall,
    )

    strata_metrics = {}
    for st in ["seen_seen", "seen_unseen", "unseen_seen"]:
        st_tot = total_by_stratum[st]
        strata_metrics[st] = compile_metrics_dict(
            st_tot,
            correct_mask_a_by_stratum[st],
            correct_mask_b_by_stratum[st],
            cat_counts_by_stratum[st],
        )

    return {
        "overall_metrics": overall_metrics,
        "strata_metrics": strata_metrics,
        "pair_records": pair_records,
    }
