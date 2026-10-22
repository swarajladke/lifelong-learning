"""
research/lifelong_01/ssrl_generator.py -- Pilot data generator, loader, and explicit contract verifier for SSRL-01.

Provides:
- Standalone pilot dataset generator for Task 1, Seed 101.
- Deterministic sampling without replacement of distractor vectors per active pattern.
- Strict disjointness checks:
  1. Valid rule coordinates (0 <= c < 16, distinct, exactly k=4).
  2. Exact pattern lengths and valid {-1, +1} values.
  3. Strict disjointness: full 16-bit input vectors in eval_seen and eval_unseen are
     completely disjoint from training pool vectors (no full-vector leakage).
  4. Unique example IDs.
  5. Content-addressed verification loader computing canonical SHA-256 before model load.
  6. Explicit semantic contract verification (Item 2 of Targeted Safety Patch):
     Validates metadata, seed, rule coordinates, exact populations, parity labels,
     pattern distributions, and disjointness independently of self-reported metadata.
     Prevents accidental overwrite when serializing datasets.

Guarded by main: this module does NOT execute generation upon import.
"""

from __future__ import annotations
import itertools
import json
import random
import hashlib
import os
from typing import List, Tuple, Dict, Any, Set


class PilotDatasetContractError(Exception):
    """Raised when dataset content violates the explicit pilot contract."""
    pass


def compute_parity(bits: List[int]) -> int:
    """
    Computes parity of a list of {-1, +1} bits.
    Returns 1 if product is +1 (even number of -1s), else 0 if product is -1 (odd number of -1s).
    """
    if len(bits) == 0:
        raise ValueError("Cannot compute parity of empty bit list")
    prod = 1
    for b in bits:
        if b not in (-1, 1):
            raise ValueError(f"Invalid bit value {b}, expected -1 or +1")
        prod *= b
    return 1 if prod == 1 else 0


def rule_to_mask(rule_indices: Tuple[int, ...] | List[int], d: int = 16) -> List[int]:
    """
    Converts active coordinate tuple into a d-dimensional binary mask {0, 1}.
    Validates that coordinates are within [0, d-1] and distinct.
    """
    if len(rule_indices) == 0:
        raise ValueError("Rule coordinates cannot be empty")
    if len(set(rule_indices)) != len(rule_indices):
        raise ValueError(f"Duplicate coordinates found in rule: {rule_indices}")
    for idx in rule_indices:
        if not (0 <= idx < d):
            raise ValueError(f"Coordinate {idx} out of range [0, {d-1}]")

    mask = [0] * d
    for idx in rule_indices:
        mask[idx] = 1
    return mask


def partition_active_patterns(seed: int = 42) -> Tuple[List[Tuple[int, ...]], List[Tuple[int, ...]]]:
    """
    Enumerates all 2^4 = 16 active patterns over {-1, +1}^4 and partitions them
    deterministically into 12 training patterns and 4 held-out evaluation patterns.
    """
    all_patterns = list(itertools.product([-1, 1], repeat=4))
    rng = random.Random(seed)
    shuffled = list(all_patterns)
    rng.shuffle(shuffled)
    train_patterns = sorted(shuffled[:12])
    held_out_patterns = sorted(shuffled[12:])
    assert len(train_patterns) == 12
    assert len(held_out_patterns) == 4
    return train_patterns, held_out_patterns


def validate_pilot_dataset_contract(data: Dict[str, Any], approved_config: Dict[str, Any]) -> None:
    """
    Exhaustively validates dataset against the explicit approved pilot contract.
    Does NOT trust self-reported metadata: audits actual examples directly.
    Raises PilotDatasetContractError on any violation.
    """
    if not isinstance(data, dict):
        raise PilotDatasetContractError("Dataset root must be a dictionary")

    for key in ("metadata", "train_examples", "eval_seen_examples", "eval_unseen_examples"):
        if key not in data:
            raise PilotDatasetContractError(f"Missing required top-level key: '{key}'")

    meta = data["metadata"]

    # 1. Study and Phase
    expected_study = approved_config.get("study_id", "SSRL-01")
    if meta.get("study_id") != expected_study:
        raise PilotDatasetContractError(f"study_id mismatch: metadata={meta.get('study_id')}, config={expected_study}")
    expected_phase = approved_config.get("phase", "learnability_pilot")
    if meta.get("phase") != expected_phase:
        raise PilotDatasetContractError(f"phase mismatch: expected '{expected_phase}', got '{meta.get('phase')}'")

    # 2. Seed and Dimensions
    expected_seed = approved_config.get("seed", 101)
    if meta.get("seed") != expected_seed:
        raise PilotDatasetContractError(f"Seed mismatch: expected {expected_seed}, got {meta.get('seed')}")

    d = approved_config.get("input_dim_features", 16)
    k = approved_config.get("active_k", 4)
    if meta.get("d") != d or meta.get("k") != k:
        raise PilotDatasetContractError(f"Dimensions mismatch: expected d={d}, k={k}; got d={meta.get('d')}, k={meta.get('k')}")

    # 3. Rule ID and Coordinates
    expected_rule_id = approved_config.get("rule_id", 1)
    if meta.get("rule_id") != expected_rule_id:
        raise PilotDatasetContractError(f"Metadata rule_id mismatch: expected {expected_rule_id}, got {meta.get('rule_id')}")

    expected_rule = list(approved_config.get("rule_indices", [0, 1, 2, 3]))
    if meta.get("rule_indices") != expected_rule:
        raise PilotDatasetContractError(f"Rule coordinates mismatch: expected {expected_rule}, got {meta.get('rule_indices')}")

    expected_mask = rule_to_mask(expected_rule, d=d)

    # 4. Population counts
    train_exs = data["train_examples"]
    eval_seen_exs = data["eval_seen_examples"]
    eval_unseen_exs = data["eval_unseen_examples"]

    exp_train_n = approved_config.get("train_pool_size", 384)
    exp_seen_n = approved_config.get("eval_seen_size", 192)
    exp_unseen_n = approved_config.get("eval_unseen_size", 64)

    if len(train_exs) != exp_train_n:
        raise PilotDatasetContractError(f"Train pool size mismatch: expected {exp_train_n}, got {len(train_exs)}")
    if len(eval_seen_exs) != exp_seen_n:
        raise PilotDatasetContractError(f"Eval seen size mismatch: expected {exp_seen_n}, got {len(eval_seen_exs)}")
    if len(eval_unseen_exs) != exp_unseen_n:
        raise PilotDatasetContractError(f"Eval unseen size mismatch: expected {exp_unseen_n}, got {len(eval_unseen_exs)}")

    # 5. Exhaustive Example Auditing
    all_ids: Set[int] = set()
    train_vectors: Set[Tuple[int, ...]] = set()
    eval_seen_vectors: Set[Tuple[int, ...]] = set()
    eval_unseen_vectors: Set[Tuple[int, ...]] = set()

    partition_seed = approved_config.get("partition_seed", 42)
    expected_train_pats, expected_unseen_pats = partition_active_patterns(seed=partition_seed)
    expected_train_pats_set = set(expected_train_pats)
    expected_unseen_pats_set = set(expected_unseen_pats)

    def audit_population(examples: List[Dict[str, Any]], expected_held_out: bool, vector_set: Set[Tuple[int, ...]], pop_name: str) -> Dict[Tuple[int, ...], int]:
        pattern_counts: Dict[Tuple[int, ...], int] = {}
        for idx, ex in enumerate(examples):
            ex_id = ex.get("example_id")
            if ex_id is None or not isinstance(ex_id, int):
                raise PilotDatasetContractError(f"[{pop_name}] Example {idx} missing integer example_id")
            if ex_id in all_ids:
                raise PilotDatasetContractError(f"[{pop_name}] Duplicate example_id {ex_id} found")
            all_ids.add(ex_id)

            if ex.get("rule_id") != expected_rule_id:
                raise PilotDatasetContractError(f"[{pop_name}] Example {ex_id} rule_id mismatch: expected {expected_rule_id}, got {ex.get('rule_id')}")
            if ex.get("rule_indices") != expected_rule:
                raise PilotDatasetContractError(f"[{pop_name}] Example {ex_id} has wrong rule_indices: {ex.get('rule_indices')}")
            if ex.get("mask") != expected_mask:
                raise PilotDatasetContractError(f"[{pop_name}] Example {ex_id} has wrong mask: {ex.get('mask')}")

            feat = ex.get("features")
            if not isinstance(feat, list) or len(feat) != d:
                raise PilotDatasetContractError(f"[{pop_name}] Example {ex_id} has invalid features length: {len(feat) if isinstance(feat, list) else type(feat)}")
            for b in feat:
                if b not in (-1, 1):
                    raise PilotDatasetContractError(f"[{pop_name}] Example {ex_id} contains non-binary feature value: {b}")

            vec_tuple = tuple(feat)
            if vec_tuple in vector_set:
                raise PilotDatasetContractError(f"[{pop_name}] Duplicate full input vector detected within population for example {ex_id}")
            vector_set.add(vec_tuple)

            # Active pattern verification
            active_tuple = tuple(feat[p] for p in expected_rule)
            if expected_held_out:
                if active_tuple not in expected_unseen_pats_set:
                    raise PilotDatasetContractError(f"[{pop_name}] Example {ex_id} active pattern {active_tuple} is not in expected unseen active pattern partition")
            else:
                if active_tuple not in expected_train_pats_set:
                    raise PilotDatasetContractError(f"[{pop_name}] Example {ex_id} active pattern {active_tuple} is not in expected train active pattern partition")

            pattern_counts[active_tuple] = pattern_counts.get(active_tuple, 0) + 1

            # Parity check
            expected_parity = compute_parity(list(active_tuple))
            if ex.get("target_class") != expected_parity:
                raise PilotDatasetContractError(f"[{pop_name}] Example {ex_id} parity label mismatch: expected {expected_parity}, got {ex.get('target_class')}")
            if ex.get("is_held_out_pattern") != expected_held_out:
                raise PilotDatasetContractError(f"[{pop_name}] Example {ex_id} is_held_out_pattern mismatch: expected {expected_held_out}, got {ex.get('is_held_out_pattern')}")

        return pattern_counts

    train_pat_counts = audit_population(train_exs, False, train_vectors, "TRAIN")
    eval_seen_pat_counts = audit_population(eval_seen_exs, False, eval_seen_vectors, "EVAL_SEEN")
    eval_unseen_pat_counts = audit_population(eval_unseen_exs, True, eval_unseen_vectors, "EVAL_UNSEEN")

    # 6. Active Pattern Coverage and Equal Allocation
    if set(train_pat_counts.keys()) != expected_train_pats_set:
        raise PilotDatasetContractError(f"Train population active patterns do not cover all 12 expected patterns")
    if set(eval_seen_pat_counts.keys()) != expected_train_pats_set:
        raise PilotDatasetContractError(f"Eval seen active patterns do not cover all 12 expected patterns")
    if set(eval_unseen_pat_counts.keys()) != expected_unseen_pats_set:
        raise PilotDatasetContractError(f"Eval unseen active patterns do not cover all 4 expected patterns")

    exp_train_draws = exp_train_n // 12
    for pat, count in train_pat_counts.items():
        if count != exp_train_draws:
            raise PilotDatasetContractError(f"Train pattern {pat} count mismatch: expected {exp_train_draws}, got {count}")

    exp_seen_draws = exp_seen_n // 12
    for pat, count in eval_seen_pat_counts.items():
        if count != exp_seen_draws:
            raise PilotDatasetContractError(f"Eval seen pattern {pat} count mismatch: expected {exp_seen_draws}, got {count}")

    exp_unseen_draws = exp_unseen_n // 4
    for pat, count in eval_unseen_pat_counts.items():
        if count != exp_unseen_draws:
            raise PilotDatasetContractError(f"Eval unseen pattern {pat} count mismatch: expected {exp_unseen_draws}, got {count}")

    # 7. Cross-Population Disjointness
    if not train_vectors.isdisjoint(eval_seen_vectors):
        raise PilotDatasetContractError("Leakage detected: train vectors overlap with eval_seen vectors!")
    if not train_vectors.isdisjoint(eval_unseen_vectors):
        raise PilotDatasetContractError("Leakage detected: train vectors overlap with eval_unseen vectors!")
    if not eval_seen_vectors.isdisjoint(eval_unseen_vectors):
        raise PilotDatasetContractError("Leakage detected: eval_seen vectors overlap with eval_unseen vectors!")


def generate_pilot_dataset_dict(
    rule_indices: Tuple[int, ...] = (0, 1, 2, 3),
    rule_id: int = 1,
    seed: int = 101,
    d: int = 16,
    k: int = 4,
    train_draws_per_pattern: int = 32,   # 12 * 32 = 384
    eval_seen_draws_per_pattern: int = 16, # 12 * 16 = 192
    eval_unseen_draws_per_pattern: int = 16, # 4 * 16 = 64
    partition_seed: int = 42,
    study_id: str = "SSRL-01",
    phase: str = "learnability_pilot"
) -> Dict[str, Any]:
    """
    Builds the complete pilot dataset dictionary in memory.
    Enforces deterministic sampling without replacement for distractor patterns,
    ensuring that full 16-bit vectors are strictly disjoint between train and eval_seen.
    """
    if len(rule_indices) != k:
        raise ValueError(f"Rule must have exactly {k} active coordinates, got {len(rule_indices)}")
    mask = rule_to_mask(rule_indices, d=d)
    inactive_indices = [i for i in range(d) if i not in rule_indices]
    num_inactive = len(inactive_indices) # 12
    total_distractor_configs = 1 << num_inactive # 4,096

    needed_per_seen = train_draws_per_pattern + eval_seen_draws_per_pattern
    if needed_per_seen > total_distractor_configs:
        raise ValueError(f"Requested {needed_per_seen} distractor draws exceeds domain of {total_distractor_configs}")

    train_patterns, held_out_patterns = partition_active_patterns(seed=partition_seed)
    rng = random.Random(seed)
    all_distractors = list(itertools.product([-1, 1], repeat=num_inactive))

    train_examples = []
    eval_seen_examples = []
    eval_unseen_examples = []

    global_ex_id = 0
    seen_full_vectors_train: Set[Tuple[int, ...]] = set()
    seen_full_vectors_eval_seen: Set[Tuple[int, ...]] = set()
    seen_full_vectors_eval_unseen: Set[Tuple[int, ...]] = set()

    for pat in train_patterns:
        sampled_distractors = rng.sample(all_distractors, needed_per_seen)
        train_distractors = sampled_distractors[:train_draws_per_pattern]
        eval_distractors = sampled_distractors[train_draws_per_pattern:]
        active_bits = list(pat)
        target_class = compute_parity(active_bits)

        for dist in train_distractors:
            feat = [0] * d
            for pos, val in zip(rule_indices, active_bits):
                feat[pos] = val
            for pos, val in zip(inactive_indices, dist):
                feat[pos] = val
            t_feat = tuple(feat)
            assert t_feat not in seen_full_vectors_train
            seen_full_vectors_train.add(t_feat)

            train_examples.append({
                "example_id": global_ex_id,
                "rule_id": rule_id,
                "rule_indices": list(rule_indices),
                "features": feat,
                "mask": mask,
                "target_class": target_class,
                "is_held_out_pattern": False,
            })
            global_ex_id += 1

        for dist in eval_distractors:
            feat = [0] * d
            for pos, val in zip(rule_indices, active_bits):
                feat[pos] = val
            for pos, val in zip(inactive_indices, dist):
                feat[pos] = val
            t_feat = tuple(feat)
            assert t_feat not in seen_full_vectors_eval_seen
            seen_full_vectors_eval_seen.add(t_feat)

            eval_seen_examples.append({
                "example_id": global_ex_id,
                "rule_id": rule_id,
                "rule_indices": list(rule_indices),
                "features": feat,
                "mask": mask,
                "target_class": target_class,
                "is_held_out_pattern": False,
            })
            global_ex_id += 1

    for pat in held_out_patterns:
        sampled_distractors = rng.sample(all_distractors, eval_unseen_draws_per_pattern)
        active_bits = list(pat)
        target_class = compute_parity(active_bits)

        for dist in sampled_distractors:
            feat = [0] * d
            for pos, val in zip(rule_indices, active_bits):
                feat[pos] = val
            for pos, val in zip(inactive_indices, dist):
                feat[pos] = val
            t_feat = tuple(feat)
            assert t_feat not in seen_full_vectors_eval_unseen
            seen_full_vectors_eval_unseen.add(t_feat)

            eval_unseen_examples.append({
                "example_id": global_ex_id,
                "rule_id": rule_id,
                "rule_indices": list(rule_indices),
                "features": feat,
                "mask": mask,
                "target_class": target_class,
                "is_held_out_pattern": True,
            })
            global_ex_id += 1

    dataset_dict = {
        "metadata": {
            "study_id": study_id,
            "phase": phase,
            "rule_id": rule_id,
            "rule_indices": list(rule_indices),
            "seed": seed,
            "partition_seed": partition_seed,
            "d": d,
            "k": k,
            "num_active_patterns_seen": 12,
            "num_active_patterns_unseen": 4,
            "train_pool_size": len(train_examples),
            "eval_seen_size": len(eval_seen_examples),
            "eval_unseen_size": len(eval_unseen_examples),
        },
        "train_examples": train_examples,
        "eval_seen_examples": eval_seen_examples,
        "eval_unseen_examples": eval_unseen_examples,
    }

    return dataset_dict


def serialize_dataset_canonical_json(dataset_dict: Dict[str, Any], filepath: str, refuse_overwrite: bool = True) -> str:
    """
    Serializes dataset to JSON with sorted keys and UTF-8 encoding.
    Refuses overwrite if refuse_overwrite=True and file exists.
    Returns computed SHA-256 hash.
    """
    filepath = os.path.abspath(filepath)
    if refuse_overwrite and os.path.exists(filepath):
        raise FileExistsError(f"Dataset file already exists and refuse_overwrite=True: {filepath}")

    raw_json = json.dumps(dataset_dict, indent=2, sort_keys=True).encode("utf-8")
    sha256 = hashlib.sha256(raw_json).hexdigest()
    with open(filepath, "wb") as f:
        f.write(raw_json)
    return sha256


def load_and_verify_dataset(filepath: str, expected_sha256: str, approved_config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Loads dataset JSON, validates exact SHA-256 byte hash, and enforces the full
    explicit pilot contract before returning the data.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    with open(filepath, "rb") as f:
        raw_bytes = f.read()

    actual_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Dataset SHA-256 hash verification failed for {filepath}!\n"
            f"Expected: {expected_sha256}\n"
            f"Actual:   {actual_sha256}\n"
            f"Execution halted before model construction."
        )

    data = json.loads(raw_bytes.decode("utf-8"))

    if approved_config is not None:
        validate_pilot_dataset_contract(data, approved_config)

    return data
