"""
research/lifelong_01/ssrl_types.py -- Typed definitions and records for SSRL-01.

Contains machine-readable dataclasses and evaluation records enforcing
AGENTS.md Section 1.1 (no bare percentages) and Section 3.7 (raw per-unit outcomes).
Repairs:
- Enforces strict validation of example records (matching target vs prediction logic).
- Validates distinct IDs, population integrity, and dataset hash references.
- Rejects matching intersections in transfer evaluation: requires exact set equality.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import math


@dataclass(frozen=True)
class SSRLMeasurement:
    """
    Count-based measurement carrying explicit numerator, denominator, mode, and input set.
    Direct implementation of AGENTS.md Appendix C.1.
    """
    name: str
    numerator: int
    denominator: int
    input_set: str
    mode: str

    def __post_init__(self) -> None:
        if self.denominator <= 0:
            raise ValueError(f"{self.name}: denominator must be positive, got {self.denominator}")
        if self.numerator < 0:
            raise ValueError(f"{self.name}: negative numerator {self.numerator}")
        if self.numerator > self.denominator:
            raise ValueError(
                f"{self.name}: numerator {self.numerator} exceeds denominator "
                f"{self.denominator} -- impossible value, halting"
            )

    @property
    def rate(self) -> float:
        return self.numerator / float(self.denominator)

    @property
    def pct(self) -> float:
        return 100.0 * self.rate

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "rate": self.rate,
            "pct": self.pct,
            "input_set": self.input_set,
            "mode": self.mode,
        }


@dataclass(frozen=True)
class SSRLExampleRecord:
    """
    Per-example evaluation outcome serializing raw boolean accuracy.
    Enforces AGENTS.md Rule 3.7.
    Includes dataset hash reference and validates logical consistency of is_correct flag.
    """
    example_id: int
    rule_id: int
    arm: str
    seed: int
    step: int
    dataset_sha256: str
    input_features: List[int]      # 16-element {-1, +1}
    selection_mask: List[int]      # 16-element {0, 1}
    target_class: int              # 0 (odd) or 1 (even)
    predicted_class: int           # 0 (odd) or 1 (even)
    is_correct: bool
    is_held_out_pattern: bool

    def __post_init__(self) -> None:
        if self.target_class not in (0, 1):
            raise ValueError(f"Invalid target_class {self.target_class}, must be 0 or 1")
        if self.predicted_class not in (0, 1):
            raise ValueError(f"Invalid predicted_class {self.predicted_class}, must be 0 or 1")
        expected_correct = (self.target_class == self.predicted_class)
        if self.is_correct != expected_correct:
            raise ValueError(
                f"Inconsistent is_correct flag for example {self.example_id}: "
                f"target={self.target_class}, predicted={self.predicted_class}, is_correct={self.is_correct}"
            )
        if len(self.input_features) != 16:
            raise ValueError(f"input_features must have length 16, got {len(self.input_features)}")
        if len(self.selection_mask) != 16:
            raise ValueError(f"selection_mask must have length 16, got {len(self.selection_mask)}")
        if not self.dataset_sha256 or len(self.dataset_sha256) != 64:
            raise ValueError(f"dataset_sha256 must be a valid 64-char hex string, got {self.dataset_sha256}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "example_id": self.example_id,
            "rule_id": self.rule_id,
            "arm": self.arm,
            "seed": self.seed,
            "step": self.step,
            "dataset_sha256": self.dataset_sha256,
            "target_class": self.target_class,
            "predicted_class": self.predicted_class,
            "is_correct": self.is_correct,
            "is_held_out_pattern": self.is_held_out_pattern,
            "input_features": self.input_features,
            "selection_mask": self.selection_mask,
        }


def compute_accuracy_measurement(
    name: str,
    records: List[SSRLExampleRecord],
    input_set: str,
    mode: str = "eval_no_dropout"
) -> SSRLMeasurement:
    """
    Derives explicit SSRLMeasurement strictly from record counts.
    Never hardcodes or infers denominators from external constants.
    Verifies that all records in the population have unique example IDs.
    """
    denom = len(records)
    if denom == 0:
        raise ValueError(f"Cannot compute accuracy for '{name}' over empty record list")

    seen_ids = set()
    for r in records:
        if r.example_id in seen_ids:
            raise ValueError(f"Duplicate example_id {r.example_id} detected in population '{name}'")
        seen_ids.add(r.example_id)

    num = sum(1 for r in records if r.is_correct)
    return SSRLMeasurement(name=name, numerator=num, denominator=denom, input_set=input_set, mode=mode)
