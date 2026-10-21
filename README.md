# Lifelong Learning — Sequential Symbolic Rule Learning (SSRL)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

An empirical research program exploring lifelong and continual learning architectures from initialization through ongoing experience. The current active study is **SSRL-01 (Sequential Symbolic Rule Learning)**, evaluating whether instruction-conditioned networks can acquire symbolic rules, isolate relevant feature subspaces under distractor noise, and retain acquired capabilities.

---

## 1. Project Lineage & Provenance

This standalone repository was migrated on 2026-10-21 from the upstream repository:
- **Upstream Source:** `https://github.com/swarajladke/Neural-Networks`
- **Source Commit Snapshot:** `9515e005bc0fbbc8776de8e18421909ea17d26bd` (`research/lifelong-01`)
- **Base Ancestor:** `e7ecf29864aa585971b7148e35f294f06e2f5dac`

For full provenance, exact byte-level file SHA-256 digests, and licensing details, see [`PROVENANCE.md`](PROVENANCE.md).

---

## 2. Research Protocol & Governance

All research in this repository operates under strict research engineering protocols codified in [`AGENTS.md`](AGENTS.md):
- **Prime Directive:** Recomputability from pinned inputs, named artifact provenance, and immutable preregistered gates.
- **Metrics Contract:** Every count-based metric returns an explicit `(numerator, denominator)` with asserted population denominators.
- **Pre-Flight Testing:** Fast model-free unit tests must execute and pass before neural compute or model weights are loaded.
- **Standing Control Arm:** Continual learning evaluations must include permanent standing controls (`MLP-FREEZE-AFTER-BASE`).

---

## 3. Project Structure

```text
lifelong-learning/
├── .gitignore                                 # Project gitignore rules
├── README.md                                  # Project overview and instructions
├── PROVENANCE.md                              # Snapshot verification and lineage ledger
├── AGENTS.md                                  # Research engineering protocol and safeguards
├── evidence/
│   └── pilot/                                 # Preserved Task 1 Pilot results and execution logs
│       ├── ssrl_pilot_t1_s101_results_bundle.zip # Complete Kaggle results archive
│       └── results.json                       # Verified pilot outcome metrics and gates
└── research/
    └── lifelong_01/                           # Active SSRL-01 study package
        ├── AUDIT.md                           # Empirical feasibility audit
        ├── context.md                         # Project orientation and status handoff
        ├── DECISIONS_REQUIRED.md              # Policy decisions and gate approval ledger
        ├── PLAN.md                            # Exploratory study design and task sequence
        ├── test_run_evidence_20261019.log     # Model-free test suite execution log (20/20 pass)
        ├── run_ssrl_learnability_pilot.py     # Master pilot runner (Task 1, Seed 101)
        ├── ssrl_evaluator.py                  # Evaluation engine and Gate G-LC logic
        ├── ssrl_generator.py                  # Parity data generator and contract verifier
        ├── ssrl_model.py                      # Instruction MLP (6,402 parameters)
        ├── ssrl_runner.py                     # Checkpoint cloning for probe evaluation
        ├── ssrl_types.py                      # Core dataclasses and measurement types
        ├── test_ssrl_model_free.py            # 20 model-free unit and regression tests
        └── data/
            ├── ssrl_pilot_t1_s101.json        # Pinned pilot dataset (SHA-256: 00a21efad1dc07b8...)
            └── ssrl_pilot_t1_s101_manifest.json # Validation manifest with byte-hash proof
```

---

## 4. Current Status: SSRL-01 Learnability Pilot

The single-rule pilot (Task 1, Seed 101, Arm `fresh_control`) evaluates the base learnability of parity over $k=4$ active bits in $D=16$ dimensions:
- **Gate $G\text{-LC}$ (Seen Active Patterns):** 192 / 192 = 100.00% (**PASSED**, threshold $\ge 90.0\%$).
- **Held-Out Pattern Generalization:** 28 / 64 = 43.75% (descriptive baseline; chance floor 50.0%).
- **Pre-Flight Tests:** 20 / 20 unit and regression tests passed.
- **Evidence Location:** Preserved in `evidence/pilot/results.json`.

---

## 5. Kaggle Execution Workflow

To ensure reproducibility and environment isolation, neural training and multi-seed sweeps are executed remotely on Kaggle under a clean clone:

```bash
%%bash
# Clean clone of pinned repository
rm -rf lifelong-learning
git clone https://github.com/swarajladke/lifelong-learning.git
cd lifelong-learning

# Pre-flight model-free verification
python3 -m unittest research/lifelong_01/test_ssrl_model_free.py
```
