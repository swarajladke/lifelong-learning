# AGENTS.md — Research Engineering Protocol for Lifelong Learning

**Scope:** Every script, experiment, commit, and report in this repository that produces, transforms,
or cites a measured empirical quantity.

**Read this before authoring code, not after.** If a rule here conflicts with a task instruction,
stop and say so in prose. Do not resolve the conflict silently.

---

## 0. The Prime Directive

> **A run that cannot be recomputed from pinned inputs is not a result. A number that cannot
> name the artifact it came from is not a measurement. A gate that can be edited to pass is
> not a gate.**

**Mechanism over promise.** A rule an agent must remember is a rule that will be broken. A rule
encoded as a type constraint, an assertion, or an immutable gate cannot be bypassed without an explicit halt.

---

## 1. Universal Scientific Safeguards

These safeguards apply to all research conducted within this repository, across all present and future phases.

### 1.1 Metrics Contract
- **No Bare Numbers:** Every count-based metric must return an explicit `Measurement` object carrying numerator, denominator, input-set identifier, and execution mode. Functions returning raw floats or bare percentages are prohibited.
- **Population Alignment:** The denominator must equal the exact population the claim addresses.
- **Assert Denominators:** Denominators must be computed from record counts and asserted in code, never written as arbitrary literals.
- **Impossible Values Halt Execution:** Ratios exceeding 100%, negative counts, `numerator > denominator`, or zero denominators must immediately raise an exception and halt. Never print an impossible value and proceed.
- **Explicit Docstrings:** Every metric must document what it counts and what condition would yield zero.

### 1.2 Gate and Threshold Integrity
- **Immutability:** Thresholds, tolerances, and reference baselines are read from content-addressed or pinned configuration dictionaries.
- **No Same-Commit Modification:** Never edit a threshold or tolerance in the same commit that observes a value. If a gate fails, the permissible actions are: diagnose the cause, fix defects, or report the failure. Changing the threshold to achieve a pass is invalid.
- **The Word "Calibrate" is Banned:** The word "calibrate" is prohibited from commit messages touching any threshold, gate, or reference value.
- **Failing Gates are Useful Information:** A failing gate is an instrument functioning correctly. Halting on failure protects subsequent experimental integrity.
- **Failures Block Downstream Compute:** A failed gate aborts subsequent execution stages.

### 1.3 Provenance and Anti-Fabrication
- **Runtime Interpolation:** Every reported number must be interpolated at runtime from a verified results object. No measured numbers may be typed directly into source files, comments, or report tables.
- **Full Run Provenance:** Results artifacts must serialize: producing commit SHA, input dataset content hash, model parameter count/checksum, seeds, optimizer steps, and total samples seen.
- **Assertion of Work Done:** Assert that optimizer steps $> 0$ and samples seen $> 0$. A script that finishes without training steps or evaluation passes is defective regardless of exit code.
- **Per-Unit Outcome Vectors:** Experiments must serialize raw per-unit boolean outcome vectors (keyed by seed and sequence index) into the results JSON. Aggregated summaries alone are insufficient for reproducibility.
- **No Expected Results in Source:** Comments or print statements predicting numeric outcomes pre-commit the experiment and are forbidden in source code.

### 1.4 Determinism and Environment
- **Pin Seeds and Versions:** Model initialization seeds, data-sampling seeds, and framework configurations must be explicitly specified and reported.
- **Environment Fingerprint:** Print framework version, accelerator type, CUDA/cuDNN status, backend determinism flags, and platform metadata on every run.
- **Execution Mode Tracking:** Distinguish training mode from evaluation mode. Quantities measured under different execution modes are different estimands and must never be pooled.
- **Kaggle Execution Standard:** Remote Kaggle execution commands must always perform a clean, fresh clone into a unique directory checking out a pinned, reviewed commit SHA.

### 1.5 Data Pinning
- **Disk-Loaded and Hashed:** Experimental datasets must be loaded from disk and asserted against an exact SHA-256 byte hash before neural compute begins. Datasets must not be generated at training runtime.
- **Generator Verification:** Any generator utility must prove byte-for-byte or field-by-field agreement with the pinned dataset file.
- **Shape and Constraint Assertions:** Assert all vector dimensions, feature types, and label invariants upon loading.

### 1.6 Tests Before Compute
- **Pre-Flight Unit Tests:** Fast, model-free unit and regression tests must execute and pass (exit code 0) before any model weights or accelerators are loaded.
- **Behavioral Testing:** Test matching rules, parity definitions, mask applications, and boundary guards against hand-constructed fixtures.
- **Guard Testing:** Verify that guards correctly raise exceptions when presented with invalid contracts, corrupt hashes, or out-of-boundary inputs.

### 1.7 Reporting Contract
- **Stdout is Measurement; Prose is Changelog:** Output streams report fingerprints, test outcomes, and computed measurements. Commit messages and prose explain changes and file diffs.
- **Per-Seed Primary Reporting:** Always report individual per-seed values alongside summary statistics. Summaries must never conceal seed-level variance.
- **Control Arm Comparators:** Continual learning evaluations must be reported against permanent standing controls (`FREEZE-AFTER-BASE`).
- **Decomposed Gap Reporting:** Report retention gap closed and acquisition gap closed as separate metrics. Never collapse them into a single pooled score.

---

## 2. Study-Specific Design Choices (SSRL-01 Pilot Boundary)

The following parameters and structures are specific design choices for the **SSRL-01 single-rule pilot (Task 1, Seed 101)** and initial multi-rule blueprint. They are **not** universal constraints for all future lifelong learning architectures:

1. **Architecture ($32 \to 64 \to 64 \to 2$, 6,402 parameters):** Chosen for minimal parameter footprint and tractability on parity rules; future studies may explore modular, sparse, or transformer-based backbones.
2. **Replication Cardinality ($N=6$ seeds):** Adopted for exploratory paired sign-flip permutation tests (permutation floor $2/64 = 0.03125$); larger sequences or confirmation tracks may require alternative sample sizes.
3. **Gate $G\text{-LC}$ Threshold ($\ge 90.0\%$ seen accuracy):** A specific learnability floor for the Task 1 pilot on active pattern fitting; subsequent tasks or complex domains will define independent, preregistered criteria.
4. **Input Topology ($D=16, k=4$):** Specific to the synthetic parity problem; does not define limits on feature dimension or sparsity for future tracks.

---

## 3. Explicit Research Boundaries & Quarantine

1. **Quarantined AGNIS Modules:** Migration to this standalone repository conveys **zero authorization** to revive, import, or execute legacy quarantined AGNIS modules (`agnis_*.py`) or unapproved historical baselines.
2. **Factual Editing Isolation:** Factual editing methods (GPT-2 Layer 1 `c_proj` localized editing, S0-16) remain entirely within the upstream repository and have no standing here.
3. **Execution Separation:** Heavy training, sweeps, and multi-seed sequences must run on remote environments (e.g. Kaggle GPU) under clean checkouts. Local execution is restricted to authoring, documentation, git management, and model-free unit tests.
