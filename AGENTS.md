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
3. **Heavy Compute Separation:** Large sweeps, unbounded multi-seed campaigns, and heavy GPU workloads belong on remote accelerator environments (e.g., Kaggle GPU) under clean, pinned checkouts. Local execution is governed strictly by Section 4.

---

## 4. Policy: Bounded Local Execution

*Clarification:* Local execution is not categorically forbidden for this project. Small workloads may be executed locally when explicitly authorized by the researcher.

### 4.1 Permitted Local Work
The agent may execute locally:
- Model-free unit and regression tests (CPU-only, lightweight);
- Dataset validation, manifest generation, and result-analysis scripts;
- Small CPU-only dataset generation or neural experiments that the researcher has explicitly authorized.

### 4.2 Authorization Boundaries
- Permission to run unit or regression tests does **not** authorize training.
- Permission for a single experiment does **not** authorize sweeps, additional seeds, hyperparameter tuning, automated retries, or follow-up experiments.
- Prior to executing any local neural compute, the agent must identify:
  1. The approved configuration or specification;
  2. The exact source commit revision;
  3. The pinned input datasets and canonical SHA-256 hashes;
  4. Explicit step and sample exposure budgets;
  5. The wall-clock timeout ceiling.
  If any of these parameters are missing, unpinned, or ambiguous, the agent must stop and ask before proceeding.

### 4.3 Resource and Timeout Limits
- Local execution must remain strictly CPU-only unless separate GPU authorization is explicitly granted.
- Local runs must operate under the approved experiment's explicit resource limits and monotonic timeout ceilings (halting execution if ceilings are exceeded).
- If no explicit limits are approved, the agent must propose them and obtain confirmation before running.
- The agent must never autonomously decide that an arbitrary workload is "small" and run it without authorization.

### 4.4 Environment and Filesystem Protection
- No package installations, upgrades, uninstalls, administrator actions, or global configuration alterations may be executed without separate approval.
- Every run must reserve a unique execution directory (`evidence/runs/<run_id>/`) and refuse execution if the destination already exists. Prior evidence artifacts must never be overwritten.
- Local operations within this repository must never modify or pollute the original `Neural-Networks` worktrees.

### 4.5 Invariant Scientific Safeguards
All scientific safeguards in Section 1 apply with equal force to local runs:
- Pinned datasets must be hash-verified before compute;
- Thresholds and reference baselines are immutable;
- Pre-flight tests must execute and pass prior to model construction;
- Full provenance, raw per-unit outcome records, and failure diagnostics must be serialized;
- Gate failures or timeout breaches immediately halt downstream execution with nonzero exit codes;
- No silent threshold changes, replacement seeds, or automated retries.

### 4.6 Accurate Provenance and Execution Environment Labeling
- Every results artifact and summary report must explicitly record its execution environment (e.g., `local_workstation` vs. `kaggle_remote`).
- Artifacts must log the actual host hardware, OS, platform flags, framework versions, commit SHA, and git working-tree dirty status.
- Local execution is not inherently invalid, and matching predictions across environments does not by itself establish tensor equality.

### 4.7 Historical Clarification
- The researcher explicitly authorized the local SSRL-02 exact-protocol replay `ssrl_02_replay_eff8aa50`.
- Execution of `ssrl_02_replay_eff8aa50` on the local workstation is classified as an authorized bounded run, not an authorization violation.
- Its authentic local provenance, environment records, captured checkpoints, and probe results are preserved in place.
