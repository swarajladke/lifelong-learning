# LIFELONG-01 / SSRL-01 Context and Status Handoff

*Document Identifier:* `research/lifelong_01/context.md`  
*Repository:* https://github.com/swarajladke/lifelong-learning  
*Workspace:* `C:\Users\Vicky\Desktop\lifelong-learning` (Branch: `main`)  
*As of:* 2026-10-21 11:20 IST  
*Status:* Descriptive orientation and status snapshot for incoming agents and reviewers. This document conveys no new authority, does not supersede governing policy (`AGENTS.md`), and does not modify authorized task boundaries.

---

## 1. Project Overview and Researcher’s Goal

The broader aspiration of this research program is to develop an artificial learner that acquires knowledge, rules, and transferable skills continually from initialization through continuing experience, potentially without a separate massive pretraining phase.

### Scientific Framing and Boundaries
1. **Aspiration vs. Realization:**
   A system developing from initialization through continuing experience—without reliance on a separate massive pretraining phase—is an unproven long-term goal, not a current empirical achievement or merely an effort to avoid repeated pretraining runs.
2. **Mechanisms of Continual Adaptation:**
   Learning fundamentally requires parameter or internal state adaptation along with allocated computational resources. The research does not assume that standard gradient descent over static weights is the only viable mechanism (explorations may involve modular allocation, state conditioning, or alternative plasticity rules), but any proposed mechanism must be tested empirically under pinned conditions.
3. **Biological Inspiration vs. Empirical Proof:**
   Biological nervous systems demonstrate that lifelong learning and forward transfer are physically possible. However, biological analogies are conceptual inspirations, not evidence of artificial general intelligence (AGI) or proof that a given artificial architecture works.
4. **Bounded Scope of the Immediate Pilot:**
   The current single-rule pilot (`SSRL-01`, Task 1) measures only whether an instruction-conditioned multilayer perceptron can fit observed active parity configurations in the presence of novel distractor features, and whether it generalizes to held-out active configurations under a static mask. It does **not** evaluate dynamic mask switching, rule composition, transfer across rules, catastrophic forgetting mitigation, or AGI.

---

## 2. Separate Research Workstreams

The repository contains distinct research workstreams that operate under different histories, protocols, and governance boundaries. They must not be conflated:

### A. S0-16 / S0 Track (Factual Editing Study)
- **Scope & Architecture:**
  Sequential factual-editing study on GPT-2, with localized parameter updates targeting Layer 1 `c_proj`.
  *(Note: SmolLM2 embedding files on disk belong to separate historical investigations. Trailing-window evaluation was formally retired in Directive S0-8 and is not assigned to S0-16 without a specific supporting source).*
- **Evaluated Factual-Editing Arms:**
  - **R / F3:** Sequential margin-targeted null-space reference.
  - **C2:** Collision-based slot consolidation with joint target optimization over colliding facts, without the multi-token template basis. (Replay-assisted).
  - **C3:** Consolidation plus multi-token template protection. (Replay-assisted).
- **Actual Status (Uncompleted / Unconfirmed):**
  - S0-16 is **not** completed, closed, or fully confirmed.
  - *Last user-reported state:* Batch 1 (Seeds 101–104) completed. Batch 2 halted at Condition C3, Seed 105, where immediate post-edit acquisition reached 178/200 (89.0%), failing Gate E1 ($\ge 180/200 = 90.0\%$). Seeds 106–108 were not reached according to the report.
  - Exact terminal-artifact availability remains to be reconciled between the producing runner and saved outputs on disk.
  - The failed seed must not be excluded, Gate E1 must not be relaxed, and no authorization to resume S0-16 is implied.
- **Boundary:** S0-16 is strictly isolated and must remain untouched by LIFELONG-01 work.

### B. Historical AGNIS Track (`agnis_*.py`)
- **Scope:** Prior exploratory continual learning scripts in the repository root.
- **Status & Restriction:** Governed by explicit quarantine under repository policy. Legacy AGNIS modules must not be imported, revived, or executed without separate, formal authorization and compliance audits.

### C. Representation and Class-Incremental Learning
- **Scope:** Continual acquisition of representations across shifting task distributions (Track B/C).
- **Status & Controls:** Distinct from localized factual editing. Claims require explicit distinction between recorded artifacts, withdrawn claims, and unresolved items. Continual learning evaluations remain governed by permanent standing controls (`FREEZE-AFTER-BASE`).

### D. SSRL-01 Track (Sequential Symbolic Rule Learning)
- **Scope:** Current exploratory study evaluating instruction-conditioned parity learning.
- **Current Authorization:** Model-free specification, documentation, and unit-test scaffolding only. Production dataset generation, neural model instantiation, local training, and remote execution remain unauthorized.

---

## 3. Current SSRL-01 Specification

`SSRL-01` explores whether explicit instruction conditioning (via binary selection masks) allows an MLP to isolate relevant bit positions and learn parity rules sequentially.

### Input Representation and Target Rule
- **Joint Input Vector:**
  The network receives $u = [x; m] \in \mathbb{R}^{32}$, where $x \in \{-1, +1\}^{16}$ is a feature vector and $m \in \{0, 1\}^{16}$ is a binary selection mask.
- **Target Parity Calculation:**
  The mask $m$ activates exactly $k=4$ positions out of $D=16$ (Rule 1 coordinates: $[0, 1, 2, 3]$). The label $y \in \{0, 1\}$ is determined by the multiplicative parity of the selected bits:
  $$P = \prod_{i:\, m_i = 1} x_i \in \{-1, +1\}$$
  $$y = \begin{cases} 1 & \text{if } P = +1 \quad (\text{even count of } -1\text{ bits}) \\ 0 & \text{if } P = -1 \quad (\text{odd count of } -1\text{ bits}) \end{cases}$$
  The 12 unselected feature positions ($m_i = 0$) are distractors and do not affect the target.

### Model Architecture
- **Architecture:** Normalization-free Multi-Layer Perceptron ($32 \to 64 \to 64 \to 2$).
- **Layers:**
  - Input: 32 dimensions (16 features + 16 mask bits).
  - Layer 1: Linear($32 \to 64$) + Bias + ReLU ($32 \times 64 + 64 = 2,112$ parameters).
  - Layer 2: Linear($64 \to 64$) + Bias + ReLU ($64 \times 64 + 64 = 4,160$ parameters).
  - Output: Linear($64 \to 2$) + Bias ($64 \times 2 + 2 = 130$ parameters).
- **Total Parameters:** Exactly 6,402 parameters (asserted in source).

### Bounded Single-Rule Learnability Pilot (Task 1, Seed 101)
- **Arm:** `fresh_control` (random initialization, no prior tasks).
- **Rule Coordinates:** $[0, 1, 2, 3]$.
- **Data Partitions & Generation Semantics:**
  - Active space: $2^4 = 16$ patterns over $\{-1, +1\}^4$.
  - Partitioned deterministically (Seed 42) into 12 observed active patterns and 4 held-out active patterns.
  - Distractor generation: For each active pattern, distractor assignments across the 12 inactive positions are sampled **without replacement** from the $2^{12} = 4,096$ possible configurations.
- **Populations:**
  - **Train Pool:** 12 observed patterns $\times$ 32 draws = 384 unique examples.
  - **Eval Seen:** 12 observed patterns $\times$ 16 draws = 192 unique examples.
  - **Eval Unseen (Held-out):** 4 held-out patterns $\times$ 16 draws = 64 unique examples.
  - **Disjointness:** Strict vector disjointness is verified across all three splits ($\mathcal{X}_{\text{train}} \cap \mathcal{X}_{\text{eval\_seen}} = \emptyset$, $\mathcal{X}_{\text{train}} \cap \mathcal{X}_{\text{eval\_unseen}} = \emptyset$, $\mathcal{X}_{\text{eval\_seen}} \cap \mathcal{X}_{\text{eval\_unseen}} = \emptyset$).
- **Optimization & Sampling Semantics:**
  - Optimizer: Adam ($\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}, \text{weight decay}=0.0$).
  - Learning Rate: $1 \times 10^{-3}$.
  - Batch Size: 16.
  - Step Budget: 2,000 steps.
  - **Minibatch Semantics:** Each step uses `random.sample(train_pool, batch_size)`, selecting 16 distinct examples uniformly without replacement within that minibatch. The pool is available again at subsequent steps, so examples can recur across steps.
  - **Presentations vs. Unique Items:** Total example presentations equal $2,000 \times 16 = 32,000$ presentations, drawn from the 384 unique training examples.
- **Resource Projections:** Wall-clock and memory requirements are analytical estimates, not empirical measurements.
- **Proposed Gate $G\text{-LC}$ (Learnability Control):**
  - Population: Eval Seen ($N=192$).
  - Proposed Threshold: $\ge 90.0\%$ (requires $\ge 173/192$ correct).
  - **Approval Status:** `"PROPOSED_PENDING_APPROVAL"`. Execution halts prior to model instantiation if approval is missing or pending.

---

## 4. Planned Exploratory Design (Unexecuted Blueprint)

Beyond the single-rule pilot, the exploratory multi-rule design outlined in `PLAN.md` remains entirely planned and unexecuted:
- **Development vs. Confirmation Rule Splits:**
  - **30 Development Rules:** Comprising 10 sequential history rules, 5 transfer probe rules, and 15 compositional evaluation rules testing held-out rule generalization across shared and orthogonal coordinate subsets.
  - **30 Future Confirmation Rules:** A separate split reserved for future confirmation, subject to formal definition in `PLAN.md`.
- **Transfer Probing:** Five independently cloned transfer probes branching from the sequential model checkpoint, evaluated against matched `fresh_control` baselines with freshly re-initialized probe optimizer states.
- **Generalization Metric:** Evaluates held-out rule generalization on unobserved coordinate masks, distinct from demonstrating generalized compositional reasoning.
- **Replication & Aggregation:** Six random model seeds, with probe scores averaged within each model seed before across-seed analysis.
- **Permutation Significance Testing:** Exact paired sign-flip permutation test across $N=6$ seeds ($2^6 = 64$ configurations, resolution floor $2/64 = 0.03125$).
- **Statistical Qualifier:** Achieving a permutation resolution floor below 0.05 is an arithmetic property of $N=6$; it is not a guarantee of empirical statistical power or validity.

---

## 5. Actual Implementation and Verification State

### Source Files and Current Content Hashes

All code files reside under `research/lifelong_01/`. Their exact SHA-256 hashes as of this snapshot are:

| File Path (Repository-Relative) | SHA-256 Digest | Status & Responsibilities |
| :--- | :--- | :--- |
| `run_ssrl_learnability_pilot.py` | `c5f0c0cca96a3b4fb775adfde954ed36d8c29aa09fee06b8367a38b89ba3df6a` | Exclusive run-directory reservation, pre-construction approval check, pre/post deadlines, effective provenance capture, and incremental evaluation persistence. |
| `ssrl_evaluator.py` | `de45c43bdfeaafad2c8cf53f97db83c01356da517c4470ae4182e7af1893d8dd` | Gate $G\text{-LC}$ evaluation (forbidding pass under unapproved status), paired sign-flip test, and atomic JSON writing. |
| `ssrl_generator.py` | `575562994cd00d07ee2245d8db1fddce46aa23b171740af5447d7ba01f7d563f` | Parity calculation, deterministic pattern partitioning, and explicit pilot contract verification (validating metadata and per-example `rule_id`). |
| `ssrl_model.py` | `662c40f61ddd3b77bed2f244837153605f96f936d4bedf493c7156e85aefd3d9` | Normalization-free MLP ($32 \to 64 \to 64 \to 2$, 6,402 parameters). Model instantiation guarded against import. |
| `ssrl_runner.py` | `b576697a3eb3bab735d3535b6b7d4252d16d60bf733e3294fcd8f3c7d11e2bd4` | Checkpoint cloning helpers for future probe evaluations. |
| `ssrl_types.py` | `d377ada116cab6904c093c3fe6b4acf254380ba2ce5361a959418e63f7e8bdac` | `SSRLMeasurement` (Appendix C.1 numerator/denominator enforcement) and `SSRLExampleRecord`. |
| `test_ssrl_model_free.py` | `b3aba73c2f59a634985598a4030de07eb93cb076eb79206d64422ddf97aa3e84` | 20 model-free unit and regression tests. |

### Evidence Categorization & Review History
1. **Static Inspection by Authoring Coding Agent:**
   - The latest safety patches were authored in code and confirmed via static file reading.
   - Guards include: explicit metadata and example `rule_id` verification, gate rejection of proposed thresholds, exclusive directory reservation via `os.makedirs(..., exist_ok=False)`, specification approval gating, pre- and post-operation deadline checks, and immediate per-example diagnostic persistence.
2. **External Review History:**
   - Multiple earlier ZIP versions underwent independent static review.
   - The latest independently inspected archive contained nine files, was 27,642 bytes, and had SHA-256:
     `28817ff2dee3d110821525ccef173b230612d6a23a54249aad32651b9f4fc47a`.
   - The later safety patches (including exclusive directory creation, approval verification, and incremental evaluation tracking) remain unreviewed externally.
   - Independent inspection of the earlier archive does **not** certify or validate the current source hashes.
3. **Remote Model-Free Test Suite Execution (Researcher-Reported):**
   - **Package Verified:** `ssrl_model_free_review_20261019.zip` (SHA-256 `f8c18d9be82eb6c637537458628e9656f6bb6dc18882a8fd0feae1df2f16f085`).
   - **Environment:** Remote Kaggle Notebook container (Python 3.13).
   - **Evidence Location:** `/kaggle/working/ssrl_val_Jrrn06vM` (preserved in `research/lifelong_01/test_run_evidence_20261019.log`).
   - **Execution Result:** All 11 manifest files verified bit-identical. All 20 unit and regression tests ran and passed in 3.784s. Actual exit code 0.
   - **Scope Qualifier:** This confirms that the 20 model-free tests passed for this specific packaged revision (`f8c18d9b`). It does **not** certify the entire instrument as defect-free, does not constitute neural validation, and does not alter the pending approval status of Gate $G\text{-LC}$.
4. **Pinned Production Dataset (Generated & Verified):**
   - **Dataset File:** `research/lifelong_01/data/ssrl_pilot_t1_s101.json`
   - **Canonical SHA-256 Byte Hash:** `00a21efad1dc07b826198d99b0b702aedddad57b6226f99301afbf16327fd0b7`
   - **Manifest File:** `research/lifelong_01/data/ssrl_pilot_t1_s101_manifest.json`
   - **Generation Run:** Remote Kaggle execution (`/kaggle/working/ssrl_data_gen_QvsnluuI`, exit code 0).
   - **Verification Status:** In-memory pilot contract passed, canonical serialization verified, and reload through `load_and_verify_dataset` passed.
5. **Pending Gate Approval & Neural Execution:**
   - Gate $G\text{-LC}$ approval remains `"PROPOSED_PENDING_APPROVAL"`.
   - Neural model instantiation and training remain unauthorized.

---

## 6. Workspace, Git History, and Research Protection

### Worktree Topology and Commit Lineage
- **Shared Base Commit:** `e7ecf29864aa585971b7148e35f294f06e2f5dac` (HEAD of `main` in the original repository).
- **Faulty Descendant:** `02e6ce68b43994c46f55075ab4a3612a9015ff6d` (initial branch commit for `research/lifelong-01`).
- **Corrective Descendant:** `2a851dd45068d85f042e3e15870d084781572921` (current HEAD of `research/lifelong-01`).

### Incident and Restoration Details
During the creation of the `research/lifelong-01` branch in the already-existing isolated lifelong worktree, an index and checkout misconfiguration resulted in commit `02e6ce6` recording the unintended deletion of 427 repository paths while adding the initial design documents.
Corrective commit `2a851dd` was executed directly within this same lifelong worktree as a forward fix, cleanly restoring all 427 deleted paths from the shared base `e7ecf29` (distinguished from its immediate parent, `02e6ce6`) while preserving the new design documents. No changes were merged or pushed to `main`.

### Current Observed Worktree Status
- **Primary Worktree (`main`):**
  - Branch: `main` at `e7ecf29864aa585971b7148e35f294f06e2f5dac`.
  - Untracked artifacts: Contains historical S0 embedding and output files (`smollm2_embeddings_*.pt`, `w1_results.json`). Their byte preservation across prior sessions has not been independently re-verified via SHA-256 hashes and must not be assumed.
- **Recovery Worktree (`research/lifelong-01`):**
  - Branch: `research/lifelong-01` at `2a851dd45068d85f042e3e15870d084781572921`.
  - Modified tracked files (`M`):
    - `research/lifelong_01/AUDIT.md`
    - `research/lifelong_01/DECISIONS_REQUIRED.md`
    - `research/lifelong_01/PLAN.md`
  - Untracked files (`??`):
    - `research/lifelong_01/context.md`
    - `research/lifelong_01/data/` (manifest and dataset location)
    - `research/lifelong_01/run_ssrl_learnability_pilot.py`
    - `research/lifelong_01/ssrl_evaluator.py`
    - `research/lifelong_01/ssrl_generator.py`
    - `research/lifelong_01/ssrl_model.py`
    - `research/lifelong_01/ssrl_model_free_review_20261019.zip`
    - `research/lifelong_01/ssrl_runner.py`
    - `research/lifelong_01/ssrl_source_review.zip`
    - `research/lifelong_01/ssrl_types.py`
    - `research/lifelong_01/test_run_evidence_20261019.log`
    - `research/lifelong_01/test_ssrl_model_free.py`

> **Environment Note (Local Paths):**  
> On the author's local workstation, the primary worktree resides at `C:\Users\Vicky\Desktop\Neural Networks` and the lifelong worktree at `C:\Users\Vicky\Desktop\Neural-Networks-lifelong-01`. Incoming agents must navigate exclusively using repository-relative paths.

---

## 7. Governance and Operational Rules

All research activities must adhere strictly to `AGENTS.md`. Key operational requirements include:

1. **Reading Order for Project Documentation:**
   1. `AGENTS.md`: Mandatory research engineering protocol, metrics contracts, and governance rules.
   2. `research/lifelong_01/context.md`: Current repository orientation and verification snapshot (this document).
   3. `research/lifelong_01/AUDIT.md`: Historical audit of legacy code, S0-16, and Track A baselines.
   4. `research/lifelong_01/PLAN.md`: Exploratory study plan, task formalization, and evaluation specifications.
   5. `research/lifelong_01/DECISIONS_REQUIRED.md`: Open architectural decisions, threshold justifications, and policy trade-offs.
2. **Worktree & Branch Precondition:**
   Before making any edits, verify that the active worktree is `Neural-Networks-lifelong-01` and the active branch is `research/lifelong-01`. Stop immediately if the working tree is on `main` or if unexpected modifications are detected.
3. **Threshold and Gate Integrity (AGENTS.md Section 2):**
   - The learnability gate threshold ($\ge 90.0\%$) is currently `"PROPOSED_PENDING_APPROVAL"`.
   - Editing an approval status string in source code does **not** constitute human authorization. A formal directive is required to change gate status.
4. **Local Execution Restrictions:**
   - The coding agent reported an explicit constraint (`RULE[user_global]`) forbidding local execution of training, testing, or sweep scripts via `run_command`.
   - If user instructions conflict with standing global rules or governing protocols, an agent must **stop and state the conflict in prose**. It must not resolve the conflict silently or bypass safeguards.
5. **Non-Destructive Remote Execution Guidelines:**
   - When remote execution on Kaggle is authorized, execution commands must use a fresh, unique clone directory (e.g., cloning into a uniquely named target directory or clean path) checking out an exact reviewed commit SHA.
   - Do **not** use destructive commands like `rm -rf` on shared or existing directories.
   - Uncommitted local files are **not** present on the remote repository; code must be formally reviewed, committed, and pushed before remote execution can run.

---

## 8. Migration Verification and SSRL-02 Status

1. **Initial Migration Verification on Kaggle (2026-10-21) [Reported]:**
   - **Tested Commit:** `3747300a20ab28724cb5d958377a10036cd9f41a`
   - **Environment:** Remote Kaggle Notebook (CPU-only, no accelerator).
   - **Run Directory:** `/kaggle/working/ssrl_verify_tslkkbjb`
   - **Reported Outcome:** 20 / 20 unit and regression tests passed in 4.303s with exit code 0 (researcher-reported terminal stream).
   - **Status & Caveat:** Reported only. The raw archive (`/kaggle/working/ssrl_verify_tslkkbjb/ssrl_verification_evidence.zip`) has not yet been downloaded to the local workstation or inspected independently. Transcript preserved in `evidence/migration_verification/README.md`.
2. **SSRL-02 Expanded Test Suite Verification on Kaggle (2026-10-21) [Reported]:**
   - **Tested Commit:** `2484408a36aa29055b15a052bf80214abf988883`
   - **Environment:** Remote Kaggle Notebook (CPU-only, no accelerator).
   - **Run Directory:** `/kaggle/working/ssrl_verify_0grt8_2x`
   - **Reported Outcome:** 26 / 26 unit and regression tests passed in 7.545s with exit code 0 (researcher-reported terminal stream).
   - **Status & Caveat:** Reported only. The raw archive (`/kaggle/working/ssrl_verify_0grt8_2x/ssrl_verification_evidence.zip`) has not yet been downloaded to the local workstation or inspected independently. This pass certifies commit `2484408a` and does not certify subsequently authored runner code.
3. **Preserved Pilot Results Bundle:**
   - Single-rule pilot results bundle preserved in `evidence/pilot/ssrl_pilot_t1_s101_results_bundle.zip` (SHA-256 `9780d34cc30f7a0639ce981671bbc87d22b5c751e62c9e2178f6a7a315579f1b`).
   - Verified empirical findings: Gate $G\text{-LC}$ PASSED (Seen active patterns: 192/192 = 100.00%), held-out pattern generalization: 28/64 = 43.75%.
4. **Rule B Dataset Generation on Kaggle & Local Verification (2026-10-22):**
   - **Canonical SHA-256 Byte Hash:** `27e813e73bf66f51e1bff07f7d2f19945e830afc6d5876b0cb4d11136cda4cd4` (Verified on disk).
   - **Dataset File:** `research/lifelong_01/data/ssrl_pilot_t2_s101.json` (388,102 bytes).
   - **Manifest File:** `research/lifelong_01/data/ssrl_pilot_t2_s101_manifest.json` (435 bytes).
   - **Evidence Bundle Archive:** `evidence/data_generation/ssrl_t2_generated_bundle.zip` (SHA-256 `59eaabc149f58d79ff4f157b83b9a98e9baa1dddad64ff67808128ca6f06e80f`).
   - **Status:** **VERIFIED** on local workstation. All 640 examples verified: counts (384 train / 192 seen-eval / 64 held-out-eval), parity labels, masks, and pattern disjointness passed.
5. **SSRL-02 Runner Implementation & Scaffolding Verification on Kaggle (2026-10-22) [Reported]:**
   - **Tested Commit:** `e4aef5042950a4329b465407f710edac72e452f2`
   - **Environment:** Remote Kaggle Notebook (CPU-only, no accelerator).
   - **Run Directory:** `/kaggle/working/ssrl_verify_72280459`
   - **Evidence Archive:** `/kaggle/working/ssrl_verify_72280459/ssrl_verification_evidence.zip`
   - **Reported Outcome:** 31 / 31 unit and regression tests passed in 5.461s with exit code 0 (researcher-reported terminal stream).
   - **Status & Caveat:** Reported only. The raw archive has not yet been downloaded to the local workstation or inspected independently. Transcript preserved in `evidence/migration_verification/README.md`.
   - **Scaffolding Scope:** Covers `seq_ab`, `fresh_b`, `freeze_after_a` arms, boundary Adam rebuild ($m=0, v=0, \text{step}=0$), shared minibatch schedule, and step 0/100/250/500 evaluations (`research/lifelong_01/run_ssrl_two_rule.py`, 401 lines, under 600-line limit).
   - **Neural Compute Status:** Zero neural compute executed. Specification approval guard verified. Ready for execution once specification is formally approved.

---

## 9. Collaboration Preferences

1. **Actionable Prompts:** Provide self-contained, copy-pastable prompts for the coding agent.
2. **Context and Contribution:** Accompany each prompt with a concise explanation of its purpose, failure modes guarded against, and exact contribution to the lifelong learning research goals.
3. **Conservative Verification:** State strictly what has been executed and evidenced; never claim a test passed or an artifact exists without verifiable execution logs.

---

## 10. Update Checklist for Incoming Agents

- [x] Verify standalone repository and workspace (`C:\Users\Vicky\Desktop\lifelong-learning`, `main`).
- [x] Record migration verification outcome (20/20 unit tests passed on Kaggle, exit code 0, commit `3747300a`).
- [x] Record SSRL-02 expanded test suite outcome (31/31 tests passed on Kaggle, exit code 0, commit `e4aef50`).
- [x] Retrieve and verify Rule B dataset on disk (SHA-256 `27e813e7...`, 640 examples).
- [x] Preserve Rule B data generation bundle (`evidence/data_generation/ssrl_t2_generated_bundle.zip`).
- [x] Preserve pilot execution results bundle (`evidence/pilot/ssrl_pilot_t1_s101_results_bundle.zip`).
- [ ] Retrieve local copy of remote test verification archives.
- [x] Complete model-free scaffolding for SSRL-02 two-rule sequential study (31 tests).
- [ ] Submit SSRL-02 specification for formal human review before any compute dispatch.

