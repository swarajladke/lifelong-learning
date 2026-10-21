# SSRL-02: Bounded Two-Rule Sequential Parity Learning Protocol (SSRL_02_SPEC.md)

**Assignment:** LIFELONG-01 / SSRL Continuation  
**Study Identifier:** SSRL-02  
**Classification:** Bounded Exploratory Sequential Feasibility Pilot (Diagnostic Only)  
**Workspace:** `C:\Users\Vicky\Desktop\lifelong-learning`  
**Repository:** `swarajladke/lifelong-learning` (Branch: `main`)  
**Base Verified Commit:** `3747300a20ab28724cb5d958377a10036cd9f41a`  
**Date:** 2026-10-21  
**Status:** **SPECIFICATION & MODEL-FREE SCAFFOLDING ONLY** — Zero neural compute executed.

---

## 1. Scientific Purpose & Rationale

The SSRL-01 single-rule pilot established baseline learnability ($G\text{-LC}$ Pass: 192/192 = 100.00% seen accuracy on Rule 1). However, a single-task pilot cannot measure continual learning dynamics, catastrophic interference, or forward transfer.

**SSRL-02** is designed as a bounded two-rule experiment ($A \to B$) serving strictly as an **exploratory interference and plasticity diagnostic**:
1. **Catastrophic Interference / Forgetting:** Measures the empirical change in Rule A performance following gradient adaptation on Rule B.
2. **Forward Transfer:** Measures whether weight adaptation to Rule A alters sample efficiency or accuracy on Rule B relative to an identically initialized fresh learner receiving the identical minibatch sequence.
3. **Subspace Routing Diagnostic:** Evaluates whether explicit binary selection mask conditioning ($m \in \{0, 1\}^{16}$) allows an MLP to preserve performance on disjoint coordinate subspaces under standard gradient descent, without introducing specialized anti-forgetting objectives.

> **Scientific Boundary:** This disjoint-rule pair is an exploratory interference/plasticity diagnostic. It does **not** constitute proof of reusable compositional structure, general rule induction, or catastrophic forgetting mitigation.

---

## 2. Rule Pair Specification & Registry Status

### 2.1 Rule Definitions
- **Rule A (Task 1, `rule_id: 1`):**
  - Active coordinates: $[0, 1, 2, 3]$ ($k=4$, $D=16$).
  - Binary selection mask: $m_A = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]$.
  - Distractor coordinates: $[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]$ (12 positions).
- **Rule B (Task 2, `rule_id: 2`):**
  - Active coordinates: $[4, 5, 6, 7]$ ($k=4$, $D=16$).
  - Binary selection mask: $m_B = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]$.
  - Distractor coordinates: $[0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15]$ (12 positions).

### 2.2 Coordinate Overlap & Rationale
- **Coordinate Overlap:** Exactly **0** shared coordinates ($\{0, 1, 2, 3\} \cap \{4, 5, 6, 7\} = \emptyset$).
- **Informativeness Rationale:**
  1. *Subspace Role Inversion:* In Rule A, dimensions $[4..7]$ are irrelevant distractors. In Rule B, $[4..7]$ carry the active parity signal, while $[0..3]$ become distractors. If Rule A training adjusts weights connecting to $[4..7]$, adapting to Rule B requires parameter updates that may interfere with previously established pathways.
  2. *Zero Feature Collision:* Because the coordinates are strictly disjoint, no input pattern on Rule B can be mistaken for a pattern on Rule A. Any observed degradation on Rule A is an empirical measure of interference from subsequent parameter updates, not label contradiction.
- **Registry Status & Audit:** In `PLAN.md` §3.3, 30 future confirmation rules are established as a reserved policy category, but their specific coordinate tuples have not yet been enumerated or hash-pinned in code. Rule A ($[0, 1, 2, 3]$) and Rule B ($[4, 5, 6, 7]$) are drawn from the 30 development rules ($T_1, T_2$). No confirmation isolation is claimed until the full 60-rule split is formally pinned in a dedicated registry artifact.

---

## 3. Compared Experimental Arms

Both `seq_ab` and `fresh_b` start from the **identical shared initial model state** $\theta_0 \sim \text{Init}(\text{seed}=101)$ to ensure that initial weight configurations do not confound adaptation differences.

| Arm ID | Protocol Name | Initial Weights | Task A Training | Optimizer at Boundary | Task B Adaptation | Lifetime Steps |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Arm 1** | `seq_ab` | $\theta_0$ | 2,000 steps on Rule A | **Reconstructed Fresh** ($m=0, v=0, \text{step}=0$) | 500 steps on Rule B | 2,500 steps (40,000 samples) |
| **Arm 2** | `fresh_b` | $\theta_0$ | None (0 steps) | Freshly Initialized ($m=0, v=0, \text{step}=0$) | 500 steps on Rule B (**Identical Batches**) | 500 steps (8,000 samples) |
| **Arm 3** | `freeze_after_a` | $\theta_0$ | 2,000 steps on Rule A | Reuses post-A checkpoint $\theta_A$ | 0 steps on Rule B (**Frozen Weights**) | 2,000 steps (32,000 samples) |

### 3.1 Optimizer State & Checkpoint Boundary Handling
1. **Fresh Optimizer at Boundary:** At the transition from Task A to Task B in `seq_ab`, a completely fresh Adam optimizer instance is constructed with learning rate $1\times 10^{-3}$, resetting both moments ($m=0, v=0$) and optimizer step counter (`step=0`). This isolates weight-history effects from optimizer-state carry-over.
2. **Reusing Post-A Checkpoint for `freeze_after_a`:** The `freeze_after_a` arm does not retrain Task A. It reuses the exact saved post-A checkpoint $\theta_A$ from Arm 1, asserting zero parameter updates throughout Task B.
3. **Identical Rule B Minibatches:** Both `seq_ab` and `fresh_b` receive the exact identical sequence of Rule B minibatches $\{B_1, B_2, \dots, B_{500}\}$, drawn from an independent minibatch sampler stream.

---

## 4. Independent RNG Streams & Frozen Populations

To prevent cross-stream contamination, random number generation is partitioned into four independent streams:
1. `model_init_seed = 101`: Controls parameter initialization $\theta_0$.
2. `data_distractor_seed_a = 42`: Controls distractor sampling without replacement for Rule A.
3. `data_distractor_seed_b = 43`: Controls distractor sampling without replacement for Rule B.
4. `minibatch_sampler_seed_b = 2027`: Controls the identical minibatch sequence for Rule B.

### Frozen Pre-Generated Evaluation Populations
Evaluation populations are generated and byte-hash pinned to disk **prior to any training**:
- **Rule A Eval Seen ($N = 192$):** 12 observed active patterns $\times$ 16 distinct distractor draws.
- **Rule A Eval Unseen ($N = 64$):** 4 held-out active patterns $\times$ 16 distinct distractor draws.
- **Rule B Eval Seen ($N = 192$):** 12 observed active patterns $\times$ 16 distinct distractor draws.
- **Rule B Eval Unseen ($N = 64$):** 4 held-out active patterns $\times$ 16 distinct distractor draws.
- Total frozen evaluation records across both rules: $256 + 256 = 512$ examples.

---

## 5. Evaluation Checkpoints & Decomposed Measurement

Evaluations occur at four checkpoints:
- **$t_0$:** Untrained Initialization ($\theta_0$).
- **$t_1$:** Post-Rule A (after 2,000 steps on Rule A).
- **$t_{\text{inter}}$:** During Rule B adaptation at step 100 and step 250.
- **$t_2$:** Post-Rule B (after 500 steps on Rule B — planned final transfer endpoint).

### 5.1 Non-Interference of Evaluation
Evaluation runs under `model.eval()`, `torch.no_grad()`, with zero optimizer calls. The training minibatch sampler maintains its independent RNG stream and is not stepped during evaluation. Model weights, optimizer state, and future batch sequences are guaranteed unchanged by evaluation passes.

### 5.2 Decomposed Metrics & Sign Conventions
1. **Rule A Interference / Forgetting:**
   - $\text{Forgetting}_{\text{seen}}(A) = \text{Acc}_{\text{seen}}(A \text{ at } t_1) - \text{Acc}_{\text{seen}}(A \text{ at } t_2)$
   - $\text{Forgetting}_{\text{unseen}}(A) = \text{Acc}_{\text{unseen}}(A \text{ at } t_1) - \text{Acc}_{\text{unseen}}(A \text{ at } t_2)$
   - *Sign Convention & Units:* Positive indicates accuracy drop (forgetting); zero indicates complete retention; negative indicates performance recovery (backward transfer). Units: percentage points.
2. **Rule B Forward Transfer Delta:**
   - $\Delta_{\text{transfer, seen}}(B, S) = \text{Acc}_{\text{seen, seq}}(B \text{ at } S) - \text{Acc}_{\text{seen, fresh}}(B \text{ at } S)$
   - Evaluated and recorded at $S \in \{0, 100, 250, 500\}$, with $S=500$ as the planned final transfer endpoint.
   - *Sign Convention & Units:* Positive indicates forward transfer; zero indicates matched performance; negative indicates interference / negative transfer. Units: percentage points.
3. **Rule B Zero-Shot Transfer:**
   - $\text{Acc}_{\text{seen}}(B \text{ at } t_1)$ (evaluated immediately after Task A training, before any Rule B gradient step).

### 5.3 Reporting Safeguards
- Comparisons enforce identical evaluation populations: matching `rule_id`, matching `dataset_sha256`, matching `target_class`, and matching per-position `example_id`.
- Seen-pattern fitting and held-out-pattern generalization are reported in separate columns with explicit denominators ($n/192$ and $n/64$).
- Seen and held-out accuracies are never combined into a single pooled headline.
- High seen-pattern accuracy alone is not cited as proof of general parity learning or robust mask routing.

---

## 6. Smallest Bounded Compute Proposal

- **Design:** Exploratory single-seed calibration (Seed 101).
- **Architecture:** Normalization-free ReLU MLP ($32 \to 64 \to 64 \to 2$, exactly 6,402 parameters).
- **Total Step Budget:**
  - Arm 1 (`seq_ab`): 2,000 (A) + 500 (B) = 2,500 steps.
  - Arm 2 (`fresh_b`): 500 (B) = 500 steps.
  - Arm 3 (`freeze_after_a`): 0 steps.
  - Total: 3,000 optimizer steps.
- **Projected Compute:** Analytical, unmeasured projection: $\approx 3\text{--}5\text{ seconds}$ on Kaggle CPU.
- **Monotonic Wall-Clock Ceiling:** 180.0 seconds.
- **Preregistration / Gate Status:** Rule A learnability gate $G\text{-LC}$ ($\ge 90.0\%$ seen active patterns at $t_1$) remains an entry requirement. No unapproved arbitrary thresholds are imposed on forgetting or transfer.
