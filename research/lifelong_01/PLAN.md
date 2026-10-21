# LIFELONG-01 — Exploratory Experimental Design (PLAN.md)

**Assignment:** LIFELONG-01 — Repository-Grounded Feasibility Audit and Experimental Design  
**Study Identifier:** SSRL-01 (Sequential Symbolic Rule Learning — Phase 1 Feasibility)  
**Study Classification:** **Exploratory Feasibility Study** (Not a Confirmatory Trial)  
**Worktree:** `C:/Users/Vicky/Desktop/Neural-Networks-lifelong-01`  
**Branch:** `research/lifelong-01`  
**Pinned Base Commit:** `e7ecf29`  
**Date:** 2026-10-19 (Revised Post-Review)  
**Status:** **DESIGN AND IMPLEMENTATION SPECIFICATION ONLY** — Zero training execution, no dataset generation.  

> **Scientific Grounding:** All quantitative figures represent prospective design parameters,
> sample populations, and unmeasured compute projections. None are empirical measurements.
> Per AGENTS.md §4.2, no expected results are asserted as facts.

---

## 1. Scientific Objective and Scope

The long-term objective of this workstream is a neural system capable of lifelong development:
acquiring structured abilities across a sequential stream, retaining earlier abilities, transferring
shared structure to novel tasks, and preserving plasticity under explicit resource constraints.

**Scope of SSRL-01:**  
SSRL-01 is an **exploratory feasibility study**. It does not claim confirmation of an anti-forgetting
mechanism, nor does it establish AGI. It investigates:
1. Whether a simple, fixed-capacity neural learner can learn an instruction-conditioned parity selection rule.
2. Whether sequential training across multiple parity selection rules exhibits forward transfer
   (accelerated target-rule adaptation) compared to matched fresh-initialized baselines.
3. Whether zero-shot generalization occurs on novel held-out rule compositions.

If an ordinary learner fails basic acquisition or suffers catastrophic interference on this domain,
that diagnostic finding establishes the baseline constraint before any complex anti-forgetting or
modular architecture is introduced.

---

## 2. Mathematical Task Formulation: Instruction-Conditioned Selection

### 2.1 Identifiability and Input-Mask Conditioning

Each observation consists of a pair $(x, m)$:
- $x \in \{-1, +1\}^D$: A binary feature vector of dimension $D = 16$.
- $m \in \{0, 1\}^D$: A binary selection mask of dimension $D = 16$, containing exactly $k = 4$ active bits ($\|m\|_1 = 4$).
- Target parity scalar $y \in \{-1, +1\}$:
  $$y = \prod_{i=1}^D x_i^{m_i} \in \{-1, +1\}$$
- Target class index for Cross-Entropy Loss:
  $$\text{target\_idx} = \begin{cases} 0 & \text{if } y = -1 \text{ (Odd parity)} \\ 1 & \text{if } y = +1 \text{ (Even parity)} \end{cases}$$

The network receives the concatenated vector $u = [x; m] \in \mathbb{R}^{32}$. The functional instruction
is: *"Compute the parity of the features indicated by the active mask coordinates."*
Because $m$ is represented in the continuous feature space $\mathbb{R}^D$, novel compositions are
well-defined as novel binary vectors $m_{\text{novel}}$ with $\|m_{\text{novel}}\|_1 = 4$.

*Labeling:* This is **instruction-conditioned learning**. It is **not** autonomous task inference.

---

## 3. Dataset Specification, Active Pattern Coverage, and Evaluation Populations

### 3.1 Combinatorial Space and Pattern Domain

- Ambient input feature space: $\mathcal{X} = \{-1, +1\}^{16}$, $|\mathcal{X}| = 2^{16} = 65,536$.
- Total available 4-bit selection rules: $|\mathcal{M}| = \binom{16}{4} = 1,820$.
- For any fixed rule $m$, parity depends strictly on the 4 active bit positions $x|_m \in \{-1, +1\}^4$.
  There are exactly $2^4 = 16$ unique active patterns.
- Distractor space: The remaining 12 bit positions take $2^{12} = 4,096$ possible configurations.
  Distractor bits are sampled independently and uniformly from $\{-1, +1\}$ on each example generation.

### 3.2 Exact Pattern Partitioning per Rule

To test within-rule generalization while ensuring mathematical identifiability:
- **12 Training Active Patterns** (75% coverage of active subspace).
- **4 Held-Out Active Patterns** (25% coverage of active subspace).

During training, examples are formed by drawing from the 12 training active patterns and pairing them
with randomly sampled distractor bits.
Full-example duplicate checks verify that identical 32-dimensional input pairs $[x; m]$ are strictly
disjoint between training sets and test sets.

### 3.3 Rule Partitions: Development Rules vs. Frozen Confirmation Rules

The 1,820 available rules are partitioned deterministically:
1. **Development Rules ($N_{\text{dev\_rules}} = 30$):**
   - **Training Sequence Rules ($N_{\text{train}} = 10$):** Tasks $T_1 \dots T_{10}$, presented sequentially.
   - **Transfer Target Rules ($N_{\text{transfer}} = 5$):** Unseen rules $T^*_{1} \dots T^*_{5}$ used to evaluate forward transfer.
   - **Composition Rules ($N_{\text{comp\_dev}} = 15$):** Unseen rules evaluated strictly zero-shot.
2. **Frozen Confirmation Rules ($N_{\text{confirm\_rules}} = 30$):**
   - Completely unreferenced and frozen during this exploratory feasibility phase.

---

## 4. Learnability Contract: Gate G-LC Specification

### 4.1 Evaluation Population and Denominators

Learnability is evaluated on Task 1 ($T_1$) using two distinct, denominated metrics:
1. **Train-Pattern Memorization / Fitting Accuracy (`acc_seen`):**
   - Evaluated on $N = 192$ examples (12 training active patterns $\times 16$ independent distractor draws).
   - Denominator = 192.
2. **Within-Rule Generalization Accuracy (`acc_unseen_patterns`):**
   - Evaluated on $N = 64$ examples (4 held-out active patterns $\times 16$ independent distractor draws).
   - Denominator = 64.

### 4.2 Attainable Increments and The 85% Threshold

- For `acc_seen` ($N=192$), each correctly classified example represents an increment of $1/192 \approx 0.5208\%$.
- For `acc_unseen_patterns` ($N=64$), each correctly classified example represents an increment of $1/64 = 1.5625\%$.
- **Prior Proposed Threshold (85.0%):** On $N=64$, 85.0% requires $\lceil 0.85 \times 64 \rceil = 55/64 \approx 85.9375\%$.
  Parity over unseen patterns is an inductive test: without specific inductive bias, a feedforward net
  may fit observed patterns without immediately generalizing to held-out parity corners.
- **Protocol Formulation of Gate G-LC:**
  - **Primary Learnability Criterion:** `acc_seen` $\ge 90.0\%$ (requires $\ge 173/192$ correct). Proves the network can represent and optimize the parity selection rule on observed patterns.
  - **Diagnostic Report (Non-Gating for Initial Run):** `acc_unseen_patterns` is measured and reported as a separate denominated value (`num/64`).
  - *Guard:* A low `acc_unseen_patterns` score on Task 1 is explicitly classified as **within-rule inductive generalization limit**, NOT catastrophic forgetting.

---

## 5. Transfer Adaptation Protocol and Arm Isolation

### 5.1 Independent Target Probe Isolation

When evaluating forward transfer across the 5 transfer rules ($T^*_1 \dots T^*_5$):
- For each transfer rule $T^*_j$, the **post-history checkpoint** (the exact state after Task 10) is
  cloned independently.
- The model adapts to $T^*_j$ for $S_{\text{target}} = 500$ steps (batch size 16).
- **Zero State Carry-Over:** The adapted weights and optimizer states from probe $T^*_j$ are
  **discarded immediately** after measuring $T^*_j$. The next probe $T^*_{j+1}$ begins from a fresh
  clone of the Task 10 post-history checkpoint.
- **Matched Fresh Comparator:** For each transfer rule $T^*_j$, a fresh learner is initialized
  from random weights (using seed-matched parameter draw) and trained on the exact identical sequence
  of mini-batches, step count, and optimizer settings.
- **Optimizer State:**
  - For `fresh_control`: Adam optimizer state ($m, v$) initializes from exact zero.
  - For `seq_sgd`: At the start of transfer probe $T^*_j$, Adam moments are **reset to zero** to match
    the fresh comparator's initial optimizer momentum, isolating weight-representation transfer from
    momentum carry-over.

### 5.2 Zero-Shot Composition vs. Adapted Transfer

- **Zero-Shot Composition (E4):** Evaluated across the 15 composition rules *immediately after Task 10*,
  with zero adaptation steps and zero gradient updates.
- **Adapted Transfer (E1):** Evaluated strictly on the 5 transfer rules *after* the 500-step adaptation phase.
  These two evaluations are never pooled.

---

## 6. Statistical Design, Analysis Units, and Uncertainty

### 6.1 Independent Analysis Unit

- The **trained-model seed** is the single independent unit of analysis ($N = 6$ independent seeds).
- Transfer performance is evaluated across 5 transfer rules per seed. The summary for seed $s$ is the
  mean transfer score across the 5 rules:
  $$\Delta_s = \frac{1}{5} \sum_{j=1}^5 \left( \text{Acc}_{\text{seq}, s}(T^*_j) - \text{Acc}_{\text{fresh}, s}(T^*_j) \right)$$
- The 5 transfer rules evaluated on the same checkpoint are **correlated repeated measures**, NOT
  independent replicates. Pooling them into $N=30$ independent observations is strictly prohibited.

### 6.2 Seed Uncertainty Scope and Resolution

- **What the $N=6$ seed design captures:** Variation in random weight initialization and stochastic
  data-sampling paths.
- **What it does NOT capture:** Variation across task presentation sequences (presentation order is
  held fixed across seeds in this feasibility phase).
- **Exact Sign-Flip Test:**
  - With $N = 6$ independent paired differences $\{\Delta_1 \dots \Delta_6\}$, the exact two-sided
    permutation floor is $2 / 2^6 = 2/64 = 0.03125$.
  - *Assumption:* Under $H_0$, the distribution of $\Delta_s$ is symmetric around zero.
  - *Exploratory Classification:* This study is exploratory. A result with $p \ge 0.05$ is reported
    descriptively with its exact interval and permutation distribution; it is not dropped or suppressed.

---

## 7. Evaluated Arms Summary

1. **`seq_sgd`:** Trains sequentially on Tasks 1–10; adapts to $T^*_j$ from post-history clone.
2. **`fresh_control`:** Randomly initialized fresh for each $T^*_j$; receives identical batches.
3. **`freeze_base`:** Trains on Task 1 only, then all parameters are frozen bitwise zero. Evaluated on Tasks 2–10 and transfer rules.
4. **`joint_offline`:** Trains on all 10 training rules simultaneously with matched total lifetime steps. Evaluated as a multi-task reference.
5. **`replay_bounded`:** Optional comparator (prospectively specified, unapproved for execution).

---

## 8. Explicit Architecture and Resource Accounting

- **Architecture:** MLP ($32 \to 64 \to 64 \to 2$), ReLU activations, zero normalization layers.
- **Parameters:**
  $$32 \times 64 + 64 + 64 \times 64 + 64 + 64 \times 2 + 2 = \mathbf{6,402}\text{ parameters}.$$
- **Optimizer Memory:** Adam requires $6,402 \times 2 = 12,804$ float32 state values ($\approx 51.2\text{ kB}$).
- **Compute Accounting:**
  - Fresh learner lifetime steps: 500 steps per probe.
  - Sequential learner lifetime steps: $10 \times 2,000 + 500 = 20,500$ steps per probe.
  - Lifetime exposure is tracked and reported separately in all tables.
- **Compute Projections:** Preliminary unmeasured estimate: $\approx 15\text{–}30\text{ s}$ per seed on CPU;
  $\approx 3\text{–}5\text{ s}$ on remote GPU. *No local execution authorized.*
