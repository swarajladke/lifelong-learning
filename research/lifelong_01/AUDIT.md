# LIFELONG-01 — Reuse and Failure Map (AUDIT.md)

**Assignment:** LIFELONG-01 — Repository-Grounded Feasibility Audit and Experimental Design  
**Worktree:** `C:/Users/Vicky/Desktop/Neural-Networks-lifelong-01`  
**Branch:** `research/lifelong-01`  
**Reviewed commit:** `e7ecf29` (HEAD on `main` as of 2026-10-19)  
**Author of this document:** Agent (LIFELONG-01, revision pass)  
**Date:** 2026-10-19  

> **Scope:** This audit inspects four research subsystems present in the repository at the
> reviewed commit (`e7ecf29`). Findings are classified strictly under the assignment taxonomy:
> - **A:** Implemented mechanism in repository code.
> - **B:** Supported empirical result tied to a committed results artifact.
> - **C:** Historical proposal, architectural aspiration, or roadmap claim.
> - **D:** Withdrawn, unverified, or unresolved result.
> - **E:** Static-code concern or untested structural assumption requiring verification.
>
> All suspected defects are treated as hypotheses until experimentally isolated.

---

## 0. Governance Boundary and Policy Ambiguities

### 0.1 Pure AGNIS Pretraining Kill Criterion

`PURE_AGNIS_ROADMAP.md` (lines 167–169) records a binding kill criterion established on 2026-09-16:

> From-scratch pure AGNIS pretraining is suspended. There is **exactly one authorized revival
> attempt**: maximum budget 5,000 optimizer steps and 10.0 GPU-hours on FineWeb-Edu. If training loss
> fails to break below 4.50 within those 5,000 steps, the revival attempt is declared a FAIL and pure
> from-scratch AGNIS pretraining is **permanently abandoned in full**; zero further GPU compute
> hours shall ever be allocated to it.

**Scope and Boundaries:**
- The text of the kill criterion explicitly addresses large-scale language pretraining from scratch
  on token-streaming corpora (FineWeb-Edu).
- *Policy Ambiguity:* The documentation does not formally specify whether this kill criterion applies
  narrowly to autoregressive language pretraining, or broadly prohibits any future small-scale
  investigation using predictive coding hierarchies (e.g. `agnis_v4_core.py`) on non-linguistic toy
  domains.
- *Status of Revival Attempt:* Unknown from committed artifacts. No commit logs a spent revival attempt
  or marks the token as active.
- *Resolution for LIFELONG-01:* To eliminate governance risk without unilaterally interpreting policy,
  the proposed next experiment (**SSRL-01**) bypasses the AGNIS hierarchy entirely, using a standard
  feedforward baseline. Scope clarification for future small-scale predictive-coding work is deferred
  to a human ruling (`DECISIONS_REQUIRED.md` D-1).

### 0.2 AGNIS Repository Quarantine

`context.md` (line 422) mandates: *"Do not touch AGNIS files (`agnis*.py`, quarantined legacy attempt)."*
While the modules exist in the working directory, this governance directive places them outside the
active experimental path.

### 0.3 Mission Disallowal of Replay vs. Replay-Assisted Implementations

- `ultimate_goal.md` §3 specifies a continuous learner operating *"without fine-tuning, replay buffers,
  or full retraining."*
- `enhanced_agnis_v2.py` implements a `ReplayBuffer` and `SurpriseBuffer`.
- `context.md` §15 formally classifies Directive S0-14/S0-15 Slot Consolidation as
  **REPLAY-ASSISTED CONSOLIDATION** because prompt texts and target tokens of earlier facts are re-fed
  to optimize joint targets ($v^*$).
- *Classification:* C (historical aspiration of zero replay) vs A (implemented replay buffers and
  replay-assisted consolidation). The repository does not contain an isolating ablation proving
  whether anti-forgetting gains in S0-14 stem from subspace geometry alone or from replay exposure.

---

## 1. Track B: S0 Sequential Factual Editing in Pretrained GPT-2

### 1.1 Actual Learning Mechanism

Closed-form or gradient-based rank-1 parametric updates applied strictly to feedforward value
projections `transformer.h.L.mlp.c_proj.weight` in GPT-2 Small (124M parameters), primarily at Layer 1.
Readout projections (`lm_head`, `wte`, `ln_f`) are frozen bitwise zero.
- **Reference (R):** Margin-targeted closed-form rank-1 writes with subject-key null-space projection.
- **Slot Consolidation (C2):** Identifies near-collinear subject keys ($\cos \ge 0.999$), forms collision
  groups $G$, and solves a joint target value $v^*$ before applying rank-1 updates.
- **Template Protection (C3):** Extends C2 by incorporating SVD-truncated prompt-template bases into the
  null-space projector.

*Classification:* **A** (implemented mechanisms verified by pre-flight unit tests in `tests/test_metrics.py`).

### 1.2 Initialization and Dependencies

- Dependent on HuggingFace pretrained `gpt2` (commit-pinned weights).
- Pinned datasets: `b1_facts.json` (1,000 synthetic facts, SHA-256 `285638ad...`), WikiText-2 capability slice.
- Pinned covariance and projection caches: `cov_l1.pt`, `v_null_l1.pt`.
- No capability for de-novo representation acquisition from random initialization.

### 1.3 Mutable State vs. Ephemeral State

- **Persistent state:** Layer 1 `c_proj.weight` tensor; running projector basis $Q_t$.
- **Tracking state:** Stored subject keys $\{k_i\}$ and target vectors $\{v^*_i\}$ accumulated in memory.
- In Arm C2/C3, stored keys and prompts are re-read during later collision-group solves.

### 1.4 Demonstrated Capabilities and Limitations

| Capability / Finding | Status | Evidence / Limits |
|:---|:---:|:---|
| Immediate injection efficacy $\ge 90\%$ | **B** | Verified across multiple directives on 200 synthetic facts per seed. |
| First-50 sequential retention above floor | **B** | Verified in S0-12, S0-13, S0-14 ($N=300$ across 6 seeds) above negative control floor. |
| Causal role of subject-key collisions | **B** | S0-14 Stage A demonstrated excluded keys suffer 96.61% loss vs 43.15% (OR = 37.54, $p < 10^{-14}$). |
| Paraphrase generalization above floor | **B** | Modest retention (10–25%) observed on fixed synthetic template variations. |
| Long-horizon scaling ($N > 200$) | **D** | Unverified. Compute ceilings prevent long-horizon testing. |
| Transfer to novel relations/domains | **D** | Not demonstrated. Limited to 4 synthetic entity relations. |
| Causal isolation of replay in C2/C3 | **D** | Unverified. Replay of earlier facts occurs during $v^*$ optimization; its contribution has not been ablated. |

### 1.5 Key Mechanism Audit Questions

- **Do gradients/updates reach intended parameters?** Yes. S0-9 Stage P verified non-zero gradient norms
  at Layer 1 `c_proj.weight`.
- **Does prediction change persistent state?** No. Autoregressive greedy generation is purely read-only.
- **Presentation order vs. Evaluation order:** Learning is strictly sequential (presentation order induces
  directional interference). Evaluation order does not alter model weights, but earlier facts suffer
  cumulative interference from later writes.
- **Task Identity:** Implicit. No task-ID input is provided; routing relies entirely on subject-key matching.
- **Can recall bypass weight updates?** In S0, inference runs strictly through the transformer weights.
  However, *training-time consolidation* accesses stored historical facts (replay).
- **Checkpoint Restoration:** Checkpoints restore model parameters, but historical covariance/projector
  caches diverged across processes in S0-11/S0-12 due to unpinned environment factors, necessitating Gate G0b.

---

## 2. Quarantined AGNIS / Predictive Coding Architecture

### 2.1 Implemented Architecture

`agnis_v4_core.py` implements a `PredictiveHierarchy` consisting of stacked `PredictiveColumn` modules:
- Recognition weights $V$ (bottom-up) and generative weights $W$ (top-down).
- Local SNAP-ATP (Synchronized Aggressive Target Propagation) settlement dynamics.
- Soft synaptic clamping replacing multiplicative weight decay (Protocol 4.9).
- Dynamic node allocation (`META_POOL_MAX_SIZE = 1024`).

*Classification:* **A** (implemented code), but quarantined under §0.2.

### 2.2 Demonstrated Empirical Status

- **Associative Settle:** Architecture demonstrates iterative convergence to fixed points in unit tests (**A**).
- **Language Generation:** Formally evaluated and failed (**B — Negative**). PURE_AGNIS_ROADMAP.md confirms
  loss stalled at 7.7 (perplexity ~2208) across 20,000 steps due to the lack of a cross-entropy objective.
- **Continual Retention & Transfer:** Unverified (**D**). No committed results artifact validates sequential
  retention or forward transfer in pure AGNIS.

---

## 3. Track A: Continual Representation Learning (Split-CIFAR-100)

### 3.1 Implemented Mechanisms and Findings

ResNet-18 evaluated across 10 sequential 10-class tasks:
- **Joint Offline:** 79.62% ± 0.21% class-IL accuracy (**B — Standing**, commit `1e4d3e6`).
- **Naive Fine-Tuning:** 9.53% ± 0.21% class-IL, task-aware 83.57%, linear probe 65.71% (**B — Standing**).
- **Frozen Backbone + NCM:** 47.12% ± 0.08% class-IL (**B — Standing**).
- **Adaptation Gap:** Validated at +20.43 pp (**B — Standing**).
- **Core Empirical Finding:** Forgetting in class-incremental classification is primarily **classifier
  readout collapse**, not representation destruction. Linear probes on catastrophically forgotten backbones
  retain high discriminative accuracy.

### 3.2 Withdrawn Arms and Structural Defects

- `1_freeze_after_base`: **WITHDRAWN** due to BatchNorm running statistics adapting during eval mode
  (`model.train()` leakage).
- `5_lwf`, `6_ewc`, `7_er_buffer500`, `8_der_plus_plus_buffer500`: **WITHDRAWN** on provenance grounds
  (missing raw records in `w3_baselines.json`).

---

## 4. Episodic Buffers and Replay Systems

### 4.1 Mechanisms Present in Repository

1. **`replay_buffer.py`:** Standalone reservoir-sampling replay buffer storing `(input, target, task_id)`.
   *Classification:* **A** (functional utility; not integrated into active S0 line).
2. **`fact_memory.py` (`EpisodicFactMemory`):** Explicit key-value retrieval store bypassing weights.
   *Classification:* **A** (implements direct external memory; not evaluated with retrieval disabled).
3. **Training-Time Replay in S0 Consolidation:** Stored prompt strings and targets are replayed during
   optimization of joint target vectors $v^*$.
   *Classification:* **A** (implemented), but distinct from test-time retrieval.

---

## 5. Candidate Components for Reuse in Lifelong-01

| Component | Location | Feasibility & Reuse Boundary |
|:---|:---|:---|
| `Measurement` Primitive | `experiments/metrics.py` | **High.** Copy verbatim (AGENTS.md §7.5). Enforces non-bare metric contracts. |
| Exact Paired Sign-Flip Test | `experiments/s0_16_evaluator.py` | **Moderate.** Usable for small paired seed comparisons, provided sample size and resolution limits are respected. |
| AST Literal Scanner | `tests/test_metrics.py` / `AGENTS.md` | **Mandatory.** Protects against hardcoded metric literals. |
| Gate Verification Harness | `AGENTS.md` Appendix C.3 | **Mandatory.** Code-defined gates with exact source hashes. |
| `EvaluationRecord` Schema | `experiments/s0_16_evaluator.py` | **High.** Adapt to record per-example inputs, selections, and outcomes. |

*Components Excluded from Reuse:*
- Pretrained GPT-2 and `b1_facts.json` (tied specifically to Track B factual editing).
- Quarantined AGNIS columns (`agnis_v4_core.py`) pending governance resolution.
