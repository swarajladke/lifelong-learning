# LIFELONG-01 — Human Decisions and Policy Ledger (DECISIONS_REQUIRED.md)

**Assignment:** LIFELONG-01 — Repository-Grounded Feasibility Audit and Experimental Design  
**Study Identifier:** SSRL-01  
**Worktree:** `C:/Users/Vicky/Desktop/Neural-Networks-lifelong-01`  
**Branch:** `research/lifelong-01`  
**Pinned Base Commit:** `e7ecf29`  
**Date:** 2026-10-19 (Revised Post-Review)  

> This document catalogues standing policy rulings, verified architectural parameters,
> and remaining authorizations required before compute execution.

---

## 1. Documented Policy Rulings for SSRL-01

The following scoped policy decisions have been established by the research lead:

1. **Rule 1 Compliance (`MLP-FREEZE-AFTER-BASE`):**  
   A normalization-free MLP frozen after the first task is accepted as this study's
   `FREEZE-AFTER-BASE` control. This resolves the AGENTS.md §13 Rule 1 mandate for this
   architecture without reinstating withdrawn results from other tracks.
2. **Seed Namespace (`SEEDS = [101, 102, 103, 104, 105, 106]`):**  
   Seeds 101–106 are approved as a study-specific exception to the historical fixed seed list
   `[42..46]`. This preserves complete separation from the S0-16 confirmation registry.
3. **Execution Boundary for Unit Tests:**  
   Local execution is authorized strictly for small, model-free unit tests using handcrafted
   synthetic fixtures. Zero local training, model instantiation, or sweeps are permitted.
4. **Replay Policy Status:**  
   Replay is recognized as an unapproved optional comparator. It is not included in the
   primary study runner or execution path.

---

## 2. Learnability Gate G-LC Specification Review

- **Evaluated Populations on Task 1 ($T_1$):**
  - Seen active patterns: $N_{\text{seen}} = 192$ (12 active patterns $\times$ 16 distractor draws). Increment: $1/192 \approx 0.52\%$.
  - Unseen active patterns: $N_{\text{unseen}} = 64$ (4 held-out active patterns $\times$ 16 distractor draws). Increment: $1/64 = 1.5625\%$.
- **Formal Gate G-LC:**
  - `acc_seen` $\ge 90.0\%$ (requires $\ge 173/192$).
  - `acc_unseen_patterns` measured and reported as a separate diagnostic ($n/64$), non-gating for initial feasibility.
  - A failure on unseen patterns is classified as an inductive bias limitation on parity, NOT catastrophic forgetting.

---

## 3. Remote Compute Proposal: Learnability-Only Calibration

Before any 10-task sequential runner or multi-seed transfer matrix is launched on Kaggle,
a **bounded learnability-only calibration run** is proposed:

- **Target Script:** `experiments/run_ssrl_learnability_pilot.py` (authoring only, not yet executed).
- **Scope:** Evaluate Arm 2 (`fresh_control`) on Task 1 across Seed 101 only.
- **Compute Budget:** $2,000$ optimizer steps (batch size 16), wall-clock projection $< 60\text{ seconds}$ on Kaggle GPU.
- **Purpose:** Confirm that Gate G-LC passes ($\text{acc\_seen} \ge 90.0\%$) and measure baseline within-rule generalization on held-out patterns before committing remote session time to the multi-task sequence.

---

## 4. Compute and Gate Blockers Ledger

| Blocker ID | Scope | Description | Status |
|:---|:---:|:---|:---:|
| **B-1** | Compute | Formal authorization to dispatch the Learnability-Only Remote Pilot to Kaggle | **Executed & Passed (2026-10-21)** |
| **B-2** | Scope | Formal approval of Gate G-LC thresholds (90% seen, diagnostic unseen) | **Approved by User (2026-10-19)** |

---

## 5. Formal Human Approvals Ledger

### Approval Record: SSRL-01 Single-Rule Learnability Pilot (2026-10-19)
- **Approved Scope:**
  - Task 1 only, seed 101, arm `fresh_control`.
  - Rule coordinates: `[0, 1, 2, 3]`, `rule_id`: 1, $D=16$, $k=4$.
  - Normalization-free ReLU MLP: $32 \to 64 \to 64 \to 2$, exactly 6,402 parameters.
  - Adam optimizer, learning rate 0.001, $\beta = (0.9, 0.999)$, $\epsilon = 10^{-8}$, weight decay 0.0.
  - 2,000 steps, batch size 16.
  - Populations: Train pool 384, seen eval 192, unseen eval 64.
  - Gate $G\text{-LC}$: seen-evaluation accuracy $\ge 90.0\%$, requiring $\ge 173/192$.
  - Held-out-pattern accuracy is descriptive, non-gating for feasibility.
  - Deadlines: 120.0s monotonic training ceiling, 60.0s monotonic evaluation ceiling.
  - Policy: Exactly one run; no automatic retries, tuning, or replacement seeds.
- **Specification Approval Status:** Transitioned from `PROPOSED_PENDING_APPROVAL` to `APPROVED`.

### Execution Outcome: SSRL-01 Pilot Remote Run (2026-10-21)
- **Status:** COMPLETED with exit code 0.
- **Environment & Execution:** Remote Kaggle run directory `/kaggle/working/ssrl_pilot_run_iOVhtk8o`.
- **Pre-flight Tests:** 20 / 20 unit and regression tests passed.
- **Gate $G\text{-LC}$ (Seen Active Patterns):** 192 / 192 = 100.00% (Passed; requirement $\ge 173/192$).
- **Held-Out Pattern Generalization:** 28 / 64 = 43.75% (Descriptive baseline; chance floor 50.0%).
- **Wall-Clock Time:** 1.92s training (ceiling 120.0s).


