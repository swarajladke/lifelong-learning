# SSRL-02 Run Evidence Record: SSRL-02 (two_rule_diagnostic)

**Run ID:** `ssrl_02_d3100b00`  
**Execution Timestamp:** 2026-10-22  
**Commit SHA:** `d087b8ba23eeef6ef8f5fcc8dd6db5f503e276d7`  
**Environment:** Kaggle Notebook (CPU-only, no accelerator)  
**Elapsed Training Wall-Clock:** 3.46s (Ceiling: 180.0s)  
**Experiment Exit Code:** 0  

---

## 1. Pinned Datasets and Provenance

- **Rule A Dataset:** `research/lifelong_01/data/ssrl_pilot_t1_s101.json`
  - Canonical SHA-256: `00a21efad1dc07b826198d99b0b702aedddad57b6226f99301afbf16327fd0b7`
  - Rule ID: 1
  - Active Coordinates: `[0, 1, 2, 3]` ($k=4$)
  - Distractor Seed: 42
- **Rule B Dataset:** `research/lifelong_01/data/ssrl_pilot_t2_s101.json`
  - Canonical SHA-256: `27e813e73bf66f51e1bff07f7d2f19945e830afc6d5876b0cb4d11136cda4cd4`
  - Rule ID: 2
  - Active Coordinates: `[4, 5, 6, 7]` ($k=4$)
  - Distractor Seed: 43
  - Coordinate Overlap with Rule A: 0 coordinates (Disjoint active coordinate partition)

---

## 2. Gate G-LC (Learnability Control) Verdict

- **Gate Name:** Gate G-LC (Learnability Control)
- **Observed Accuracy on Rule A Seen Patterns:** 192/192 (100.00%)
- **Reference Threshold:** >= 172/192 (90.0%)
- **Signed Deviation:** +10.00 percentage points
- **Verdict:** **PASSED** (Execution unblocked downstream)

---

## 3. Catastrophic Forgetting Evaluation on Rule A

| Arm | Step / Time | Seen Patterns ($n=192$) | Held-out Patterns ($n=64$) |
|---|---|---|---|
| `seq_ab` | $t_0$ (Init) | 96/192 (50.00%) | — |
| `seq_ab` | $t_1$ (Post-A: 2000 steps) | 192/192 (100.00%) | 29/64 (45.31%) |
| `seq_ab` | $t_2$ (Post-B: 500 steps) | 111/192 (57.81%) | 16/64 (25.00%) |
| **`seq_ab` Forgetting** | **$t_1 \to t_2$ Drop** | **42.19% drop (192/192 -> 111/192)** | **20.31% drop (29/64 -> 16/64)** |
| `freeze_after_a` (Standing Control) | $t_2$ (Post-B: 0 steps) | 192/192 (100.00%) | 29/64 (45.31%) |

> **Empirical Finding:** Sequential training on Rule B without parameter isolation induces severe catastrophic forgetting on Rule A seen patterns (42.19 percentage points loss, dropping from 100.0% to 57.81%). The standing control arm (`freeze_after_a`) preserves 100.0% seen accuracy with zero degradation.

---

## 4. Forward Transfer and Acquisition Trajectory on Rule B

| Step | Sequential AB Seen ($n=192$) | Fresh B Seen ($n=192$) | Seen Transfer Delta | Sequential AB Unseen ($n=64$) | Fresh B Unseen ($n=64$) | Unseen Transfer Delta |
|---|---|---|---|---|---|---|
| Step 0 | 98/192 (51.04%) | 80/192 (41.67%) | +9.37% | 34/64 (53.12%) | 48/64 (75.00%) | -21.88% |
| Step 100 | 102/192 (53.12%) | 124/192 (64.58%) | -11.46% | 23/64 (35.94%) | 6/64 (9.38%) | +26.56% |
| Step 250 | 120/192 (62.50%) | 185/192 (96.35%) | -33.85% | 11/64 (17.19%) | 2/64 (3.12%) | +14.06% |
| Step 500 | 155/192 (80.73%) | 192/192 (100.00%) | -19.27% | 1/64 (1.56%) | 0/64 (0.00%) | +1.56% |

> **Empirical Finding:** Sequential prior training on Rule A creates non-trivial dynamics on Rule B:
> 1. At Step 0 (initial transfer), seen patterns show a modest positive offset (+9.38%), while held-out patterns show a negative offset (-21.88%).
> 2. At Step 100, held-out pattern accuracy shows higher transient generalization in the sequential arm (+26.56%), while seen pattern fitting lags (-11.46%).
> 3. By Step 250 and Step 500, `fresh_b` rapidly masters seen patterns (reaching 96.35% at Step 250 and 100.0% at Step 500), whereas `seq_ab` lags significantly (reaching only 62.50% at Step 250 and 80.73% at Step 500, a negative acquisition delta of -19.27% at Step 500). Prior learning on Rule A thus interferes with the acquisition rate of Rule B compared to learning from a fresh initialization.

---

## 5. Exposures and Budget Accountability

- **`seq_ab` Arm:** 2500 total optimizer steps (40000 sample presentations: 32,000 Rule A + 8,000 Rule B).
- **`fresh_b` Arm:** 500 total optimizer steps (8000 sample presentations: 8,000 Rule B only).
- **`freeze_after_a` Arm:** 2000 total optimizer steps (32000 sample presentations: 32,000 Rule A only, 0 updates during Rule B phase).
