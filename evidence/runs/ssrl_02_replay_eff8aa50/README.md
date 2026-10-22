# SSRL-02 Run Evidence Record: Checkpoint Capture Replay & Counterfactual Mask Probe

**Run ID:** `ssrl_02_replay_eff8aa50`  
**Study ID:** `SSRL-02`  
**Execution Environment:** Local Workstation (CPU-only, Windows)  
**Execution Mode:** Local Exploratory Replay (Deviation from authorized user-run Kaggle remote workflow; authorized retroactively by researcher)  
**Execution Timestamp:** 2026-10-22  
**Elapsed Training Wall-Clock:** 14.73s (Ceiling: 180.0s)  
**Elapsed Probe Wall-Clock:** 0.30s (Ceiling: 60.0s)  
**Experiment Exit Code:** 0  

---

## 1. Protocol Deviation & Environment Provenance

- **Workflow Deviation:** This replay was executed on the local development workstation rather than via the authorized manual user-executed Kaggle remote cell. The researcher explicitly authorized this local execution post-hoc due to the small, lightweight footprint of the model and datasets. All raw artifacts, saved checkpoint tensors, and local environment fingerprints are preserved verbatim.
- **Hardware / Platform:** CPU-only (no accelerator), Windows 10/11, Python 3.11.
- **Git Commit:** `43e0530df1008e22c3a34925c6cf8113659eccc3` (pinned HEAD at execution time).
- **Working Tree State:** Clean at execution start (reproduction ran against committed tree).

---

## 2. Pinned Datasets and Provenance

- **Rule A Dataset:** `research/lifelong_01/data/ssrl_pilot_t1_s101.json`
  - Canonical SHA-256: `00a21efad1dc07b826198d99b0b702aedddad57b6226f99301afbf16327fd0b7`
  - Rule ID: 1
  - Active Coordinates: `[0, 1, 2, 3]` (k=4)
- **Rule B Dataset:** `research/lifelong_01/data/ssrl_pilot_t2_s101.json`
  - Canonical SHA-256: `27e813e73bf66f51e1bff07f7d2f19945e830afc6d5876b0cb4d11136cda4cd4`
  - Rule ID: 2
  - Active Coordinates: `[4, 5, 6, 7]` (k=4)
- **Counterfactual Mask Probe Dataset:** `research/lifelong_01/data/ssrl_probe_counterfactual_128.json`
  - Canonical SHA-256: `1c07d50fed772cfcff86e3bf30cc74a95cdc24d190ec767aa15365671fe06319`
  - Total Unique Vectors: 128 (paired under both Mask A and Mask B = 256 presentations)
  - Quotas: 64 seen/seen, 32 seen/held-out, 32 held-out/seen, 0 held-out/held-out
  - Training Pool Intersections: 0 with Rule A, 0 with Rule B

---

## 3. Reproduction Verification Against Original SSRL-02 Run

- **Reference Run ID:** `ssrl_02_d3100b00` (`evidence/runs/ssrl_02_d3100b00/results.json`)
- **Outcome Reproduction Verified:** `True`
- **Evaluation Splits Checked:** 25 splits across all training phases and transfer evaluation steps
- **Total Individual Predictions Checked:** 3,264
- **Bit-for-Bit Prediction Discrepancies:** 0 / 3,264
- **Prediction Reproduction vs. Tensor Reproduction:** While all 3,264 individual prediction outcomes match bit-for-bit between the Kaggle run `ssrl_02_d3100b00` and this local replay, exact tensor equality across environments cannot be directly proven because the original Kaggle run did not serialize `.pt` weight files to disk. Matching discrete predictions does not by itself establish identical underlying weight tensors. The physical checkpoint fingerprints established in this run represent the verified baseline for this local replay.

---

## 4. Size-Bounded Checkpoint Capture Integrity

| Checkpoint Name | Target Milestone | File SHA-256 | Size (Bytes) | Tensor Fingerprint |
|---|---|---|---|---|
| `checkpoint_t0_init.pt` | `checkpoint_t0_init` | `42a065e127f9b2f8...` | 28342 | `acb9143e56b062e5...` |
| `checkpoint_t1_post_a.pt` | `checkpoint_t1_post_a` | `350deff9412af6b0...` | 28362 | `9ce6ab7020b1e7c2...` |
| `checkpoint_t2_freeze_a.pt` | `checkpoint_t2_freeze_a` | `73f271fb355b3d18...` | 28446 | `9ce6ab7020b1e7c2...` |
| `checkpoint_t2_fresh_b.pt` | `checkpoint_t2_fresh_b` | `44932e1787178bc2...` | 28372 | `10bb1d489358ff15...` |
| `checkpoint_t2_seq_ab.pt` | `checkpoint_t2_seq_ab` | `4e4cce888341b058...` | 28362 | `cf185302d328dc4b...` |

> **Verification Note:** All five checkpoints are strictly below the 100 KB limit (28,446 bytes max). Checkpoint reload-and-verify confirmed zero state mutation and identical parameter tensors.
> Notice that `checkpoint_t1_post_a` and `checkpoint_t2_freeze_a` possess identical tensor fingerprints (`9ce6ab7020b1e7c2...`), proving zero parameter updates during the freeze phase.

---

## 5. Gate G-LC (Learnability Control) Verdict

- **Gate Name:** Gate G-LC (Learnability Control)
- **Observed Accuracy on Rule A Seen Patterns:** 192/192 (100.00%)
- **Reference Threshold:** >= 172/192 (90.0%)
- **Signed Deviation:** +10.00 percentage points
- **Verdict:** **PASSED** (Execution unblocked downstream)

---

## 6. Counterfactual Mask Probe Diagnostic Results

Each of the 128 unique $x$ vectors was evaluated twice: once with Mask A and once with Mask B.
Because $	ext{parity}_A(x) 
eq 	ext{parity}_B(x)$ on all 128 pairs, a model cannot achieve joint correctness without changing its output based on the mask.

### 6.1 Overall Metrics Across Captured Checkpoints ($n=128$)

| Checkpoint State | Mask A Acc ($n=128$) | Mask B Acc ($n=128$) | Switch Rate ($n=128$) | Joint Correctness ($n=128$) | Unchanged Match A | Unchanged Match B |
|---|---|---|---|---|---|---|
| `checkpoint_t0_init` | 64/128 (50.0%) | 64/128 (50.0%) | 0/128 (0.0%) | 0/128 (0.0%) | 64/128 (50.0%) | 64/128 (50.0%) |
| `checkpoint_t1_post_a` | 111/128 (86.7%) | 18/128 (14.1%) | 3/128 (2.3%) | 2/128 (1.6%) | 109/128 (85.2%) | 16/128 (12.5%) |
| `checkpoint_t2_seq_ab` | 60/128 (46.9%) | 81/128 (63.3%) | 47/128 (36.7%) | 30/128 (23.4%) | 30/128 (23.4%) | 51/128 (39.8%) |
| `checkpoint_t2_fresh_b` | 32/128 (25.0%) | 96/128 (75.0%) | 0/128 (0.0%) | 0/128 (0.0%) | 32/128 (25.0%) | 96/128 (75.0%) |
| `checkpoint_t2_freeze_a` | 111/128 (86.7%) | 18/128 (14.1%) | 3/128 (2.3%) | 2/128 (1.6%) | 109/128 (85.2%) | 16/128 (12.5%) |

### 6.2 Strata Breakdown Across Key Checkpoints

| Checkpoint | Stratum | Total Pairs | Acc(Mask A) | Acc(Mask B) | Switch Rate | Joint Correctness | Inverted (Switched Both Incorrect) |
|---|---|---|---|---|---|---|---|
| `checkpoint_t1_post_a` | Stratum 1: Seen(A) / Seen(B) | 64 | 64/64 (100.0%) | 1/64 (1.6%) | 1/64 (1.6%) | 1/64 (1.6%) | 0/64 (0.0%) |
| `checkpoint_t1_post_a` | Stratum 2: Seen(A) / Unseen(B) | 32 | 32/32 (100.0%) | 0/32 (0.0%) | 0/32 (0.0%) | 0/32 (0.0%) | 0/32 (0.0%) |
| `checkpoint_t1_post_a` | Stratum 3: Unseen(A) / Seen(B) | 32 | 15/32 (46.9%) | 17/32 (53.1%) | 2/32 (6.2%) | 1/32 (3.1%) | 1/32 (3.1%) |
| `checkpoint_t2_seq_ab` | Stratum 1: Seen(A) / Seen(B) | 64 | 32/64 (50.0%) | 54/64 (84.4%) | 32/64 (50.0%) | 27/64 (42.2%) | 5/64 (7.8%) |
| `checkpoint_t2_seq_ab` | Stratum 2: Seen(A) / Unseen(B) | 32 | 24/32 (75.0%) | 0/32 (0.0%) | 8/32 (25.0%) | 0/32 (0.0%) | 8/32 (25.0%) |
| `checkpoint_t2_seq_ab` | Stratum 3: Unseen(A) / Seen(B) | 32 | 4/32 (12.5%) | 27/32 (84.4%) | 7/32 (21.9%) | 3/32 (9.4%) | 4/32 (12.5%) |
| `checkpoint_t2_fresh_b` | Stratum 1: Seen(A) / Seen(B) | 64 | 0/64 (0.0%) | 64/64 (100.0%) | 0/64 (0.0%) | 0/64 (0.0%) | 0/64 (0.0%) |
| `checkpoint_t2_fresh_b` | Stratum 2: Seen(A) / Unseen(B) | 32 | 32/32 (100.0%) | 0/32 (0.0%) | 0/32 (0.0%) | 0/32 (0.0%) | 0/32 (0.0%) |
| `checkpoint_t2_fresh_b` | Stratum 3: Unseen(A) / Seen(B) | 32 | 0/32 (0.0%) | 32/32 (100.0%) | 0/32 (0.0%) | 0/32 (0.0%) | 0/32 (0.0%) |

### 6.3 Descriptive Summary & Interpretive Boundaries

1. **Observed Switch and Joint Correctness Counts:**
   - Under single-task training ($t_1$ post-A), the model switches predictions on only 3/128 (2.3%) instances. On 109/128 pairs (85.2%), predictions are unchanged matching the Rule A target.
   - Under fresh single-task training ($t_2	ext{-fresh}$ on B), the model switches predictions on 0/128 (0.0%) instances. On 96/128 pairs (75.0%), predictions are unchanged matching the Rule B target.
   - In sequential training (`checkpoint_t2_seq_ab`), overall switch count rises to **47/128 (36.7%)**, and **30/128 (23.4%)** pairs are jointly correct.
   - Within Stratum 1 (Seen in A / Seen in B, $n=64$):
     - **32/64 (50.0%)** switch predictions;
     - **27/64 (42.2%)** are jointly correct (both rules predicted accurately);
     - **5/64 (7.8%)** are inverted (predictions switch, but both are incorrect);
     - **5/64 (7.8%)** remain unchanged matching the Rule A target;
     - **27/64 (42.2%)** remain unchanged matching the Rule B target.
2. **Interpretive Guard (Argmax Switching vs. Representation Sensitivity):**
   - Low or zero argmax switching does **not** prove that the network has "ignored" the mask or that internal routing weights are zero.
   - Argmax switching measures only whether the discrete classification decision flips across the decision boundary. Pre-activation logit margins or internal representations could still exhibit continuous variation with respect to mask coordinates without crossing the decision threshold.
3. **Scope and Cardinality Limitations:**
   - These findings are observed under a single initialization seed (`seed=101`) and a sampled probe population of 128 counterfactual pairs.
   - They describe behavior for this specific architecture ($32 	o 64 	o 64 	o 2$ MLP) on disjoint active parity rules ($k=4$). They do not establish general architectural properties across multiple seeds or alternative rule configurations.

---

## 7. Standard Catastrophic Forgetting & Forward Transfer Metrics

### Rule A Accuracy and Forgetting

| Metric | Value | Details |
|---|---|---|
| `acc_t0_rule_a_seen` | 96/192 (50.00%) | Mode: `eval_no_dropout` |
| `acc_t1_seq_rule_a_seen` | 192/192 (100.00%) | Mode: `eval_no_dropout` |
| `acc_t1_seq_rule_a_unseen` | 29/64 (45.31%) | Mode: `eval_no_dropout` |
| `acc_t2_seq_rule_a_seen` | 111/192 (57.81%) | Mode: `eval_no_dropout` |
| `acc_t2_seq_rule_a_unseen` | 16/64 (25.00%) | Mode: `eval_no_dropout` |
| `acc_t2_frozen_rule_a_seen` | 192/192 (100.00%) | Mode: `eval_no_dropout` |
| `acc_t2_frozen_rule_a_unseen` | 29/64 (45.31%) | Mode: `eval_no_dropout` |
| `forgetting_rule_a_seen` | **42.19% drop** | Drop: 192/192 -> 111/192 (n=192) |
| `forgetting_rule_a_unseen` | **20.31% drop** | Drop: 29/64 -> 16/64 (n=64) |

### Rule B Transfer Trajectory

| Step | Transfer Delta (Seen, n=192) | Transfer Delta (Unseen, n=64) |
|---|---|---|
| Step 0 | +9.37% (Seq: 98/192 vs Fresh: 80/192) | -21.88% (Seq: 34/64 vs Fresh: 48/64) |
| Step 100 | -11.46% (Seq: 102/192 vs Fresh: 124/192) | +26.56% (Seq: 23/64 vs Fresh: 6/64) |
| Step 250 | -33.85% (Seq: 120/192 vs Fresh: 185/192) | +14.06% (Seq: 11/64 vs Fresh: 2/64) |
| Step 500 | -19.27% (Seq: 155/192 vs Fresh: 192/192) | +1.56% (Seq: 1/64 vs Fresh: 0/64) |
