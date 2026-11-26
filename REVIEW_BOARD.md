# Lifelong Learning — Human Review Board

> **How to use this file:** This is your **one permanent review file** (`AGENTS.md` §5). Every time a study finishes, I update this single file in place—moving the finished study into the **Completed Study Log** at the bottom and putting the next proposal here at the top—so you never have to search through separate proposal files.

---

## Current Status: SSRL-14 STAGE 1 COMPLETE — WAITING FOR YOUR YES / NO FOR STAGE 2 RUN

**Proposed Next Decision:** **SSRL-14 — Stage 2 Execution (`SSRLInputGatedMLP` + Matched `2,000`-Step/Rule `L2-Init + DER++` 3-Rule Sequential Stream `Rule 1 -> Rule 2 -> Rule 3` — Approve Spec, Commit & Push Stage 1 to `main`, Run 3-Seed CPU Batch & Auto-Publish)**  
**Goal Served:** Cross from solved **2-rule** continual learning (`SSRL-12`) to solved **3-rule** continual learning ($\ge 90\%$ on **Rule 1, Rule 2, AND Rule 3 simultaneously across all 3 seeds `102, 103, 104`**) by removing the exact distractor-bit interference defect uncovered by SSRL-13 forensics.

### Stage 1 Pre-Flight Verification (Completed Locally — Zero Production Training Run Yet)
- **New Input-Gated Architecture Added to [ssrl_model.py](file:///c:/Users/Vicky/Desktop/lifelong-learning/research/lifelong_01/ssrl_model.py):** `SSRLInputGatedMLP` (`6,658` parameters = exact `6,402`-param `SSRLInstructionMLP` backbone instantiated first + `256`-param zero-initialized bias-free mask-to-feature gate `fc_in_gate(mask)`). Step-0 forward pass is **bit-identical** to bare `SSRLInstructionMLP` under the same seed, while nonzero asymmetric gradients activate from Step 1 to suppress inactive distractor coordinates without any hardcoded coordinate mask.
- **New Runner Created:** [run_ssrl_input_gated_three_rule.py](file:///c:/Users/Vicky/Desktop/lifelong-learning/research/lifelong_01/run_ssrl_input_gated_three_rule.py) — locked at `gate_approval_status = "PROPOSED_PENDING_APPROVAL"` (`532` lines `< 600` ceiling).
- **Allowlists Synced:** Added all `27` `checkpoint_ig3_*_s{102,103,104}.pt` filenames (`9` per seed) cleanly in [ssrl_checkpoints.py](file:///c:/Users/Vicky/Desktop/lifelong-learning/research/lifelong_01/ssrl_checkpoints.py) and [publish_results.py](file:///c:/Users/Vicky/Desktop/lifelong-learning/research/lifelong_01/publish_results.py) with zero threshold or `10 MB` cap modifications.
- **Test Suites Passed (`5` new tests added in [test_ssrl_model_free.py](file:///c:/Users/Vicky/Desktop/lifelong-learning/research/lifelong_01/test_ssrl_model_free.py)):**
  - `TestSSRLModelFree`: **`76/76` passed** (`exit 0`)
  - `TestSSRLCPUIntegration`: **`32/32` passed** (`exit 0`)

### Where We Stand Right Now (Plain-English Scoreboard)

| Study & Method (Seeds `102, 103, 104`) | Rule 1 Seen Retained (`/192`) | Rule 2 Seen (`/192`) | Rule 3 Seen (`/192`) | Plain-English Lesson |
| :--- | :---: | :---: | :---: | :--- |
| **SSRL-03** *(Offline Joint, 2-Rule, Seed 101)* | `100.0%` | `98.96%` | *N/A* | The `6,402`-param MLP **has enough capacity** for multi-rule parity at once. |
| **SSRL-07** *(Naive Sequential, 2-Rule)* | `51.0%–61.5%` | `74.0%–80.2%` | *N/A* | Forgets Rule 1 almost completely AND loses plasticity (`-23.1 pp` on Rule 2). |
| **SSRL-08** *(`L2-Init` $\lambda=0.01$ + Replay, 2-Rule)* | `58.9%–68.8%` | **`91.2%–95.8%`** | *N/A* | Solved 2-rule plasticity, but pulling toward $\theta_0$ loosened Rule 1 retention. |
| **SSRL-09** *(`LayerNorm` + Hidden Mask Gate, 2-Rule)* | `60.4%–63.5%` | `81.8%–92.2%` | *N/A* | Single-mask Phase 1 left `fc3` margins high and hidden gate collapsed (`gate_cos=0.98`). |
| **SSRL-12** *(`L2-Init` + `DER++` `M=64`, `1k` R2 steps)* | **`93.2%–97.9%`** | **`96.4%–100.0%`** | *N/A* | **TWO-RULE SOLVED (`3/3` RESULT 1):** `184, 188, 179 / 192` on R1 AND `192, 192, 185 / 192` on R2 (`64/64` buffer match). |
| **SSRL-13** *(Bare `6,402` MLP + `L2-Init` + `DER++`, `1k` R3 steps)* | **`85.9%–91.1%`** (`165, 173, 175`) | **`76.6%–92.2%`** (`177, 175, 147`) | **`64.6%–69.3%`** (`130, 133, 124`) | **RESULT 3 TRIGGERED (FORENSICALLY DIAGNOSED):** Training acc on `384` R3 items hit **`95.3%–96.1%`** (`366–369/384`), while unseen-distractor `eval_seen` (`192`) stalled at `64.6%–69.3%`. Zeroing `R1+R2` distractor bits restored `R1` to **`192/192`** (all 3 seeds) and `seq_r3` `R3` to **`176, 192, 176 / 192`**. Additive `fc1([x; m])` overfit prior active feature bits (`8` bits = `256` states across `384` samples) instead of routing by mask! |

**Bottom line (Why `SSRL-14` Directly Targets the Exact SSRL-13 Root Cause):** In `SSRL-13`, `fresh_r3` learned Rule 3 cleanly (**`189, 188, 179 / 192` = `93.2%–98.4%`**), and `stream_der_r3` had **`0/64` dead ReLUs** while hitting **`366–369/384` (`95.3%–96.1%`) on the Rule 3 training pool**. The failure on held-out `Rule 3` distractors happened strictly because `fc1([x; m])` had built large Rule-1 & Rule-2 feature columns (`||W_x1|| = 2.63`, `||W_x2|| = 2.67` vs `||W_x3|| = 2.00`), so the `2^8 = 256` combinations of distractor bits `(x_{0:4}, x_{4:8})` shattered the `384` training examples rather than isolating the active `Rule 3` bits `x_{8:12}`. Meanwhile, `SSRL-13` only gave Phase 2 & Phase 3 `1,000` steps (`125` replay updates/rule in Phase 3) instead of `PLAN.md`'s canonical `2,000` steps (`250` replay updates/rule). Adding a **multiplicative input gate** `g_in = 2 * sigmoid(4 * W_gate * mask)` lets the mask directly scale down inactive rule bits before additive mixing in `fc1` while keeping exact Step-0 bit-identity with `SSRLInstructionMLP`.

---

## 1. Stage 1 Code Implemented & What Stage 2 Will Execute

### What SSRL-14 Tests (`Rule 1 -> Rule 2 -> Rule 3`, `2,000` Steps/Rule, Seeds `102, 103, 104`)
1. **Primary Gated Stream (`SSRLInputGatedMLP`, `6,658` params):**
   - `g_in = 2 * sigmoid(4.0 * W_gate @ mask)` with zero-initialized `(16, 16)` `W_gate` (no bias), applied elementwise `feat * g_in` before concatenating `[feat_gated; mask]`.
   - Arms per seed: **`freeze_after_r1`** (standing control), **`fresh_r3`** (`L2-Init` from $\theta_0$, G-LC gate $\ge 90\%$), **`seq_r1_r2_r3`** (`L2-Init`, no replay), and **`stream_der_m64`** (`L2-Init` $\lambda=0.01$ + `DER++` $\alpha=0.5$, `M=64`/rule cyclic `4:1` replay = `2,500` total steps in Phases 2 & 3).
2. **Matched Bare Control (`SSRLInstructionMLP`, `6,402` params) at `2,000` Steps/Rule:**
   - Runs **`bare_post_r1` $\to$ `bare_der_post_r2` $\to$ `bare_stream_der_r3`** on the exact same `2,000`-step/rule `DER++` schedule so we cleanly separate how much gain comes from **step-budget parity (`1,000` $\to$ `2,000` steps/rule)** vs the **learned input gate (`+256` params)**.
3. **Mechanistic Gate & Retention Diagnostics:**
   - Records active-coordinate gain vs inactive-coordinate suppression (`all_distractor_gain_mean`, `prior_or_other_rules_gain_mean`) for Rules `1, 2, 3`, pairwise gate cosine similarities (`cos_r1_r2`, `cos_r1_r3`, `cos_r2_r3`), `64`-item buffer logit drift (`R1` & `R2`), and R3 transfer milestones `[0, 25, 50, 100, 250, 500, 750, 1000, 1500, 2000]`.

### What Stays Fixed
- **Pinned Datasets & Hashes:** Rule 1 (`00a21efad1...`), Rule 2 (`27e813e73b...`), Rule 3 (`3a9b7da13f...`), Counterfactual Probe (`1c07d50fed...`).
- **Regularization & Replay:** `Adam(lr=0.001)`, `L2-Init` $\lambda=0.01$, `DER++` $\alpha=0.5$, `M=64`/rule stratified buffers, cyclic `4:1` ratio.
- **Seeds, Gates & Resource Accounting:** Seeds `[102, 103, 104]`, CPU-only, `60.0s` stage ceilings, Gate G-LC $\ge 90.0\%$ (`173/192`), `27` small `.pt` checkpoints (`~29 KB` each $\approx 0.78\text{ MB}$ + `~5–6 MB` `results.json` $\approx 6.9\text{ MB}$ total `< 10 MB` cap).

---

## 2. All Possible Results of SSRL-14 (When Executed in Stage 2)

> *Per `AGENTS.md` §1.3, these list every possible outcome in advance without asserting which one will happen.*

### Result 1 — Best Case: `SSRLInputGatedMLP` Achieves `>= 90%` on All 3 Rules Across `3/3` Seeds (`bare_stream_der` Still `< 90%`)
* **What the numbers would look like:** `stream_der_r3` clears **`>= 173/192` (`>= 90%`) on `R1`, `R2`, AND `R3`** in all 3 seeds, input gates suppress prior-rule distractors below `1.0`, while `bare_stream_der_r3` still stalls `< 90%` on `R3` (or forgets `R1/R2`).
* **What it proves:** Learned mask-to-feature multiplicative gating solves cross-rule distractor interference in continual streams where additive `fc1([x; m])` + `DER++` saturated.
* **Next step after Result 1:** Ready to scale directly to **5–10 sequential rules + held-out composition rules (`PLAN.md`)**.

### Result 2 — Step-Budget Parity Alone Solves It (`2,000` Steps/Rule Lifts Both `stream_der_r3` AND `bare_stream_der_r3` to `>= 90%`)
* **What the numbers would look like:** Both `stream_der_r3` and `bare_stream_der_r3` reach `>= 90%` across `R1, R2, R3` by step `2,000`, showing step `1,000` in `SSRL-13` was simply under-trained under `4:1` replay constraints.
* **What it proves:** `SSRL-13`'s deficit was horizon truncation relative to `PLAN.md`'s `2,000`-step rule budget (`250` vs `125` replay updates/rule), not an absolute capacity barrier at 3 rules (though gating speed/margin differences still quantify transfer efficiency).
* **Next step after Result 2:** Adopt `2,000` steps/rule and advance to **5–10 sequential rules (`PLAN.md`)** comparing bare vs input-gated backbones.

### Result 3 — Partial Trade-Off or Residual Stall (`< 90%` on Either `R3` or Prior `R1/R2` for Gated Stream)
* **What the numbers would look like:** Input gating improves `R3` (e.g., `75%–88%`) or `R1/R2` without clearing `>= 90%` on all 3 rules in `3/3` seeds, OR `fc_in_gate` drifts toward `1.0` under `L2-Init` penalty.
* **What it proves:** Quantifies whether `L2-Init` pulling `W_gate` toward zero (or insufficient replay ratio / width `64` hidden units) caps selective gating strength.
* **Next step after Result 3:** Inspect saved `checkpoint_ig3_*.pt` gate weights and run targeted ablation (e.g., exempting `W_gate` from `L2-Init` shrinkage toward zero or `fc_in_gate` + hidden `LayerNorm`).

### Result 4 — Pre-Gate Failure (`< 90%` on `post_r1`, `bare_post_r1`, or `fresh_r3`) or Runtime/Deadline Fault
* **What the numbers would look like:** Pre-gate G-LC fails or monotonic deadline triggers.
* **What it proves:** Runner, Step-0 neutrality, or budget defect caught prior to downstream interpretation.
* **Next step after Result 4:** Halt immediately with `failure_diagnostics.json` saved and zero automatic retries.

---

## 3. Your Decision Needed Now

* Reply **`Yes`** $\rightarrow$ I will approve **SSRL-14 (`gate_approval_status = "APPROVED"`)**, re-verify tests, commit & push Stage 1 (`ssrl_model.py`, `run_ssrl_input_gated_three_rule.py`, allowlists, tests) to `main`, run the full **3-seed SSRL-14 CPU batch** into a unique run directory, auto-publish if $\le 10$ MB, and update this `REVIEW_BOARD.md` with exact measured numbers.
* Reply **`No`** $\rightarrow$ Keep Stage 1 uncommitted/unrun (or specify a different modification).

---

## 4. Completed Decisions & Study Log (Immutable History)

| Study | Commit / Artifact | Plain-English Result (Verified) |
| :--- | :--- | :--- |
| **SSRL-01** *(Pilot)* | [evidence/pilot/results.json](file:///c:/Users/Vicky/Desktop/lifelong-learning/evidence/pilot/results.json) | Bare MLP (`6,402` params) learned Rule A seen patterns (**`192/192` = `100.00%`**); held-out unseen patterns scored **`28/64` (`43.75%`)**. |
| **SSRL-02** *(Sequential Baseline)* | [ssrl_02_d3100b00/results.json](file:///c:/Users/Vicky/Desktop/lifelong-learning/evidence/runs/ssrl_02_d3100b00/results.json) | Training A then B with no replay caused catastrophic forgetting on Seed 101: Rule A Seen fell **`192/192` $\to$ `111/192` (`57.81%`)**. |
| **SSRL-03** *(Joint Reference)* | `73f8764` \| [ssrl_03_joint_recovery_73f87646/results.json](file:///c:/Users/Vicky/Desktop/lifelong-learning/evidence/runs/ssrl_03_joint_recovery_73f87646/results.json) | Jointly training A + B hit **`192/192` (`100.00%`)** A Seen and **`190/192` (`98.96%`)** B Seen — proving capacity (`6,402` params) is sufficient for both rules. |
| **SSRL-04 & 05** *(Replay Pilots, Seed 101)* | `ed4960c`, `8c49e1d` \| `ssrl_04_replay_ed4960c0`, `ssrl_05_diverse_replay_8c49e1db` | Diverse replay (`M=16`, all 12 patterns) retained **`152/192` (`79.17%`)** vs lowest-ID replay **`129/192` (`67.19%`)** on Rule A Seen; B Seen reached `164/192` vs `166/192`. |
| **SSRL-06** *(3-Seed Replay Replication + Post-A Audit)* | `f208e4d` (Published) \| [ssrl_06_paired_replication_f208e4da/results.json](file:///c:/Users/Vicky/Desktop/lifelong-learning/evidence/runs/ssrl_06_paired_replication_f208e4da/results.json) | Across Seeds `102, 103, 104`, diverse replay beat lowest-ID on Rule A Seen (**`75.00%–85.42%`** vs **`63.02%–71.35%`**), while B Seen stalled (`74.48%–82.81%`). Audit showed held-out A was already `26.56%–48.44%` right after Phase A. |
| **SSRL-07** *(Sequential Transfer Controls + Forensics)* | `33178de` (Local: `17.6 MB > 10 MB` upload refused) \| [ssrl_07_transfer_controls_33178deb/results.json](file:///c:/Users/Vicky/Desktop/lifelong-learning/evidence/runs/ssrl_07_transfer_controls_33178deb/results.json) & [forensics_results.json](file:///c:/Users/Vicky/Desktop/lifelong-learning/evidence/runs/ssrl_07_checkpoint_forensics_readonly_r2/forensics_results.json) | Fresh learners learned Rule B to **`98.44%–100.00%`**, while `seq_b` stalled at **`73.96%–80.21%`** (`-23.09 pp` mean deficit) and forgot Rule A to `51.04%–61.46%`. Forensics proved **`0/64` `h1` dead ReLUs**; instead, Phase A created **`7.41–8.97` Task-B logit margins** (`3.56–4.16` CE loss) and `17.9×–28.5×` Step-0 gradients concentrated on wrong Rule-A bits. |
| **SSRL-08** *(`L2-Init`, $\lambda=0.01$)* | `af1553a7` (Published, `3.09 MB`) \| [ssrl_08_l2_init_plasticity_af1553a7/results.json](file:///c:/Users/Vicky/Desktop/lifelong-learning/evidence/runs/ssrl_08_l2_init_plasticity_af1553a7/results.json) | Halved `post_a` Rule B margins (`2.81–4.24`) and boosted `diverse_replay` Rule B Seen to **`91.15%–95.83%`** (`184, 184, 175 / 192`, passing `>= 90%` in all 3 seeds), **but** Rule A replay retention dropped to **`58.85%–68.75%`** (`113, 132, 126 / 192`). |
| **SSRL-09** *(`SSRLConditionedMLP`: LayerNorm + Mask Gate)* | `f15e5137` (Published, `3.14 MB`) \| [ssrl_09_conditioned_mlp_f15e5137/results.json](file:///c:/Users/Vicky/Desktop/lifelong-learning/evidence/runs/ssrl_09_conditioned_mlp_f15e5137/results.json) | Pre-gates passed (`post_a`: `192, 190, 190 / 192`; `fresh_b`: `189, 183, 189 / 192`), `0/64` dead ReLUs. However, single-mask Phase A left the gate unseparated (`gate_cos = 0.978–0.980` vs `0.983–0.984` init) and `post_a` Rule B margins high (`7.07–7.46`, `CE_B = 3.59–3.81`). Final `diverse_replay` reached `90%` B in `1/3` seeds (**`177, 162, 157 / 192`** = `81.77%–92.19%`; step-500 delta `-11.28 pp`) while Rule A retention stayed low (**`116, 120, 122 / 192`** = `60.42%–63.54%`). |
| **SSRL-10** *(`L2-Init` $\lambda=0.01$ + `DER++` $\alpha=0.5$, `M=16`)* | `986788d` (Published, `3.10 MB` / `3,255,534 B` at [experiment-results](https://github.com/swarajladke/lifelong-learning/tree/experiment-results/runs/ssrl_10/ssrl_10_l2init_der_986788d3)) \| [ssrl_10_l2init_der_986788d3/results.json](file:///c:/Users/Vicky/Desktop/lifelong-learning/evidence/runs/ssrl_10_l2init_der_986788d3/results.json) | Pre-gates passed (`post_a`: `192/192` in all 3 seeds; `fresh_b`: `188, 192, 192 / 192`), bit-exact match with `SSRL-08` on `fresh_b` and `seq_b`. `DER++` collapsed `M=16` buffer logit drift MSE from `10.1810` (`seq_b`, `argmax 8, 9, 9 / 16`) down to `0.009045` (`diverse_der`, `0.0067, 0.0080, 0.0124`; **`16/16` argmax match** across all 3 seeds) and improved Rule A Seen by `+5.21` to `+6.77 pp` over `SSRL-08` (**`126, 142, 136 / 192`** = `65.62%–73.96%`; `+16.49 pp` mean vs `seq_b`). **Hybrid Result 3 / Seed-104 Result 4:** all 3 seeds still `< 75%` on Rule A Seen and Seed `104` Rule B Seen dipped to `168/192` (`87.50%` vs `93.75%` Seed `102`, `96.35%` Seed `103`). |
| **SSRL-11** *(`L2-Init` $\lambda=0.01$ + `DER++` $\alpha=0.5$, `M=64`)* | `9ad5057` (Published, `3.10 MB` / `3,254,113 B` at [experiment-results](https://github.com/swarajladke/lifelong-learning/tree/experiment-results/runs/ssrl_11/ssrl_11_l2init_der_m64_9ad50574)) \| [ssrl_11_l2init_der_m64_9ad50574/results.json](file:///c:/Users/Vicky/Desktop/lifelong-learning/evidence/runs/ssrl_11/ssrl_11_l2init_der_m64_9ad50574/results.json) | Pre-gates passed (`post_a`: `192/192` all seeds; `fresh_b`: `188, 192, 192 / 192`). Expanding buffer from `M=16` to **`M=64`** solved Rule A forgetting: **`188, 191, 189 / 192`** = **`97.92%–99.48%`** (`1.39 pp` mean forgetting; `+44.97 pp` mean vs `seq_b`; `64/64` buffer argmax match, `0.0614` mean MSE vs `7.8074` on `seq_b`; probe `both_correct` **`73, 67, 57 / 128`**). On Rule B Seen at `500` steps, Seeds `102` (`180/192 = 93.75%`) and `103` (`185/192 = 96.35%`) cleared `>= 90%`, while Seed `104` reached **`166/192` (`86.46%`)** (`seq_b` `171/192 = 89.06%`), giving `2/3 >= 90%` (near-Result-1 breakthrough / Result-2 step-500 trigger). |
| **SSRL-12** *(`L2-Init` $\lambda=0.01$ + `DER++` $\alpha=0.5$, `M=64`, `1,000` B steps)* | `a262763` (Published, `3.43 MB` / `3,595,782 B` at [experiment-results](https://github.com/swarajladke/lifelong-learning/tree/experiment-results/runs/ssrl_12/ssrl_12_l2init_der_m64_b1000_a2627630)) \| [ssrl_12_l2init_der_m64_b1000_a2627630/results.json](file:///c:/Users/Vicky/Desktop/lifelong-learning/evidence/runs/ssrl_12/ssrl_12_l2init_der_m64_b1000_a2627630/results.json) | **RESULT 1 (3/3 SEEDS PASSED BOTH):** Prefix hash & step-`500` milestones matched `SSRL-11` bit-for-bit. Extending Task B to `1,000` steps lifted Rule B Seen to **`192/192` (`100.00%`), `192/192` (`100.00%`), `185/192` (`96.35%`)** (Seed `104` cleared `>= 90%` by step `750` at `181/192 = 94.27%`; mean delta `-1.22 pp`), while retaining Rule A Seen at **`184, 188, 179 / 192` (`93.23%–97.92%`)** (`4.34 pp` mean forgetting; `+43.06 pp` vs `seq_b`; `64/64` buffer match, `0.0486` mean MSE vs `9.1913` `seq_b`; probe `both_correct` **`72, 67, 63 / 128`**). |
| **SSRL-13** *(3-Rule Sequential Stream `R1 -> R2 -> R3`, Bare `6,402` MLP, `1,000` R3 steps)* | `8d4863e` (Published, `4.89 MB` / `5,126,833 B` at [experiment-results](https://github.com/swarajladke/lifelong-learning/tree/experiment-results/runs/ssrl_13/ssrl_13_three_rule_stream_8d4863ef)) \| [ssrl_13_three_rule_stream_8d4863ef/results.json](file:///c:/Users/Vicky/Desktop/lifelong-learning/evidence/runs/ssrl_13/ssrl_13_three_rule_stream_8d4863ef/results.json) | **RESULT 3 TRIGGERED & DIAGNOSED:** Pre-gates passed (`post_r1`: `192/192` all seeds; `fresh_r3`: **`189, 188, 179 / 192` = `93.23%–98.44%`**), and `der_post_r2` matched `SSRL-12` bit-for-bit (`R1`: `184, 188, 179 / 192`; `R2`: `192, 192, 185 / 192`). After adding Phase 3 (`1,000` R3 + `125` R1 + `125` R2 `DER++` steps), `stream_der_r3` pinned both `R1` and `R2` replay buffers (`64/64` argmax match on both rules; mean MSE `0.1020` `R1`, `0.1451` `R2`; `R1` Seen = **`165, 173, 175 / 192`**, `R2` Seen = **`177, 175, 147 / 192`**, probe `joint_correctness_rate` = **`51, 48, 47 / 128`** vs `seq_r3` `11, 5, 17 / 128`), **but `R3` Seen stalled at `130, 133, 124 / 192` (`67.71%, 69.27%, 64.58%`; `-29.86 pp` mean step-1000 delta vs `fresh_r3`; `seq_r3` `133, 155, 142 / 192`)** with `0/64` dead ReLUs. Read-only forensics proved the exact cause: `stream_der_r3` & `seq_r3` fit `366–379 / 384` (`95.3%–98.7%`) on the training pool, yet inactive `R1+R2` distractor coordinates (`2^8 = 256` patterns) overwhelmed held-out `eval_seen` (`192`), whereas zeroing inactive distractor bits restored `R1` to **`192/192`** across all 3 seeds and `seq_r3` `R3` to **`176, 192, 176 / 192`**. |
