# Lifelong Learning — Human Review Board

> **How to use this file:** This is your **one permanent review file** (`AGENTS.md` §5). Every time a decision is finished, I will update this exact file in place—moving finished work into the **Completed Study Log** at the bottom and putting the next proposal here at the top—so you never have to hunt through new proposal files.

---

## Current Status: WAITING FOR YOUR YES / NO (Stage 2: Approve, Commit, Run & Auto-Publish)

**Proposed Next Decision:** **SSRL-09 — Stage 2 (Approve `APPROVED`, Commit SSRL-09 + Governance Docs, Push to `main`, Run Single CPU Batch, and Auto-Publish)**  
**Goal Served:** Break the **Retention vs. Plasticity trade-off** so the same network can keep Rule A ($\ge 90\%$) *and* learn Rule B cleanly ($\ge 90\%$) before scaling to the multi-rule sequence in `PLAN.md`.

### Stage 1 Verification Complete (Done — Zero Production Training Run Yet)
| Item | Verified State |
| :--- | :--- |
| **Model Added** | `SSRLConditionedMLP` in [ssrl_model.py](file:///c:/Users/Vicky/Desktop/lifelong-learning/research/lifelong_01/ssrl_model.py) — **`6,722` parameters** (`+320` / `+5.0%` vs `6,402` baseline); `SSRLInstructionMLP` untouched |
| **Runner Added** | [run_ssrl_conditioned_mlp.py](file:///c:/Users/Vicky/Desktop/lifelong-learning/research/lifelong_01/run_ssrl_conditioned_mlp.py) — **`506` lines** (`< 600` limit); `gate_approval_status = "PROPOSED_PENDING_APPROVAL"` |
| **Allowlists Updated** | 12 `checkpoint_cond_*_s{102,103,104}.pt` entries added to [ssrl_checkpoints.py](file:///c:/Users/Vicky/Desktop/lifelong-learning/research/lifelong_01/ssrl_checkpoints.py) & [publish_results.py](file:///c:/Users/Vicky/Desktop/lifelong-learning/research/lifelong_01/publish_results.py) (`10 MB` limit unchanged) |
| **Model-Free Tests** | `TestSSRLModelFree`: **`61/61` passed** (`1.580s`, Exit `0`, including 3 new SSRL-09 checks) in [test_ssrl_model_free.py](file:///c:/Users/Vicky/Desktop/lifelong-learning/research/lifelong_01/test_ssrl_model_free.py) |
| **CPU Integration Tests** | `TestSSRLCPUIntegration`: **`22/22` passed** (`14.448s`, Exit `0`, including forward/shape, tiny E2E `< 1 MB`, Gate G-LC & deadline rejection) |

### Where We Stand Right Now (Plain English)
Our last three studies mapped a clear deadlock on the plain ReLU MLP (`32 → 64 → 64 → 2`, `6,402` parameters):
1. **Without regularization (`SSRL-06` & `SSRL-07`):** Pattern-diverse replay keeps Rule A strong (**`75.00%–85.42%`**), but Phase A grows large weights (`fc3.W` and Rule-A columns) that lock the network into high-confidence wrong predictions on Rule B (`7.41–8.97` margin). As a result, learning Rule B stalls at **`74.48%–82.81%`** (missing the `90%` bar, while a fresh learner hits `98.44%–100%`).
2. **With `L2-Init` weight pulling (`SSRL-08`):** Pulling weights gently toward their random starting values (`θ₀`) halved those starting Rule B margins and lifted `diverse_replay` Rule B acquisition to **`91.15%–95.83%`** (passing `90%` in all 3 seeds). **However**, pulling Rule A weights back toward random initialization softened stability—Rule A retention dropped from `75.00%–85.42%` down to **`58.85%–68.75%`**.

**Bottom line:** Both tricks on the bare concatenated MLP push a single blanket parameter pool in opposite directions. To advance toward lifelong learning without cheating (we must **not** hand-hardcode `x * mask`), the network needs an architectural way to normalize activations and route features cleanly using the instruction mask `m`.

---

## 1. Exact Plan If You Say `Yes` (SSRL-09 Stage 2 Execution Spec)

### What We Are Testing
**SSRL-09** compares `SSRLConditionedMLP` (**LayerNorm + Learned Mask-Conditioned Hidden Gating**) against the prior bare `SSRLInstructionMLP` results:
- `LayerNorm` on hidden layers 1 and 2 (`ln1`, `ln2`) keeps internal activations and logit scales bounded naturally without pulling weights back toward random `θ₀` (addressing the `SSRL-08` retention loss).
- Learned multiplicative gating on hidden layer 1 (`gate = sigmoid(fc1_gate(m))` modulating `h1 = gate * relu(ln1(fc1_feat(x)))`) lets the network learn *on its own* how to suppress irrelevant input bits and activate task-appropriate hidden units depending on mask `m`—with **zero hardcoded `x * m`**.

### Exact Execution Steps & Resource Bounds (If You Say `Yes`)
1. Flip `"gate_approval_status"` from `"PROPOSED_PENDING_APPROVAL"` to `"APPROVED"` in `run_ssrl_conditioned_mlp.py` and align its guard unit test.
2. Re-run both test suites (`TestSSRLModelFree` `61/61` & `TestSSRLCPUIntegration` `22/22`) and verify `< 600` line limit.
3. Commit the SSRL-09 files (`ssrl_model.py`, `run_ssrl_conditioned_mlp.py`, `ssrl_checkpoints.py`, `publish_results.py`, `test_ssrl_model_free.py`, plus `AGENTS.md` and `REVIEW_BOARD.md`) and push to `origin/main`.
4. Run **1 CPU-only production batch** across Seeds `102, 103, 104` with **4 matched arms per seed**:
   - `FREEZE-AFTER-BASE` (`0` updates on `post_a`, permanent standing control)
   - `fresh_b` (`500` Rule B updates from fresh init)
   - `seq_b` (`500` Rule B updates from `post_a`, zero replay, zeroed Adam moments)
   - `diverse_replay` (`625` total updates at `4 B : 1 A` with `M = 16` pattern-diverse buffer)
5. Auto-publish results (expected `< 4 MB` `< 10 MB` limit) to `experiment-results` and upgrade this `REVIEW_BOARD.md` file in place with the measured results.

### What Stays Fixed
- **Pinned Datasets (zero new data):** Task A (`ssrl_pilot_t1_s101.json`), Task B (`ssrl_pilot_t2_s101.json`), Probe (`ssrl_probe_counterfactual_128.json`).
- **Seeds & Schedules:** Seeds `[102, 103, 104]`; sampler seeds `2026` (A) and `2027` (B); batch size `16`.
- **Budget & Ceilings:** `2,000` Phase A + `500` `fresh_b` + `500` `seq_b` + `625` `diverse_replay` = `3,625` updates/seed (`10,875` updates / `174,000` presentations total); `Adam(lr=0.001, weight_decay=0.0)`; Gate G-LC `>= 90%` (`173/192`) enforced on `post_a` and `fresh_b`; monotonic `60.0s` ceiling per stage; zero retries or tuning.

---

## 2. All Possible Results of This Test (When Executed Later in Stage 2)

> *Per `AGENTS.md` §1.3, these list every possible outcome in advance without predicting which one will happen.*

### Result 1 — Best Case: Breakthrough on Both Retention & Plasticity
* **What the numbers would look like:** `diverse_replay` reaches **$\ge 90\%$ ($173/192$)** on **Rule B Seen** *and* keeps **Rule A Seen** stronger than unregularized `SSRL-06` (`> 85%`, ideally $\ge 90\%$) across all 3 seeds, while narrowing or eliminating the mid-training transfer lag.
* **What it proves:** Separating `x` and `m` through learned normalization and multiplicative gating breaks the Stability-Plasticity deadlock on the parity domain without hardcoding the mask.
* **Next step after Result 1:** Ready to scale to the original **10-rule sequential transfer + unseen-rule composition study (`PLAN.md`)**, where seeing multiple masks should let learned gating generalize to new masks.

### Result 2 — Plasticity Fixed, Retention Still Drops
* **What the numbers would look like:** Both `seq_b` and `diverse_replay` reach **$\ge 90\%$** on Rule B Seen (`SSRL-08`-level plasticity or better, with bounded margins), **but** Rule A retention with `4:1` diverse replay still falls noticeably (e.g., `< 80%`).
* **What it proves:** Normalization + gating prevents the Task A weights from interfering with Task B, *but* `1` replay batch per `4` new batches (`M=16`) is not strong enough to preserve shared readout/hidden weights (`fc2`/`fc3`).
* **Next step after Result 2:** Test an explicit **parameter-stability or replay-strength lever on top of the conditioned architecture** (e.g., `2:1` replay ratio or EWC/consolidation on `fc2`/`fc3`), rather than changing the architecture again.

### Result 3 — Retention Preserved, Negative Transfer Still Persists
* **What the numbers would look like:** Rule A stays high under replay ($\ge 80\%\text{–}90\%$), **but** `seq_b` / `diverse_replay` still stall **`< 90%`** on Rule B (behind `fresh_b`).
* **What it proves:** Single-task Phase A (`2,000` steps seeing **only one constant mask** `[1,1,1,1,0,...]`) gives the mask-gate encoder zero incentive to stay sensitive to unused mask bits `[4:8]`, so the gate saturates closed/constant during Phase A.
* **Next step after Result 3:** Test **multi-rule pre-exposure / mask-diverse warm-up** (teaching 2–3 masks first so the gate actually learns mask-sensitivity before evaluating forward transfer to a new rule).

### Result 4 — Mixed Across Seeds or Pre-Gate Failure (`< 90%` on `post_a` or `fresh_b`)
* **What the numbers would look like:** Either the conditioned model fails the single-rule `90%` gate on `post_a` or `fresh_b`, or results flip direction across seeds `102/103/104`.
* **What it proves:** The new model either fails basic parity learnability under `lr=0.001` / 500–2,000 steps, or behavior is seed-sensitive at $N=3$.
* **Next step after Result 4:** Run immediately halts nonzero (`LearnabilityGateError`), preserves failure diagnostics with **zero automatic tuning or retries**, and comes back here for diagnosis.

---

## 3. Your Decision Needed Now

* Reply **`Yes`** $\rightarrow$ I will approve **Stage 2**, re-verify tests, commit & push the SSRL-09 code + governance docs (`AGENTS.md`, `REVIEW_BOARD.md`) to `main`, run the single CPU-only batch across seeds `102, 103, 104`, auto-publish to `experiment-results`, and upgrade this file with the exact measured outcome.
* Reply **`No`** (or tell me what change you want before running training) $\rightarrow$ I will stop and update this board without running compute or pushing commits.

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
| **SSRL-08** *(`L2-Init`, $\lambda=0.01$)* | `af1553a7` (Published, `3.09 MB`) \| [ssrl_08_l2_init_plasticity_af1553a7/results.json](file:///c:/Users/Vicky/Desktop/lifelong-learning/evidence/runs/ssrl_08_l2_init_plasticity_af1553a7/results.json) | Halved `post_a` Rule B margins (`2.81–4.24`) and boosted `diverse_replay` Rule B Seen to **`91.15%–95.83%`** (passing `>= 90%` in all 3 seeds, cutting step-500 deficit to `-5.38 pp`), **but** Rule A replay retention dropped to **`58.85%–68.75%`** — exposing the Stability-Plasticity trade-off on the shared bare MLP. |
