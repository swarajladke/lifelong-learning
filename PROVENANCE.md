# Repository Provenance and Migration Ledger

**Repository:** `lifelong-learning`  
**Date of Migration:** 2026-10-21  
**Authorizing Directives:** LIFELONG-01 Standalone Migration Directive (2026-10-21)

---

## 1. Upstream Lineage

This repository represents an independent, standalone continuation of the lifelong learning and Sequential Symbolic Rule Learning (SSRL) research track extracted from the upstream multi-track repository:

- **Upstream Repository:** `https://github.com/swarajladke/Neural-Networks`
- **Upstream Base Commit:** `e7ecf29864aa585971b7148e35f294f06e2f5dac`
- **Source Snapshot Commit:** `9515e005bc0fbbc8776de8e18421909ea17d26bd` (branch `research/lifelong-01`)
- **Git History Relationship:** This repository initiates a fresh, dedicated Git history rooted at the snapshot commit `9515e00`. The upstream repository (`swarajladke/Neural-Networks`) remains intact, unaltered, and completely independent.

---

## 2. Licensing and Ownership Notice

- **Upstream Licensing Status:** The upstream repository `swarajladke/Neural-Networks` contains no explicit `LICENSE` file.
- **Ownership:** All code, specifications, documentation, and data structures originated by or for `swarajladke`.
- **Policy on Migration:** No new license grant is created during this migration. All existing author notices and attributions are preserved verbatim. Full rights and intellectual property remain reserved under upstream ownership until an explicit license is formally designated by the repository owner.

---

## 3. Migration Note on Preserved Documentation

All foundational documents in `research/lifelong_01/` (`AUDIT.md`, `context.md`, `DECISIONS_REQUIRED.md`, `PLAN.md`, `test_run_evidence_20261019.log`) have been preserved bit-identically from the source snapshot.

> **Important Note for Incoming Agents and Reviewers:**  
> References to prior local paths (such as `C:\Users\Vicky\Desktop\Neural-Networks-lifelong-01` or `/kaggle/working/ssrl_val_Jrrn06vM`) and upstream Git branches (`research/lifelong-01` in `swarajladke/Neural-Networks`) document the upstream environment at the time those artifacts were produced.
>
> All future LIFELONG research, sequence extensions, and verification runs belong strictly to this standalone repository (`lifelong-learning`). Historical evidence documents are preserved for auditability and must not be retroactively modified or assumed to reflect newly executed local runs.

---

## 4. Copied File Integrity Verification

The 14 whitelisted files extracted from commit `9515e005bc0fbbc8776de8e18421909ea17d26bd` match the source snapshot byte-for-byte:

| File Path | SHA-256 Digest | Status |
| :--- | :--- | :---: |
| `research/lifelong_01/AUDIT.md` | `871665d32580944751dca78903844fcb947c398b8c74472e60b07781f5b2026f` | Verified Bit-Identical |
| `research/lifelong_01/context.md` | `deb2c38f0e37c76c826e05bcf85362f27faa8d92f3312b98df9d1e9cfc04bae2` | Verified Bit-Identical |
| `research/lifelong_01/DECISIONS_REQUIRED.md` | `f043a945c19d3e3c40c7cbfe19a73119ff0a609d8e28dfb07e41cf1af553eba1` | Verified Bit-Identical |
| `research/lifelong_01/PLAN.md` | `98a3a16b853913bb18d174b852d1f14fc02ea2551132b4af50e061e25003252f` | Verified Bit-Identical |
| `research/lifelong_01/test_run_evidence_20261019.log` | `4a161faab30ceec736fd2cf35336573eacc1bf0b4b78870e82b8d8e0eeed2bef` | Verified Bit-Identical |
| `research/lifelong_01/run_ssrl_learnability_pilot.py` | `a66177ea85d8a265d47af670ffd076facc1d2f03f7f6c18db271771d8f66638e` | Verified Bit-Identical |
| `research/lifelong_01/ssrl_evaluator.py` | `de45c43bdfeaafad2c8cf53f97db83c01356da517c4470ae4182e7af1893d8dd` | Verified Bit-Identical |
| `research/lifelong_01/ssrl_generator.py` | `575562994cd00d07ee2245d8db1fddce46aa23b171740af5447d7ba01f7d563f` | Verified Bit-Identical |
| `research/lifelong_01/ssrl_model.py` | `662c40f61ddd3b77bed2f244837153605f96f936d4bedf493c7156e85aefd3d9` | Verified Bit-Identical |
| `research/lifelong_01/ssrl_runner.py` | `b576697a3eb3bab735d3535b6b7d4252d16d60bf733e3294fcd8f3c7d11e2bd4` | Verified Bit-Identical |
| `research/lifelong_01/ssrl_types.py` | `d377ada116cab6904c093c3fe6b4acf254380ba2ce5361a959418e63f7e8bdac` | Verified Bit-Identical |
| `research/lifelong_01/test_ssrl_model_free.py` | `b3aba73c2f59a634985598a4030de07eb93cb076eb79206d64422ddf97aa3e84` | Verified Bit-Identical |
| `research/lifelong_01/data/ssrl_pilot_t1_s101.json` | `00a21efad1dc07b826198d99b0b702aedddad57b6226f99301afbf16327fd0b7` | Verified Bit-Identical |
| `research/lifelong_01/data/ssrl_pilot_t1_s101_manifest.json` | `7c3b685b37565477601afb2bacfe883c7ec79add3fba6bc6d244a2cd5daa6a80` | Verified Bit-Identical |

---

## 5. Evidence Inventory and Status

1. **Model-Free Pre-Flight Test Suite:**
   - **Artifact:** `research/lifelong_01/test_run_evidence_20261019.log`
   - **Result:** 20/20 unit and regression tests passed on Kaggle (exit code 0).
   - **Status:** Preserved and verified.

2. **Single-Rule Learnability Pilot Execution:**
   - **Archive:** `evidence/pilot/ssrl_pilot_t1_s101_results_bundle.zip`
   - **SHA-256 Digest:** `9780d34cc30f7a0639ce981671bbc87d22b5c751e62c9e2178f6a7a315579f1b`
   - **Contents:** `results.json`, `pilot_stdout.log`, `pilot_stderr.log`, `unittest_stdout.log`, `unittest_stderr.log`, `pilot_exit_code.txt`.
   - **Findings:** Gate $G\text{-LC}$ PASSED (seen active patterns: 192/192 = 100.00%, threshold $\ge 90.0\%$). Held-out pattern generalization: 28/64 = 43.75% (descriptive baseline). Execution wall-clock: 1.92s.
   - **Status:** Located in workstation Downloads, verified against recorded log output, and preserved in `evidence/pilot/`.
