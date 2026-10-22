# Data Generation Evidence Ledger

**Study:** SSRL-02 / Rule B Generation  
**Target Repository:** `swarajladke/lifelong-learning`  
**Date Generated:** 2026-10-22  
**Execution Environment:** Kaggle Notebook (CPU-only, no accelerator)  

---

## 1. Rule B Dataset Verification

- **Artifact File:** `research/lifelong_01/data/ssrl_pilot_t2_s101.json`
- **Manifest File:** `research/lifelong_01/data/ssrl_pilot_t2_s101_manifest.json`
- **Canonical SHA-256 Byte Hash:** `27e813e73bf66f51e1bff07f7d2f19945e830afc6d5876b0cb4d11136cda4cd4`
- **Raw Evidence Archive:** `evidence/data_generation/ssrl_t2_generated_bundle.zip`
- **Raw Archive SHA-256:** `59eaabc149f58d79ff4f157b83b9a98e9baa1dddad64ff67808128ca6f06e80f`
- **Local Verification Status:** **VERIFIED** on disk.

### Dataset Specifications
- **Rule ID:** 2
- **Rule Coordinates:** `[4, 5, 6, 7]` ($k=4$, $D=16$)
- **Selection Mask:** `[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]`
- **Coordinate Overlap with Rule A:** Exactly 0 shared coordinates
- **Sampling Seed:** 101
- **Partition Seed:** 43
- **Total Population:** 640 examples
  - Training Pool: 384 examples (12 active patterns $\times$ 32 distractor draws)
  - Eval Seen: 192 examples (12 active patterns $\times$ 16 distractor draws)
  - Eval Unseen (Held-out): 64 examples (4 active patterns $\times$ 16 distractor draws)
- **Disjointness:** Full 16-bit input vectors across train, eval_seen, and eval_unseen are strictly disjoint.
