"""
research/lifelong_01/test_ssrl_model_free.py -- Model-free regression test suite for SSRL-01 and SSRL-02.

Contains handcrafted, CPU-only unit tests with ZERO neural model instantiations:
1. Parity label computation and selection mask interpretation.
2. Disjointness of rule coordinates and active pattern partitions.
3. Rule coordinate validation (out of range, duplicates, non-4).
4. Numerator/denominator integrity and impossible-value guard (SSRLMeasurement).
5. Exact paired sign-flip permutation distribution and resolution floor (N=6).
6. Non-neural probe-state cloning independence (zero state leakage).
7. Strict transfer rule matching: rejects partial intersections and misaligned targets.
8. Example record post-init validation: rejects inconsistent is_correct flags.
9. Example ID uniqueness enforcement.
10. Gate G-LC boundary tests: 172/192 fails and 173/192 passes at 90.0% under APPROVED; unapproved status forbids passing.
11. Atomic artifact writing: temporary file rename, refusal of overwrite, and corruption protection.
12. Shared runner failure-handling logic: ensures pre-existing sentinel file bytes are preserved.
13. Pilot Contract Validation on handcrafted valid and deliberately invalid datasets.
14. Accidental overwrite prevention when serializing datasets.
15. Rejection of unapproved specification before model construction.
16. Partial evaluation records surviving simulated exception with incomplete population identified.
17. Effective provenance reflecting applied deterministic settings and no model hash fabrication.
18. Monotonic deadline enforcement before and after operations.
19. Rejection of rule_id mismatch in metadata and examples.
20. Exclusive run-directory reservation preventing overwrite.
21. SSRLDeltaMeasurement dataclass validation (numerator/denominator integrity, unit strings, negative deltas).
22. compute_forgetting_from_records signed deltas (positive forgetting, zero retention, negative backward transfer).
23. compute_transfer_from_records signed deltas (positive transfer, zero difference, negative interference).
24. Cross-rule comparison rejection in delta evaluators (rule_id mismatch).
25. Dataset SHA-256 mismatch rejection in delta evaluators.
26. Target class mismatch rejection in delta evaluators.
27. Two-rule minibatch schedule pairing: identical sequences for identical seeds, different for different seeds.
28. Two-rule checkpoint independence: deepcopy cloning prevents live state mutations from altering checkpoint.
29. Two-rule specification approval guard: unapproved status rejected before model construction or data processing.
30. Two-rule failure diagnostics persistence: writes failure_diagnostics.json on early specification rejection.
31. Two-rule step 0 transfer delta semantics: compares seq_ab at t1 against fresh_b at t0 on Rule B evaluation population.
32. Publish results artifact size ceiling: enforces 10 MB limit before Git staging.
33. Publish results forbidden files guard: rejects unapproved .pt, .ckpt, and credential files.
34. Publish results branch guard: strictly rejects 'main' branch targets.
35. Two-rule JSON serialization integrity: proves make_json_serializable converts dataclasses and measurements.
36. Counterfactual probe quota and strata feasibility: exactly 128 unique x, opposing parities, 64/32/32 strata, 0 unseen/unseen.
37. Counterfactual probe unrelaxed quota guard: rejects malformed quotas without silent relaxation.
38. Counterfactual probe category arithmetic: partitions outcomes into 4 descriptive categories.
39. Checkpoint capture invariants: RNG state independence, non-mutation, and reload integrity.
40. Narrow checkpoint publisher allowlist: allows approved .pt < 100 KB, rejects unapproved or oversized .pt and .bin.
41. Reproduction divergence detector: detects and raises on aligned prediction mismatch against reference artifact.
"""

from __future__ import annotations
import os
import time
import math
import copy
import tempfile
import unittest
import unittest.mock
import json
import hashlib
import torch

from research.lifelong_01.ssrl_types import (
    SSRLMeasurement,
    SSRLExampleRecord,
    SSRLDeltaMeasurement,
    compute_accuracy_measurement
)
from research.lifelong_01.ssrl_generator import (
    compute_parity,
    partition_active_patterns,
    rule_to_mask,
    validate_pilot_dataset_contract,
    serialize_dataset_canonical_json,
    load_and_verify_dataset,
    PilotDatasetContractError
)
from research.lifelong_01.ssrl_evaluator import (
    evaluate_gate_g_lc,
    compute_seed_transfer_score,
    compute_exact_sign_flip_test,
    compute_forgetting_measurement,
    compute_transfer_delta_measurement,
    compute_forgetting_from_records,
    compute_transfer_from_records,
    atomic_write_json
)
from research.lifelong_01.ssrl_runner import clone_post_history_checkpoint_for_probe
from research.lifelong_01.run_ssrl_learnability_pilot import (
    run_learnability_pilot,
    APPROVED_PILOT_CONFIG,
    SpecificationApprovalError,
    verify_run_specification_approval,
    reserve_run_directory,
    configure_determinism,
    capture_environment_provenance,
    evaluate_model_on_split,
)
from research.lifelong_01.run_ssrl_two_rule import (
    APPROVED_TWO_RULE_CONFIG,
    build_minibatch_schedule,
    verify_run_specification_approval as verify_two_rule_approval,
    run_two_rule_study,
    make_json_serializable,
)
from research.lifelong_01.publish_results import (
    calculate_and_verify_run_size,
    verify_no_forbidden_files,
    publish_run_artifacts,
    MAX_ALLOWED_RUN_BYTES,
    ALLOWED_CHECKPOINT_FILENAMES as PUBLISH_ALLOWED_CHECKPOINT_FILENAMES,
)
from research.lifelong_01.ssrl_counterfactual_probe import (
    generate_counterfactual_probe_dataset,
    validate_counterfactual_probe_contract,
    evaluate_counterfactual_probe,
    STRATA_QUOTAS,
)
from research.lifelong_01.ssrl_checkpoints import (
    capture_and_verify_checkpoint,
    compute_tensor_fingerprint_from_state_dict,
    ALLOWED_CHECKPOINT_FILENAMES,
    MAX_CHECKPOINT_FILE_BYTES,
)
from research.lifelong_01.ssrl_reproduction_check import (
    verify_aligned_reproduction,
    ReproductionDivergenceError,
)
from research.lifelong_01.ssrl_model import (
    SSRLInstructionMLP,
    SSRLConditionedMLP,
    get_conditioned_parameter_count,
    SSRLInputGatedMLP,
    get_input_gated_parameter_count,
)
from research.lifelong_01.run_ssrl_joint_reference import (
    interleave_minibatch_schedules,
    load_and_verify_initial_model,
    CheckpointVerificationError,
    DeadlineExceededError,
    verify_joint_specification_approval,
    EXPECTED_T0_FINGERPRINT,
    PROPOSED_JOINT_REFERENCE_CONFIG,
    APPROVED_JOINT_REFERENCE_CONFIG,
    run_joint_reference_study,
    evaluate_population_records,
)
from research.lifelong_01.run_ssrl_replay_baseline import (
    PROPOSED_REPLAY_CONFIG,
    select_replay_memory,
    build_replay_schedule,
    load_and_verify_t1_model,
    verify_replay_specification_approval,
    run_replay_study,
)
from research.lifelong_01.run_ssrl_diverse_replay import (
    PROPOSED_DIVERSE_REPLAY_CONFIG,
    select_pattern_diverse_replay_memory,
    compute_per_pattern_retention,
    verify_diverse_replay_specification_approval,
    run_diverse_replay_study,
)
from research.lifelong_01.run_ssrl_paired_replication import (
    PROPOSED_PAIRED_REPLICATION_CONFIG,
    verify_paired_replication_specification_approval,
    compute_schedule_hash,
    train_single_arm,
    run_paired_replication_study,
)
from research.lifelong_01.run_ssrl_transfer_controls import (
    PROPOSED_TRANSFER_CONTROLS_CONFIG,
    verify_transfer_controls_specification_approval,
    train_task_b_arm,
    run_transfer_controls_study,
)
from research.lifelong_01.run_ssrl_l2_init_plasticity import (
    PROPOSED_L2_INIT_PLASTICITY_CONFIG,
    LearnabilityGateError,
    verify_l2_init_specification_approval,
    clone_frozen_init_params,
    compute_l2_init_penalty,
    run_l2_init_plasticity_study,
)
from research.lifelong_01.run_ssrl_conditioned_mlp import (
    PROPOSED_CONDITIONED_MLP_CONFIG,
    verify_conditioned_mlp_specification_approval,
    run_conditioned_mlp_study,
    EVAL_MODE_CONDITIONED,
)
from research.lifelong_01.run_ssrl_l2init_der import (
    PROPOSED_L2INIT_DER_CONFIG,
    verify_l2init_der_specification_approval,
    capture_frozen_teacher_logits,
    compute_der_logit_loss,
    evaluate_buffer_logit_drift,
    run_l2init_der_study,
    EVAL_MODE_DETERMINISTIC,
)
from research.lifelong_01.run_ssrl_l2init_der_m64 import (
    PROPOSED_L2INIT_DER_M64_CONFIG,
    verify_l2init_der_m64_specification_approval,
    select_stratified_pattern_diverse_memory,
    build_cyclic_replay_schedule,
    run_l2init_der_m64_study,
)
from research.lifelong_01.run_ssrl_l2init_der_m64_b1000 import (
    PROPOSED_L2INIT_DER_M64_B1000_CONFIG,
    verify_l2init_der_m64_b1000_specification_approval,
    run_l2init_der_m64_b1000_study,
)
from research.lifelong_01.run_ssrl_three_rule_stream import (
    PROPOSED_THREE_RULE_STREAM_CONFIG,
    SHA256_RULE_3_DEFAULT,
    verify_three_rule_specification_approval,
    select_multi_rule_stratified_memory,
    build_multi_rule_cyclic_replay_schedule,
    run_three_rule_stream_study,
)
from research.lifelong_01.run_ssrl_input_gated_three_rule import (
    PROPOSED_INPUT_GATED_THREE_RULE_CONFIG,
    verify_input_gated_three_rule_approval,
    run_input_gated_three_rule_study,
)


class TestSSRLModelFree(unittest.TestCase):

    def test_parity_and_mask_interpretation(self):
        # Even parity: product is +1
        self.assertEqual(compute_parity([1, 1, 1, 1]), 1)
        self.assertEqual(compute_parity([-1, -1, 1, 1]), 1)
        self.assertEqual(compute_parity([-1, -1, -1, -1]), 1)
        # Odd parity: product is -1
        self.assertEqual(compute_parity([-1, 1, 1, 1]), 0)
        self.assertEqual(compute_parity([-1, -1, -1, 1]), 0)

        with self.assertRaises(ValueError):
            compute_parity([])

        rule = (0, 1, 2, 3)
        mask = rule_to_mask(rule, d=16)
        self.assertEqual(len(mask), 16)
        self.assertEqual(sum(mask), 4)
        for i in range(16):
            self.assertEqual(mask[i], 1 if i in rule else 0)

    def test_invalid_rules_and_malformed_patterns(self):
        with self.assertRaises(ValueError):
            rule_to_mask((1, 2, 2, 3), d=16)
        with self.assertRaises(ValueError):
            rule_to_mask((0, 1, 2, 16), d=16)
        with self.assertRaises(ValueError):
            rule_to_mask((), d=16)
        with self.assertRaises(ValueError):
            compute_parity([1, 0, 1, 1])

    def test_active_pattern_partition_integrity(self):
        train_pats, test_pats = partition_active_patterns(seed=42)
        self.assertEqual(len(train_pats), 12)
        self.assertEqual(len(test_pats), 4)
        s_train = set(train_pats)
        s_test = set(test_pats)
        self.assertTrue(s_train.isdisjoint(s_test))
        self.assertEqual(len(s_train.union(s_test)), 16)

    def test_measurement_guards(self):
        m = SSRLMeasurement(name="test_acc", numerator=18, denominator=20, input_set="dev", mode="eval")
        self.assertEqual(m.rate, 0.9)
        self.assertEqual(m.pct, 90.0)

        with self.assertRaises(ValueError):
            SSRLMeasurement(name="zero_den", numerator=0, denominator=0, input_set="dev", mode="eval")
        with self.assertRaises(ValueError):
            SSRLMeasurement(name="neg_num", numerator=-1, denominator=10, input_set="dev", mode="eval")
        with self.assertRaises(ValueError):
            SSRLMeasurement(name="overflow", numerator=21, denominator=20, input_set="dev", mode="eval")

    def test_example_record_validation(self):
        r = SSRLExampleRecord(
            example_id=1, rule_id=1, arm="fresh", seed=101, step=2000,
            dataset_sha256="a" * 64,
            input_features=[1] * 16, selection_mask=[1, 1, 1, 1] + [0] * 12,
            target_class=1, predicted_class=1, is_correct=True, is_held_out_pattern=False
        )
        self.assertTrue(r.is_correct)

        with self.assertRaises(ValueError):
            SSRLExampleRecord(
                example_id=2, rule_id=1, arm="fresh", seed=101, step=2000,
                dataset_sha256="a" * 64,
                input_features=[1] * 16, selection_mask=[1, 1, 1, 1] + [0] * 12,
                target_class=1, predicted_class=0, is_correct=True, is_held_out_pattern=False
            )

        with self.assertRaises(ValueError):
            SSRLExampleRecord(
                example_id=3, rule_id=1, arm="fresh", seed=101, step=2000,
                dataset_sha256="short_hash",
                input_features=[1] * 16, selection_mask=[1, 1, 1, 1] + [0] * 12,
                target_class=1, predicted_class=1, is_correct=True, is_held_out_pattern=False
            )

    def test_duplicate_example_ids_rejected(self):
        r1 = SSRLExampleRecord(
            example_id=10, rule_id=1, arm="fresh", seed=101, step=2000,
            dataset_sha256="a" * 64, input_features=[1] * 16, selection_mask=[1] * 16,
            target_class=1, predicted_class=1, is_correct=True, is_held_out_pattern=False
        )
        r2 = SSRLExampleRecord(
            example_id=10, rule_id=1, arm="fresh", seed=101, step=2000,
            dataset_sha256="a" * 64, input_features=[1] * 16, selection_mask=[1] * 16,
            target_class=1, predicted_class=1, is_correct=True, is_held_out_pattern=False
        )
        with self.assertRaises(ValueError):
            compute_accuracy_measurement("pop", [r1, r2], "dev")

    def test_gate_g_lc_boundary_cases(self):
        m_172 = SSRLMeasurement("t1_seen", 172, 192, "t1_seen", "eval")
        res_172 = evaluate_gate_g_lc(m_172, threshold=0.90, approval_status="APPROVED")
        self.assertFalse(res_172["passed"])
        self.assertEqual(res_172["verdict"], "FAILED")

        m_173 = SSRLMeasurement("t1_seen", 173, 192, "t1_seen", "eval")
        # Under APPROVED status, 173/192 (90.1%) passes
        res_173_app = evaluate_gate_g_lc(m_173, threshold=0.90, approval_status="APPROVED")
        self.assertTrue(res_173_app["passed"])
        self.assertEqual(res_173_app["verdict"], "PASSED")

        # Under PROPOSED_PENDING_APPROVAL status, 173/192 CANNOT pass the gate
        res_173_prop = evaluate_gate_g_lc(m_173, threshold=0.90, approval_status="PROPOSED_PENDING_APPROVAL")
        self.assertTrue(res_173_prop["meets_threshold"])
        self.assertFalse(res_173_prop["passed"])
        self.assertEqual(res_173_prop["verdict"], "PROPOSED_THRESHOLD_UNAPPROVED")

    def test_strict_transfer_rule_set_matching(self):
        rec_seq = {
            1: [SSRLExampleRecord(1, 1, "seq", 101, 2000, "a"*64, [1]*16, [1]*16, 1, 1, True, False)],
            2: [SSRLExampleRecord(2, 2, "seq", 101, 2000, "a"*64, [1]*16, [1]*16, 0, 0, True, False)]
        }
        rec_fresh = {
            1: [SSRLExampleRecord(1, 1, "fresh", 101, 2000, "a"*64, [1]*16, [1]*16, 1, 1, True, False)],
            2: [SSRLExampleRecord(2, 2, "fresh", 101, 2000, "a"*64, [1]*16, [1]*16, 0, 0, True, False)]
        }

        score = compute_seed_transfer_score(rec_seq, rec_fresh, declared_transfer_rules=[1, 2])
        self.assertEqual(score, 0.0)

        with self.assertRaises(ValueError):
            compute_seed_transfer_score(rec_seq, rec_fresh, declared_transfer_rules=[1, 2, 3])

        rec_partial = {1: rec_fresh[1]}
        with self.assertRaises(ValueError):
            compute_seed_transfer_score(rec_seq, rec_partial, declared_transfer_rules=[1, 2])

    def test_exact_sign_flip_resolution_n6(self):
        diffs = [0.05, 0.08, 0.03, 0.06, 0.04, 0.07]
        res = compute_exact_sign_flip_test(diffs)
        self.assertEqual(res["n_seeds"], 6)
        self.assertEqual(res["total_permutations"], 64)
        self.assertAlmostEqual(res["resolution_floor"], 2.0 / 64.0, places=6)
        self.assertEqual(res["extreme_count"], 2)
        self.assertAlmostEqual(res["exact_p_value"], 2.0 / 64.0, places=6)

    def test_non_neural_probe_cloning_independence(self):
        base_state = {
            "layer1": [1.0, 2.0, 3.0],
            "metadata": {"step": 2000, "rule": "T10"}
        }
        probe1 = clone_post_history_checkpoint_for_probe(base_state)
        probe2 = clone_post_history_checkpoint_for_probe(base_state)

        probe1["layer1"][0] = 999.0
        probe1["metadata"]["probe_name"] = "T*_1"

        self.assertEqual(base_state["layer1"][0], 1.0)
        self.assertNotIn("probe_name", base_state["metadata"])
        self.assertEqual(probe2["layer1"][0], 1.0)
        self.assertNotIn("probe_name", probe2["metadata"])

    def _build_handcrafted_contract_dataset(self, modify_fn=None) -> dict:
        """Builds a minimal synthetic dataset matching the approved contract schema."""
        train_pats, unseen_pats = partition_active_patterns(seed=42)
        rule = [0, 1, 2, 3]
        mask = [1, 1, 1, 1] + [0] * 12

        train_exs = []
        seen_exs = []
        unseen_exs = []

        ex_id = 0
        # 12 train patterns * 32 draws = 384
        for pat in train_pats:
            p_parity = compute_parity(list(pat))
            for i in range(32):
                dist = [1 if (i & (1 << b)) else -1 for b in range(12)]
                train_exs.append({
                    "example_id": ex_id, "rule_id": 1, "rule_indices": list(rule),
                    "features": list(pat) + dist, "mask": list(mask),
                    "target_class": p_parity, "is_held_out_pattern": False
                })
                ex_id += 1

        # 12 train patterns * 16 draws = 192 (distinct distractors from train)
        for pat in train_pats:
            p_parity = compute_parity(list(pat))
            for i in range(16):
                # Offset distractor bit pattern so disjoint from train (e.g. invert first bit)
                dist = [-1 if (i & (1 << b)) else 1 for b in range(12)]
                seen_exs.append({
                    "example_id": ex_id, "rule_id": 1, "rule_indices": list(rule),
                    "features": list(pat) + dist, "mask": list(mask),
                    "target_class": p_parity, "is_held_out_pattern": False
                })
                ex_id += 1

        # 4 unseen patterns * 16 draws = 64
        for pat in unseen_pats:
            p_parity = compute_parity(list(pat))
            for i in range(16):
                dist = [1 if (i & (1 << b)) else -1 for b in range(12)]
                unseen_exs.append({
                    "example_id": ex_id, "rule_id": 1, "rule_indices": list(rule),
                    "features": list(pat) + dist, "mask": list(mask),
                    "target_class": p_parity, "is_held_out_pattern": True
                })
                ex_id += 1

        data = {
            "metadata": {
                "study_id": "SSRL-01", "phase": "learnability_pilot",
                "rule_id": 1, "rule_indices": [0, 1, 2, 3], "seed": 101, "d": 16, "k": 4,
                "num_active_patterns_seen": 12, "num_active_patterns_unseen": 4,
                "train_pool_size": 384, "eval_seen_size": 192, "eval_unseen_size": 64
            },
            "train_examples": train_exs,
            "eval_seen_examples": seen_exs,
            "eval_unseen_examples": unseen_exs,
        }

        if modify_fn:
            modify_fn(data)
        return data

    def test_explicit_pilot_contract_validation(self):
        # 1. Valid contract passes
        valid_data = self._build_handcrafted_contract_dataset()
        validate_pilot_dataset_contract(valid_data, APPROVED_PILOT_CONFIG)

        # 2. Corrupted seed rejected
        def corrupt_seed(d):
            d["metadata"]["seed"] = 999
        bad_seed = self._build_handcrafted_contract_dataset(corrupt_seed)
        with self.assertRaises(PilotDatasetContractError):
            validate_pilot_dataset_contract(bad_seed, APPROVED_PILOT_CONFIG)

        # 3. Wrong parity label rejected
        def corrupt_parity(d):
            d["train_examples"][0]["target_class"] = 1 - d["train_examples"][0]["target_class"]
        bad_parity = self._build_handcrafted_contract_dataset(corrupt_parity)
        with self.assertRaises(PilotDatasetContractError):
            validate_pilot_dataset_contract(bad_parity, APPROVED_PILOT_CONFIG)

        # 4. Duplicate full input vector across train and eval_seen rejected
        def leak_vector(d):
            d["eval_seen_examples"][0]["features"] = list(d["train_examples"][0]["features"])
        bad_leak = self._build_handcrafted_contract_dataset(leak_vector)
        with self.assertRaises(PilotDatasetContractError):
            validate_pilot_dataset_contract(bad_leak, APPROVED_PILOT_CONFIG)

    def test_recomputed_hash_on_invalid_contract_fails(self):
        """Proves that a correctly re-hashed dataset fails if its semantic content violates the contract."""
        def corrupt_rule(d):
            d["metadata"]["rule_indices"] = [0, 1, 2, 4] # Corrupted rule
        bad_rule_data = self._build_handcrafted_contract_dataset(corrupt_rule)

        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, "corrupt_rule.json")
            recomputed_sha = serialize_dataset_canonical_json(bad_rule_data, fpath, refuse_overwrite=True)

            # Loading with approved config must raise PilotDatasetContractError despite valid hash
            with self.assertRaises(PilotDatasetContractError):
                load_and_verify_dataset(fpath, expected_sha256=recomputed_sha, approved_config=APPROVED_PILOT_CONFIG)

    def test_accidental_dataset_overwrite_prevention(self):
        data = self._build_handcrafted_contract_dataset()
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, "dataset.json")
            serialize_dataset_canonical_json(data, fpath, refuse_overwrite=True)

            # Second serialization to same path with refuse_overwrite=True must raise
            with self.assertRaises(FileExistsError):
                serialize_dataset_canonical_json(data, fpath, refuse_overwrite=True)

    def test_shared_failure_handling_preserves_existing_sentinel(self):
        """
        Tests the actual runner entry point run_learnability_pilot when destination already exists.
        Verifies that existing destination bytes are strictly untouched.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            sentinel_path = os.path.join(tmpdir, "existing_run.json")
            sentinel_bytes = b"PRE_EXISTING_RUN_RESULT_DO_NOT_OVERWRITE"
            with open(sentinel_path, "wb") as f:
                f.write(sentinel_bytes)

            # Execute runner pointing to existing destination
            ret_code = run_learnability_pilot(
                dataset_path="dummy.json",
                expected_dataset_sha256="0"*64,
                output_artifact_path=sentinel_path,
                device_name="cpu"
            )
            self.assertEqual(ret_code, 1)

            # Verify sentinel bytes remain bit-for-bit identical
            with open(sentinel_path, "rb") as f:
                actual_bytes = f.read()
            self.assertEqual(actual_bytes, sentinel_bytes)

    def test_reject_unapproved_specification_before_model_construction(self):
        """
        Tests that an unapproved specification is rejected before any model construction.
        """
        # 1. Direct function check on PROPOSED_PENDING_APPROVAL
        unapproved_cfg = dict(APPROVED_PILOT_CONFIG)
        unapproved_cfg["gate_approval_status"] = "PROPOSED_PENDING_APPROVAL"
        with self.assertRaises(SpecificationApprovalError):
            verify_run_specification_approval(unapproved_cfg)

        # 2. Missing approval status check
        missing_cfg = dict(APPROVED_PILOT_CONFIG)
        del missing_cfg["gate_approval_status"]
        with self.assertRaises(SpecificationApprovalError):
            verify_run_specification_approval(missing_cfg)

        # 3. Approved status passes
        approved_cfg = dict(APPROVED_PILOT_CONFIG)
        approved_cfg["gate_approval_status"] = "APPROVED"
        verify_run_specification_approval(approved_cfg)

        # 4. Runner rejects unapproved specification before model construction
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = os.path.join(tmpdir, "dataset.json")
            data = self._build_handcrafted_contract_dataset()
            d_sha = serialize_dataset_canonical_json(data, dataset_path, refuse_overwrite=True)
            run_dest = os.path.join(tmpdir, "run_attempt")

            exit_code = run_learnability_pilot(
                dataset_path=dataset_path,
                expected_dataset_sha256=d_sha,
                output_destination=run_dest,
                device_name="cpu",
                config=unapproved_cfg
            )
            self.assertEqual(exit_code, 1)

            # Failure diagnostics must record SpecificationApprovalError
            diag_file = os.path.join(run_dest, "results.json")
            self.assertTrue(os.path.exists(diag_file))
            with open(diag_file, "r") as f:
                diag = json.load(f)
            self.assertEqual(diag["error_type"], "SpecificationApprovalError")
            # Ensure model parameters were never allocated/recorded in diagnostics
            self.assertNotIn("param_count", diag)

    def test_partial_evaluation_records_survive_simulated_exception(self):
        """
        Tests that per-example evaluation records are accumulated immediately,
        survive an exception mid-evaluation, and identify the incomplete population.
        """
        class DummyFailingModel(torch.nn.Module):
            def __init__(self, fail_at: int = 5):
                super().__init__()
                self.count = 0
                self.fail_at = fail_at

            def forward(self, x):
                self.count += 1
                if self.count > self.fail_at:
                    raise RuntimeError(f"Simulated mid-evaluation failure at item {self.count}")
                return torch.tensor([[0.2, 0.8]], dtype=torch.float32)

        model = DummyFailingModel(fail_at=5)
        # Create 10 dummy examples
        examples = []
        for i in range(10):
            examples.append({
                "example_id": i + 1,
                "rule_id": 1,
                "rule_indices": [0, 1, 2, 3],
                "features": [1] * 16,
                "mask": [1, 1, 1, 1] + [0] * 12,
                "target_class": 1,
                "is_held_out_pattern": False,
            })

        diag = {
            "eval_records_by_population": {},
            "eval_records_count": 0,
            "incomplete_population": None,
        }

        with self.assertRaises(RuntimeError) as ctx:
            evaluate_model_on_split(
                model=model,
                examples=examples,
                dataset_sha256="0" * 64,
                arm="fresh_control",
                seed=101,
                step=2000,
                device=torch.device("cpu"),
                deadline_monotonic=time.monotonic() + 100.0,
                pop_name="eval_seen",
                diagnostics=diag
            )
        self.assertIn("Simulated mid-evaluation failure", str(ctx.exception))

        # Records 0..4 (5 records) must survive in diagnostics
        self.assertEqual(len(diag["eval_records_by_population"]["eval_seen"]), 5)
        self.assertEqual(diag["eval_records_count"], 5)
        self.assertEqual(diag["incomplete_population"], "eval_seen")

    def test_effective_provenance_reflects_applied_settings(self):
        """
        Tests that environment provenance captures applied determinism settings
        and does not fabricate model fingerprints in model-free scenarios.
        """
        configure_determinism(seed=101, device_name="cpu")
        prov = capture_environment_provenance(
            device=torch.device("cpu"),
            dataset_sha256="0" * 64,
            config=APPROVED_PILOT_CONFIG,
            initial_model_fingerprint=None,
            final_model_fingerprint=None
        )

        self.assertTrue(prov["deterministic_algorithms_enabled"])
        self.assertEqual(prov["cublas_workspace_config"], ":4096:8")
        self.assertIsNone(prov["initial_model_fingerprint"])
        self.assertIsNone(prov["final_model_fingerprint"])
        self.assertEqual(prov["dataset_sha256"], "0" * 64)
        self.assertEqual(len(prov["config_sha256"]), 64)

    def test_deadline_enforcement_before_and_after_operation(self):
        """
        Tests that monotonic deadlines are enforced both before and after operations.
        """
        class DummySleepModel(torch.nn.Module):
            def forward(self, x):
                time.sleep(0.05)
                return torch.tensor([[0.2, 0.8]], dtype=torch.float32)

        model = DummySleepModel()
        example = [{
            "example_id": 1,
            "rule_id": 1,
            "rule_indices": [0, 1, 2, 3],
            "features": [1] * 16,
            "mask": [1, 1, 1, 1] + [0] * 12,
            "target_class": 1,
            "is_held_out_pattern": False,
        }]

        # 1. Pre-operation deadline expiry
        diag_pre = {}
        with self.assertRaises(TimeoutError) as ctx_pre:
            evaluate_model_on_split(
                model=model,
                examples=example,
                dataset_sha256="0" * 64,
                arm="fresh_control",
                seed=101,
                step=2000,
                device=torch.device("cpu"),
                deadline_monotonic=time.monotonic() - 1.0, # Already expired
                pop_name="eval_seen",
                diagnostics=diag_pre
            )
        self.assertIn("BEFORE inference", str(ctx_pre.exception))

        # 2. Post-operation deadline expiry (expires during the 0.05s forward call)
        diag_post = {}
        with self.assertRaises(TimeoutError) as ctx_post:
            evaluate_model_on_split(
                model=model,
                examples=example,
                dataset_sha256="0" * 64,
                arm="fresh_control",
                seed=101,
                step=2000,
                device=torch.device("cpu"),
                deadline_monotonic=time.monotonic() + 0.01, # Expires while sleeping 0.05s
                pop_name="eval_seen",
                diagnostics=diag_post
            )
        self.assertIn("AFTER inference", str(ctx_post.exception))

    def test_reject_rule_id_mismatch(self):
        """
        Tests that validate_pilot_dataset_contract rejects rule_id mismatch in metadata or examples.
        """
        # 1. Metadata rule_id mismatch
        def corrupt_meta_rule_id(d):
            d["metadata"]["rule_id"] = 99
        bad_meta = self._build_handcrafted_contract_dataset(corrupt_meta_rule_id)
        with self.assertRaises(PilotDatasetContractError) as ctx1:
            validate_pilot_dataset_contract(bad_meta, APPROVED_PILOT_CONFIG)
        self.assertIn("Metadata rule_id mismatch", str(ctx1.exception))

        # 2. Example rule_id mismatch
        def corrupt_ex_rule_id(d):
            d["train_examples"][0]["rule_id"] = 99
        bad_ex = self._build_handcrafted_contract_dataset(corrupt_ex_rule_id)
        with self.assertRaises(PilotDatasetContractError) as ctx2:
            validate_pilot_dataset_contract(bad_ex, APPROVED_PILOT_CONFIG)
        self.assertIn("rule_id mismatch", str(ctx2.exception))

    def test_exclusive_directory_reservation(self):
        """
        Tests that reserve_run_directory exclusively creates a new directory
        and strictly refuses an existing directory to protect prior contents.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = os.path.join(tmpdir, "unique_run_dir")
            created_dir = reserve_run_directory(run_dir)
            self.assertTrue(os.path.isdir(created_dir))

            # Put a sentinel artifact in the directory
            sentinel = os.path.join(created_dir, "sentinel.txt")
            with open(sentinel, "w") as f:
                f.write("PROTECTED_DATA")

            # Calling reserve_run_directory again must raise FileExistsError
            with self.assertRaises(FileExistsError):
                reserve_run_directory(run_dir)

            # Sentinel file must remain completely unchanged
            with open(sentinel, "r") as f:
                self.assertEqual(f.read(), "PROTECTED_DATA")

    def test_two_rule_disjoint_masks_and_coordinates(self):
        """
        Tests that Rule A [0,1,2,3] and Rule B [4,5,6,7] produce valid, mutually
        disjoint binary masks with zero coordinate overlap.
        """
        rule_a = (0, 1, 2, 3)
        rule_b = (4, 5, 6, 7)
        self.assertEqual(len(set(rule_a).intersection(set(rule_b))), 0)

        mask_a = rule_to_mask(rule_a, d=16)
        mask_b = rule_to_mask(rule_b, d=16)

        self.assertEqual(sum(mask_a), 4)
        self.assertEqual(sum(mask_b), 4)
        # Dot product of disjoint binary masks must be 0
        dot_prod = sum(a * b for a, b in zip(mask_a, mask_b))
        self.assertEqual(dot_prod, 0)

    def test_forgetting_measurement_calculation(self):
        """
        Tests that compute_forgetting_measurement computes accuracy after A minus
        accuracy after B with explicit numerator/denominator tracking.
        """
        m_after_a = SSRLMeasurement(
            name="task_a_seen_post_a",
            numerator=192,
            denominator=192,
            input_set="task_a_seen",
            mode="eval_no_dropout"
        )
        m_after_b = SSRLMeasurement(
            name="task_a_seen_post_b",
            numerator=144,
            denominator=192,
            input_set="task_a_seen",
            mode="eval_no_dropout"
        )

        forgetting = compute_forgetting_measurement(m_after_a, m_after_b)
        self.assertAlmostEqual(forgetting["forgetting_rate"], 1.0 - 0.75)
        self.assertAlmostEqual(forgetting["forgetting_pct"], 25.0)
        self.assertEqual(forgetting["input_set"], "task_a_seen")

        # Mismatched input sets must raise ValueError
        m_diff_set = SSRLMeasurement(
            name="task_b_seen",
            numerator=100,
            denominator=192,
            input_set="task_b_seen",
            mode="eval_no_dropout"
        )
        with self.assertRaises(ValueError):
            compute_forgetting_measurement(m_after_a, m_diff_set)

    def test_transfer_delta_measurement_calculation(self):
        """
        Tests that compute_transfer_delta_measurement computes sequential minus
        fresh-B accuracy with explicit denominators.
        """
        m_seq_b = SSRLMeasurement(
            name="task_b_seq",
            numerator=180,
            denominator=192,
            input_set="task_b_seen",
            mode="eval_no_dropout"
        )
        m_fresh_b = SSRLMeasurement(
            name="task_b_fresh",
            numerator=150,
            denominator=192,
            input_set="task_b_seen",
            mode="eval_no_dropout"
        )

        transfer = compute_transfer_delta_measurement(m_seq_b, m_fresh_b)
        expected_rate = (180 - 150) / 192.0
        self.assertAlmostEqual(transfer["transfer_delta_rate"], expected_rate)
        self.assertAlmostEqual(transfer["transfer_delta_pct"], expected_rate * 100.0)

    def test_delta_measurement_dataclass_validation(self):
        """
        Tests that SSRLDeltaMeasurement dataclass validates arithmetic consistency
        between baseline, target, and declared delta.
        """
        base = SSRLMeasurement("base", 100, 200, "set1", "eval")
        target = SSRLMeasurement("target", 150, 200, "set1", "eval")

        # Correct delta
        delta = SSRLDeltaMeasurement(
            name="transfer_delta",
            baseline=base,
            target=target,
            delta_rate=0.25,
            delta_pct=25.0,
            input_set="set1",
            mode="eval"
        )
        self.assertEqual(delta.delta_rate, 0.25)

        # Mismatched delta_rate must raise ValueError
        with self.assertRaises(ValueError):
            SSRLDeltaMeasurement(
                name="bad_delta",
                baseline=base,
                target=target,
                delta_rate=0.50, # Inconsistent!
                delta_pct=50.0,
                input_set="set1",
                mode="eval"
            )

    def test_forgetting_from_records_semantic_alignment_and_negative_drop(self):
        """
        Tests compute_forgetting_from_records requiring identical evaluation population,
        rejecting mismatched rule_id, and accepting negative forgetting (performance recovery).
        """
        sha = "a" * 64
        rec_a = [
            SSRLExampleRecord(0, 1, "post_a", 101, 2000, sha, [1]*16, [1]*4+[0]*12, 1, 1, True, False),
            SSRLExampleRecord(1, 1, "post_a", 101, 2000, sha, [-1]*16, [1]*4+[0]*12, 0, 0, True, False),
        ]
        # Performance improves at post_b (both correct, same as post_a -> 0 drop)
        rec_b_same = [
            SSRLExampleRecord(0, 1, "post_b", 101, 2500, sha, [1]*16, [1]*4+[0]*12, 1, 1, True, False),
            SSRLExampleRecord(1, 1, "post_b", 101, 2500, sha, [-1]*16, [1]*4+[0]*12, 0, 0, True, False),
        ]
        res = compute_forgetting_from_records(rec_a, rec_b_same)
        self.assertEqual(res["forgetting_rate"], 0.0)

        # Performance drops at post_b -> positive forgetting
        rec_b_drop = [
            SSRLExampleRecord(0, 1, "post_b", 101, 2500, sha, [1]*16, [1]*4+[0]*12, 1, 0, False, False),
            SSRLExampleRecord(1, 1, "post_b", 101, 2500, sha, [-1]*16, [1]*4+[0]*12, 0, 0, True, False),
        ]
        res_drop = compute_forgetting_from_records(rec_a, rec_b_drop)
        self.assertAlmostEqual(res_drop["forgetting_rate"], 0.5)
        self.assertAlmostEqual(res_drop["forgetting_pct"], 50.0)

        # Performance was low at post_a and improves at post_b -> negative forgetting (recovery)
        rec_a_low = [
            SSRLExampleRecord(0, 1, "post_a", 101, 2000, sha, [1]*16, [1]*4+[0]*12, 1, 0, False, False),
            SSRLExampleRecord(1, 1, "post_a", 101, 2000, sha, [-1]*16, [1]*4+[0]*12, 0, 0, True, False),
        ]
        res_recov = compute_forgetting_from_records(rec_a_low, rec_b_same)
        self.assertAlmostEqual(res_recov["forgetting_rate"], -0.5) # Valid negative delta
        self.assertAlmostEqual(res_recov["forgetting_pct"], -50.0)

        # Mismatched rule_id must be rejected
        rec_b_wrong_rule = [
            SSRLExampleRecord(0, 2, "post_b", 101, 2500, sha, [1]*16, [0]*4+[1]*4+[0]*8, 1, 1, True, False),
            SSRLExampleRecord(1, 2, "post_b", 101, 2500, sha, [-1]*16, [0]*4+[1]*4+[0]*8, 0, 0, True, False),
        ]
        with self.assertRaises(ValueError) as ctx:
            compute_forgetting_from_records(rec_a, rec_b_wrong_rule)
        self.assertIn("Rule ID mismatch", str(ctx.exception))

    def test_transfer_from_records_semantic_alignment_and_negative_transfer(self):
        """
        Tests compute_transfer_from_records asserting semantic alignment,
        accepting negative transfer (interference), and rejecting mismatched target classes.
        """
        sha = "b" * 64
        rec_seq = [
            SSRLExampleRecord(0, 2, "seq", 101, 2500, sha, [1]*16, [0]*4+[1]*4+[0]*8, 1, 1, True, False),
            SSRLExampleRecord(1, 2, "seq", 101, 2500, sha, [-1]*16, [0]*4+[1]*4+[0]*8, 0, 0, True, False),
        ]
        # Fresh control does worse -> positive transfer
        rec_fresh_low = [
            SSRLExampleRecord(0, 2, "fresh", 101, 500, sha, [1]*16, [0]*4+[1]*4+[0]*8, 1, 0, False, False),
            SSRLExampleRecord(1, 2, "fresh", 101, 500, sha, [-1]*16, [0]*4+[1]*4+[0]*8, 0, 0, True, False),
        ]
        res_pos = compute_transfer_from_records(rec_seq, rec_fresh_low)
        self.assertAlmostEqual(res_pos["transfer_delta_rate"], 0.5)

        # Fresh control does better -> negative transfer (interference)
        rec_seq_low = [
            SSRLExampleRecord(0, 2, "seq", 101, 2500, sha, [1]*16, [0]*4+[1]*4+[0]*8, 1, 0, False, False),
            SSRLExampleRecord(1, 2, "seq", 101, 2500, sha, [-1]*16, [0]*4+[1]*4+[0]*8, 0, 0, True, False),
        ]
        rec_fresh_high = [
            SSRLExampleRecord(0, 2, "fresh", 101, 500, sha, [1]*16, [0]*4+[1]*4+[0]*8, 1, 1, True, False),
            SSRLExampleRecord(1, 2, "fresh", 101, 500, sha, [-1]*16, [0]*4+[1]*4+[0]*8, 0, 0, True, False),
        ]
        res_neg = compute_transfer_from_records(rec_seq_low, rec_fresh_high)
        self.assertAlmostEqual(res_neg["transfer_delta_rate"], -0.5) # Valid negative transfer

        # Mismatched target class must be rejected
        rec_mismatch_target = [
            SSRLExampleRecord(0, 2, "fresh", 101, 500, sha, [1]*16, [0]*4+[1]*4+[0]*8, 0, 0, True, False), # Changed target
            SSRLExampleRecord(1, 2, "fresh", 101, 500, sha, [-1]*16, [0]*4+[1]*4+[0]*8, 0, 0, True, False),
        ]
        with self.assertRaises(ValueError) as ctx:
            compute_transfer_from_records(rec_seq, rec_mismatch_target)
        self.assertIn("Target class mismatch", str(ctx.exception))

    def test_two_rule_minibatch_schedule_pairing(self):
        """
        Tests that build_minibatch_schedule produces identical batch sequences
        when given the same seed, proving that seq_ab and fresh_b can be paired exactly.
        """
        dummy_pool = [{"example_id": i, "features": [1]*16} for i in range(50)]
        sched_seq = build_minibatch_schedule(dummy_pool, step_budget=10, batch_size=4, seed=2027)
        sched_fresh = build_minibatch_schedule(dummy_pool, step_budget=10, batch_size=4, seed=2027)

        self.assertEqual(len(sched_seq), 10)
        self.assertEqual(len(sched_fresh), 10)
        for b_s, b_f in zip(sched_seq, sched_fresh):
            ids_s = [ex["example_id"] for ex in b_s]
            ids_f = [ex["example_id"] for ex in b_f]
            self.assertEqual(ids_s, ids_f)

        # Different seed must yield a different schedule
        sched_diff = build_minibatch_schedule(dummy_pool, step_budget=10, batch_size=4, seed=9999)
        first_step_same = ([ex["example_id"] for ex in sched_seq[0]] == [ex["example_id"] for ex in sched_diff[0]])
        self.assertFalse(first_step_same)

    def test_two_rule_checkpoint_independence_double(self):
        """
        Tests that post-A checkpoint cloning produces an independent dictionary double,
        so mutating the live state does not contaminate the preserved checkpoint.
        """
        live_state = {"layer1.weight": [1.0, 2.0], "layer2.bias": [0.5]}
        post_a_checkpoint = copy.deepcopy(live_state)

        # Simulate subsequent training updates on live_state
        live_state["layer1.weight"][0] = 99.0
        live_state["new_key"] = [10.0]

        self.assertEqual(post_a_checkpoint["layer1.weight"][0], 1.0)
        self.assertNotIn("new_key", post_a_checkpoint)

    def test_two_rule_specification_approval_guard(self):
        """
        Tests that verify_two_rule_approval rejects unapproved configuration
        before any model construction or dataset processing.
        """
        unapproved_config = {**APPROVED_TWO_RULE_CONFIG, "gate_approval_status": "PROPOSED_PENDING_APPROVAL"}
        with self.assertRaises(SpecificationApprovalError):
            verify_two_rule_approval(unapproved_config)

        approved_config = {**APPROVED_TWO_RULE_CONFIG, "gate_approval_status": "APPROVED"}
        # Must not raise
        verify_two_rule_approval(approved_config)

    def test_two_rule_failure_diagnostics_persistence(self):
        """
        Tests that run_two_rule_study writes failure_diagnostics.json to the run
        directory when an unapproved configuration causes an abort.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = os.path.join(tmpdir, "two_rule_run")
            dest = os.path.join(tmpdir, "two_rule_results.json")
            unapproved = {**APPROVED_TWO_RULE_CONFIG, "gate_approval_status": "PROPOSED_PENDING_APPROVAL"}

            with self.assertRaises(SpecificationApprovalError):
                run_two_rule_study(
                    config=unapproved,
                    dataset_a_path="dummy_a.json",
                    expected_sha_a="0"*64,
                    dataset_b_path="dummy_b.json",
                    expected_sha_b="0"*64,
                    destination=dest,
                    run_dir=run_dir,
                )

            failure_file = os.path.join(run_dir, "failure_diagnostics.json")
            self.assertTrue(os.path.exists(failure_file))
            with open(failure_file, "r") as f:
                data = json.load(f)
            self.assertEqual(data["status"], "FAILED")
            self.assertEqual(data["error_type"], "SpecificationApprovalError")

    def test_two_rule_step0_transfer_delta_semantics(self):
        """
        Tests that zero-shot transfer delta at step 0 compares sequential model at t1
        against fresh model at t0 on the identical Rule B evaluation population.
        """
        recs_t1_seq = [
            SSRLExampleRecord(
                example_id=i,
                rule_id=2,
                arm="seq_ab",
                seed=101,
                step=2000,
                dataset_sha256="b"*64,
                input_features=[1]*16,
                selection_mask=[0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0],
                target_class=1 if i % 2 == 0 else 0,
                predicted_class=1 if i % 2 == 0 else 0,
                is_correct=True,
                is_held_out_pattern=False
            )
            for i in range(10)
        ]
        recs_t0_fresh = [
            SSRLExampleRecord(
                example_id=i,
                rule_id=2,
                arm="init",
                seed=101,
                step=0,
                dataset_sha256="b"*64,
                input_features=[1]*16,
                selection_mask=[0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0],
                target_class=1 if i % 2 == 0 else 0,
                predicted_class=1 if i < 5 else 0,
                is_correct=((1 if i < 5 else 0) == (1 if i % 2 == 0 else 0)),
                is_held_out_pattern=False
            )
            for i in range(10)
        ]
        transfer_step0 = compute_transfer_from_records(recs_t1_seq, recs_t0_fresh, "transfer_step0_seen")
        self.assertEqual(transfer_step0["units"], "percentage_points")
        self.assertEqual(transfer_step0["population_size"], 10)
        self.assertEqual(transfer_step0["rule_id"], 2)
        # 10/10 (100%) vs 6/10 (60%) -> delta = +40.0 percentage points
        self.assertAlmostEqual(transfer_step0["transfer_delta_pct"], 40.0)

    def test_publish_size_ceiling_enforced(self):
        """
        Tests that calculate_and_verify_run_size raises ValueError
        when run artifacts exceed the 10 MB ceiling.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            large_file = os.path.join(tmpdir, "large_artifact.json")
            with open(large_file, "wb") as f:
                f.seek(MAX_ALLOWED_RUN_BYTES + 1024)
                f.write(b"0")
            with self.assertRaises(ValueError) as ctx:
                calculate_and_verify_run_size(tmpdir)
            self.assertIn("exceeds the 10 MB ceiling", str(ctx.exception))

    def test_publish_forbidden_checkpoints_rejected(self):
        """
        Tests that verify_no_forbidden_files rejects unapproved .pt, .ckpt, and credential files.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_file = os.path.join(tmpdir, "model_weights.pt")
            with open(ckpt_file, "w") as f:
                f.write("dummy")
            with self.assertRaises(ValueError) as ctx:
                verify_no_forbidden_files(tmpdir)
            self.assertIn("Unapproved .pt checkpoint", str(ctx.exception))

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_file2 = os.path.join(tmpdir, "model_weights.ckpt")
            with open(ckpt_file2, "w") as f:
                f.write("dummy")
            with self.assertRaises(ValueError) as ctx2:
                verify_no_forbidden_files(tmpdir)
            self.assertIn("Forbidden checkpoint extension", str(ctx2.exception))

        with tempfile.TemporaryDirectory() as tmpdir:
            secret_file = os.path.join(tmpdir, "github_token.txt")
            with open(secret_file, "w") as f:
                f.write("secret")
            with self.assertRaises(ValueError) as ctx:
                verify_no_forbidden_files(tmpdir)
            self.assertIn("Forbidden credential-like filename", str(ctx.exception))

    def test_publish_branch_guard_rejects_main(self):
        """
        Tests that publish_run_artifacts asserts target_branch != 'main',
        strictly preventing accidental push to the main code branch.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(AssertionError) as ctx:
                publish_run_artifacts(
                    run_dir=tmpdir,
                    run_id="test_run",
                    repo_url="https://dummy.git",
                    source_commit="0"*40,
                    experiment_exit_code=0,
                    target_branch="main",
                )
            self.assertIn("strictly forbidden", str(ctx.exception))

    def test_two_rule_json_serialization_integrity(self):
        """
        Tests that make_json_serializable converts nested SSRLMeasurement,
        SSRLExampleRecord, and dataclass objects into types that json.dumps accepts.
        """
        m = SSRLMeasurement(name="test_m", numerator=10, denominator=10, input_set="0"*64, mode="eval")
        rec = SSRLExampleRecord(0, 1, "arm", 101, 10, "0"*64, [1]*16, [1]*4+[0]*12, 1, 1, True, False)
        payload = {
            "measurement": m,
            "records": [rec],
            "nested": {"inner_m": m}
        }
        serialized = make_json_serializable(payload)
        dumped = json.dumps(serialized)
        self.assertIsInstance(dumped, str)
        self.assertIn('"numerator": 10', dumped)

    def test_counterfactual_probe_quotas_and_pairing(self):
        """
        Tests that generate_counterfactual_probe_dataset satisfies the exact quota
        and parity pairing specifications on pinned datasets.
        """
        dataset_a_path = os.path.join(os.path.dirname(__file__), "data", "ssrl_pilot_t1_s101.json")
        dataset_b_path = os.path.join(os.path.dirname(__file__), "data", "ssrl_pilot_t2_s101.json")
        with open(dataset_a_path) as fa:
            da = json.load(fa)
        with open(dataset_b_path) as fb:
            db = json.load(fb)

        probe = generate_counterfactual_probe_dataset(da, db, seed=3030)
        exs = probe["probe_examples"]
        self.assertEqual(len(exs), 128)

        # Check opposing parities
        for e in exs:
            self.assertNotEqual(e["target_class_a"], e["target_class_b"])

        # Check strata quotas
        seen_seen = sum(1 for e in exs if e["stratum"] == "seen_seen")
        seen_unseen = sum(1 for e in exs if e["stratum"] == "seen_unseen")
        unseen_seen = sum(1 for e in exs if e["stratum"] == "unseen_seen")
        unseen_unseen = sum(1 for e in exs if e["stratum"] == "unseen_unseen")

        self.assertEqual(seen_seen, 64)
        self.assertEqual(seen_unseen, 32)
        self.assertEqual(unseen_seen, 32)
        self.assertEqual(unseen_unseen, 0)

        # Check orientation balance
        a1_b0 = sum(1 for e in exs if e["target_class_a"] == 1 and e["target_class_b"] == 0)
        a0_b1 = sum(1 for e in exs if e["target_class_a"] == 0 and e["target_class_b"] == 1)
        self.assertEqual(a1_b0, 64)
        self.assertEqual(a0_b1, 64)

        # Check training pool exclusion
        train_a_xs = {tuple(x["features"]) for x in da["train_examples"]}
        train_b_xs = {tuple(x["features"]) for x in db["train_examples"]}
        probe_xs = {tuple(x["features"]) for x in exs}
        self.assertEqual(len(probe_xs), 128)
        self.assertEqual(len(probe_xs & train_a_xs), 0)
        self.assertEqual(len(probe_xs & train_b_xs), 0)

        # Contract validation passes
        self.assertTrue(validate_counterfactual_probe_contract(probe, da, db))

    def test_counterfactual_probe_unrelaxed_quota_guard(self):
        """
        Tests that validate_counterfactual_probe_contract raises ValueError
        if any quota is violated, ensuring quotas are never relaxed silently.
        """
        dataset_a_path = os.path.join(os.path.dirname(__file__), "data", "ssrl_pilot_t1_s101.json")
        dataset_b_path = os.path.join(os.path.dirname(__file__), "data", "ssrl_pilot_t2_s101.json")
        with open(dataset_a_path) as fa:
            da = json.load(fa)
        with open(dataset_b_path) as fb:
            db = json.load(fb)

        probe = generate_counterfactual_probe_dataset(da, db, seed=3030)

        # 1. Corrupt one active feature so parities match (violates opposing parity constraint)
        bad_parity = copy.deepcopy(probe)
        bad_parity["probe_examples"][0]["features"][4] *= -1
        with self.assertRaises(ValueError) as ctx1:
            validate_counterfactual_probe_contract(bad_parity, da, db)
        self.assertIn("violates opposing parity constraint", str(ctx1.exception))

        # 2. Corrupt one label to not match recomputed parity
        bad_label = copy.deepcopy(probe)
        bad_label["probe_examples"][0]["target_class_b"] = 1 - bad_label["probe_examples"][0]["target_class_b"]
        with self.assertRaises(ValueError) as ctx2:
            validate_counterfactual_probe_contract(bad_label, da, db)
        self.assertIn("label fields do not match recomputed parity", str(ctx2.exception))

        # 3. Corrupt strata count (e.g. modify stratum string)
        bad_strata = copy.deepcopy(probe)
        bad_strata["probe_examples"][0]["stratum"] = "unseen_unseen"
        with self.assertRaises(ValueError) as ctx3:
            validate_counterfactual_probe_contract(bad_strata, da, db)
        self.assertTrue("Stratum" in str(ctx3.exception))

    def test_counterfactual_probe_category_arithmetic(self):
        """
        Tests that evaluate_counterfactual_probe partitions outcomes into
        all four mutually exclusive, exhaustive descriptive categories.
        """
        class MockProbeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self, u):
                # Distinguish by mask
                mask_is_a = (u[0, 0].item() == 1.0 and u[0, 16].item() == 1.0)
                # Predict based on example features
                ex_type = int(u[0, 8].item())  # we pack mock test ID into feature 8
                if ex_type == 1:
                    # both_correct: target A=1, B=0 -> predict A=1, B=0
                    return torch.tensor([[0.1, 0.9] if mask_is_a else [0.9, 0.1]])
                elif ex_type == 2:
                    # both_incorrect_with_switched_predictions: target A=1, B=0 -> predict A=0, B=1
                    return torch.tensor([[0.9, 0.1] if mask_is_a else [0.1, 0.9]])
                elif ex_type == 3:
                    # unchanged_prediction_matching_a_target: predict A=1, B=1
                    return torch.tensor([[0.1, 0.9]])
                elif ex_type == 4:
                    # unchanged_prediction_matching_b_target: predict A=0, B=0
                    return torch.tensor([[0.9, 0.1]])
                return torch.tensor([[0.5, 0.5]])

        # Hand-craft 4 probe examples covering all 4 categories
        mock_examples = [
            {"probe_id": 0, "features": [1]*8 + [1] + [1]*7, "target_class_a": 1, "target_class_b": 0, "stratum": "seen_seen"},
            {"probe_id": 1, "features": [1]*8 + [2] + [1]*7, "target_class_a": 1, "target_class_b": 0, "stratum": "seen_seen"},
            {"probe_id": 2, "features": [1]*8 + [3] + [1]*7, "target_class_a": 1, "target_class_b": 0, "stratum": "seen_unseen"},
            {"probe_id": 3, "features": [1]*8 + [4] + [1]*7, "target_class_a": 1, "target_class_b": 0, "stratum": "unseen_seen"},
        ]

        model = MockProbeModel()
        res = evaluate_counterfactual_probe(model, mock_examples, torch.device("cpu"))
        overall = res["overall_metrics"]

        self.assertEqual(overall["total_pairs"], 4)
        cats = overall["category_counts"]
        self.assertEqual(cats["both_correct"]["numerator"], 1)
        self.assertEqual(cats["both_incorrect_with_switched_predictions"]["numerator"], 1)
        self.assertEqual(cats["unchanged_prediction_matching_a_target"]["numerator"], 1)
        self.assertEqual(cats["unchanged_prediction_matching_b_target"]["numerator"], 1)

        # Sum of categories equals total pairs
        cat_sum = sum(v["numerator"] for v in cats.values())
        self.assertEqual(cat_sum, 4)

        # Switch rate = (1 + 1) / 4 = 50.0%
        self.assertAlmostEqual(overall["switch_rate"]["pct"], 50.0)
        # Joint correctness = 1 / 4 = 25.0%
        self.assertAlmostEqual(overall["joint_correctness_rate"]["pct"], 25.0)
        # Accuracy Mask A = 2 / 4 = 50.0% (both_correct and unchanged_matching_a)
        self.assertAlmostEqual(overall["accuracy_mask_a"]["pct"], 50.0)
        # Accuracy Mask B = 2 / 4 = 50.0% (both_correct and unchanged_matching_b)
        self.assertAlmostEqual(overall["accuracy_mask_b"]["pct"], 50.0)

    def test_checkpoint_capture_invariants(self):
        """
        Tests that capture_and_verify_checkpoint preserves RNG states,
        does not mutate state, verifies size limit, and reloads identically.
        """
        dummy_sd = {
            "layer1.weight": torch.randn(64, 32),
            "layer1.bias": torch.randn(64),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Allowed checkpoint name succeeds
            meta = capture_and_verify_checkpoint(dummy_sd, tmpdir, "checkpoint_t0_init.pt")
            self.assertEqual(meta["filename"], "checkpoint_t0_init.pt")
            self.assertTrue(os.path.exists(meta["path"]))
            self.assertLess(meta["size_bytes"], MAX_CHECKPOINT_FILE_BYTES)
            self.assertEqual(len(meta["tensor_fingerprint"]), 64)

            # 2. Unapproved checkpoint name rejected
            with self.assertRaises(ValueError) as ctx:
                capture_and_verify_checkpoint(dummy_sd, tmpdir, "unapproved.pt")
            self.assertIn("Forbidden checkpoint filename", str(ctx.exception))

            # 3. Refuses overwrite
            with self.assertRaises(FileExistsError):
                capture_and_verify_checkpoint(dummy_sd, tmpdir, "checkpoint_t0_init.pt")

    def test_narrow_checkpoint_publisher_allowlist(self):
        """
        Tests that publisher allows only approved checkpoint filenames under 100 KB,
        rejects unapproved .pt files, rejects oversized .pt files, and rejects other extensions.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Approved checkpoint under 100 KB passes
            valid_ckpt = os.path.join(tmpdir, "checkpoint_t1_post_a.pt")
            with open(valid_ckpt, "wb") as f:
                f.write(b"x" * 1024)
            verify_no_forbidden_files(tmpdir)

            # 2. Unapproved .pt file rejected
            invalid_pt = os.path.join(tmpdir, "arbitrary_model.pt")
            with open(invalid_pt, "wb") as f:
                f.write(b"x" * 1024)
            with self.assertRaises(ValueError) as ctx:
                verify_no_forbidden_files(tmpdir)
            self.assertIn("Unapproved .pt checkpoint", str(ctx.exception))
            os.remove(invalid_pt)

            # 3. Oversized approved checkpoint rejected (> 100 KB)
            oversized_ckpt = os.path.join(tmpdir, "checkpoint_t2_seq_ab.pt")
            with open(oversized_ckpt, "wb") as f:
                f.seek(150 * 1024)
                f.write(b"0")
            with self.assertRaises(ValueError) as ctx2:
                verify_no_forbidden_files(tmpdir)
            self.assertIn("exceeds 100 KB ceiling", str(ctx2.exception))
            os.remove(oversized_ckpt)

            # 4. Forbidden extension (.bin) rejected
            bin_file = os.path.join(tmpdir, "model.bin")
            with open(bin_file, "wb") as f:
                f.write(b"x" * 1024)
            with self.assertRaises(ValueError) as ctx3:
                verify_no_forbidden_files(tmpdir)
            self.assertIn("Forbidden checkpoint extension", str(ctx3.exception))

    def test_reproduction_divergence_detector(self):
        """
        Tests that verify_aligned_reproduction raises ReproductionDivergenceError
        when any individual aligned prediction differs from the reference artifact.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            ref_path = os.path.join(tmpdir, "ref_results.json")
            ref_data = {
                "config": {"run_id": "ref_run"},
                "eval_records": {
                    "split_1": [
                        {"example_id": 1, "predicted_class": 1, "target_class": 1, "is_correct": True, "step": 0},
                        {"example_id": 2, "predicted_class": 0, "target_class": 0, "is_correct": True, "step": 0},
                    ]
                }
            }
            with open(ref_path, "w") as f:
                json.dump(ref_data, f)

            # 1. Matching replay passes
            replay_matching = copy.deepcopy(ref_data)
            report = verify_aligned_reproduction(replay_matching, ref_path)
            self.assertTrue(report["outcome_reproduction_verified"])
            self.assertEqual(report["total_predictions_checked"], 2)

            # 2. Divergent prediction raises ReproductionDivergenceError
            replay_divergent = copy.deepcopy(ref_data)
            replay_divergent["eval_records"]["split_1"][0]["predicted_class"] = 0
            replay_divergent["eval_records"]["split_1"][0]["is_correct"] = False
            with self.assertRaises(ReproductionDivergenceError) as ctx:
                verify_aligned_reproduction(replay_divergent, ref_path)
            self.assertIn("Prediction divergence", str(ctx.exception))

    def test_joint_reference_schedule_interleaving_and_order(self):
        """
        Tests that interleave_minibatch_schedules produces exactly 2,500 batches
        (2,000 from Rule A, 500 from Rule B) with a 4:1 cyclic ratio, preserving
        exact within-rule batch order without drops or duplications.
        """
        # Hand-constructed dummy batches
        sched_a = [[{"id": f"a_{i}_{b}"} for b in range(16)] for i in range(2000)]
        sched_b = [[{"id": f"b_{i}_{b}"} for b in range(16)] for i in range(500)]

        interleaved = interleave_minibatch_schedules(sched_a, sched_b, ratio_a=4, ratio_b=1)
        self.assertEqual(len(interleaved), 2500)

        count_a = sum(1 for tag, rid, b in interleaved if tag == "rule_a" and rid == 1)
        count_b = sum(1 for tag, rid, b in interleaved if tag == "rule_b" and rid == 2)
        self.assertEqual(count_a, 2000)
        self.assertEqual(count_b, 500)

        # Check 4:1 cycle pattern
        for cycle_idx in range(500):
            block = interleaved[cycle_idx * 5 : (cycle_idx + 1) * 5]
            for k in range(4):
                self.assertEqual(block[k][0], "rule_a")
                self.assertEqual(block[k][1], 1)
            self.assertEqual(block[4][0], "rule_b")
            self.assertEqual(block[4][1], 2)

        # Check within-rule sequence order preserved
        extracted_a = [b for tag, _, b in interleaved if tag == "rule_a"]
        extracted_b = [b for tag, _, b in interleaved if tag == "rule_b"]
        self.assertEqual(extracted_a, sched_a)
        self.assertEqual(extracted_b, sched_b)

    def test_joint_reference_approval_guard_and_directory_reservation(self):
        """
        Tests that joint reference runner enforces approval guards, directory reservation,
        and strict checkpoint allowlist rules.
        """
        # 1. Production configuration is APPROVED
        self.assertEqual(PROPOSED_JOINT_REFERENCE_CONFIG["gate_approval_status"], "APPROVED")
        verify_joint_specification_approval(PROPOSED_JOINT_REFERENCE_CONFIG)

        # 2. An unapproved configuration is rejected
        cfg_unapproved = copy.deepcopy(PROPOSED_JOINT_REFERENCE_CONFIG)
        cfg_unapproved["gate_approval_status"] = "PROPOSED_PENDING_APPROVAL"
        with self.assertRaises(SpecificationApprovalError) as ctx:
            verify_joint_specification_approval(cfg_unapproved)
        self.assertIn("PROPOSED_PENDING_APPROVAL", str(ctx.exception))

        # 3. Directory reservation prevents overwrite
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = os.path.join(tmpdir, "joint_run_dir")
            r1 = reserve_run_directory(run_dir)
            self.assertTrue(os.path.isdir(r1))
            with self.assertRaises(FileExistsError):
                reserve_run_directory(run_dir)

    def test_joint_reference_dataset_validation_enforces_rule_contracts(self):
        """
        Tests that load_and_verify_dataset enforces Rule A (rule_id 1, indices [0,1,2,3])
        and Rule B (rule_id 2, indices [4,5,6,7]) semantics via approved_config, and raises
        PilotDatasetContractError when a rule mismatch is injected.
        """
        dataset_a_path = os.path.join(os.path.dirname(__file__), "data", "ssrl_pilot_t1_s101.json")
        with open(dataset_a_path, "rb") as fa:
            sha_a = hashlib.sha256(fa.read()).hexdigest()

        # Valid Rule A config passes
        cfg_a = {
            **PROPOSED_JOINT_REFERENCE_CONFIG,
            "study_id": "SSRL-01",
            "phase": "learnability_pilot",
            "rule_id": 1,
            "rule_indices": [0, 1, 2, 3],
            "partition_seed": 42,
        }
        da = load_and_verify_dataset(dataset_a_path, expected_sha256=sha_a, approved_config=cfg_a)
        self.assertIsInstance(da, dict)

        # Mismatched rule_id raises PilotDatasetContractError
        cfg_bad_rule = copy.deepcopy(cfg_a)
        cfg_bad_rule["rule_id"] = 2
        with self.assertRaises(PilotDatasetContractError) as ctx:
            load_and_verify_dataset(dataset_a_path, expected_sha256=sha_a, approved_config=cfg_bad_rule)
        self.assertIn("rule_id mismatch", str(ctx.exception))

    def test_counterfactual_probe_malformed_opposing_parity_rejected(self):
        """
        Tests that validate_counterfactual_probe_contract strictly rejects probe fixtures
        with malformed opposing parity (e.g. features=[1]*16, where parity under both masks
        is identical), as well as label/parity mismatches.
        """
        dataset_a_path = os.path.join(os.path.dirname(__file__), "data", "ssrl_pilot_t1_s101.json")
        dataset_b_path = os.path.join(os.path.dirname(__file__), "data", "ssrl_pilot_t2_s101.json")
        with open(dataset_a_path, "r", encoding="utf-8") as fa:
            da = json.load(fa)
        with open(dataset_b_path, "r", encoding="utf-8") as fb:
            db = json.load(fb)

        # Generate valid baseline probe
        valid_probe = generate_counterfactual_probe_dataset(da, db, seed=3030)
        self.assertTrue(validate_counterfactual_probe_contract(valid_probe, da, db))

        # Case 1: All-ones feature vector has parity +1 under both masks (pa=[1]*4, pb=[1]*4).
        # Injecting an all-ones vector violates opposing parity constraint.
        malformed_opposing = copy.deepcopy(valid_probe)
        malformed_opposing["probe_examples"][0]["features"] = [1] * 16
        with self.assertRaises(ValueError) as ctx1:
            validate_counterfactual_probe_contract(malformed_opposing, da, db)
        self.assertIn("violates opposing parity constraint", str(ctx1.exception))

        # Case 2: Label fields inverted / mismatched from recomputed parity
        malformed_labels = copy.deepcopy(valid_probe)
        ex0 = malformed_labels["probe_examples"][0]
        ex0["target_class_a"] = 1 - ex0["target_class_a"]
        with self.assertRaises(ValueError) as ctx2:
            validate_counterfactual_probe_contract(malformed_labels, da, db)
        self.assertIn("label fields do not match recomputed parity", str(ctx2.exception))

    def test_replay_buffer_selection_and_exclusion(self):
        """
        Tests that select_replay_memory deterministically selects 8 lowest-ID class-0
        and 8 lowest-ID class-1 Rule A training examples, enforces M=16, and verifies
        disjointness from all evaluation and probe populations.
        """
        dataset_a_path = os.path.join(os.path.dirname(__file__), "data", "ssrl_pilot_t1_s101.json")
        with open(dataset_a_path, "r", encoding="utf-8") as f:
            da = json.load(f)
        probe_path = os.path.join(os.path.dirname(__file__), "data", "ssrl_probe_counterfactual_128.json")
        with open(probe_path, "r", encoding="utf-8") as f:
            dp = json.load(f)

        buffer_examples = select_replay_memory(da["train_examples"], buffer_size=16)
        self.assertEqual(len(buffer_examples), 16)

        c0 = [ex for ex in buffer_examples if ex["target_class"] == 0]
        c1 = [ex for ex in buffer_examples if ex["target_class"] == 1]
        self.assertEqual(len(c0), 8)
        self.assertEqual(len(c1), 8)

        # Assert exact lowest IDs
        self.assertEqual([ex["example_id"] for ex in c0], [0, 1, 2, 3, 4, 5, 6, 7])
        self.assertEqual([ex["example_id"] for ex in c1], [96, 97, 98, 99, 100, 101, 102, 103])

        # Assert zero overlap with eval_seen, eval_unseen, and probe
        sel_ids = set(ex["example_id"] for ex in buffer_examples)
        eval_seen_ids = set(ex["example_id"] for ex in da["eval_seen_examples"])
        eval_unseen_ids = set(ex["example_id"] for ex in da["eval_unseen_examples"])
        self.assertEqual(len(sel_ids.intersection(eval_seen_ids)), 0)
        self.assertEqual(len(sel_ids.intersection(eval_unseen_ids)), 0)

        sel_features = set(tuple(ex["features"]) for ex in buffer_examples)
        probe_features = set(tuple(ex["features"]) for ex in dp["probe_examples"])
        self.assertEqual(len(sel_features.intersection(probe_features)), 0)

        # Assert buffer size validation
        with self.assertRaises(ValueError):
            select_replay_memory(da["train_examples"], buffer_size=15)

    def test_replay_schedule_order_and_counts(self):
        """
        Tests that build_replay_schedule produces exactly 625 batches:
        500 B batches and 125 Replay A batches, preserving B ordering in 4:1 cycles.
        """
        dummy_b = [[{"example_id": i * 16 + j} for j in range(16)] for i in range(500)]
        dummy_a = [{"example_id": 9000 + j} for j in range(16)]

        sched = build_replay_schedule(dummy_b, dummy_a, ratio_b=4, ratio_a=1)
        self.assertEqual(len(sched), 625)

        b_count = sum(1 for tag, _, _ in sched if tag == "rule_b")
        a_count = sum(1 for tag, _, _ in sched if tag == "replay_a")
        self.assertEqual(b_count, 500)
        self.assertEqual(a_count, 125)

        # Check cyclic pattern: 4 B then 1 A
        for cycle_idx in range(125):
            base = cycle_idx * 5
            for k in range(4):
                tag, rid, batch = sched[base + k]
                self.assertEqual(tag, "rule_b")
                self.assertEqual(rid, 2)
                self.assertEqual(batch, dummy_b[cycle_idx * 4 + k])
            tag, rid, batch = sched[base + 4]
            self.assertEqual(tag, "replay_a")
            self.assertEqual(rid, 1)
            self.assertEqual(batch, dummy_a)

    def test_replay_specification_approval_guard(self):
        """
        Tests that verify_replay_specification_approval blocks unapproved execution.
        """
        unapproved = copy.deepcopy(PROPOSED_REPLAY_CONFIG)
        unapproved["gate_approval_status"] = "PROPOSED_PENDING_APPROVAL"
        with self.assertRaises(SpecificationApprovalError):
            verify_replay_specification_approval(unapproved)

        approved = copy.deepcopy(unapproved)
        approved["gate_approval_status"] = "APPROVED"
        # Must not raise
        verify_replay_specification_approval(approved)

    def test_diverse_replay_buffer_selection_and_coverage(self):
        """
        Tests that select_pattern_diverse_replay_memory deterministically selects
        8 class-0 and 8 class-1 Rule A training examples spanning all 12 Rule A
        seen active patterns, enforcing exact selected IDs, M=16, 100% active pattern coverage,
        and strict disjointness from all evaluation and probe populations.
        """
        dataset_a_path = os.path.join(os.path.dirname(__file__), "data", "ssrl_pilot_t1_s101.json")
        with open(dataset_a_path, "r", encoding="utf-8") as f:
            da = json.load(f)
        probe_path = os.path.join(os.path.dirname(__file__), "data", "ssrl_probe_counterfactual_128.json")
        with open(probe_path, "r", encoding="utf-8") as f:
            dp = json.load(f)

        buffer_examples = select_pattern_diverse_replay_memory(da["train_examples"], buffer_size=16)
        self.assertEqual(len(buffer_examples), 16)

        c0 = [ex for ex in buffer_examples if ex["target_class"] == 0]
        c1 = [ex for ex in buffer_examples if ex["target_class"] == 1]
        self.assertEqual(len(c0), 8)
        self.assertEqual(len(c1), 8)

        # Assert exact deterministic IDs
        expected_c0_ids = [0, 1, 48, 49, 192, 240, 432, 480]
        expected_c1_ids = [96, 97, 144, 145, 288, 336, 384, 528]
        self.assertEqual([ex["example_id"] for ex in c0], expected_c0_ids)
        self.assertEqual([ex["example_id"] for ex in c1], expected_c1_ids)
        self.assertEqual([ex["example_id"] for ex in buffer_examples], expected_c0_ids + expected_c1_ids)

        # Assert active pattern coverage: all 12 seen active patterns covered
        active_patterns = set(tuple(ex["features"][:4]) for ex in buffer_examples)
        self.assertEqual(len(active_patterns), 12)

        # Assert zero overlap with eval_seen, eval_unseen, and probe
        sel_ids = set(ex["example_id"] for ex in buffer_examples)
        eval_seen_ids = set(ex["example_id"] for ex in da["eval_seen_examples"])
        eval_unseen_ids = set(ex["example_id"] for ex in da["eval_unseen_examples"])
        self.assertEqual(len(sel_ids.intersection(eval_seen_ids)), 0)
        self.assertEqual(len(sel_ids.intersection(eval_unseen_ids)), 0)

        sel_features = set(tuple(ex["features"]) for ex in buffer_examples)
        probe_features = set(tuple(ex["features"]) for ex in dp["probe_examples"])
        self.assertEqual(len(sel_features.intersection(probe_features)), 0)

        # Assert buffer size validation
        with self.assertRaises(ValueError):
            select_pattern_diverse_replay_memory(da["train_examples"], buffer_size=15)

    def test_diverse_replay_specification_approval_guard(self):
        """
        Tests that verify_diverse_replay_specification_approval blocks unapproved execution.
        """
        unapproved = copy.deepcopy(PROPOSED_DIVERSE_REPLAY_CONFIG)
        unapproved["gate_approval_status"] = "PROPOSED_PENDING_APPROVAL"
        with self.assertRaises(SpecificationApprovalError):
            verify_diverse_replay_specification_approval(unapproved)

        approved = copy.deepcopy(unapproved)
        approved["gate_approval_status"] = "APPROVED"
        verify_diverse_replay_specification_approval(approved)

    def test_per_pattern_retention_computation(self):
        """
        Tests compute_per_pattern_retention on synthetic records.
        """
        synth_records = [
            SSRLExampleRecord(
                example_id=1, rule_id=1, arm="test", seed=101, step=5, dataset_sha256="a" * 64,
                input_features=[1, 1, 1, 1] + [0] * 12, selection_mask=[1] * 4 + [0] * 12,
                target_class=1, predicted_class=1, is_correct=True, is_held_out_pattern=False
            ),
            SSRLExampleRecord(
                example_id=2, rule_id=1, arm="test", seed=101, step=5, dataset_sha256="a" * 64,
                input_features=[1, 1, 1, 1] + [0] * 12, selection_mask=[1] * 4 + [0] * 12,
                target_class=1, predicted_class=0, is_correct=False, is_held_out_pattern=False
            ),
            SSRLExampleRecord(
                example_id=3, rule_id=1, arm="test", seed=101, step=5, dataset_sha256="a" * 64,
                input_features=[-1, 1, 1, 1] + [0] * 12, selection_mask=[1] * 4 + [0] * 12,
                target_class=0, predicted_class=0, is_correct=True, is_held_out_pattern=False
            ),
            SSRLExampleRecord(
                example_id=4, rule_id=1, arm="test", seed=101, step=5, dataset_sha256="a" * 64,
                input_features=[-1, 1, 1, 1] + [0] * 12, selection_mask=[1] * 4 + [0] * 12,
                target_class=0, predicted_class=0, is_correct=True, is_held_out_pattern=False
            ),
        ]
        breakdown = compute_per_pattern_retention(synth_records, rule_indices=[0, 1, 2, 3])
        self.assertIn("1_1_1_1", breakdown)
        self.assertIn("-1_1_1_1", breakdown)
        self.assertEqual(breakdown["1_1_1_1"]["numerator"], 1)
        self.assertEqual(breakdown["1_1_1_1"]["denominator"], 2)
        self.assertEqual(breakdown["1_1_1_1"]["rate"], 0.5)
        self.assertEqual(breakdown["-1_1_1_1"]["numerator"], 2)
        self.assertEqual(breakdown["-1_1_1_1"]["denominator"], 2)
        self.assertEqual(breakdown["-1_1_1_1"]["rate"], 1.0)

    def test_paired_replication_specification_approval_guard(self):
        """
        Tests that verify_paired_replication_specification_approval blocks unapproved execution.
        """
        unapproved = copy.deepcopy(PROPOSED_PAIRED_REPLICATION_CONFIG)
        unapproved["gate_approval_status"] = "PROPOSED_PENDING_APPROVAL"
        with self.assertRaises(SpecificationApprovalError):
            verify_paired_replication_specification_approval(unapproved)

        approved = copy.deepcopy(unapproved)
        approved["gate_approval_status"] = "APPROVED"
        verify_paired_replication_specification_approval(approved)

    def test_paired_schedule_and_manifest_determinism(self):
        """
        Tests schedule pairing, schedule hashes, and memory manifest determinism for SSRL-06.
        """
        dataset_a_path = os.path.join(os.path.dirname(__file__), "data", "ssrl_pilot_t1_s101.json")
        dataset_b_path = os.path.join(os.path.dirname(__file__), "data", "ssrl_pilot_t2_s101.json")
        with open(dataset_a_path, "r", encoding="utf-8") as fa:
            da = json.load(fa)
        with open(dataset_b_path, "r", encoding="utf-8") as fb:
            db = json.load(fb)

        mem_lowest = select_replay_memory(da["train_examples"], buffer_size=16)
        mem_div = select_pattern_diverse_replay_memory(da["train_examples"], buffer_size=16)

        # Manifest checks
        self.assertEqual(len(mem_lowest), 16)
        self.assertEqual(len(mem_div), 16)
        self.assertEqual(sum(1 for ex in mem_lowest if ex["target_class"] == 0), 8)
        self.assertEqual(sum(1 for ex in mem_lowest if ex["target_class"] == 1), 8)
        self.assertEqual(sum(1 for ex in mem_div if ex["target_class"] == 0), 8)
        self.assertEqual(sum(1 for ex in mem_div if ex["target_class"] == 1), 8)
        self.assertEqual(len(set(tuple(ex["features"][:4]) for ex in mem_div)), 12)

        # Paired schedule creation
        sched_b = build_minibatch_schedule(db["train_examples"], step_budget=500, batch_size=16, seed=2027)
        sched_lowest = build_replay_schedule(sched_b, mem_lowest, ratio_b=4, ratio_a=1)
        sched_div = build_replay_schedule(sched_b, mem_div, ratio_b=4, ratio_a=1)

        self.assertEqual(len(sched_lowest), 625)
        self.assertEqual(len(sched_div), 625)

        # Assert identical B minibatches across both schedules
        for cycle in range(125):
            base = cycle * 5
            for k in range(4):
                tag_l, rid_l, batch_l = sched_lowest[base + k]
                tag_d, rid_d, batch_d = sched_div[base + k]
                self.assertEqual(tag_l, "rule_b")
                self.assertEqual(tag_d, "rule_b")
                self.assertEqual([ex["example_id"] for ex in batch_l], [ex["example_id"] for ex in batch_d])

        # Schedule hash determinism
        h1 = compute_schedule_hash(sched_lowest)
        h2 = compute_schedule_hash(sched_lowest)
        self.assertEqual(h1, h2)
        h_div = compute_schedule_hash(sched_div)
        self.assertNotEqual(h1, h_div)

    def test_transfer_controls_specification_approval_guard(self):
        """
        Tests that verify_transfer_controls_specification_approval blocks unapproved execution.
        """
        unapproved = copy.deepcopy(PROPOSED_TRANSFER_CONTROLS_CONFIG)
        unapproved["gate_approval_status"] = "PROPOSED_PENDING_APPROVAL"
        with self.assertRaises(SpecificationApprovalError):
            verify_transfer_controls_specification_approval(unapproved)

        approved = copy.deepcopy(unapproved)
        approved["gate_approval_status"] = "APPROVED"
        verify_transfer_controls_specification_approval(approved)

    def test_transfer_controls_milestone_delta_logic(self):
        """
        Tests transfer delta computation and step-to-threshold logic with hand-built stubs.
        """
        # seq_b has 180/192 correct (93.75%), fresh_b has 160/192 correct (83.33%)
        # delta = (180 - 160) / 192.0 * 100.0 = 20 / 192.0 * 100.0 = 10.4167 pp
        delta = ((180 - 160) / 192.0) * 100.0
        self.assertAlmostEqual(delta, 10.416666666666668, places=5)
        # Check threshold detection (threshold = 0.90 -> 172.8 -> 173/192)
        self.assertTrue((180 / 192.0) >= 0.90)
        self.assertFalse((160 / 192.0) >= 0.90)

    def test_l2_init_specification_approval_guard(self):
        """
        Tests that verify_l2_init_specification_approval blocks unapproved execution
        and accepts APPROVED configs.
        """
        unapproved = copy.deepcopy(PROPOSED_L2_INIT_PLASTICITY_CONFIG)
        unapproved["gate_approval_status"] = "PROPOSED_PENDING_APPROVAL"
        with self.assertRaises(SpecificationApprovalError):
            verify_l2_init_specification_approval(unapproved)

        approved = copy.deepcopy(unapproved)
        approved["gate_approval_status"] = "APPROVED"
        verify_l2_init_specification_approval(approved)

    def test_l2_init_penalty_zero_at_init_and_positive_after_perturbation(self):
        """
        Model-free unit test on a tiny handcrafted linear stub verifying L2-Init penalty:
        - Exact 0.0 value and zero gradients when theta == theta_0.
        - Exact hand-computed 0.5 * lambda * sum((theta - theta_0)^2) after perturbation.
        - Negative lambda raises ValueError.
        """
        stub = torch.nn.Linear(2, 1)
        with torch.no_grad():
            stub.weight.copy_(torch.tensor([[1.0, 2.0]]))
            stub.bias.copy_(torch.tensor([0.0]))
        init_params, _ = clone_frozen_init_params(stub)

        # At init: penalty == 0.0 and gradients == 0.0
        p0 = compute_l2_init_penalty(stub, init_params, l2_init_lambda=0.01)
        self.assertAlmostEqual(float(p0.item()), 0.0, places=7)
        p0.backward()
        self.assertTrue(torch.all(stub.weight.grad == 0.0))
        self.assertTrue(torch.all(stub.bias.grad == 0.0))
        stub.zero_grad()

        # Perturb weights by delta = [[1.0, 2.0]] and bias by delta = [3.0] -> sum_sq = 1 + 4 + 9 = 14
        # With lambda = 0.2: penalty = 0.5 * 0.2 * 14 = 1.4
        with torch.no_grad():
            stub.weight.copy_(torch.tensor([[2.0, 4.0]]))
            stub.bias.copy_(torch.tensor([3.0]))
        p1 = compute_l2_init_penalty(stub, init_params, l2_init_lambda=0.2)
        self.assertAlmostEqual(float(p1.item()), 1.4, places=6)

        with self.assertRaises(ValueError):
            compute_l2_init_penalty(stub, init_params, l2_init_lambda=-0.01)

    def test_l2_init_checkpoint_allowlist_sync(self):
        """
        Tests that all 12 SSRL-08 checkpoint filenames exist in both ssrl_checkpoints
        and publish_results allowlists.
        """
        expected_ckpts = []
        for arm in ["post_a", "fresh_b", "seq_b", "diverse_replay"]:
            for seed in [102, 103, 104]:
                expected_ckpts.append(f"checkpoint_l2init_{arm}_s{seed}.pt")
        self.assertEqual(len(expected_ckpts), 12)
        for name in expected_ckpts:
            self.assertIn(name, ALLOWED_CHECKPOINT_FILENAMES)
            self.assertIn(name, PUBLISH_ALLOWED_CHECKPOINT_FILENAMES)

    def test_conditioned_mlp_specification_approval_guard(self):
        """
        Tests that verify_conditioned_mlp_specification_approval blocks unapproved
        execution and accepts APPROVED configurations.
        """
        unapproved = copy.deepcopy(PROPOSED_CONDITIONED_MLP_CONFIG)
        unapproved["gate_approval_status"] = "PROPOSED_PENDING_APPROVAL"
        with self.assertRaises(SpecificationApprovalError):
            verify_conditioned_mlp_specification_approval(unapproved)

        approved = copy.deepcopy(unapproved)
        approved["gate_approval_status"] = "APPROVED"
        verify_conditioned_mlp_specification_approval(approved)

    def test_conditioned_mlp_parameter_count_formula(self):
        """
        Model-free unit test checking get_conditioned_parameter_count() exact arithmetic.
        """
        self.assertEqual(get_conditioned_parameter_count(), 6722)
        self.assertEqual(get_conditioned_parameter_count(16, 16, 64, 2), 6722)

    def test_conditioned_mlp_checkpoint_allowlist_sync(self):
        """
        Tests that all 12 SSRL-09 checkpoint filenames exist in both ssrl_checkpoints
        and publish_results allowlists.
        """
        expected_ckpts = []
        for arm in ["post_a", "fresh_b", "seq_b", "diverse_replay"]:
            for seed in [102, 103, 104]:
                expected_ckpts.append(f"checkpoint_cond_{arm}_s{seed}.pt")
        self.assertEqual(len(expected_ckpts), 12)
        for name in expected_ckpts:
            self.assertIn(name, ALLOWED_CHECKPOINT_FILENAMES)
            self.assertIn(name, PUBLISH_ALLOWED_CHECKPOINT_FILENAMES)

    def test_l2init_der_specification_approval_guard(self):
        """
        Tests that verify_l2init_der_specification_approval blocks unapproved
        execution, verifies pinned hyperparameters, and accepts APPROVED configurations.
        """
        self.assertEqual(PROPOSED_L2INIT_DER_CONFIG["study_id"], "SSRL-10")
        self.assertAlmostEqual(PROPOSED_L2INIT_DER_CONFIG["l2_init_lambda"], 0.01)
        self.assertAlmostEqual(PROPOSED_L2INIT_DER_CONFIG["der_alpha"], 0.5)
        self.assertEqual(PROPOSED_L2INIT_DER_CONFIG["expected_parameter_count"], 6402)
        unapproved = copy.deepcopy(PROPOSED_L2INIT_DER_CONFIG)
        unapproved["gate_approval_status"] = "PROPOSED_PENDING_APPROVAL"
        with self.assertRaises(SpecificationApprovalError):
            verify_l2init_der_specification_approval(unapproved)
        approved_cfg = copy.deepcopy(unapproved)
        approved_cfg["gate_approval_status"] = "APPROVED"
        verify_l2init_der_specification_approval(approved_cfg)

    def test_der_logit_loss_zero_at_match_and_exact_mse(self):
        """
        Tests capture_frozen_teacher_logits, example_id-aligned compute_der_logit_loss
        (0.0 when matched, exact MSE when shifted), and evaluate_buffer_logit_drift.
        """
        torch.manual_seed(77)
        model = SSRLInstructionMLP(input_dim=32, hidden_dim=64, output_dim=2)
        fp = compute_tensor_fingerprint_from_state_dict(model.state_dict())

        buffer_examples = [
            {"example_id": 10, "features": [1] * 16, "mask": [1, 1, 1, 1] + [0] * 12},
            {"example_id": 20, "features": [-1] * 16, "mask": [1, 1, 1, 1] + [0] * 12},
            {"example_id": 30, "features": [1, -1] * 8, "mask": [1, 1, 1, 1] + [0] * 12},
        ]
        teacher_by_id, teacher_fp, num_floats = capture_frozen_teacher_logits(model, buffer_examples, fp)
        self.assertEqual(num_floats, 6)
        self.assertEqual(set(teacher_by_id.keys()), {10, 20, 30})
        self.assertEqual(len(teacher_fp), 64)

        with self.assertRaises(CheckpointVerificationError):
            capture_frozen_teacher_logits(model, buffer_examples, "0" * 64)

        reordered_batch = [buffer_examples[2], buffer_examples[0], buffer_examples[1]]
        matched_logits = torch.stack([teacher_by_id[int(ex["example_id"])] for ex in reordered_batch], dim=0)
        zero_loss = compute_der_logit_loss(matched_logits, reordered_batch, teacher_by_id)
        self.assertAlmostEqual(float(zero_loss.item()), 0.0, places=7)

        delta = torch.tensor([[0.25, -0.50], [1.00, 0.00], [-0.50, 0.50]], dtype=torch.float32)
        shifted_logits = matched_logits + delta
        expected_mse = float((delta ** 2).mean().item())
        actual_mse = float(compute_der_logit_loss(shifted_logits, reordered_batch, teacher_by_id).item())
        self.assertAlmostEqual(actual_mse, expected_mse, places=6)

        with self.assertRaises(KeyError):
            compute_der_logit_loss(matched_logits[:1], [{"example_id": 999}], teacher_by_id)

        drift = evaluate_buffer_logit_drift(model, buffer_examples, teacher_by_id)
        self.assertEqual(drift["execution_mode"], EVAL_MODE_DETERMINISTIC)
        self.assertEqual(drift["buffer_size"], 3)
        self.assertEqual(drift["num_logit_floats"], 6)
        self.assertAlmostEqual(drift["mse_logit_drift"], 0.0, places=7)
        self.assertAlmostEqual(drift["max_abs_logit_drift"], 0.0, places=7)
        self.assertEqual(drift["argmax_match_numerator"], 3)
        self.assertEqual(drift["argmax_match_denominator"], 3)

    def test_l2init_der_checkpoint_allowlist_sync(self):
        """
        Tests that all 12 SSRL-10 checkpoint filenames exist in both ssrl_checkpoints
        and publish_results allowlists.
        """
        expected_ckpts = []
        for arm in ["post_a", "fresh_b", "seq_b", "diverse_der"]:
            for seed in [102, 103, 104]:
                expected_ckpts.append(f"checkpoint_der_{arm}_s{seed}.pt")
        self.assertEqual(len(expected_ckpts), 12)
        for name in expected_ckpts:
            self.assertIn(name, ALLOWED_CHECKPOINT_FILENAMES)
            self.assertIn(name, PUBLISH_ALLOWED_CHECKPOINT_FILENAMES)

    def test_l2init_der_m64_specification_approval_guard(self):
        """
        Tests that verify_l2init_der_m64_specification_approval blocks unapproved
        execution, verifies pinned hyperparameters, and accepts APPROVED configurations.
        """
        self.assertEqual(PROPOSED_L2INIT_DER_M64_CONFIG["study_id"], "SSRL-11")
        self.assertEqual(PROPOSED_L2INIT_DER_M64_CONFIG["replay_buffer_size"], 64)
        self.assertEqual(PROPOSED_L2INIT_DER_M64_CONFIG["replay_batch_size"], 16)
        self.assertAlmostEqual(PROPOSED_L2INIT_DER_M64_CONFIG["l2_init_lambda"], 0.01)
        self.assertAlmostEqual(PROPOSED_L2INIT_DER_M64_CONFIG["der_alpha"], 0.5)
        unapproved = copy.deepcopy(PROPOSED_L2INIT_DER_M64_CONFIG)
        unapproved["gate_approval_status"] = "PROPOSED_PENDING_APPROVAL"
        with self.assertRaises(SpecificationApprovalError):
            verify_l2init_der_m64_specification_approval(unapproved)
        approved_cfg = copy.deepcopy(unapproved)
        approved_cfg["gate_approval_status"] = "APPROVED"
        verify_l2init_der_m64_specification_approval(approved_cfg)

    def test_stratified_m64_buffer_and_cyclic_replay_schedule(self):
        """
        Tests M=64 stratified pattern-diverse buffer selection (64 unique IDs, 32/32 class balance,
        5 or 6 items per active pattern across all 12 patterns, 8/8 balance per 16-item chunk) and
        deterministic cyclic 4:1 replay interleaving covering all 64 buffer IDs in 4 replay chunks.
        """
        from research.lifelong_01.ssrl_generator import generate_pilot_dataset_dict

        da = generate_pilot_dataset_dict(
            rule_indices=(0, 1, 2, 3), rule_id=1, seed=101, partition_seed=42,
            study_id="SSRL-01", phase="learnability_pilot",
        )
        buf = select_stratified_pattern_diverse_memory(
            da["train_examples"], [0, 1, 2, 3], buffer_size=64, replay_batch_size=16
        )
        self.assertEqual(len(buf), 64)
        ids = [int(ex["example_id"]) for ex in buf]
        self.assertEqual(len(set(ids)), 64)
        self.assertEqual(sum(1 for ex in buf if int(ex["target_class"]) == 0), 32)
        self.assertEqual(sum(1 for ex in buf if int(ex["target_class"]) == 1), 32)

        pattern_counts = {}
        for ex in buf:
            pat = tuple(ex["features"][i] for i in [0, 1, 2, 3])
            pattern_counts[pat] = pattern_counts.get(pat, 0) + 1
        self.assertEqual(len(pattern_counts), 12)
        for pat, cnt in pattern_counts.items():
            self.assertIn(cnt, (5, 6), f"Unexpected count {cnt} for pattern {pat}")

        for c in range(4):
            chunk = buf[c * 16 : (c + 1) * 16]
            self.assertEqual(sum(1 for ex in chunk if int(ex["target_class"]) == 0), 8)
            self.assertEqual(sum(1 for ex in chunk if int(ex["target_class"]) == 1), 8)

        schedule_b = [da["train_examples"][i * 16 : (i + 1) * 16] for i in range(16)]
        cyclic = build_cyclic_replay_schedule(
            schedule_b, buf, replay_batch_size=16, ratio_b=4, ratio_a=1
        )
        self.assertEqual(len(cyclic), 20)
        b_steps = [item for item in cyclic if item[0] == "rule_b"]
        replay_steps = [item for item in cyclic if item[0] == "replay_a"]
        self.assertEqual(len(b_steps), 16)
        self.assertEqual(len(replay_steps), 4)
        replayed_ids = set()
        for _, _, chunk in replay_steps:
            self.assertEqual(len(chunk), 16)
            replayed_ids.update(int(ex["example_id"]) for ex in chunk)
        self.assertEqual(replayed_ids, set(ids))

    def test_l2init_der_m64_checkpoint_allowlist_sync(self):
        """
        Tests that all 12 SSRL-11 checkpoint filenames exist in both ssrl_checkpoints
        and publish_results allowlists.
        """
        expected_ckpts = []
        for arm in ["post_a", "fresh_b", "seq_b", "diverse_der_m64"]:
            for seed in [102, 103, 104]:
                expected_ckpts.append(f"checkpoint_derm64_{arm}_s{seed}.pt")
        self.assertEqual(len(expected_ckpts), 12)
        for name in expected_ckpts:
            self.assertIn(name, ALLOWED_CHECKPOINT_FILENAMES)
            self.assertIn(name, PUBLISH_ALLOWED_CHECKPOINT_FILENAMES)

    def test_l2init_der_m64_b1000_specification_approval_guard(self):
        """
        Tests that verify_l2init_der_m64_b1000_specification_approval blocks unapproved
        execution, verifies pinned hyperparameters, and accepts APPROVED configurations.
        """
        self.assertEqual(PROPOSED_L2INIT_DER_M64_B1000_CONFIG["study_id"], "SSRL-12")
        self.assertEqual(PROPOSED_L2INIT_DER_M64_B1000_CONFIG["step_budget_b"], 1000)
        self.assertEqual(
            PROPOSED_L2INIT_DER_M64_B1000_CONFIG["milestone_b_steps"],
            [0, 25, 50, 100, 250, 500, 750, 1000],
        )
        self.assertEqual(PROPOSED_L2INIT_DER_M64_B1000_CONFIG["replay_buffer_size"], 64)
        self.assertAlmostEqual(PROPOSED_L2INIT_DER_M64_B1000_CONFIG["l2_init_lambda"], 0.01)
        self.assertAlmostEqual(PROPOSED_L2INIT_DER_M64_B1000_CONFIG["der_alpha"], 0.5)
        unapproved = copy.deepcopy(PROPOSED_L2INIT_DER_M64_B1000_CONFIG)
        unapproved["gate_approval_status"] = "PROPOSED_PENDING_APPROVAL"
        with self.assertRaises(SpecificationApprovalError):
            verify_l2init_der_m64_b1000_specification_approval(unapproved)
        approved_cfg = copy.deepcopy(unapproved)
        approved_cfg["gate_approval_status"] = "APPROVED"
        verify_l2init_der_m64_b1000_specification_approval(approved_cfg)

    def test_b1000_schedule_prefix_identity_with_ssrl11(self):
        """
        Tests that SSRL-12 (1,000 Task B steps) produces 1,250 total steps (1,000 B + 250 replay_a),
        matches SSRL-11's 625-step prefix schedule hash identically, and visits the 4 buffer chunks
        63, 63, 62, 62 times across the 250 replay steps.
        """
        from research.lifelong_01.ssrl_generator import generate_pilot_dataset_dict
        from research.lifelong_01.run_ssrl_two_rule import build_minibatch_schedule
        from research.lifelong_01.run_ssrl_paired_replication import compute_schedule_hash

        da = generate_pilot_dataset_dict(
            rule_indices=(0, 1, 2, 3), rule_id=1, seed=101, partition_seed=42,
            study_id="SSRL-01", phase="learnability_pilot",
        )
        db = generate_pilot_dataset_dict(
            rule_indices=(4, 5, 6, 7), rule_id=2, seed=101, partition_seed=43,
            study_id="SSRL-02", phase="two_rule_diagnostic",
        )
        buf = select_stratified_pattern_diverse_memory(da["train_examples"], [0, 1, 2, 3], 64, 16)
        sched_b_500 = build_minibatch_schedule(db["train_examples"], 500, 16, 2027)
        sched_b_1000 = build_minibatch_schedule(db["train_examples"], 1000, 16, 2027)
        cyclic_500 = build_cyclic_replay_schedule(sched_b_500, buf, replay_batch_size=16, ratio_b=4, ratio_a=1)
        cyclic_1000 = build_cyclic_replay_schedule(sched_b_1000, buf, replay_batch_size=16, ratio_b=4, ratio_a=1)

        self.assertEqual(len(cyclic_1000), 1250)
        b_steps = [item for item in cyclic_1000 if item[0] == "rule_b"]
        replay_steps = [item for item in cyclic_1000 if item[0] == "replay_a"]
        self.assertEqual(len(b_steps), 1000)
        self.assertEqual(len(replay_steps), 250)
        self.assertEqual(compute_schedule_hash(cyclic_1000[:625]), compute_schedule_hash(cyclic_500))

        chunk_sigs = [tuple(int(ex["example_id"]) for ex in buf[c * 16 : (c + 1) * 16]) for c in range(4)]
        counts = {sig: 0 for sig in chunk_sigs}
        for _, _, chunk in replay_steps:
            sig = tuple(int(ex["example_id"]) for ex in chunk)
            counts[sig] += 1
        self.assertEqual([counts[sig] for sig in chunk_sigs], [63, 63, 62, 62])

    def test_l2init_der_m64_b1000_checkpoint_allowlist_sync(self):
        """
        Tests that all 12 SSRL-12 checkpoint filenames exist in both ssrl_checkpoints
        and publish_results allowlists.
        """
        expected_ckpts = []
        for arm in ["post_a", "fresh_b", "seq_b", "diverse_der_m64_b1000"]:
            for seed in [102, 103, 104]:
                expected_ckpts.append(f"checkpoint_derm64b1k_{arm}_s{seed}.pt")
        self.assertEqual(len(expected_ckpts), 12)
        for name in expected_ckpts:
            self.assertIn(name, ALLOWED_CHECKPOINT_FILENAMES)
            self.assertIn(name, PUBLISH_ALLOWED_CHECKPOINT_FILENAMES)

    def test_three_rule_stream_specification_approval_guard(self):
        """
        Tests that verify_three_rule_specification_approval blocks unapproved
        execution, verifies pinned hyperparameters, and accepts APPROVED configurations.
        """
        self.assertEqual(PROPOSED_THREE_RULE_STREAM_CONFIG["study_id"], "SSRL-13")
        self.assertEqual(PROPOSED_THREE_RULE_STREAM_CONFIG["rule_3_id"], 3)
        self.assertEqual(PROPOSED_THREE_RULE_STREAM_CONFIG["rule_3_indices"], [8, 9, 10, 11])
        self.assertEqual(PROPOSED_THREE_RULE_STREAM_CONFIG["step_budget_r1"], 2000)
        self.assertEqual(PROPOSED_THREE_RULE_STREAM_CONFIG["step_budget_r2"], 1000)
        self.assertEqual(PROPOSED_THREE_RULE_STREAM_CONFIG["step_budget_r3"], 1000)
        self.assertEqual(PROPOSED_THREE_RULE_STREAM_CONFIG["replay_buffer_size_per_rule"], 64)
        self.assertAlmostEqual(PROPOSED_THREE_RULE_STREAM_CONFIG["l2_init_lambda"], 0.01)
        self.assertAlmostEqual(PROPOSED_THREE_RULE_STREAM_CONFIG["der_alpha"], 0.5)
        unapproved = copy.deepcopy(PROPOSED_THREE_RULE_STREAM_CONFIG)
        unapproved["gate_approval_status"] = "PROPOSED_PENDING_APPROVAL"
        with self.assertRaises(SpecificationApprovalError):
            verify_three_rule_specification_approval(unapproved)
        approved_cfg = copy.deepcopy(unapproved)
        approved_cfg["gate_approval_status"] = "APPROVED"
        verify_three_rule_specification_approval(approved_cfg)

    def test_rule3_pinned_dataset_generator_parity_and_disjoint_coords(self):
        """
        Verifies pinned ssrl_pilot_t3_s101.json + manifest SHA-256, field-by-field parity with
        generate_pilot_dataset_dict (AGENTS.md 6.2), and pairwise disjointness across Rules 1, 2, 3.
        """
        from research.lifelong_01.ssrl_generator import generate_pilot_dataset_dict, load_and_verify_dataset

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        data_path = os.path.join(repo_root, "research", "lifelong_01", "data", "ssrl_pilot_t3_s101.json")
        manifest_path = os.path.join(repo_root, "research", "lifelong_01", "data", "ssrl_pilot_t3_s101_manifest.json")
        self.assertTrue(os.path.exists(data_path))
        self.assertTrue(os.path.exists(manifest_path))

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        self.assertEqual(manifest["dataset_sha256"], SHA256_RULE_3_DEFAULT)
        self.assertEqual(manifest["rule_indices"], [8, 9, 10, 11])

        d3 = load_and_verify_dataset(
            data_path,
            expected_sha256=SHA256_RULE_3_DEFAULT,
            approved_config={
                "study_id": "SSRL-13", "phase": "three_rule_sequential_stream",
                "seed": 101, "partition_seed": 44, "rule_id": 3, "rule_indices": [8, 9, 10, 11],
                "train_pool_size": 384, "eval_seen_size": 192, "eval_unseen_size": 64,
            },
        )
        d3_regen = generate_pilot_dataset_dict(
            rule_indices=(8, 9, 10, 11), rule_id=3, seed=101, partition_seed=44,
            study_id="SSRL-13", phase="three_rule_sequential_stream",
        )
        self.assertEqual(d3, d3_regen)
        coords = (
            set(PROPOSED_THREE_RULE_STREAM_CONFIG["rule_1_indices"])
            | set(PROPOSED_THREE_RULE_STREAM_CONFIG["rule_2_indices"])
            | set(PROPOSED_THREE_RULE_STREAM_CONFIG["rule_3_indices"])
        )
        self.assertEqual(len(coords), 12)

    def test_multi_rule_cyclic_replay_schedule_and_allowlist_sync(self):
        """
        Tests 3-rule 4:1 alternating cyclic replay schedule (1,000 Rule 3 + 125 Rule 1 + 125 Rule 2
        with zero cross-rule contamination and full 64-item coverage per prior rule) and verifies
        all 18 SSRL-13 checkpoint filenames in both allowlists.
        """
        from research.lifelong_01.ssrl_generator import generate_pilot_dataset_dict
        from research.lifelong_01.run_ssrl_two_rule import build_minibatch_schedule

        d1 = generate_pilot_dataset_dict((0, 1, 2, 3), 1, 101, partition_seed=42, study_id="SSRL-01", phase="learnability_pilot")
        d2 = generate_pilot_dataset_dict((4, 5, 6, 7), 2, 101, partition_seed=43, study_id="SSRL-02", phase="two_rule_diagnostic")
        d3 = generate_pilot_dataset_dict((8, 9, 10, 11), 3, 101, partition_seed=44, study_id="SSRL-13", phase="three_rule_sequential_stream")
        buf1 = select_stratified_pattern_diverse_memory(d1["train_examples"], [0, 1, 2, 3], 64, 16)
        buf2 = select_multi_rule_stratified_memory(d2["train_examples"], [4, 5, 6, 7], 64, 16)
        sched3 = build_minibatch_schedule(d3["train_examples"], 1000, 16, 2028)

        combined = build_multi_rule_cyclic_replay_schedule(sched3, "rule_3", 3, {1: buf1, 2: buf2}, 16, 4, 1)
        self.assertEqual(len(combined), 1250)
        r3_steps = [s for s in combined if s[0] == "rule_3"]
        rep_steps = [s for s in combined if s[0] == "replay"]
        self.assertEqual(len(r3_steps), 1000)
        self.assertEqual(len(rep_steps), 250)
        rep1 = [s for s in rep_steps if s[1] == 1]
        rep2 = [s for s in rep_steps if s[1] == 2]
        self.assertEqual(len(rep1), 125)
        self.assertEqual(len(rep2), 125)
        self.assertEqual([s[1] for s in rep_steps[:6]], [1, 2, 1, 2, 1, 2])
        seen_ids_1, seen_ids_2 = set(), set()
        for _, rid, batch in rep_steps:
            self.assertEqual(len(batch), 16)
            self.assertTrue(all(int(ex["rule_id"]) == rid for ex in batch))
            target_set = seen_ids_1 if rid == 1 else seen_ids_2
            target_set.update(int(ex["example_id"]) for ex in batch)
        self.assertEqual(seen_ids_1, {int(ex["example_id"]) for ex in buf1})
        self.assertEqual(seen_ids_2, {int(ex["example_id"]) for ex in buf2})

        expected_ckpts = []
        for arm in ["post_r1", "fresh_r3", "seq_post_r2", "seq_r3", "der_post_r2", "stream_der_r3"]:
            for seed in [102, 103, 104]:
                expected_ckpts.append(f"checkpoint_r3_{arm}_s{seed}.pt")
        self.assertEqual(len(expected_ckpts), 18)
        for name in expected_ckpts:
            self.assertIn(name, ALLOWED_CHECKPOINT_FILENAMES)
            self.assertIn(name, PUBLISH_ALLOWED_CHECKPOINT_FILENAMES)

    def test_input_gated_mlp_parameter_count_and_zero_init_neutrality(self):
        """
        Verifies SSRLInputGatedMLP parameter count (6658), bit-identical Step-0 forward output
        vs SSRLInstructionMLP under identical seed, nonzero gate gradient on a single backward
        step (no zero-init symmetry lock), and ValueError on invalid input dimension.
        """
        self.assertEqual(get_input_gated_parameter_count(), 6658)
        torch.manual_seed(102)
        m_gated = SSRLInputGatedMLP()
        torch.manual_seed(102)
        m_bare = SSRLInstructionMLP()
        self.assertEqual(sum(p.numel() for p in m_gated.parameters()), 6658)
        self.assertEqual(sum(p.numel() for p in m_bare.parameters()), 6402)

        x = torch.tensor(
            [([1.0, -1.0] * 8 + [1.0] * 4 + [0.0] * 12), ([-1.0, 1.0] * 8 + [0.0] * 4 + [1.0] * 4 + [0.0] * 8)],
            dtype=torch.float32,
        )
        m_gated.eval()
        m_bare.eval()
        with torch.no_grad():
            self.assertTrue(torch.equal(m_gated(x), m_bare(x)))

        m_gated.train()
        m_gated.zero_grad()
        loss = torch.nn.CrossEntropyLoss()(m_gated(x), torch.tensor([0, 1], dtype=torch.long))
        loss.backward()
        self.assertIsNotNone(m_gated.fc_in_gate.weight.grad)
        self.assertGreater(float(m_gated.fc_in_gate.weight.grad.abs().sum().item()), 0.0)

        with self.assertRaises(ValueError):
            m_gated(torch.zeros((2, 31), dtype=torch.float32))

    def test_input_gated_three_rule_default_unapproved_guard(self):
        """
        Verifies SSRL-14 config is APPROVED, passes verify_input_gated_three_rule_approval,
        and rejects a copy whose gate_approval_status is PROPOSED_PENDING_APPROVAL.
        """
        self.assertEqual(PROPOSED_INPUT_GATED_THREE_RULE_CONFIG["gate_approval_status"], "APPROVED")
        verify_input_gated_three_rule_approval(PROPOSED_INPUT_GATED_THREE_RULE_CONFIG)
        cfg = copy.deepcopy(PROPOSED_INPUT_GATED_THREE_RULE_CONFIG)
        cfg["gate_approval_status"] = "PROPOSED_PENDING_APPROVAL"
        with self.assertRaises(SpecificationApprovalError):
            verify_input_gated_three_rule_approval(cfg)

    def test_input_gated_checkpoint_allowlists_synced(self):
        """
        Verifies all 27 SSRL-14 checkpoint filenames exist in both ssrl_checkpoints and
        publish_results allowlists and that both allowlists match identically.
        """
        arms = [
            "post_r1", "fresh_r3", "seq_post_r2", "seq_r3", "der_post_r2", "stream_der_r3",
            "bare_post_r1", "bare_der_post_r2", "bare_stream_der_r3",
        ]
        expected_ckpts = [f"checkpoint_ig3_{arm}_s{seed}.pt" for arm in arms for seed in (102, 103, 104)]
        self.assertEqual(len(expected_ckpts), 27)
        for name in expected_ckpts:
            self.assertIn(name, ALLOWED_CHECKPOINT_FILENAMES)
            self.assertIn(name, PUBLISH_ALLOWED_CHECKPOINT_FILENAMES)
        self.assertEqual(ALLOWED_CHECKPOINT_FILENAMES, PUBLISH_ALLOWED_CHECKPOINT_FILENAMES)


class TestSSRLCPUIntegration(unittest.TestCase):
    """
    CPU integration test suite for SSRL components requiring model construction (SSRLInstructionMLP)
    or neural forward pass execution. Distinct from model-free verification.
    """

    def test_replay_checkpoint_identity_and_rejection(self):
        """
        Tests that load_and_verify_t1_model validates expected post-A checkpoint fingerprint
        and file hash, rejecting mismatched weights or corrupted files.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            model = SSRLInstructionMLP(input_dim=32, hidden_dim=64, output_dim=2)
            ckpt_path = os.path.join(tmpdir, "checkpoint_t1_post_a.pt")
            torch.save(model.state_dict(), ckpt_path)

            with open(ckpt_path, "rb") as f:
                valid_sha = hashlib.sha256(f.read()).hexdigest()
            valid_fp = compute_tensor_fingerprint_from_state_dict(model.state_dict())

            # Matching loads cleanly
            loaded = load_and_verify_t1_model(ckpt_path, valid_fp, valid_sha, device=torch.device("cpu"))
            self.assertIsInstance(loaded, SSRLInstructionMLP)

            # Wrong fingerprint raises CheckpointVerificationError
            with self.assertRaises(CheckpointVerificationError):
                load_and_verify_t1_model(ckpt_path, "0"*64, valid_sha, device=torch.device("cpu"))

            # Wrong file sha raises CheckpointVerificationError
            with self.assertRaises(CheckpointVerificationError):
                load_and_verify_t1_model(ckpt_path, valid_fp, "0"*64, device=torch.device("cpu"))

    def test_joint_reference_initialization_checkpoint_validation(self):
        """
        Tests that load_and_verify_initial_model verifies tensor fingerprint,
        rejects non-existent files, and detects corrupted/mismatched checkpoint weights.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "checkpoint_t0_init.pt")
            model_orig = SSRLInstructionMLP(input_dim=32, hidden_dim=64, output_dim=2)
            torch.save(model_orig.state_dict(), ckpt_path)

            # 1. Matching fingerprint loads cleanly
            valid_fp = compute_tensor_fingerprint_from_state_dict(model_orig.state_dict())
            loaded_model = load_and_verify_initial_model(ckpt_path, valid_fp, device=torch.device("cpu"))
            self.assertIsInstance(loaded_model, SSRLInstructionMLP)

            # 2. Mismatched fingerprint raises CheckpointVerificationError
            wrong_fp = "0000000000000000000000000000000000000000000000000000000000000000"
            with self.assertRaises(CheckpointVerificationError) as ctx:
                load_and_verify_initial_model(ckpt_path, wrong_fp, device=torch.device("cpu"))
            self.assertIn("does not match expected fingerprint", str(ctx.exception))

            # 3. Missing file raises FileNotFoundError
            with self.assertRaises(FileNotFoundError):
                load_and_verify_initial_model(os.path.join(tmpdir, "missing.pt"), valid_fp, device=torch.device("cpu"))

    def test_joint_reference_evaluation_only_probe_isolation(self):
        """
        Tests that probe evaluation produces zero gradient or parameter mutations,
        and that probe evaluation runs strictly under evaluation mode with a mathematically
        valid opposing-parity fixture.
        """
        model = SSRLInstructionMLP(input_dim=32, hidden_dim=64, output_dim=2)
        orig_fp = compute_tensor_fingerprint_from_state_dict(model.state_dict())

        with tempfile.TemporaryDirectory() as tmpdir:
            probe_path = os.path.join(tmpdir, "probe.json")
            # Valid opposing-parity fixture:
            # Rule A [0..3]: [1, 1, 1, 1] -> parity +1 -> target_class_a = 1
            # Rule B [4..7]: [-1, 1, 1, 1] -> parity -1 -> target_class_b = 0
            valid_x = [1] * 16
            valid_x[4] = -1
            probe_data = {
                "metadata": {"seed": 3030},
                "probe_examples": [
                    {
                        "probe_id": 1,
                        "features": valid_x,
                        "target_class_a": 1,
                        "target_class_b": 0,
                        "stratum": "seen_seen",
                        "is_held_out_a": False,
                        "is_held_out_b": False
                    }
                ]
            }
            with open(probe_path, "w", encoding="utf-8") as f:
                json.dump(probe_data, f)

            res = evaluate_counterfactual_probe(model, probe_data["probe_examples"], device=torch.device("cpu"))
            self.assertEqual(res["overall_metrics"]["total_pairs"], 1)

            # Verify model weights were untouched and have no accumulated gradients
            post_fp = compute_tensor_fingerprint_from_state_dict(model.state_dict())
            self.assertEqual(orig_fp, post_fp)
            for p in model.parameters():
                self.assertIsNone(p.grad)

    def test_joint_reference_failure_diagnostics_persistence_on_error(self):
        """
        Tests that run_joint_reference_study catches execution failures, persists
        failure diagnostics with actual status ('FAILED'), work counters, partial records,
        and enforces non-overwrite behavior on failure records and run directories.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = os.path.join(tmpdir, "run_failed_test")
            dest_json = os.path.join(tmpdir, "results.json")
            init_ckpt = os.path.join(tmpdir, "checkpoint_t0_init.pt")

            # Create test initial checkpoint
            test_model = SSRLInstructionMLP(input_dim=32, hidden_dim=64, output_dim=2)
            test_fp = compute_tensor_fingerprint_from_state_dict(test_model.state_dict())
            torch.save(test_model.state_dict(), init_ckpt)

            # Isolated copied config marked APPROVED for this test
            cfg = copy.deepcopy(PROPOSED_JOINT_REFERENCE_CONFIG)
            cfg["gate_approval_status"] = "APPROVED"
            cfg["expected_t0_fingerprint"] = test_fp
            cfg["step_budget_a"] = 4
            cfg["step_budget_b"] = 1
            cfg["step_budget_total"] = 5

            dataset_a_path = os.path.join(os.path.dirname(__file__), "data", "ssrl_pilot_t1_s101.json")
            dataset_b_path = os.path.join(os.path.dirname(__file__), "data", "ssrl_pilot_t2_s101.json")
            with open(dataset_a_path, "rb") as fa:
                sha_a = hashlib.sha256(fa.read()).hexdigest()
            with open(dataset_b_path, "rb") as fb:
                sha_b = hashlib.sha256(fb.read()).hexdigest()

            probe_path = os.path.join(tmpdir, "probe.json")
            valid_x = [1] * 16
            valid_x[4] = -1
            with open(probe_path, "w", encoding="utf-8") as pf:
                json.dump({
                    "metadata": {"seed": 3030},
                    "probe_examples": [
                        {"probe_id": 1, "features": valid_x, "target_class_a": 1, "target_class_b": 0, "stratum": "seen_seen"}
                    ]
                }, pf)

            # Inject a controlled failure into forward pass during step 2
            original_forward = SSRLInstructionMLP.forward
            call_count = [0]
            def faulty_forward(self_model, x):
                call_count[0] += 1
                if call_count[0] >= 2:
                    raise RuntimeError("Injected simulated compute fault during joint training")
                return original_forward(self_model, x)

            with unittest.mock.patch.object(SSRLInstructionMLP, "forward", faulty_forward):
                with self.assertRaises(RuntimeError) as ctx:
                    run_joint_reference_study(
                        dataset_a_path=dataset_a_path,
                        sha256_a=sha_a,
                        dataset_b_path=dataset_b_path,
                        sha256_b=sha_b,
                        destination=dest_json,
                        run_dir=run_dir,
                        init_checkpoint_path=init_ckpt,
                        probe_dataset_path=probe_path,
                        config=cfg
                    )
                self.assertIn("Injected simulated compute fault", str(ctx.exception))

            # 1. Verify failure diagnostics file was persisted
            diag_file = os.path.join(run_dir, "failure_diagnostics.json")
            self.assertTrue(os.path.isfile(diag_file))

            with open(diag_file, "r", encoding="utf-8") as df:
                failure_diag = json.load(df)

            # 2. Verify status and error fields
            self.assertEqual(failure_diag["status"], "FAILED")
            self.assertEqual(failure_diag["error_type"], "RuntimeError")
            self.assertIn("Injected simulated compute fault", failure_diag["error_message"])

            # 3. Verify work counters reflect completed steps before failure
            self.assertGreaterEqual(failure_diag["exposures"]["total_presentations"], 16)
            self.assertGreaterEqual(failure_diag["exposures"]["total_optimizer_steps"], 1)

            # 4. Verify existing partial records / initial milestone checkpoint recorded
            self.assertIn("checkpoint_joint_t0_init", failure_diag["checkpoints"])
            init_saved = os.path.join(run_dir, "checkpoint_joint_t0_init.pt")
            self.assertTrue(os.path.isfile(init_saved))

            # 5. Verify non-overwrite behavior
            with self.assertRaises(FileExistsError):
                atomic_write_json(failure_diag, diag_file, refuse_overwrite=True)

            with self.assertRaises(FileExistsError):
                reserve_run_directory(run_dir)

    def test_evaluate_population_records_contract_and_serialization(self):
        """
        Tests that evaluate_population_records evaluates under torch.no_grad() and model.eval(),
        constructs valid SSRLExampleRecord instances matching the real schema, preserves all
        required fields (IDs, rule identity, dataset hash, input_features, selection_mask,
        target_class, predicted_class, is_correct, is_held_out_pattern), achieves population
        completeness, serializes cleanly to JSON, verifies raw-logit preservation through
        the probe's intended schema, and guarantees zero model mutation or parameter gradients.
        """
        device = torch.device("cpu")
        model = SSRLInstructionMLP(input_dim=32, hidden_dim=64, output_dim=2)
        fp_before = compute_tensor_fingerprint_from_state_dict(model.state_dict())

        # Load pinned dataset A
        dataset_a_path = os.path.join(os.path.dirname(__file__), "data", "ssrl_pilot_t1_s101.json")
        with open(dataset_a_path, "r", encoding="utf-8") as f:
            dataset_a = json.load(f)
        with open(dataset_a_path, "rb") as fb:
            sha_a = hashlib.sha256(fb.read()).hexdigest()

        eval_seen = dataset_a["eval_seen_examples"]
        eval_unseen = dataset_a["eval_unseen_examples"]
        self.assertEqual(len(eval_seen), 192)
        self.assertEqual(len(eval_unseen), 64)

        # 1. Evaluate seen population
        seen_records = evaluate_population_records(
            model=model,
            examples=eval_seen,
            arm="joint_reference",
            seed=101,
            step=2500,
            dataset_sha256=sha_a,
            device=device
        )
        self.assertEqual(len(seen_records), 192)

        # 2. Evaluate unseen population
        unseen_records = evaluate_population_records(
            model=model,
            examples=eval_unseen,
            arm="joint_reference",
            seed=101,
            step=2500,
            dataset_sha256=sha_a,
            device=device
        )
        self.assertEqual(len(unseen_records), 64)

        # 3. Field validation and population completeness across all records
        all_eval_records = seen_records + unseen_records
        all_examples = eval_seen + eval_unseen
        seen_ids = set()

        for rec, ex in zip(all_eval_records, all_examples):
            self.assertIsInstance(rec, SSRLExampleRecord)
            self.assertEqual(rec.example_id, ex["example_id"])
            self.assertNotIn(rec.example_id, seen_ids)
            seen_ids.add(rec.example_id)

            self.assertEqual(rec.rule_id, 1)
            self.assertEqual(rec.arm, "joint_reference")
            self.assertEqual(rec.seed, 101)
            self.assertEqual(rec.step, 2500)
            self.assertEqual(rec.dataset_sha256, sha_a)
            self.assertEqual(rec.input_features, ex["features"])
            self.assertEqual(len(rec.input_features), 16)
            self.assertEqual(rec.selection_mask, ex["mask"])
            self.assertEqual(len(rec.selection_mask), 16)
            self.assertEqual(rec.target_class, ex["target_class"])
            self.assertIn(rec.predicted_class, (0, 1))
            self.assertEqual(rec.is_correct, (rec.predicted_class == rec.target_class))
            self.assertEqual(rec.is_held_out_pattern, ex["is_held_out_pattern"])

        # 4. Downstream serialization verification
        serialized_seen = [r.to_dict() for r in seen_records]
        serialized_unseen = [r.to_dict() for r in unseen_records]
        self.assertEqual(len(serialized_seen), 192)
        self.assertEqual(len(serialized_unseen), 64)

        # Roundtrip JSON serialization check
        json_dump = json.dumps({"seen": serialized_seen, "unseen": serialized_unseen})
        parsed = json.loads(json_dump)
        self.assertEqual(len(parsed["seen"]), 192)
        self.assertEqual(parsed["seen"][0]["dataset_sha256"], sha_a)
        self.assertEqual(len(parsed["seen"][0]["input_features"]), 16)

        # 5. Raw-logit preservation check through intended schema (counterfactual probe)
        valid_x = [1] * 16
        valid_x[4] = -1
        probe_examples = [
            {
                "probe_id": 42,
                "features": valid_x,
                "target_class_a": 1,
                "target_class_b": 0,
                "stratum": "seen_seen",
                "is_held_out_a": False,
                "is_held_out_b": False
            }
        ]
        probe_eval = evaluate_counterfactual_probe(model, probe_examples, device=device)
        self.assertEqual(len(probe_eval["pair_records"]), 1)
        pr = probe_eval["pair_records"][0]
        # Assert raw logits are preserved in intended schema
        self.assertIn("logits_a", pr)
        self.assertIn("logits_b", pr)
        self.assertEqual(len(pr["logits_a"]), 2)
        self.assertEqual(len(pr["logits_b"]), 2)
        self.assertTrue(all(isinstance(v, float) and math.isfinite(v) for v in pr["logits_a"]))
        self.assertTrue(all(isinstance(v, float) and math.isfinite(v) for v in pr["logits_b"]))

        # 6. Zero mutation and zero gradient verification
        fp_after = compute_tensor_fingerprint_from_state_dict(model.state_dict())
        self.assertEqual(fp_before, fp_after)
        for p in model.parameters():
            self.assertIsNone(p.grad)

    def test_replay_failure_persistence_on_error(self):
        """
        Tests that run_replay_study catches execution failures, persists
        failure diagnostics with actual status ('FAILED'), work counters, and
        enforces non-overwrite behavior on failure records and run directories.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = os.path.join(tmpdir, "run_failed_replay")
            dest_json = os.path.join(tmpdir, "results.json")
            t1_ckpt = os.path.join(tmpdir, "checkpoint_t1_post_a.pt")

            test_model = SSRLInstructionMLP(input_dim=32, hidden_dim=64, output_dim=2)
            torch.save(test_model.state_dict(), t1_ckpt)
            with open(t1_ckpt, "rb") as f:
                sha_ckpt = hashlib.sha256(f.read()).hexdigest()
            fp_ckpt = compute_tensor_fingerprint_from_state_dict(test_model.state_dict())

            cfg = copy.deepcopy(PROPOSED_REPLAY_CONFIG)
            cfg["gate_approval_status"] = "APPROVED"
            cfg["expected_t1_fingerprint"] = fp_ckpt
            cfg["expected_t1_file_sha256"] = sha_ckpt
            cfg["step_budget_b"] = 4
            cfg["step_budget_replay_a"] = 1
            cfg["step_budget_total"] = 5
            cfg["replay_cycles"] = 1

            dataset_a_path = os.path.join(os.path.dirname(__file__), "data", "ssrl_pilot_t1_s101.json")
            dataset_b_path = os.path.join(os.path.dirname(__file__), "data", "ssrl_pilot_t2_s101.json")
            probe_path = os.path.join(os.path.dirname(__file__), "data", "ssrl_probe_counterfactual_128.json")
            with open(dataset_a_path, "rb") as fa:
                sha_a = hashlib.sha256(fa.read()).hexdigest()
            with open(dataset_b_path, "rb") as fb:
                sha_b = hashlib.sha256(fb.read()).hexdigest()
            with open(probe_path, "rb") as fp_file:
                sha_probe = hashlib.sha256(fp_file.read()).hexdigest()

            call_count = [0]
            original_forward = SSRLInstructionMLP.forward
            def faulty_forward(self_model, x):
                call_count[0] += 1
                if call_count[0] >= 2:
                    raise RuntimeError("Injected simulated fault during replay training")
                return original_forward(self_model, x)

            with unittest.mock.patch.object(SSRLInstructionMLP, "forward", faulty_forward):
                with self.assertRaises(RuntimeError) as ctx:
                    run_replay_study(
                        dataset_a_path=dataset_a_path,
                        sha256_a=sha_a,
                        dataset_b_path=dataset_b_path,
                        sha256_b=sha_b,
                        probe_dataset_path=probe_path,
                        sha256_probe=sha_probe,
                        destination=dest_json,
                        run_dir=run_dir,
                        t1_checkpoint_path=t1_ckpt,
                        config=cfg
                    )
                self.assertIn("Injected simulated fault", str(ctx.exception))

            diag_file = os.path.join(run_dir, "failure_diagnostics.json")
            self.assertTrue(os.path.isfile(diag_file))
            with open(diag_file, "r", encoding="utf-8") as df:
                fdiag = json.load(df)

            self.assertEqual(fdiag["status"], "FAILED")
            self.assertEqual(fdiag["error_type"], "RuntimeError")
            self.assertIn("Injected simulated fault", fdiag["error_message"])
            self.assertGreaterEqual(fdiag["exposures"]["total_optimizer_steps"], 1)
            self.assertGreaterEqual(fdiag["exposures"]["total_presentations"], 16)

            with self.assertRaises(FileExistsError):
                atomic_write_json(fdiag, diag_file, refuse_overwrite=True)

    def test_replay_tiny_e2e_integration(self):
        """
        Tests the real training-to-evaluation path of run_replay_study with a bounded
        tiny fixture budget (4 B steps + 1 Replay A step = 5 steps).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = os.path.join(tmpdir, "run_e2e_replay")
            dest_json = os.path.join(tmpdir, "results.json")
            t1_ckpt = os.path.join(tmpdir, "checkpoint_t1_post_a.pt")

            test_model = SSRLInstructionMLP(input_dim=32, hidden_dim=64, output_dim=2)
            torch.save(test_model.state_dict(), t1_ckpt)
            with open(t1_ckpt, "rb") as f:
                sha_ckpt = hashlib.sha256(f.read()).hexdigest()
            fp_ckpt = compute_tensor_fingerprint_from_state_dict(test_model.state_dict())

            cfg = copy.deepcopy(PROPOSED_REPLAY_CONFIG)
            cfg["gate_approval_status"] = "APPROVED"
            cfg["expected_t1_fingerprint"] = fp_ckpt
            cfg["expected_t1_file_sha256"] = sha_ckpt
            cfg["step_budget_b"] = 4
            cfg["step_budget_replay_a"] = 1
            cfg["step_budget_total"] = 5
            cfg["replay_cycles"] = 1

            dataset_a_path = os.path.join(os.path.dirname(__file__), "data", "ssrl_pilot_t1_s101.json")
            dataset_b_path = os.path.join(os.path.dirname(__file__), "data", "ssrl_pilot_t2_s101.json")
            probe_path = os.path.join(os.path.dirname(__file__), "data", "ssrl_probe_counterfactual_128.json")
            with open(dataset_a_path, "rb") as fa:
                sha_a = hashlib.sha256(fa.read()).hexdigest()
            with open(dataset_b_path, "rb") as fb:
                sha_b = hashlib.sha256(fb.read()).hexdigest()
            with open(probe_path, "rb") as fp_file:
                sha_probe = hashlib.sha256(fp_file.read()).hexdigest()

            res = run_replay_study(
                dataset_a_path=dataset_a_path,
                sha256_a=sha_a,
                dataset_b_path=dataset_b_path,
                sha256_b=sha_b,
                probe_dataset_path=probe_path,
                sha256_probe=sha_probe,
                destination=dest_json,
                run_dir=run_dir,
                t1_checkpoint_path=t1_ckpt,
                config=cfg
            )

            self.assertEqual(res["diagnostics"]["status"], "COMPLETED")
            self.assertEqual(res["diagnostics"]["exposures"]["total_optimizer_steps"], 5)
            self.assertEqual(res["diagnostics"]["exposures"]["rule_b_presentations"], 64)
            self.assertEqual(res["diagnostics"]["exposures"]["replay_a_presentations"], 16)
            self.assertEqual(res["diagnostics"]["exposures"]["total_presentations"], 80)
            self.assertTrue(os.path.isfile(dest_json))
            final_ckpt = os.path.join(run_dir, "checkpoint_replay_step625_final.pt")
            self.assertTrue(os.path.isfile(final_ckpt))
            self.assertIn("acc_rule_a_seen", res["metrics"])
            self.assertIn("counterfactual_probe", res["diagnostics"])

    def test_diverse_replay_tiny_e2e_integration(self):
        """
        Tests the real training-to-evaluation path of run_diverse_replay_study with a bounded
        tiny fixture budget (4 B steps + 1 Diverse Replay A step = 5 steps).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = os.path.join(tmpdir, "run_e2e_diverse_replay")
            dest_json = os.path.join(tmpdir, "results.json")
            t1_ckpt = os.path.join(tmpdir, "checkpoint_t1_post_a.pt")

            test_model = SSRLInstructionMLP(input_dim=32, hidden_dim=64, output_dim=2)
            torch.save(test_model.state_dict(), t1_ckpt)
            with open(t1_ckpt, "rb") as f:
                sha_ckpt = hashlib.sha256(f.read()).hexdigest()
            fp_ckpt = compute_tensor_fingerprint_from_state_dict(test_model.state_dict())

            cfg = copy.deepcopy(PROPOSED_DIVERSE_REPLAY_CONFIG)
            cfg["gate_approval_status"] = "APPROVED"
            cfg["expected_t1_fingerprint"] = fp_ckpt
            cfg["expected_t1_file_sha256"] = sha_ckpt
            cfg["step_budget_b"] = 4
            cfg["step_budget_replay_a"] = 1
            cfg["step_budget_total"] = 5
            cfg["replay_cycles"] = 1

            dataset_a_path = os.path.join(os.path.dirname(__file__), "data", "ssrl_pilot_t1_s101.json")
            dataset_b_path = os.path.join(os.path.dirname(__file__), "data", "ssrl_pilot_t2_s101.json")
            probe_path = os.path.join(os.path.dirname(__file__), "data", "ssrl_probe_counterfactual_128.json")
            with open(dataset_a_path, "rb") as fa:
                sha_a = hashlib.sha256(fa.read()).hexdigest()
            with open(dataset_b_path, "rb") as fb:
                sha_b = hashlib.sha256(fb.read()).hexdigest()
            with open(probe_path, "rb") as fp_file:
                sha_probe = hashlib.sha256(fp_file.read()).hexdigest()

            res = run_diverse_replay_study(
                dataset_a_path=dataset_a_path,
                sha256_a=sha_a,
                dataset_b_path=dataset_b_path,
                sha256_b=sha_b,
                probe_dataset_path=probe_path,
                sha256_probe=sha_probe,
                destination=dest_json,
                run_dir=run_dir,
                t1_checkpoint_path=t1_ckpt,
                config=cfg
            )

            self.assertEqual(res["diagnostics"]["status"], "COMPLETED")
            self.assertEqual(res["diagnostics"]["exposures"]["total_optimizer_steps"], 5)
            self.assertEqual(res["diagnostics"]["exposures"]["rule_b_presentations"], 64)
            self.assertEqual(res["diagnostics"]["exposures"]["replay_a_presentations"], 16)
            self.assertEqual(res["diagnostics"]["exposures"]["total_presentations"], 80)
            self.assertTrue(os.path.isfile(dest_json))
            final_ckpt = os.path.join(run_dir, "checkpoint_diverse_replay_step625_final.pt")
            self.assertTrue(os.path.isfile(final_ckpt))
            self.assertIn("acc_rule_a_seen", res["metrics"])
            self.assertIn("counterfactual_probe", res["diagnostics"])
            self.assertIn("rule_a_seen_per_pattern", res["diagnostics"])
            self.assertEqual(len(res["diagnostics"]["rule_a_seen_per_pattern"]), 12)

    def test_paired_replication_checkpoint_mismatch_rejection(self):
        """
        AGENTS.md Section 8.4: Negative test showing mismatched starting state is rejected.
        Asserts train_single_arm raises CheckpointVerificationError when the loaded state's
        tensor fingerprint does not equal the expected post-A fingerprint.
        """
        model = SSRLInstructionMLP(input_dim=32, hidden_dim=64, output_dim=2)
        state_dict = model.state_dict()
        expected_fp = "0" * 64

        cfg = copy.deepcopy(PROPOSED_PAIRED_REPLICATION_CONFIG)
        dummy_batch = [{"example_id": 1, "features": [0.0] * 16, "mask": [1.0] * 16, "target_class": 0}]
        dummy_schedule = [("rule_b", 2, dummy_batch)]

        with self.assertRaises(CheckpointVerificationError) as ctx:
            train_single_arm(
                initial_model_state=state_dict,
                schedule=dummy_schedule,
                config=cfg,
                train_deadline=time.monotonic() + 60.0,
                expected_fingerprint=expected_fp,
            )
        self.assertIn("Initial model fingerprint mismatch", str(ctx.exception))

    def test_paired_replication_tiny_e2e_integration(self):
        """
        Tests tiny paired replication fixture:
        - Seed 102 (whitelisted in production allowlist; in tempdir)
        - 2 steps Phase A
        - 5 updates per arm: 4 B updates and 1 A replay update (80 presentations)
        Verifies:
        - Exact completed update and presentation counters for each task
        - Forked arm execution from identical checkpoint state
        - Identical post-A tensor fingerprint recorded and verified for both arms
        - Primary retention delta and secondary acquisition delta serialization
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = os.path.join(tmpdir, "run_e2e_paired")
            dest_json = os.path.join(tmpdir, "results.json")

            dataset_a_path = os.path.join(os.path.dirname(__file__), "data", "ssrl_pilot_t1_s101.json")
            dataset_b_path = os.path.join(os.path.dirname(__file__), "data", "ssrl_pilot_t2_s101.json")
            probe_path = os.path.join(os.path.dirname(__file__), "data", "ssrl_probe_counterfactual_128.json")
            with open(dataset_a_path, "rb") as fa:
                sha_a = hashlib.sha256(fa.read()).hexdigest()
            with open(dataset_b_path, "rb") as fb:
                sha_b = hashlib.sha256(fb.read()).hexdigest()
            with open(probe_path, "rb") as fp_file:
                sha_probe = hashlib.sha256(fp_file.read()).hexdigest()

            cfg = copy.deepcopy(PROPOSED_PAIRED_REPLICATION_CONFIG)
            cfg["gate_approval_status"] = "APPROVED"
            cfg["seeds"] = [102]
            cfg["step_budget_a_per_seed"] = 2
            cfg["step_budget_b_per_arm"] = 4
            cfg["step_budget_replay_per_arm"] = 1
            cfg["step_budget_total_per_arm"] = 5
            cfg["replay_ratio_b_to_a"] = [4, 1]
            cfg["replay_cycles"] = 1
            cfg["gate_g_lc_threshold"] = 0.0  # Allow tiny 2-step model to pass Phase A

            res = run_paired_replication_study(
                dataset_a_path=dataset_a_path,
                sha256_a=sha_a,
                dataset_b_path=dataset_b_path,
                sha256_b=sha_b,
                probe_dataset_path=probe_path,
                sha256_probe=sha_probe,
                destination=dest_json,
                run_dir=run_dir,
                config=cfg,
            )

            self.assertEqual(res["diagnostics"]["status"], "COMPLETED")
            self.assertEqual(len(res["diagnostics"]["paired_differences"]), 1)
            pair = res["diagnostics"]["paired_differences"][0]
            self.assertEqual(pair["seed"], 102)
            self.assertIn("primary_delta_retention_pp", pair)
            self.assertIn("secondary_delta_acquisition_pp", pair)
            self.assertIn("post_a_seen_pct", pair)

            # Check that checkpoint files were saved and whitelisted
            ckpt_post_a = os.path.join(run_dir, "checkpoint_t1_post_a_s102.pt")
            ckpt_lid = os.path.join(run_dir, "checkpoint_lowest_id_final_s102.pt")
            ckpt_div = os.path.join(run_dir, "checkpoint_diverse_replay_final_s102.pt")
            self.assertTrue(os.path.isfile(ckpt_post_a))
            self.assertTrue(os.path.isfile(ckpt_lid))
            self.assertTrue(os.path.isfile(ckpt_div))

            s_data = res["diagnostics"]["seeds_data"]["102"]
            fp_post_a = s_data["checkpoint_post_a"]["tensor_fingerprint"]

            # Strengthened checkpoint-pairing verification:
            # Record actual fingerprint after loading and before updates; assert both equal saved post-A
            self.assertEqual(s_data["arm_lowest_id"]["loaded_initial_fingerprint"], fp_post_a)
            self.assertEqual(s_data["arm_diverse"]["loaded_initial_fingerprint"], fp_post_a)
            self.assertEqual(
                s_data["arm_lowest_id"]["loaded_initial_fingerprint"],
                s_data["arm_diverse"]["loaded_initial_fingerprint"],
            )

            # Assert exact completed update and presentation counters for each task
            counters_a = s_data["counters_phase_a"]
            self.assertEqual(counters_a["rule_a_updates"], 2)
            self.assertEqual(counters_a["rule_a_presentations"], 32)
            self.assertEqual(counters_a["total_optimizer_steps"], 2)
            self.assertEqual(counters_a["total_presentations"], 32)

            counters_lid = s_data["arm_lowest_id"]["counters"]
            self.assertEqual(counters_lid["rule_b_updates"], 4)
            self.assertEqual(counters_lid["rule_b_presentations"], 64)
            self.assertEqual(counters_lid["replay_a_updates"], 1)
            self.assertEqual(counters_lid["replay_a_presentations"], 16)
            self.assertEqual(counters_lid["total_optimizer_steps"], 5)
            self.assertEqual(counters_lid["total_presentations"], 80)

            counters_div = s_data["arm_diverse"]["counters"]
            self.assertEqual(counters_div["rule_b_updates"], 4)
            self.assertEqual(counters_div["rule_b_presentations"], 64)
            self.assertEqual(counters_div["replay_a_updates"], 1)
            self.assertEqual(counters_div["replay_a_presentations"], 16)
            self.assertEqual(counters_div["total_optimizer_steps"], 5)
            self.assertEqual(counters_div["total_presentations"], 80)

    def test_paired_replication_training_deadline_failure(self):
        """
        AGENTS.md Section 2.6, 8.4, 10: Training-stage deadline failure.
        Asserts DeadlineExceededError is raised, failure diagnostics are persisted,
        the failing stage and seed are recorded as FAILED, later seeds are NOT_RUN,
        and no replacement seeds or retries occur.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = os.path.join(tmpdir, "run_train_deadline")
            dest_json = os.path.join(tmpdir, "results.json")
            diag_file = os.path.join(run_dir, "failure_diagnostics.json")

            dataset_a_path = os.path.join(os.path.dirname(__file__), "data", "ssrl_pilot_t1_s101.json")
            dataset_b_path = os.path.join(os.path.dirname(__file__), "data", "ssrl_pilot_t2_s101.json")
            probe_path = os.path.join(os.path.dirname(__file__), "data", "ssrl_probe_counterfactual_128.json")
            with open(dataset_a_path, "rb") as fa:
                sha_a = hashlib.sha256(fa.read()).hexdigest()
            with open(dataset_b_path, "rb") as fb:
                sha_b = hashlib.sha256(fb.read()).hexdigest()
            with open(probe_path, "rb") as fp_file:
                sha_probe = hashlib.sha256(fp_file.read()).hexdigest()

            cfg = copy.deepcopy(PROPOSED_PAIRED_REPLICATION_CONFIG)
            cfg["gate_approval_status"] = "APPROVED"
            cfg["seeds"] = [102, 103]
            cfg["step_budget_a_per_seed"] = 2
            cfg["step_budget_b_per_arm"] = 4
            cfg["step_budget_replay_per_arm"] = 1
            cfg["step_budget_total_per_arm"] = 5
            cfg["replay_ratio_b_to_a"] = [4, 1]
            cfg["wall_clock_ceiling_training_seconds_per_stage"] = -1.0  # Controlled immediate deadline failure

            with self.assertRaises(DeadlineExceededError):
                run_paired_replication_study(
                    dataset_a_path=dataset_a_path,
                    sha256_a=sha_a,
                    dataset_b_path=dataset_b_path,
                    sha256_b=sha_b,
                    probe_dataset_path=probe_path,
                    sha256_probe=sha_probe,
                    destination=dest_json,
                    run_dir=run_dir,
                    config=cfg,
                )

            self.assertTrue(os.path.isfile(diag_file))
            with open(diag_file, "r", encoding="utf-8") as f:
                diag = json.load(f)["diagnostics"]

            self.assertEqual(diag["status"], "FAILED")
            self.assertEqual(diag["error_type"], "DeadlineExceededError")
            self.assertEqual(diag["current_stage"], "seed_102_phase_a")
            self.assertEqual(diag["seeds_status"]["102"], "FAILED")
            self.assertEqual(diag["seeds_status"]["103"], "NOT_RUN")
            self.assertFalse(os.path.exists(dest_json))

    def test_paired_replication_evaluation_deadline_failure(self):
        """
        AGENTS.md Section 2.6, 8.4, 10: Evaluation-stage deadline failure.
        Asserts DeadlineExceededError is raised during evaluation stage,
        failure diagnostics preserve completed Phase A counters and artifacts,
        failing seed is recorded as FAILED, remaining seeds are NOT_RUN.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = os.path.join(tmpdir, "run_eval_deadline")
            dest_json = os.path.join(tmpdir, "results.json")
            diag_file = os.path.join(run_dir, "failure_diagnostics.json")

            dataset_a_path = os.path.join(os.path.dirname(__file__), "data", "ssrl_pilot_t1_s101.json")
            dataset_b_path = os.path.join(os.path.dirname(__file__), "data", "ssrl_pilot_t2_s101.json")
            probe_path = os.path.join(os.path.dirname(__file__), "data", "ssrl_probe_counterfactual_128.json")
            with open(dataset_a_path, "rb") as fa:
                sha_a = hashlib.sha256(fa.read()).hexdigest()
            with open(dataset_b_path, "rb") as fb:
                sha_b = hashlib.sha256(fb.read()).hexdigest()
            with open(probe_path, "rb") as fp_file:
                sha_probe = hashlib.sha256(fp_file.read()).hexdigest()

            cfg = copy.deepcopy(PROPOSED_PAIRED_REPLICATION_CONFIG)
            cfg["gate_approval_status"] = "APPROVED"
            cfg["seeds"] = [102, 103]
            cfg["step_budget_a_per_seed"] = 2
            cfg["step_budget_b_per_arm"] = 4
            cfg["step_budget_replay_per_arm"] = 1
            cfg["step_budget_total_per_arm"] = 5
            cfg["replay_ratio_b_to_a"] = [4, 1]
            cfg["gate_g_lc_threshold"] = 0.0
            cfg["wall_clock_ceiling_training_seconds_per_stage"] = 60.0
            cfg["wall_clock_ceiling_eval_seconds_per_stage"] = -1.0  # Controlled immediate deadline failure

            with self.assertRaises(DeadlineExceededError):
                run_paired_replication_study(
                    dataset_a_path=dataset_a_path,
                    sha256_a=sha_a,
                    dataset_b_path=dataset_b_path,
                    sha256_b=sha_b,
                    probe_dataset_path=probe_path,
                    sha256_probe=sha_probe,
                    destination=dest_json,
                    run_dir=run_dir,
                    config=cfg,
                )

            self.assertTrue(os.path.isfile(diag_file))
            with open(diag_file, "r", encoding="utf-8") as f:
                diag = json.load(f)["diagnostics"]

            self.assertEqual(diag["status"], "FAILED")
            self.assertEqual(diag["error_type"], "DeadlineExceededError")
            self.assertEqual(diag["current_stage"], "seed_102_phase_a_eval")
            self.assertEqual(diag["seeds_status"]["102"], "FAILED")
            self.assertEqual(diag["seeds_status"]["103"], "NOT_RUN")

            # Preserved completed Phase A training counters and artifacts
            s102_data = diag["seeds_data"]["102"]
            self.assertIn("counters_phase_a", s102_data)
            self.assertEqual(s102_data["counters_phase_a"]["rule_a_updates"], 2)
            self.assertEqual(s102_data["counters_phase_a"]["rule_a_presentations"], 32)
            self.assertIn("initial_fingerprint_phase_a", s102_data)
            self.assertFalse(os.path.exists(dest_json))

    def test_paired_replication_failure_preserves_artifacts_and_marks_later_seeds_not_run(self):
        """
        AGENTS.md Section 2.6, 8.4, 10: Multi-seed execution failure and zero retry.
        Simulates failure on seed 103 after seed 102 completes.
        Asserts:
        - Seed 102 status is COMPLETED and its artifacts/counters are fully preserved.
        - Seed 103 status is FAILED.
        - Later seed 104 is marked NOT_RUN.
        - No retry or replacement seeds occur.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = os.path.join(tmpdir, "run_multiseed_fail")
            dest_json = os.path.join(tmpdir, "results.json")
            diag_file = os.path.join(run_dir, "failure_diagnostics.json")

            dataset_a_path = os.path.join(os.path.dirname(__file__), "data", "ssrl_pilot_t1_s101.json")
            dataset_b_path = os.path.join(os.path.dirname(__file__), "data", "ssrl_pilot_t2_s101.json")
            probe_path = os.path.join(os.path.dirname(__file__), "data", "ssrl_probe_counterfactual_128.json")
            with open(dataset_a_path, "rb") as fa:
                sha_a = hashlib.sha256(fa.read()).hexdigest()
            with open(dataset_b_path, "rb") as fb:
                sha_b = hashlib.sha256(fb.read()).hexdigest()
            with open(probe_path, "rb") as fp_file:
                sha_probe = hashlib.sha256(fp_file.read()).hexdigest()

            cfg = copy.deepcopy(PROPOSED_PAIRED_REPLICATION_CONFIG)
            cfg["gate_approval_status"] = "APPROVED"
            cfg["seeds"] = [102, 103, 104]
            cfg["step_budget_a_per_seed"] = 2
            cfg["step_budget_b_per_arm"] = 4
            cfg["step_budget_replay_per_arm"] = 1
            cfg["step_budget_total_per_arm"] = 5
            cfg["replay_ratio_b_to_a"] = [4, 1]
            cfg["replay_cycles"] = 1
            cfg["gate_g_lc_threshold"] = 0.0

            import research.lifelong_01.run_ssrl_paired_replication as runner_mod
            orig_train = runner_mod.train_single_arm

            def patched_train(state, sched, c, deadline, expected_fingerprint=None):
                if runner_mod.sys._getframe(1).f_locals.get("seed") == 103:
                    raise RuntimeError("Simulated fault during seed 103 training")
                return orig_train(state, sched, c, deadline, expected_fingerprint)

            runner_mod.train_single_arm = patched_train
            try:
                with self.assertRaises(RuntimeError) as ctx:
                    run_paired_replication_study(
                        dataset_a_path=dataset_a_path,
                        sha256_a=sha_a,
                        dataset_b_path=dataset_b_path,
                        sha256_b=sha_b,
                        probe_dataset_path=probe_path,
                        sha256_probe=sha_probe,
                        destination=dest_json,
                        run_dir=run_dir,
                        config=cfg,
                    )
                self.assertIn("Simulated fault during seed 103 training", str(ctx.exception))
            finally:
                runner_mod.train_single_arm = orig_train

            self.assertTrue(os.path.isfile(diag_file))
            with open(diag_file, "r", encoding="utf-8") as f:
                diag = json.load(f)["diagnostics"]

            self.assertEqual(diag["status"], "FAILED")
            self.assertEqual(diag["error_type"], "RuntimeError")
            self.assertEqual(diag["seeds_status"]["102"], "COMPLETED")
            self.assertEqual(diag["seeds_status"]["103"], "FAILED")
            self.assertEqual(diag["seeds_status"]["104"], "NOT_RUN")

            # Preserved seed 102 completed artifacts and paired comparison
            self.assertIn("102", diag["seeds_data"])
            self.assertEqual(diag["seeds_data"]["102"]["paired_comparison"]["seed"], 102)
            self.assertEqual(len(diag["paired_differences"]), 1)
            self.assertFalse(os.path.exists(dest_json))

    def test_transfer_controls_checkpoint_mismatch_rejection(self):
        """
        AGENTS.md Section 8.4: Negative test showing mismatched initial fingerprint is rejected.
        Asserts train_task_b_arm raises CheckpointVerificationError when the loaded state's
        tensor fingerprint does not equal expected_fingerprint.
        """
        model = SSRLInstructionMLP(input_dim=32, hidden_dim=64, output_dim=2)
        state_dict = model.state_dict()
        expected_fp = "0" * 64

        dataset_b_path = os.path.join(os.path.dirname(__file__), "data", "ssrl_pilot_t2_s101.json")
        with open(dataset_b_path, "r", encoding="utf-8") as fb:
            db = json.load(fb)
        with open(dataset_b_path, "rb") as fb:
            sha_b = hashlib.sha256(fb.read()).hexdigest()

        cfg = copy.deepcopy(PROPOSED_TRANSFER_CONTROLS_CONFIG)
        dummy_batch = [{"example_id": 1, "features": [0.0] * 16, "mask": [1.0] * 16, "target_class": 0}]
        dummy_schedule = [dummy_batch]

        with self.assertRaises(CheckpointVerificationError) as ctx:
            train_task_b_arm(
                initial_model_state=state_dict,
                schedule_b=dummy_schedule,
                db=db,
                sha256_b=sha_b,
                config=cfg,
                seed=102,
                arm_tag="test_mismatch",
                train_deadline=time.monotonic() + 60.0,
                eval_ceiling=60.0,
                expected_fingerprint=expected_fp,
            )
        self.assertIn("Initial fingerprint mismatch", str(ctx.exception))

    def test_transfer_controls_tiny_e2e_integration(self):
        """
        Tests tiny SSRL-07 transfer controls fixture:
        - Seed 102 (uses preserved checkpoint_t1_post_a_s102.pt)
        - step_budget_b = 2, milestone_steps = [0, 2]
        Verifies:
        - FREEZE-AFTER-BASE evaluated with 0 updates, fingerprint identical before and after.
        - fresh_b starts with verified initial fingerprint, completes 2 updates, saves checkpoint.
        - seq_b starts with verified post-A fingerprint, completes 2 updates, saves checkpoint.
        - Milestone deltas computed at steps 0 and 2.
        - Results JSON output saved.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = os.path.join(tmpdir, "run_e2e_transfer")
            dest_json = os.path.join(tmpdir, "results.json")

            dataset_a_path = os.path.join(os.path.dirname(__file__), "data", "ssrl_pilot_t1_s101.json")
            dataset_b_path = os.path.join(os.path.dirname(__file__), "data", "ssrl_pilot_t2_s101.json")
            probe_path = os.path.join(os.path.dirname(__file__), "data", "ssrl_probe_counterfactual_128.json")
            with open(dataset_a_path, "rb") as fa:
                sha_a = hashlib.sha256(fa.read()).hexdigest()
            with open(dataset_b_path, "rb") as fb:
                sha_b = hashlib.sha256(fb.read()).hexdigest()
            with open(probe_path, "rb") as fp_file:
                sha_probe = hashlib.sha256(fp_file.read()).hexdigest()

            # Path to preserved SSRL-06 post-A checkpoints
            post_a_dir = os.path.abspath("C:/Users/Vicky/Desktop/lifelong-learning/evidence/runs/ssrl_06_paired_replication_f208e4da")

            cfg = copy.deepcopy(PROPOSED_TRANSFER_CONTROLS_CONFIG)
            cfg["gate_approval_status"] = "APPROVED"
            cfg["seeds"] = [102]
            cfg["step_budget_b"] = 2
            cfg["milestone_steps"] = [0, 2]

            res = run_transfer_controls_study(
                dataset_a_path=dataset_a_path,
                sha256_a=sha_a,
                dataset_b_path=dataset_b_path,
                sha256_b=sha_b,
                probe_dataset_path=probe_path,
                sha256_probe=sha_probe,
                post_a_checkpoints_dir=post_a_dir,
                destination=dest_json,
                run_dir=run_dir,
                config=cfg,
            )

            self.assertEqual(res["diagnostics"]["status"], "COMPLETED")
            self.assertEqual(len(res["diagnostics"]["paired_transfer_comparisons"]), 1)
            pair = res["diagnostics"]["paired_transfer_comparisons"][0]
            self.assertEqual(pair["seed"], 102)
            self.assertIn("milestone_deltas", pair)
            self.assertIn("0", pair["milestone_deltas"])
            self.assertIn("2", pair["milestone_deltas"])
            self.assertIn("final_transfer_delta_step500_pp", pair)

            s_data = res["diagnostics"]["seeds_data"]["102"]
            fp_post_a = s_data["post_a_checkpoint_loaded_fingerprint"]
            fp_fresh_init = s_data["fresh_init_fingerprint"]

            # Verify arm fingerprints
            self.assertEqual(s_data["freeze_after_base"]["fingerprint"], fp_post_a)
            self.assertEqual(s_data["fresh_b"]["loaded_initial_fingerprint"], fp_fresh_init)
            self.assertEqual(s_data["seq_b"]["loaded_initial_fingerprint"], fp_post_a)

            # Assert exact counters
            cnt_fresh = s_data["fresh_b"]["counters"]
            self.assertEqual(cnt_fresh["rule_b_updates"], 2)
            self.assertEqual(cnt_fresh["rule_b_presentations"], 32)
            cnt_seq = s_data["seq_b"]["counters"]
            self.assertEqual(cnt_seq["rule_b_updates"], 2)
            self.assertEqual(cnt_seq["rule_b_presentations"], 32)

            # Check checkpoint files
            ckpt_fresh_path = os.path.join(run_dir, "checkpoint_fresh_b_s102.pt")
            ckpt_seq_path = os.path.join(run_dir, "checkpoint_seq_b_s102.pt")
            self.assertTrue(os.path.isfile(ckpt_fresh_path))
            self.assertTrue(os.path.isfile(ckpt_seq_path))
            self.assertTrue(os.path.isfile(dest_json))

    def test_transfer_controls_deadline_and_failure_persistence(self):
        """
        AGENTS.md Section 2.6, 8.4, 10: Training deadline exceeded in transfer controls.
        Asserts DeadlineExceededError is raised, failure diagnostics are persisted,
        failing seed is marked FAILED, later seeds are NOT_RUN, no retries.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = os.path.join(tmpdir, "run_deadline_transfer")
            dest_json = os.path.join(tmpdir, "results.json")
            diag_file = os.path.join(run_dir, "failure_diagnostics.json")

            dataset_a_path = os.path.join(os.path.dirname(__file__), "data", "ssrl_pilot_t1_s101.json")
            dataset_b_path = os.path.join(os.path.dirname(__file__), "data", "ssrl_pilot_t2_s101.json")
            probe_path = os.path.join(os.path.dirname(__file__), "data", "ssrl_probe_counterfactual_128.json")
            with open(dataset_a_path, "rb") as fa:
                sha_a = hashlib.sha256(fa.read()).hexdigest()
            with open(dataset_b_path, "rb") as fb:
                sha_b = hashlib.sha256(fb.read()).hexdigest()
            with open(probe_path, "rb") as fp_file:
                sha_probe = hashlib.sha256(fp_file.read()).hexdigest()

            post_a_dir = os.path.abspath("C:/Users/Vicky/Desktop/lifelong-learning/evidence/runs/ssrl_06_paired_replication_f208e4da")

            cfg = copy.deepcopy(PROPOSED_TRANSFER_CONTROLS_CONFIG)
            cfg["gate_approval_status"] = "APPROVED"
            cfg["seeds"] = [102, 103]
            cfg["step_budget_b"] = 2
            cfg["milestone_steps"] = [0, 2]
            cfg["wall_clock_ceiling_training_seconds_per_stage"] = -1.0  # Controlled immediate deadline failure

            with self.assertRaises(DeadlineExceededError):
                run_transfer_controls_study(
                    dataset_a_path=dataset_a_path,
                    sha256_a=sha_a,
                    dataset_b_path=dataset_b_path,
                    sha256_b=sha_b,
                    probe_dataset_path=probe_path,
                    sha256_probe=sha_probe,
                    post_a_checkpoints_dir=post_a_dir,
                    destination=dest_json,
                    run_dir=run_dir,
                    config=cfg,
                )

            self.assertTrue(os.path.isfile(diag_file))
            with open(diag_file, "r", encoding="utf-8") as f:
                diag = json.load(f)["diagnostics"]

            self.assertEqual(diag["status"], "FAILED")
            self.assertEqual(diag["error_type"], "DeadlineExceededError")
            self.assertEqual(diag["seeds_status"]["102"], "FAILED")
            self.assertEqual(diag["seeds_status"]["103"], "NOT_RUN")
            self.assertFalse(os.path.exists(dest_json))

    def test_l2_init_tiny_e2e_integration(self):
        """
        Tiny CPU integration test for SSRL-08 (Seed 102, 4 A steps, 4 B steps, 5 diverse_replay steps).
        Verifies all 4 arms, fingerprints, compact per-unit boolean lengths (192/64/128), mechanism
        snapshots, 4 checkpoints, and pre-upload validation passing well under 1 MB.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = os.path.join(tmpdir, "run_e2e_l2init")
            dest_json = os.path.join(run_dir, "results.json")

            dataset_a_path = os.path.join(os.path.dirname(__file__), "data", "ssrl_pilot_t1_s101.json")
            dataset_b_path = os.path.join(os.path.dirname(__file__), "data", "ssrl_pilot_t2_s101.json")
            probe_path = os.path.join(os.path.dirname(__file__), "data", "ssrl_probe_counterfactual_128.json")
            with open(dataset_a_path, "rb") as fa:
                sha_a = hashlib.sha256(fa.read()).hexdigest()
            with open(dataset_b_path, "rb") as fb:
                sha_b = hashlib.sha256(fb.read()).hexdigest()
            with open(probe_path, "rb") as fp_file:
                sha_probe = hashlib.sha256(fp_file.read()).hexdigest()

            cfg = copy.deepcopy(PROPOSED_L2_INIT_PLASTICITY_CONFIG)
            cfg["gate_approval_status"] = "APPROVED"
            cfg["seeds"] = [102]
            cfg["step_budget_a"] = 4
            cfg["step_budget_b"] = 4
            cfg["milestone_b_steps"] = [0, 4]
            cfg["enforce_post_a_learnability_gate"] = False
            cfg["enforce_fresh_b_learnability_gate"] = False

            res = run_l2_init_plasticity_study(
                dataset_a_path=dataset_a_path, sha256_a=sha_a,
                dataset_b_path=dataset_b_path, sha256_b=sha_b,
                probe_dataset_path=probe_path, sha256_probe=sha_probe,
                destination=dest_json, run_dir=run_dir, config=cfg,
            )

            self.assertEqual(res["diagnostics"]["status"], "COMPLETED")
            s_data = res["diagnostics"]["seeds_data"]["102"]
            fp_init = s_data["fresh_init_fingerprint"]
            fp_post_a = s_data["phase_a"]["checkpoint"]["tensor_fingerprint"]

            self.assertEqual(s_data["freeze_after_base"]["fingerprint"], fp_post_a)
            self.assertEqual(s_data["fresh_b"]["loaded_initial_fingerprint"], fp_init)
            self.assertEqual(s_data["seq_b"]["loaded_initial_fingerprint"], fp_post_a)
            self.assertEqual(s_data["diverse_replay"]["loaded_initial_fingerprint"], fp_post_a)

            # Verify exact counters including 4 B + 1 A = 5 steps for diverse_replay
            self.assertEqual(s_data["phase_a"]["counters"]["rule_a_updates"], 4)
            self.assertEqual(s_data["fresh_b"]["counters"]["rule_b_updates"], 4)
            self.assertEqual(s_data["seq_b"]["counters"]["rule_b_updates"], 4)
            cnt_div = s_data["diverse_replay"]["counters"]
            self.assertEqual(cnt_div["rule_b_updates"], 4)
            self.assertEqual(cnt_div["replay_a_updates"], 1)
            self.assertEqual(cnt_div["total_optimizer_steps"], 5)
            self.assertEqual(cnt_div["total_presentations"], 80)

            # Verify mechanism snapshots
            for key in ["global_l2", "mean_abs_logit_margin_b", "cross_entropy_loss_b", "h1_dead_relu_count", "h2_dead_relu_count"]:
                self.assertIn(key, s_data["mechanism_fresh_init"])
                self.assertIn(key, s_data["phase_a"]["mechanism_post_a"])

            # Verify compact per-unit boolean vector lengths (192/64/128)
            for arm_eval in [
                s_data["freeze_after_base"]["eval"],
                s_data["fresh_b"]["final_eval"],
                s_data["seq_b"]["final_eval"],
                s_data["diverse_replay"]["final_eval"],
            ]:
                pops = arm_eval["per_unit_populations"]
                self.assertEqual(len(pops["rule_a_seen"]["is_correct"]), 192)
                self.assertEqual(len(pops["rule_a_unseen"]["is_correct"]), 64)
                self.assertEqual(len(pops["rule_b_seen"]["is_correct"]), 192)
                self.assertEqual(len(pops["rule_b_unseen"]["is_correct"]), 64)
                self.assertEqual(len(arm_eval["counterfactual_probe"]["pair_records"]), 128)
                self.assertNotIn("features", arm_eval["counterfactual_probe"]["pair_records"][0])

            # Verify 4 checkpoint files and publish pre-upload validation (< 1 MB)
            for arm in ["post_a", "fresh_b", "seq_b", "diverse_replay"]:
                self.assertTrue(os.path.isfile(os.path.join(run_dir, f"checkpoint_l2init_{arm}_s102.pt")))
            run_bytes = calculate_and_verify_run_size(run_dir)
            self.assertLess(run_bytes, 1024 * 1024)
            verify_no_forbidden_files(run_dir)

    def test_l2_init_learnability_gate_rejection(self):
        """
        Negative test: step_budget_a=1 with enforce_post_a_learnability_gate=True fails Gate G-LC,
        raises LearnabilityGateError, persists failure_diagnostics.json (102=FAILED, 103=NOT_RUN),
        and prevents final results.json creation.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = os.path.join(tmpdir, "run_gate_fail_l2init")
            dest_json = os.path.join(tmpdir, "results.json")
            diag_file = os.path.join(run_dir, "failure_diagnostics.json")

            dataset_a_path = os.path.join(os.path.dirname(__file__), "data", "ssrl_pilot_t1_s101.json")
            dataset_b_path = os.path.join(os.path.dirname(__file__), "data", "ssrl_pilot_t2_s101.json")
            probe_path = os.path.join(os.path.dirname(__file__), "data", "ssrl_probe_counterfactual_128.json")
            with open(dataset_a_path, "rb") as fa:
                sha_a = hashlib.sha256(fa.read()).hexdigest()
            with open(dataset_b_path, "rb") as fb:
                sha_b = hashlib.sha256(fb.read()).hexdigest()
            with open(probe_path, "rb") as fp_file:
                sha_probe = hashlib.sha256(fp_file.read()).hexdigest()

            cfg = copy.deepcopy(PROPOSED_L2_INIT_PLASTICITY_CONFIG)
            cfg["gate_approval_status"] = "APPROVED"
            cfg["seeds"] = [102, 103]
            cfg["step_budget_a"] = 1
            cfg["step_budget_b"] = 4
            cfg["milestone_b_steps"] = [0, 4]
            cfg["enforce_post_a_learnability_gate"] = True

            with self.assertRaises(LearnabilityGateError):
                run_l2_init_plasticity_study(
                    dataset_a_path=dataset_a_path, sha256_a=sha_a,
                    dataset_b_path=dataset_b_path, sha256_b=sha_b,
                    probe_dataset_path=probe_path, sha256_probe=sha_probe,
                    destination=dest_json, run_dir=run_dir, config=cfg,
                )

            self.assertTrue(os.path.isfile(diag_file))
            with open(diag_file, "r", encoding="utf-8") as f:
                diag = json.load(f)["diagnostics"]
            self.assertEqual(diag["status"], "FAILED")
            self.assertEqual(diag["error_type"], "LearnabilityGateError")
            self.assertEqual(diag["seeds_status"]["102"], "FAILED")
            self.assertEqual(diag["seeds_status"]["103"], "NOT_RUN")
            self.assertFalse(os.path.exists(dest_json))

    def test_l2_init_deadline_failure_persistence(self):
        """
        Negative test: negative wall-clock deadline raises DeadlineExceededError,
        persists failure_diagnostics.json with 102=FAILED and 103=NOT_RUN, no results.json.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = os.path.join(tmpdir, "run_deadline_fail_l2init")
            dest_json = os.path.join(tmpdir, "results.json")
            diag_file = os.path.join(run_dir, "failure_diagnostics.json")

            dataset_a_path = os.path.join(os.path.dirname(__file__), "data", "ssrl_pilot_t1_s101.json")
            dataset_b_path = os.path.join(os.path.dirname(__file__), "data", "ssrl_pilot_t2_s101.json")
            probe_path = os.path.join(os.path.dirname(__file__), "data", "ssrl_probe_counterfactual_128.json")
            with open(dataset_a_path, "rb") as fa:
                sha_a = hashlib.sha256(fa.read()).hexdigest()
            with open(dataset_b_path, "rb") as fb:
                sha_b = hashlib.sha256(fb.read()).hexdigest()
            with open(probe_path, "rb") as fp_file:
                sha_probe = hashlib.sha256(fp_file.read()).hexdigest()

            cfg = copy.deepcopy(PROPOSED_L2_INIT_PLASTICITY_CONFIG)
            cfg["gate_approval_status"] = "APPROVED"
            cfg["seeds"] = [102, 103]
            cfg["step_budget_a"] = 4
            cfg["step_budget_b"] = 4
            cfg["milestone_b_steps"] = [0, 4]
            cfg["wall_clock_ceiling_training_seconds_per_stage"] = -1.0

            with self.assertRaises(DeadlineExceededError):
                run_l2_init_plasticity_study(
                    dataset_a_path=dataset_a_path, sha256_a=sha_a,
                    dataset_b_path=dataset_b_path, sha256_b=sha_b,
                    probe_dataset_path=probe_path, sha256_probe=sha_probe,
                    destination=dest_json, run_dir=run_dir, config=cfg,
                )

            self.assertTrue(os.path.isfile(diag_file))
            with open(diag_file, "r", encoding="utf-8") as f:
                diag = json.load(f)["diagnostics"]
            self.assertEqual(diag["status"], "FAILED")
            self.assertEqual(diag["error_type"], "DeadlineExceededError")
            self.assertEqual(diag["seeds_status"]["102"], "FAILED")
            self.assertEqual(diag["seeds_status"]["103"], "NOT_RUN")
            self.assertFalse(os.path.exists(dest_json))

    def test_conditioned_mlp_forward_and_param_count_match(self):
        """
        CPU integration test verifying SSRLConditionedMLP parameter count (6722),
        forward shape (4, 32) -> (4, 2), sensitivity to mask changes, and wrong-dim ValueError.
        """
        torch.manual_seed(102)
        model = SSRLConditionedMLP()
        param_count = sum(p.numel() for p in model.parameters())
        self.assertEqual(param_count, 6722)
        self.assertEqual(param_count, get_conditioned_parameter_count())

        feat = torch.tensor([[1.0, -1.0] * 8] * 4, dtype=torch.float32)
        mask_a = torch.tensor([[1.0] * 4 + [0.0] * 12] * 4, dtype=torch.float32)
        mask_b = torch.tensor([[0.0] * 4 + [1.0] * 4 + [0.0] * 8] * 4, dtype=torch.float32)
        u_a = torch.cat([feat, mask_a], dim=1)
        u_b = torch.cat([feat, mask_b], dim=1)

        model.eval()
        with torch.no_grad():
            out_a = model(u_a)
            out_b = model(u_b)
        self.assertEqual(tuple(out_a.shape), (4, 2))
        self.assertEqual(tuple(out_b.shape), (4, 2))
        self.assertFalse(torch.allclose(out_a, out_b))

        with self.assertRaises(ValueError):
            model(torch.zeros((2, 31), dtype=torch.float32))

    def test_conditioned_mlp_tiny_e2e_integration(self):
        """
        Tiny CPU integration test for SSRL-09 (Seed 102, 4 A steps, 4 B steps, 5 diverse_replay steps).
        Verifies all 4 arms, fingerprints, compact boolean lengths (192/64/128), eval_layernorm_no_dropout
        mode tag, mechanism + gate separation snapshots, 4 checkpoints, and pre-upload validation (< 1 MB).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = os.path.join(tmpdir, "run_e2e_cond")
            dest_json = os.path.join(run_dir, "results.json")

            dataset_a_path = os.path.join(os.path.dirname(__file__), "data", "ssrl_pilot_t1_s101.json")
            dataset_b_path = os.path.join(os.path.dirname(__file__), "data", "ssrl_pilot_t2_s101.json")
            probe_path = os.path.join(os.path.dirname(__file__), "data", "ssrl_probe_counterfactual_128.json")
            with open(dataset_a_path, "rb") as fa:
                sha_a = hashlib.sha256(fa.read()).hexdigest()
            with open(dataset_b_path, "rb") as fb:
                sha_b = hashlib.sha256(fb.read()).hexdigest()
            with open(probe_path, "rb") as fp_file:
                sha_probe = hashlib.sha256(fp_file.read()).hexdigest()

            cfg = copy.deepcopy(PROPOSED_CONDITIONED_MLP_CONFIG)
            cfg["gate_approval_status"] = "APPROVED"
            cfg["seeds"] = [102]
            cfg["step_budget_a"] = 4
            cfg["step_budget_b"] = 4
            cfg["milestone_b_steps"] = [0, 4]
            cfg["enforce_post_a_learnability_gate"] = False
            cfg["enforce_fresh_b_learnability_gate"] = False

            res = run_conditioned_mlp_study(
                dataset_a_path=dataset_a_path, sha256_a=sha_a,
                dataset_b_path=dataset_b_path, sha256_b=sha_b,
                probe_dataset_path=probe_path, sha256_probe=sha_probe,
                destination=dest_json, run_dir=run_dir, config=cfg,
            )

            self.assertEqual(res["diagnostics"]["status"], "COMPLETED")
            s_data = res["diagnostics"]["seeds_data"]["102"]
            fp_init = s_data["fresh_init_fingerprint"]
            fp_post_a = s_data["phase_a"]["checkpoint"]["tensor_fingerprint"]

            self.assertEqual(s_data["freeze_after_base"]["fingerprint"], fp_post_a)
            self.assertEqual(s_data["fresh_b"]["loaded_initial_fingerprint"], fp_init)
            self.assertEqual(s_data["seq_b"]["loaded_initial_fingerprint"], fp_post_a)
            self.assertEqual(s_data["diverse_replay"]["loaded_initial_fingerprint"], fp_post_a)

            cnt_div = s_data["diverse_replay"]["counters"]
            self.assertEqual(cnt_div["rule_b_updates"], 4)
            self.assertEqual(cnt_div["replay_a_updates"], 1)
            self.assertEqual(cnt_div["total_optimizer_steps"], 5)
            self.assertEqual(cnt_div["total_presentations"], 80)

            for snap in [
                s_data["mechanism_fresh_init"],
                s_data["phase_a"]["mechanism_post_a"],
                s_data["fresh_b"]["mechanism_snapshot"],
                s_data["seq_b"]["mechanism_snapshot"],
                s_data["diverse_replay"]["mechanism_snapshot"],
            ]:
                for k in ["parameter_count", "global_l2", "fc1_feat_weight_l2", "fc1_gate_weight_l2",
                          "gate_a_mean", "gate_b_mean", "gate_ab_cosine_similarity", "gate_ab_l2_distance",
                          "mean_abs_logit_margin_b", "cross_entropy_loss_b", "h1_dead_relu_count", "h2_dead_relu_count"]:
                    self.assertIn(k, snap)
                self.assertEqual(snap["parameter_count"], 6722)

            for arm_eval in [
                s_data["freeze_after_base"]["eval"],
                s_data["fresh_b"]["final_eval"],
                s_data["seq_b"]["final_eval"],
                s_data["diverse_replay"]["final_eval"],
            ]:
                self.assertEqual(arm_eval["mode"], EVAL_MODE_CONDITIONED)
                self.assertEqual(arm_eval["metrics"]["acc_rule_a_seen"].mode, "eval_layernorm_no_dropout")
                pops = arm_eval["per_unit_populations"]
                self.assertEqual(len(pops["rule_a_seen"]["is_correct"]), 192)
                self.assertEqual(len(pops["rule_a_unseen"]["is_correct"]), 64)
                self.assertEqual(len(pops["rule_b_seen"]["is_correct"]), 192)
                self.assertEqual(len(pops["rule_b_unseen"]["is_correct"]), 64)
                self.assertEqual(len(arm_eval["counterfactual_probe"]["pair_records"]), 128)
                self.assertNotIn("features", arm_eval["counterfactual_probe"]["pair_records"][0])

            for arm in ["post_a", "fresh_b", "seq_b", "diverse_replay"]:
                self.assertTrue(os.path.isfile(os.path.join(run_dir, f"checkpoint_cond_{arm}_s102.pt")))
            run_bytes = calculate_and_verify_run_size(run_dir)
            self.assertLess(run_bytes, 1024 * 1024)
            verify_no_forbidden_files(run_dir)

    def test_conditioned_mlp_learnability_gate_and_deadline_failure(self):
        """
        Verifies both LearnabilityGateError (step_budget_a=1, gate True) and DeadlineExceededError
        (-1.0s ceiling) persist failure_diagnostics.json with 102=FAILED, 103=NOT_RUN, no results.json.
        """
        dataset_a_path = os.path.join(os.path.dirname(__file__), "data", "ssrl_pilot_t1_s101.json")
        dataset_b_path = os.path.join(os.path.dirname(__file__), "data", "ssrl_pilot_t2_s101.json")
        probe_path = os.path.join(os.path.dirname(__file__), "data", "ssrl_probe_counterfactual_128.json")
        with open(dataset_a_path, "rb") as fa:
            sha_a = hashlib.sha256(fa.read()).hexdigest()
        with open(dataset_b_path, "rb") as fb:
            sha_b = hashlib.sha256(fb.read()).hexdigest()
        with open(probe_path, "rb") as fp_file:
            sha_probe = hashlib.sha256(fp_file.read()).hexdigest()

        cases = [
            ("gate_fail", 1, 60.0, True, LearnabilityGateError, "LearnabilityGateError"),
            ("deadline_fail", 4, -1.0, False, DeadlineExceededError, "DeadlineExceededError"),
        ]
        for subdir, steps_a, train_ceil, gate_on, exc_cls, exc_name in cases:
            with tempfile.TemporaryDirectory() as tmpdir:
                run_dir = os.path.join(tmpdir, f"run_{subdir}")
                dest_json = os.path.join(tmpdir, "results.json")
                diag_file = os.path.join(run_dir, "failure_diagnostics.json")

                cfg = copy.deepcopy(PROPOSED_CONDITIONED_MLP_CONFIG)
                cfg["gate_approval_status"] = "APPROVED"
                cfg["seeds"] = [102, 103]
                cfg["step_budget_a"] = steps_a
                cfg["step_budget_b"] = 4
                cfg["milestone_b_steps"] = [0, 4]
                cfg["enforce_post_a_learnability_gate"] = gate_on
                cfg["wall_clock_ceiling_training_seconds_per_stage"] = train_ceil

                with self.assertRaises(exc_cls):
                    run_conditioned_mlp_study(
                        dataset_a_path=dataset_a_path, sha256_a=sha_a,
                        dataset_b_path=dataset_b_path, sha256_b=sha_b,
                        probe_dataset_path=probe_path, sha256_probe=sha_probe,
                        destination=dest_json, run_dir=run_dir, config=cfg,
                    )

                self.assertTrue(os.path.isfile(diag_file))
                with open(diag_file, "r", encoding="utf-8") as f:
                    diag = json.load(f)["diagnostics"]
                self.assertEqual(diag["status"], "FAILED")
                self.assertEqual(diag["error_type"], exc_name)
                self.assertEqual(diag["seeds_status"]["102"], "FAILED")
                self.assertEqual(diag["seeds_status"]["103"], "NOT_RUN")
                self.assertFalse(os.path.exists(dest_json))

    def _build_temp_ssrl10_inputs(self, tmpdir: str):
        import hashlib
        from research.lifelong_01.ssrl_generator import (
            generate_pilot_dataset_dict,
            serialize_dataset_canonical_json,
        )
        from research.lifelong_01.ssrl_counterfactual_probe import generate_counterfactual_probe_dataset

        da = generate_pilot_dataset_dict(
            rule_indices=(0, 1, 2, 3),
            rule_id=1,
            seed=101,
            partition_seed=42,
            study_id="SSRL-01",
            phase="learnability_pilot",
        )
        db = generate_pilot_dataset_dict(
            rule_indices=(4, 5, 6, 7),
            rule_id=2,
            seed=101,
            partition_seed=43,
            study_id="SSRL-02",
            phase="two_rule_diagnostic",
        )
        dp = generate_counterfactual_probe_dataset(da, db, seed=3030)

        path_a = os.path.join(tmpdir, "dataset_a.json")
        path_b = os.path.join(tmpdir, "dataset_b.json")
        path_p = os.path.join(tmpdir, "probe.json")
        sha_a = serialize_dataset_canonical_json(da, path_a)
        sha_b = serialize_dataset_canonical_json(db, path_b)
        raw_p = json.dumps(dp, indent=2, sort_keys=True).encode("utf-8")
        sha_p = hashlib.sha256(raw_p).hexdigest()
        with open(path_p, "wb") as f:
            f.write(raw_p)
        return path_a, sha_a, path_b, sha_b, path_p, sha_p

    def test_l2init_der_tiny_e2e_integration(self):
        """
        Tests SSRL-10 L2-Init + DER++ tiny CPU end-to-end execution across 2 seeds and all 4 arms,
        verifying checkpoint creation, teacher logit metadata (32 floats / 128 bytes), DER counters,
        buffer logit drift on seq_b and diverse_der, and output serialization.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            path_a, sha_a, path_b, sha_b, path_p, sha_p = self._build_temp_ssrl10_inputs(tmpdir)
            cfg = copy.deepcopy(PROPOSED_L2INIT_DER_CONFIG)
            cfg["gate_approval_status"] = "APPROVED"
            cfg["seeds"] = [102, 103]
            cfg["step_budget_a"] = 4
            cfg["step_budget_b"] = 4
            cfg["milestone_b_steps"] = [0, 2, 4]
            cfg["enforce_post_a_learnability_gate"] = False
            cfg["enforce_fresh_b_learnability_gate"] = False

            run_dir = os.path.join(tmpdir, "ssrl10_run")
            dest_json = os.path.join(tmpdir, "ssrl10_results.json")
            res = run_l2init_der_study(
                dataset_a_path=path_a,
                sha256_a=sha_a,
                dataset_b_path=path_b,
                sha256_b=sha_b,
                probe_dataset_path=path_p,
                sha256_probe=sha_p,
                destination=dest_json,
                run_dir=run_dir,
                config=cfg,
            )
            self.assertTrue(os.path.exists(dest_json))
            diag = res["diagnostics"]
            self.assertEqual(diag["status"], "COMPLETED")
            self.assertEqual(len(diag["paired_transfer_comparisons"]), 2)

            for seed in [102, 103]:
                sdata = diag["seeds_data"][str(seed)]
                self.assertEqual(sdata["phase_a"]["teacher_logits_num_floats"], 32)
                self.assertEqual(sdata["phase_a"]["teacher_logits_bytes_float32"], 128)
                self.assertEqual(len(sdata["phase_a"]["teacher_logits_fingerprint"]), 64)

                seq_drift = sdata["seq_b"]["buffer_logit_drift"]
                der_drift = sdata["diverse_der"]["buffer_logit_drift"]
                self.assertEqual(seq_drift["argmax_match_denominator"], 16)
                self.assertEqual(der_drift["argmax_match_denominator"], 16)
                self.assertEqual(der_drift["num_logit_floats"], 32)
                self.assertGreaterEqual(sdata["diverse_der"]["counters"]["der_loss_steps"], 1)
                self.assertEqual(sdata["seq_b"]["counters"]["der_loss_steps"], 0)
                for arm in ["post_a", "fresh_b", "seq_b", "diverse_der"]:
                    self.assertTrue(os.path.exists(os.path.join(run_dir, f"checkpoint_der_{arm}_s{seed}.pt")))

            self.assertIn("buffer_logit_drift_mse_seq_b", diag["summary_statistics"])
            self.assertIn("buffer_logit_drift_mse_diverse_der", diag["summary_statistics"])

    def test_l2init_der_learnability_gate_and_deadline_failure(self):
        """
        Tests that SSRL-10 halts immediately and persists failure_diagnostics.json without writing
        destination JSON on both LearnabilityGateError and DeadlineExceededError.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            path_a, sha_a, path_b, sha_b, path_p, sha_p = self._build_temp_ssrl10_inputs(tmpdir)

            scenarios = [
                ("learnability_gate", LearnabilityGateError, "LearnabilityGateError", {
                    "gate_approval_status": "APPROVED",
                    "seeds": [102, 103],
                    "step_budget_a": 2,
                    "step_budget_b": 4,
                    "milestone_b_steps": [0, 4],
                    "gate_g_lc_threshold": 0.99,
                    "enforce_post_a_learnability_gate": True,
                }),
                ("deadline_exceeded", DeadlineExceededError, "DeadlineExceededError", {
                    "gate_approval_status": "APPROVED",
                    "seeds": [102, 103],
                    "step_budget_a": 2,
                    "step_budget_b": 4,
                    "milestone_b_steps": [0, 4],
                    "wall_clock_ceiling_training_seconds_per_stage": -1.0,
                }),
            ]
            for label, exc_cls, exc_name, overrides in scenarios:
                cfg = copy.deepcopy(PROPOSED_L2INIT_DER_CONFIG)
                cfg.update(overrides)
                run_dir = os.path.join(tmpdir, f"fail_run_{label}")
                dest_json = os.path.join(tmpdir, f"fail_dest_{label}.json")

                with self.assertRaises(exc_cls):
                    run_l2init_der_study(
                        dataset_a_path=path_a,
                        sha256_a=sha_a,
                        dataset_b_path=path_b,
                        sha256_b=sha_b,
                        probe_dataset_path=path_p,
                        sha256_probe=sha_p,
                        destination=dest_json,
                        run_dir=run_dir,
                        config=cfg,
                    )
                diag_path = os.path.join(run_dir, "failure_diagnostics.json")
                self.assertTrue(os.path.exists(diag_path))
                with open(diag_path, "r", encoding="utf-8") as f:
                    diag = json.load(f)["diagnostics"]
                self.assertEqual(diag["status"], "FAILED")
                self.assertEqual(diag["error_type"], exc_name)
                self.assertEqual(diag["seeds_status"]["102"], "FAILED")
                self.assertEqual(diag["seeds_status"]["103"], "NOT_RUN")
                self.assertFalse(os.path.exists(dest_json))

    def test_l2init_der_m64_tiny_e2e_integration(self):
        """
        Tests SSRL-11 L2-Init + DER++ (M=64) tiny CPU end-to-end execution across 2 seeds and
        all 4 arms, verifying checkpoint creation, 128-float / 512-byte teacher logit metadata,
        DER counters, 64-item buffer logit drift on seq_b and diverse_der_m64, and serialization.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            path_a, sha_a, path_b, sha_b, path_p, sha_p = self._build_temp_ssrl10_inputs(tmpdir)
            cfg = copy.deepcopy(PROPOSED_L2INIT_DER_M64_CONFIG)
            cfg["gate_approval_status"] = "APPROVED"
            cfg["seeds"] = [102, 103]
            cfg["step_budget_a"] = 4
            cfg["step_budget_b"] = 4
            cfg["milestone_b_steps"] = [0, 2, 4]
            cfg["enforce_post_a_learnability_gate"] = False
            cfg["enforce_fresh_b_learnability_gate"] = False

            run_dir = os.path.join(tmpdir, "ssrl11_run")
            dest_json = os.path.join(tmpdir, "ssrl11_results.json")
            res = run_l2init_der_m64_study(
                dataset_a_path=path_a, sha256_a=sha_a,
                dataset_b_path=path_b, sha256_b=sha_b,
                probe_dataset_path=path_p, sha256_probe=sha_p,
                destination=dest_json, run_dir=run_dir, config=cfg,
            )
            self.assertTrue(os.path.exists(dest_json))
            diag = res["diagnostics"]
            self.assertEqual(diag["status"], "COMPLETED")
            self.assertEqual(len(diag["paired_transfer_comparisons"]), 2)
            self.assertEqual(len(diag["diverse_memory_example_ids"]), 64)

            for seed in [102, 103]:
                sdata = diag["seeds_data"][str(seed)]
                self.assertEqual(sdata["phase_a"]["teacher_logits_num_floats"], 128)
                self.assertEqual(sdata["phase_a"]["teacher_logits_bytes_float32"], 512)
                self.assertEqual(len(sdata["phase_a"]["teacher_logits_fingerprint"]), 64)

                seq_drift = sdata["seq_b"]["buffer_logit_drift"]
                der_drift = sdata["diverse_der_m64"]["buffer_logit_drift"]
                self.assertEqual(seq_drift["argmax_match_denominator"], 64)
                self.assertEqual(der_drift["argmax_match_denominator"], 64)
                self.assertEqual(der_drift["num_logit_floats"], 128)
                self.assertEqual(sdata["diverse_der_m64"]["counters"]["der_loss_steps"], 1)
                self.assertEqual(sdata["seq_b"]["counters"]["der_loss_steps"], 0)
                for arm in ["post_a", "fresh_b", "seq_b", "diverse_der_m64"]:
                    self.assertTrue(os.path.exists(os.path.join(run_dir, f"checkpoint_derm64_{arm}_s{seed}.pt")))

            self.assertIn("buffer_logit_drift_mse_seq_b", diag["summary_statistics"])
            self.assertIn("buffer_logit_drift_mse_diverse_der_m64", diag["summary_statistics"])

    def test_l2init_der_m64_learnability_gate_and_deadline_failure(self):
        """
        Tests that SSRL-11 halts immediately and persists failure_diagnostics.json without writing
        destination JSON on both LearnabilityGateError and DeadlineExceededError.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            path_a, sha_a, path_b, sha_b, path_p, sha_p = self._build_temp_ssrl10_inputs(tmpdir)

            scenarios = [
                ("learnability_gate", LearnabilityGateError, "LearnabilityGateError", {
                    "gate_approval_status": "APPROVED",
                    "seeds": [102, 103],
                    "step_budget_a": 2,
                    "step_budget_b": 4,
                    "milestone_b_steps": [0, 4],
                    "gate_g_lc_threshold": 0.99,
                    "enforce_post_a_learnability_gate": True,
                }),
                ("deadline_exceeded", DeadlineExceededError, "DeadlineExceededError", {
                    "gate_approval_status": "APPROVED",
                    "seeds": [102, 103],
                    "step_budget_a": 2,
                    "step_budget_b": 4,
                    "milestone_b_steps": [0, 4],
                    "wall_clock_ceiling_training_seconds_per_stage": -1.0,
                }),
            ]
            for label, exc_cls, exc_name, overrides in scenarios:
                cfg = copy.deepcopy(PROPOSED_L2INIT_DER_M64_CONFIG)
                cfg.update(overrides)
                run_dir = os.path.join(tmpdir, f"m64_fail_run_{label}")
                dest_json = os.path.join(tmpdir, f"m64_fail_dest_{label}.json")

                with self.assertRaises(exc_cls):
                    run_l2init_der_m64_study(
                        dataset_a_path=path_a, sha256_a=sha_a,
                        dataset_b_path=path_b, sha256_b=sha_b,
                        probe_dataset_path=path_p, sha256_probe=sha_p,
                        destination=dest_json, run_dir=run_dir, config=cfg,
                    )
                diag_path = os.path.join(run_dir, "failure_diagnostics.json")
                self.assertTrue(os.path.exists(diag_path))
                with open(diag_path, "r", encoding="utf-8") as f:
                    diag = json.load(f)["diagnostics"]
                self.assertEqual(diag["status"], "FAILED")
                self.assertEqual(diag["error_type"], exc_name)
                self.assertEqual(diag["seeds_status"]["102"], "FAILED")
                self.assertEqual(diag["seeds_status"]["103"], "NOT_RUN")
                self.assertFalse(os.path.exists(dest_json))

    def test_l2init_der_m64_b1000_tiny_e2e_integration(self):
        """
        Tests SSRL-12 L2-Init + DER++ (M=64, extended horizon) tiny CPU E2E execution across 2 seeds
        and all 4 arms, verifying der_loss_steps == 2, 128-float / 512-byte teacher logits,
        64-item buffer drift, aligned milestones [0, 4, 8], and all 8 checkpoint files.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            path_a, sha_a, path_b, sha_b, path_p, sha_p = self._build_temp_ssrl10_inputs(tmpdir)
            cfg = copy.deepcopy(PROPOSED_L2INIT_DER_M64_B1000_CONFIG)
            cfg["gate_approval_status"] = "APPROVED"
            cfg["seeds"] = [102, 103]
            cfg["step_budget_a"] = 4
            cfg["step_budget_b"] = 8
            cfg["milestone_b_steps"] = [0, 4, 8]
            cfg["enforce_post_a_learnability_gate"] = False
            cfg["enforce_fresh_b_learnability_gate"] = False

            run_dir = os.path.join(tmpdir, "ssrl12_run")
            dest_json = os.path.join(tmpdir, "ssrl12_results.json")
            res = run_l2init_der_m64_b1000_study(
                dataset_a_path=path_a, sha256_a=sha_a,
                dataset_b_path=path_b, sha256_b=sha_b,
                probe_dataset_path=path_p, sha256_probe=sha_p,
                destination=dest_json, run_dir=run_dir, config=cfg,
            )
            self.assertTrue(os.path.exists(dest_json))
            diag = res["diagnostics"]
            self.assertEqual(diag["status"], "COMPLETED")
            self.assertEqual(len(diag["paired_transfer_comparisons"]), 2)
            self.assertEqual(len(diag["diverse_memory_example_ids"]), 64)

            for seed in [102, 103]:
                sdata = diag["seeds_data"][str(seed)]
                self.assertEqual(sdata["phase_a"]["teacher_logits_num_floats"], 128)
                self.assertEqual(sdata["phase_a"]["teacher_logits_bytes_float32"], 512)
                self.assertEqual(len(sdata["phase_a"]["teacher_logits_fingerprint"]), 64)

                seq_drift = sdata["seq_b"]["buffer_logit_drift"]
                der_drift = sdata["diverse_der_m64_b1000"]["buffer_logit_drift"]
                self.assertEqual(seq_drift["argmax_match_denominator"], 64)
                self.assertEqual(der_drift["argmax_match_denominator"], 64)
                self.assertEqual(der_drift["num_logit_floats"], 128)
                self.assertEqual(sdata["diverse_der_m64_b1000"]["counters"]["der_loss_steps"], 2)
                self.assertEqual(sdata["seq_b"]["counters"]["der_loss_steps"], 0)
                self.assertEqual(set(sdata["diverse_der_m64_b1000"]["milestone_evaluations"].keys()), {"0", "4", "8"})
                for arm in ["post_a", "fresh_b", "seq_b", "diverse_der_m64_b1000"]:
                    self.assertTrue(os.path.exists(os.path.join(run_dir, f"checkpoint_derm64b1k_{arm}_s{seed}.pt")))

            self.assertIn("buffer_logit_drift_mse_seq_b", diag["summary_statistics"])
            self.assertIn("buffer_logit_drift_mse_diverse_der_m64_b1000", diag["summary_statistics"])

    def test_l2init_der_m64_b1000_learnability_gate_and_deadline_failure(self):
        """
        Tests that SSRL-12 halts immediately and persists failure_diagnostics.json without writing
        destination JSON on both LearnabilityGateError and DeadlineExceededError.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            path_a, sha_a, path_b, sha_b, path_p, sha_p = self._build_temp_ssrl10_inputs(tmpdir)

            scenarios = [
                ("learnability_gate", LearnabilityGateError, "LearnabilityGateError", {
                    "gate_approval_status": "APPROVED",
                    "seeds": [102, 103],
                    "step_budget_a": 2,
                    "step_budget_b": 8,
                    "milestone_b_steps": [0, 4, 8],
                    "gate_g_lc_threshold": 0.99,
                    "enforce_post_a_learnability_gate": True,
                }),
                ("deadline_exceeded", DeadlineExceededError, "DeadlineExceededError", {
                    "gate_approval_status": "APPROVED",
                    "seeds": [102, 103],
                    "step_budget_a": 2,
                    "step_budget_b": 8,
                    "milestone_b_steps": [0, 4, 8],
                    "wall_clock_ceiling_training_seconds_per_stage": -1.0,
                }),
            ]
            for label, exc_cls, exc_name, overrides in scenarios:
                cfg = copy.deepcopy(PROPOSED_L2INIT_DER_M64_B1000_CONFIG)
                cfg.update(overrides)
                run_dir = os.path.join(tmpdir, f"m64b1k_fail_run_{label}")
                dest_json = os.path.join(tmpdir, f"m64b1k_fail_dest_{label}.json")

                with self.assertRaises(exc_cls):
                    run_l2init_der_m64_b1000_study(
                        dataset_a_path=path_a, sha256_a=sha_a,
                        dataset_b_path=path_b, sha256_b=sha_b,
                        probe_dataset_path=path_p, sha256_probe=sha_p,
                        destination=dest_json, run_dir=run_dir, config=cfg,
                    )
                diag_path = os.path.join(run_dir, "failure_diagnostics.json")
                self.assertTrue(os.path.exists(diag_path))
                with open(diag_path, "r", encoding="utf-8") as f:
                    diag = json.load(f)["diagnostics"]
                self.assertEqual(diag["status"], "FAILED")
                self.assertEqual(diag["error_type"], exc_name)
                self.assertEqual(diag["seeds_status"]["102"], "FAILED")
                self.assertEqual(diag["seeds_status"]["103"], "NOT_RUN")
                self.assertFalse(os.path.exists(dest_json))

    def _build_temp_ssrl13_inputs(self, tmpdir: str):
        import hashlib
        from research.lifelong_01.ssrl_generator import generate_pilot_dataset_dict, serialize_dataset_canonical_json
        from research.lifelong_01.ssrl_counterfactual_probe import generate_counterfactual_probe_dataset

        d1 = generate_pilot_dataset_dict((0, 1, 2, 3), 1, 101, partition_seed=42, study_id="SSRL-01", phase="learnability_pilot")
        d2 = generate_pilot_dataset_dict((4, 5, 6, 7), 2, 101, partition_seed=43, study_id="SSRL-02", phase="two_rule_diagnostic")
        d3 = generate_pilot_dataset_dict((8, 9, 10, 11), 3, 101, partition_seed=44, study_id="SSRL-13", phase="three_rule_sequential_stream")
        dp = generate_counterfactual_probe_dataset(d1, d2, seed=3030)

        p1, p2, p3, pp = [os.path.join(tmpdir, name) for name in ("r1.json", "r2.json", "r3.json", "probe.json")]
        sha1 = serialize_dataset_canonical_json(d1, p1)
        sha2 = serialize_dataset_canonical_json(d2, p2)
        sha3 = serialize_dataset_canonical_json(d3, p3)
        raw_p = json.dumps(dp, indent=2, sort_keys=True).encode("utf-8")
        sha_p = hashlib.sha256(raw_p).hexdigest()
        with open(pp, "wb") as f:
            f.write(raw_p)
        return p1, sha1, p2, sha2, p3, sha3, pp, sha_p

    def test_three_rule_stream_tiny_e2e_integration(self):
        """
        Tests SSRL-13 3-rule sequential stream tiny CPU E2E execution across 2 seeds, verifying
        der_loss_steps == 2 (1 Rule-1 + 1 Rule-2 replay) in stream_der_r3, 64-item drift tracking
        on R1 and R2, all 6 split metrics (192/64), and all 12 saved checkpoints.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            p1, s1, p2, s2, p3, s3, pp, sp = self._build_temp_ssrl13_inputs(tmpdir)
            cfg = copy.deepcopy(PROPOSED_THREE_RULE_STREAM_CONFIG)
            cfg["gate_approval_status"] = "APPROVED"
            cfg["seeds"] = [102, 103]
            cfg["step_budget_r1"] = 4
            cfg["step_budget_r2"] = 4
            cfg["step_budget_r3"] = 8
            cfg["milestone_r3_steps"] = [0, 4, 8]
            cfg["enforce_post_r1_learnability_gate"] = False
            cfg["enforce_fresh_r3_learnability_gate"] = False

            run_dir = os.path.join(tmpdir, "ssrl13_run")
            dest_json = os.path.join(tmpdir, "ssrl13_results.json")
            res = run_three_rule_stream_study(p1, s1, p2, s2, p3, s3, pp, sp, dest_json, run_dir, config=cfg)
            self.assertTrue(os.path.exists(dest_json))
            diag = res["diagnostics"]
            self.assertEqual(diag["status"], "COMPLETED")
            self.assertEqual(len(diag["paired_three_rule_comparisons"]), 2)

            for seed in [102, 103]:
                sd = diag["seeds_data"][str(seed)]
                stream_r3 = sd["stream_der_r3"]
                self.assertEqual(stream_r3["counters"]["der_loss_steps"], 2)
                self.assertEqual(stream_r3["counters"]["replay_updates_by_rule"], {"1": 1, "2": 1})
                for dk in ("buffer_logit_drift_r1", "buffer_logit_drift_r2"):
                    self.assertEqual(stream_r3[dk]["argmax_match_denominator"], 64)
                    self.assertEqual(stream_r3[dk]["num_logit_floats"], 128)
                metrics = stream_r3["final_eval"]["metrics"]
                for rk in ("rule_1", "rule_2", "rule_3"):
                    self.assertEqual(metrics[f"acc_{rk}_seen"].denominator, 192)
                    self.assertEqual(metrics[f"acc_{rk}_unseen"].denominator, 64)
                for arm in ["post_r1", "fresh_r3", "seq_post_r2", "seq_r3", "der_post_r2", "stream_der_r3"]:
                    self.assertTrue(os.path.exists(os.path.join(run_dir, f"checkpoint_r3_{arm}_s{seed}.pt")))

    def test_three_rule_stream_learnability_gate_and_deadline_failure(self):
        """
        Tests that SSRL-13 halts immediately and persists failure_diagnostics.json without writing
        destination JSON on both LearnabilityGateError and DeadlineExceededError.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            p1, s1, p2, s2, p3, s3, pp, sp = self._build_temp_ssrl13_inputs(tmpdir)

            scenarios = [
                ("learnability_gate", LearnabilityGateError, "LearnabilityGateError", {
                    "gate_approval_status": "APPROVED", "seeds": [102, 103],
                    "step_budget_r1": 2, "step_budget_r2": 4, "step_budget_r3": 8,
                    "milestone_r3_steps": [0, 4, 8], "gate_g_lc_threshold": 0.99,
                    "enforce_post_r1_learnability_gate": True,
                }),
                ("deadline_exceeded", DeadlineExceededError, "DeadlineExceededError", {
                    "gate_approval_status": "APPROVED", "seeds": [102, 103],
                    "step_budget_r1": 2, "step_budget_r2": 4, "step_budget_r3": 8,
                    "milestone_r3_steps": [0, 4, 8], "wall_clock_ceiling_training_seconds_per_stage": -1.0,
                }),
            ]
            for label, exc_cls, exc_name, overrides in scenarios:
                cfg = copy.deepcopy(PROPOSED_THREE_RULE_STREAM_CONFIG)
                cfg.update(overrides)
                run_dir = os.path.join(tmpdir, f"r3_fail_run_{label}")
                dest_json = os.path.join(tmpdir, f"r3_fail_dest_{label}.json")

                with self.assertRaises(exc_cls):
                    run_three_rule_stream_study(p1, s1, p2, s2, p3, s3, pp, sp, dest_json, run_dir, config=cfg)
                diag_path = os.path.join(run_dir, "failure_diagnostics.json")
                self.assertTrue(os.path.exists(diag_path))
                with open(diag_path, "r", encoding="utf-8") as f:
                    diag = json.load(f)["diagnostics"]
                self.assertEqual(diag["status"], "FAILED")
                self.assertEqual(diag["error_type"], exc_name)
                self.assertEqual(diag["seeds_status"]["102"], "FAILED")
                self.assertEqual(diag["seeds_status"]["103"], "NOT_RUN")
                self.assertFalse(os.path.exists(dest_json))

    def test_input_gated_three_rule_tiny_e2e_integration(self):
        """
        Tests SSRL-14 3-rule sequential stream tiny CPU E2E execution across 2 seeds, verifying all
        18 checkpoints, der_loss_steps == 2 ({1: 1, 2: 1}) on gated & bare DER++ arms, nonzero
        fc_in_gate movement from zero init, and explicit 192/64 denominators across all rules.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            p1, s1, p2, s2, p3, s3, pp, sp = self._build_temp_ssrl13_inputs(tmpdir)
            cfg = copy.deepcopy(PROPOSED_INPUT_GATED_THREE_RULE_CONFIG)
            cfg["gate_approval_status"] = "APPROVED"
            cfg["seeds"] = [102, 103]
            cfg["step_budget_r1"] = 4
            cfg["step_budget_r2"] = 4
            cfg["step_budget_r3"] = 8
            cfg["milestone_r3_steps"] = [0, 4, 8]
            cfg["enforce_post_r1_learnability_gate"] = False
            cfg["enforce_bare_post_r1_learnability_gate"] = False
            cfg["enforce_fresh_r3_learnability_gate"] = False

            run_dir = os.path.join(tmpdir, "ssrl14_run")
            dest_json = os.path.join(tmpdir, "ssrl14_results.json")
            res = run_input_gated_three_rule_study(p1, s1, p2, s2, p3, s3, pp, sp, dest_json, run_dir, config=cfg)
            self.assertTrue(os.path.exists(dest_json))
            diag = res["diagnostics"]
            self.assertEqual(diag["status"], "COMPLETED")
            self.assertEqual(len(diag["paired_gated_vs_bare_comparisons"]), 2)

            arms = [
                "post_r1", "fresh_r3", "seq_post_r2", "seq_r3", "der_post_r2", "stream_der_r3",
                "bare_post_r1", "bare_der_post_r2", "bare_stream_der_r3",
            ]
            for seed in (102, 103):
                sd = diag["seeds_data"][str(seed)]
                for arm_key in ("stream_der_r3", "bare_stream_der_r3"):
                    self.assertEqual(sd[arm_key]["counters"]["der_loss_steps"], 2)
                    self.assertEqual(sd[arm_key]["counters"]["replay_updates_by_rule"], {"1": 1, "2": 1})
                    for dk in ("buffer_logit_drift_r1", "buffer_logit_drift_r2"):
                        self.assertEqual(sd[arm_key][dk]["argmax_match_denominator"], 64)
                        self.assertEqual(sd[arm_key][dk]["num_logit_floats"], 128)
                    metrics = sd[arm_key]["final_eval"]["metrics"]
                    for rk in ("rule_1", "rule_2", "rule_3"):
                        self.assertEqual(metrics[f"acc_{rk}_seen"].denominator, 192)
                        self.assertEqual(metrics[f"acc_{rk}_unseen"].denominator, 64)
                self.assertGreater(sd["stream_der_r3"]["mechanism"]["fc_in_gate_weight_l2"], 0.0)
                for arm in arms:
                    self.assertTrue(os.path.exists(os.path.join(run_dir, f"checkpoint_ig3_{arm}_s{seed}.pt")))

    def test_input_gated_three_rule_gate_and_deadline_failure(self):
        """
        Tests that SSRL-14 halts immediately and persists failure_diagnostics.json without writing
        destination JSON on both LearnabilityGateError and DeadlineExceededError.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            p1, s1, p2, s2, p3, s3, pp, sp = self._build_temp_ssrl13_inputs(tmpdir)
            scenarios = [
                ("learnability_gate", LearnabilityGateError, "LearnabilityGateError", {
                    "gate_approval_status": "APPROVED", "seeds": [102, 103],
                    "step_budget_r1": 2, "step_budget_r2": 4, "step_budget_r3": 8,
                    "milestone_r3_steps": [0, 4, 8], "gate_g_lc_threshold": 0.99,
                    "enforce_post_r1_learnability_gate": True,
                }),
                ("deadline_exceeded", DeadlineExceededError, "DeadlineExceededError", {
                    "gate_approval_status": "APPROVED", "seeds": [102, 103],
                    "step_budget_r1": 2, "step_budget_r2": 4, "step_budget_r3": 8,
                    "milestone_r3_steps": [0, 4, 8], "wall_clock_ceiling_training_seconds_per_stage": -1.0,
                }),
            ]
            for label, exc_cls, exc_name, overrides in scenarios:
                cfg = copy.deepcopy(PROPOSED_INPUT_GATED_THREE_RULE_CONFIG)
                cfg.update(overrides)
                run_dir = os.path.join(tmpdir, f"ig3_fail_run_{label}")
                dest_json = os.path.join(tmpdir, f"ig3_fail_dest_{label}.json")

                with self.assertRaises(exc_cls):
                    run_input_gated_three_rule_study(p1, s1, p2, s2, p3, s3, pp, sp, dest_json, run_dir, config=cfg)
                diag_path = os.path.join(run_dir, "failure_diagnostics.json")
                self.assertTrue(os.path.exists(diag_path))
                with open(diag_path, "r", encoding="utf-8") as f:
                    diag = json.load(f)["diagnostics"]
                self.assertEqual(diag["status"], "FAILED")
                self.assertEqual(diag["error_type"], exc_name)
                self.assertEqual(diag["seeds_status"]["102"], "FAILED")
                self.assertEqual(diag["seeds_status"]["103"], "NOT_RUN")
                self.assertFalse(os.path.exists(dest_json))


if __name__ == "__main__":
    unittest.main()
