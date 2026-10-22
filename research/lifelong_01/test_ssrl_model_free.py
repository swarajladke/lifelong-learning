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
"""

from __future__ import annotations
import os
import time
import copy
import tempfile
import unittest
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


if __name__ == "__main__":
    unittest.main()
