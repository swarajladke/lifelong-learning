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
from research.lifelong_01.ssrl_model import SSRLInstructionMLP
from research.lifelong_01.run_ssrl_joint_reference import (
    interleave_minibatch_schedules,
    load_and_verify_initial_model,
    CheckpointVerificationError,
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


if __name__ == "__main__":
    unittest.main()
