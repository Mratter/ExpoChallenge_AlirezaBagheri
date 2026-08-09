from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import onnx
import onnxruntime as ort

import scripts.select_policy_v3 as select_v3
import scripts.train_policy_v3 as train_v3
import scripts.evaluate_policy_v3 as evaluate_v3
from model.ppo_v3 import ArtifactError, _create_session
from scripts.evaluate_policy_v3 import (
    FinalEvaluationError,
    SELECTED_POLICY_ID,
    _aggregate,
    _expected_case_order,
    _load_durable_rows,
    _publish_or_verify_final_terminal,
    _validate_selected_session,
    _validate_ordered_row,
)
from scripts.select_policy_v3 import (
    SelectionError,
    _export_selected_onnx,
    _materialize_selected_deployment,
    _onnx_interface_receipt,
)
from scripts.train_policy_v3 import (
    _commit_completed_stage,
    _load_completed_stage_records,
    _load_curriculum_receipt,
    _publish_sealed_release,
    publish_new,
)
from scripts.v3_protocol import (
    SCIENTIFIC_SOURCE_PATHS,
    canonical_hash,
    expected_protocol,
)


def _summary(*, solved: bool, trajectory: str) -> dict[str, object]:
    return {
        "solved": solved,
        "status": "solved" if solved else "failed",
        "reason_codes": [],
        "resilience_auc": 0.75,
        "critical_service_days": 3,
        "hard_violation_count": 0,
        "max_conservation_residual": 0.0,
        "trajectory_sha256": trajectory,
    }


class V3EvidenceToolTests(unittest.TestCase):
    def test_runtime_and_evaluator_sources_are_frozen(self) -> None:
        for relative in (
            ".python-version",
            "requirements.txt",
            "training/v3/requirements-training.txt",
            "scripts/select_policy_v3.py",
            "scripts/evaluate_policy_v3.py",
        ):
            self.assertIn(relative, SCIENTIFIC_SOURCE_PATHS)

    def test_final_contract_pins_selected_onnx_and_lock(self) -> None:
        final = expected_protocol()["final_evaluation"]
        self.assertEqual(
            final["candidate_runtime"],
            "onnxruntime_cpu_selected_deployment_only",
        )
        self.assertEqual(
            final["selected_deployment_requirements"]["onnx_path"],
            "artifacts/city_recovery_ppo.v3.selected.onnx",
        )
        self.assertTrue(final["ordered_rows_must_be_exact_final_split_prefix"])
        self.assertEqual(
            final["exclusive_lock_path"],
            "internal/training_runs/v3/final.lock",
        )

    def test_trainer_has_no_legacy_direct_production_writes(self) -> None:
        root = Path(__file__).resolve().parents[1]
        source = (root / "scripts" / "train_policy_v3.py").read_text(encoding="utf-8")
        for forbidden in (
            "LEDGER_PATH.unlink",
            "model.save(SB3_PATH)",
            "export_onnx(model)",
            "write_json(RECEIPT_PATH",
            "write_json(MANIFEST_PATH",
            "write_json(PARITY_PATH",
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn("DEVELOPMENT_ROOT / campaign_identity", source)
        self.assertIn("write_new_json(TRAINING_USE_PATH", source)
        self.assertIn("write_new_json(TERMINAL_MARKER_PATH", source)

    def test_final_order_is_exactly_forty_unique_bindings(self) -> None:
        order = _expected_case_order()
        self.assertEqual(len(order), 40)
        self.assertEqual(len({row["row_id"] for row in order}), 40)

    def test_ordered_row_validator_rejects_reordered_case(self) -> None:
        identity_sha256 = canonical_hash({"identity": "test"})
        expected = _expected_case_order()[0]
        candidate = _summary(solved=True, trajectory="a" * 64)
        baseline = _summary(solved=False, trajectory="b" * 64)
        row = {
            **expected,
            "identity_sha256": identity_sha256,
            "scenario_sha256": "c" * 64,
            "shock_schedule_sha256": "d" * 64,
            "candidate": candidate,
            "baseline": baseline,
            "absolute_outcome_pair": "ppo_only",
            "secondary_rauc_candidate_minus_baseline": 0.0,
            "deterministic_replay_mismatches": 0,
            "previous_record_sha256": "0" * 64,
            "record_sha256": "e" * 64,
        }
        _validate_ordered_row(row, expected, identity_sha256)
        tampered = {**row, "row_id": _expected_case_order()[1]["row_id"]}
        with self.assertRaises(FinalEvaluationError):
            _validate_ordered_row(tampered, expected, identity_sha256)
        boolean_delta = {
            **row,
            "candidate": {**candidate, "resilience_auc": 0.75},
            "baseline": {**baseline, "resilience_auc": 0.75},
            "secondary_rauc_candidate_minus_baseline": False,
        }
        with self.assertRaises(FinalEvaluationError):
            _validate_ordered_row(boolean_delta, expected, identity_sha256)
        negative_replay = {**row, "deterministic_replay_mismatches": -1}
        with self.assertRaises(FinalEvaluationError):
            _validate_ordered_row(negative_replay, expected, identity_sha256)

    def test_torn_complete_final_row_is_reconciled_without_real_final_cases(
        self,
    ) -> None:
        identity = {"selected_policy_id": SELECTED_POLICY_ID, "fixture": "torn"}
        identity_sha256 = canonical_hash(identity)
        expected = {
            "row_id": "fake-family:1",
            "family_id": "fake-family",
            "case_seed": 1,
            "tape_seed": 2,
        }
        candidate = _summary(solved=True, trajectory="1" * 64)
        baseline = _summary(solved=False, trajectory="2" * 64)
        row = {
            **expected,
            "identity_sha256": identity_sha256,
            "scenario_sha256": canonical_hash({}),
            "shock_schedule_sha256": canonical_hash([]),
            "candidate": candidate,
            "baseline": baseline,
            "absolute_outcome_pair": "ppo_only",
            "secondary_rauc_candidate_minus_baseline": 0.0,
            "deterministic_replay_mismatches": 0,
            "previous_record_sha256": "0" * 64,
        }
        row["record_sha256"] = canonical_hash(row)

        class FakeFamily:
            def build(self, seed: int) -> SimpleNamespace:
                self.assert_seed = seed
                return SimpleNamespace(model_dump=lambda mode: {})

            def tape_seed(self, seed: int) -> int:
                return seed + 1

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ledger = root / "final-ledger.jsonl"
            ledger.write_text(
                json.dumps(row, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            with (
                patch.object(evaluate_v3, "ROOT", root),
                patch.object(evaluate_v3, "LEDGER_PATH", ledger),
                patch.object(
                    evaluate_v3,
                    "LEDGER_RECOVERY_ROOT",
                    root / "final-ledger-recovery",
                ),
                patch.object(evaluate_v3, "FINAL_FAMILIES_V3", (FakeFamily(),)),
                patch.object(evaluate_v3, "FINAL_SEEDS_V3", (1,)),
                patch.object(evaluate_v3, "_expected_case_order", lambda: [expected]),
                patch.object(
                    evaluate_v3, "generate_disaster_tape_v3", lambda *args: []
                ),
            ):
                recovered = _load_durable_rows(identity)
            self.assertEqual(recovered, [row])
            self.assertTrue(ledger.read_bytes().endswith(b"\n"))
            receipts = list((root / "final-ledger-recovery").glob("*.json"))
            self.assertEqual(len(receipts), 1)
            receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
            self.assertEqual(receipt["action"], "append_missing_newline")
            self.assertEqual(receipt["recovered_record_sha256"], row["record_sha256"])

    def test_final_aggregate_rejects_excess_conservation_residual(self) -> None:
        identity = {
            "identity": "residual-test",
            "selected_policy_id": SELECTED_POLICY_ID,
        }
        identity_sha256 = canonical_hash(identity)
        rows = []
        for index, expected in enumerate(_expected_case_order()):
            candidate = _summary(solved=True, trajectory=f"{index:064x}")
            baseline = _summary(solved=False, trajectory=f"{index + 40:064x}")
            if index == 0:
                candidate["max_conservation_residual"] = 1.000001e-6
            rows.append(
                {
                    **expected,
                    "identity_sha256": identity_sha256,
                    "scenario_sha256": "a" * 64,
                    "shock_schedule_sha256": "b" * 64,
                    "candidate": candidate,
                    "baseline": baseline,
                    "absolute_outcome_pair": "ppo_only",
                    "secondary_rauc_candidate_minus_baseline": 0.0,
                    "deterministic_replay_mismatches": 0,
                    "previous_record_sha256": "c" * 64,
                    "record_sha256": "d" * 64,
                }
            )
        with self.assertRaises(FinalEvaluationError):
            _aggregate(identity, rows)

    def test_final_aggregate_uses_selected_deployment_policy_id(self) -> None:
        identity = {
            "identity": "selected-policy-id-test",
            "selected_policy_id": SELECTED_POLICY_ID,
        }
        identity_sha256 = canonical_hash(identity)
        rows = []
        for index, expected in enumerate(_expected_case_order()):
            rows.append(
                {
                    **expected,
                    "identity_sha256": identity_sha256,
                    "scenario_sha256": f"{index + 80:064x}",
                    "shock_schedule_sha256": f"{index + 120:064x}",
                    "candidate": _summary(
                        solved=True, trajectory=f"{index + 160:064x}"
                    ),
                    "baseline": _summary(
                        solved=False, trajectory=f"{index + 200:064x}"
                    ),
                    "absolute_outcome_pair": "ppo_only",
                    "secondary_rauc_candidate_minus_baseline": 0.0,
                    "deterministic_replay_mismatches": 0,
                    "previous_record_sha256": f"{index + 240:064x}",
                    "record_sha256": f"{index + 280:064x}",
                }
            )
        with tempfile.TemporaryDirectory() as temporary:
            ledger = Path(temporary) / "final-ledger.jsonl"
            ledger.write_text("test-ledger\n", encoding="utf-8")
            with patch("scripts.evaluate_policy_v3.LEDGER_PATH", ledger):
                report = _aggregate(identity, rows)
        self.assertEqual(report["candidate"]["id"], "city-recovery-sb3-ppo-v3-selected")

    def test_final_terminal_publication_is_write_once_and_recoverable(self) -> None:
        identity = {
            "selected_policy_id": SELECTED_POLICY_ID,
            "protocol_sha256": "1" * 64,
            "source_seal_sha256": "2" * 64,
            "scientific_source_sha256": "3" * 64,
            "selection_receipt_sha256": "4" * 64,
            "authorization_sha256": "5" * 64,
            "checkpoint_sha256": "6" * 64,
            "completed_transitions": 92_160,
            "onnx_path": "artifacts/city_recovery_ppo.v3.selected.onnx",
            "onnx_sha256": "7" * 64,
            "parity_sha256": "8" * 64,
            "manifest_sha256": "9" * 64,
            "candidate_runtime": "onnxruntime_cpu_selected_deployment_only",
            "engine_spec_sha256": "a" * 64,
            "outcome_definition_sha256": "b" * 64,
        }
        identity_sha256 = canonical_hash(identity)
        rows = []
        for index, expected in enumerate(_expected_case_order()):
            row = {
                **expected,
                "identity_sha256": identity_sha256,
                "scenario_sha256": f"{index + 400:064x}",
                "shock_schedule_sha256": f"{index + 500:064x}",
                "candidate": _summary(solved=True, trajectory=f"{index + 600:064x}"),
                "baseline": _summary(solved=False, trajectory=f"{index + 700:064x}"),
                "absolute_outcome_pair": "ppo_only",
                "secondary_rauc_candidate_minus_baseline": 0.0,
                "deterministic_replay_mismatches": 0,
                "previous_record_sha256": f"{index + 800:064x}",
                "record_sha256": f"{index + 900:064x}",
            }
            rows.append(row)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ledger = root / "internal/training_runs/v3/final-ledger.jsonl"
            result_path = root / "benchmarks/v3/final-40.json"
            use_path = root / "training/v3/final-use-receipt.json"
            terminal_path = root / "training/v3/final-terminal.json"
            ledger.parent.mkdir(parents=True)
            ledger.write_text(
                "".join(
                    json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                    for row in rows
                ),
                encoding="utf-8",
            )
            use_path.parent.mkdir(parents=True)
            use_path.write_text("{}\n", encoding="utf-8")
            with (
                patch.object(evaluate_v3, "ROOT", root),
                patch.object(evaluate_v3, "LEDGER_PATH", ledger),
                patch.object(evaluate_v3, "RESULT_PATH", result_path),
                patch.object(evaluate_v3, "USE_RECEIPT_PATH", use_path),
                patch.object(evaluate_v3, "FINAL_TERMINAL_PATH", terminal_path),
            ):
                result = _aggregate(identity, rows)
                result_path.parent.mkdir(parents=True)
                result_path.write_text(
                    json.dumps(result, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                first = _publish_or_verify_final_terminal(identity, rows, result)
                second = _publish_or_verify_final_terminal(identity, rows, result)
                self.assertEqual(first, second)
                self.assertEqual(first["selected_policy_id"], SELECTED_POLICY_ID)
                self.assertEqual(first["final_ledger"]["record_count"], 40)
                terminal_path.write_text("{}\n", encoding="utf-8")
                with self.assertRaises(FinalEvaluationError):
                    _publish_or_verify_final_terminal(identity, rows, result)

    def test_selected_onnx_contract_is_exact_cpu_opset17_73_to_22(self) -> None:
        graph_input = onnx.helper.make_tensor_value_info(
            "observation", onnx.TensorProto.FLOAT, ["batch", 73]
        )
        graph_output = onnx.helper.make_tensor_value_info(
            "action", onnx.TensorProto.FLOAT, ["batch", 22]
        )
        weights = onnx.numpy_helper.from_array(
            np.zeros((73, 22), dtype=np.float32), name="weights"
        )
        graph = onnx.helper.make_graph(
            [onnx.helper.make_node("MatMul", ["observation", "weights"], ["action"])],
            "v3-test-interface",
            [graph_input],
            [graph_output],
            [weights],
        )
        model = onnx.helper.make_model(
            graph, opset_imports=[onnx.helper.make_opsetid("", 17)]
        )
        model.ir_version = 10
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "policy.onnx"
            onnx.save_model(model, path)
            session = ort.InferenceSession(
                path.read_bytes(), providers=["CPUExecutionProvider"]
            )
            receipt = _onnx_interface_receipt(path, session)
            self.assertEqual(receipt["execution_providers"], ["CPUExecutionProvider"])
            self.assertEqual(receipt["input_shape"], ["batch", 73])
            self.assertEqual(receipt["output_shape"], ["batch", 22])
            _validate_selected_session(path, session)
            _create_session(path.read_bytes())

            model.opset_import[0].version = 16
            bad_path = Path(temporary) / "policy-opset16.onnx"
            onnx.save_model(model, bad_path)
            bad_session = ort.InferenceSession(
                bad_path.read_bytes(), providers=["CPUExecutionProvider"]
            )
            with self.assertRaises(SelectionError):
                _onnx_interface_receipt(bad_path, bad_session)
            with self.assertRaises(FinalEvaluationError):
                _validate_selected_session(bad_path, bad_session)
            with self.assertRaises(ArtifactError):
                _create_session(bad_path.read_bytes())

    def test_selected_export_pins_symbolic_action_dimension(self) -> None:
        class FakePolicy:
            def eval(self) -> FakePolicy:
                return self

        class FakeModel:
            policy = object()

        graph_input = onnx.helper.make_tensor_value_info(
            "observation", onnx.TensorProto.FLOAT, ["batch", 73]
        )
        graph_output = onnx.helper.make_tensor_value_info(
            "action", onnx.TensorProto.FLOAT, ["batch", "Clipaction_dim_1"]
        )
        weights = onnx.numpy_helper.from_array(
            np.zeros((73, 22), dtype=np.float32), name="weights"
        )
        symbolic = onnx.helper.make_model(
            onnx.helper.make_graph(
                [
                    onnx.helper.make_node(
                        "MatMul", ["observation", "weights"], ["action"]
                    )
                ],
                "symbolic-selected-export",
                [graph_input],
                [graph_output],
                [weights],
            ),
            opset_imports=[onnx.helper.make_opsetid("", 17)],
        )
        symbolic.ir_version = 10
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "selected.onnx"
            exported_temporary = output.with_suffix(output.suffix + ".tmp")

            def fake_export(*_args: object, **_kwargs: object) -> None:
                onnx.save_model(symbolic, exported_temporary)

            with (
                patch(
                    "scripts.select_policy_v3._OnnxablePolicy",
                    return_value=FakePolicy(),
                ),
                patch("scripts.select_policy_v3.torch.zeros", return_value=object()),
                patch(
                    "scripts.select_policy_v3.torch.onnx.export",
                    side_effect=fake_export,
                ),
            ):
                _export_selected_onnx(FakeModel(), output)

            session = ort.InferenceSession(
                output.read_bytes(), providers=["CPUExecutionProvider"]
            )
            receipt = _onnx_interface_receipt(output, session)
            self.assertEqual(receipt["output_shape"], ["batch", 22])

    def test_selected_publication_resumes_after_every_artifact_boundary(self) -> None:
        class SimulatedOutage(RuntimeError):
            pass

        for published_before_outage in range(4):
            with self.subTest(published_before_outage=published_before_outage):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    protocol = root / "training" / "v3" / "protocol.json"
                    seal = root / "training" / "v3" / "source-seal.json"
                    selected_onnx = root / "artifacts" / "selected.onnx"
                    selected_parity = root / "training" / "v3" / "selected-parity.json"
                    selected_manifest = root / "artifacts" / "selected-manifest.json"
                    checkpoint = root / "checkpoints" / "selected.zip"
                    protocol.parent.mkdir(parents=True)
                    checkpoint.parent.mkdir(parents=True)
                    protocol.write_bytes(b"protocol\n")
                    seal.write_bytes(b"seal\n")
                    checkpoint.write_bytes(b"checkpoint\n")
                    selected = {
                        "path": checkpoint.relative_to(root).as_posix(),
                        "sha256": "a" * 64,
                        "completed_transitions": 92_160,
                        "candidate_solved_count": 25,
                    }
                    verification = {
                        "scientific_source_sha256": "b" * 64,
                        "observation_order_sha256": "c" * 64,
                        "action_order_sha256": "d" * 64,
                    }

                    def fake_export(_model: object, path: Path) -> None:
                        path.write_bytes(b"deterministic-selected-onnx\n")

                    def fake_parity(
                        _model: object,
                        _session: object,
                        chosen: dict[str, object],
                        path: Path,
                    ) -> dict[str, object]:
                        return {
                            "schema_version": 1,
                            "status": "passed",
                            "passed": True,
                            "checkpoint_sha256": chosen["sha256"],
                            "onnx_sha256": select_v3._sha256_file(path),
                        }

                    destinations = {selected_onnx, selected_parity, selected_manifest}
                    real_publish = select_v3._publish_exact_file
                    published = 0

                    def crash_after_boundary(source: Path, destination: Path) -> None:
                        nonlocal published
                        if destination in destinations:
                            if published == published_before_outage:
                                raise SimulatedOutage
                            real_publish(source, destination)
                            published += 1
                            return
                        real_publish(source, destination)

                    path_bindings = {
                        "ROOT": root,
                        "PROTOCOL_PATH": protocol,
                        "SOURCE_SEAL_PATH": seal,
                        "SELECTED_ONNX_PATH": selected_onnx,
                        "SELECTED_PARITY_PATH": selected_parity,
                        "SELECTED_MANIFEST_PATH": selected_manifest,
                    }

                    def enter_shared_patches(stack: ExitStack) -> None:
                        stack.enter_context(patch.multiple(select_v3, **path_bindings))
                        stack.enter_context(
                            patch(
                                "scripts.select_policy_v3.PPO.load",
                                return_value=object(),
                            )
                        )
                        stack.enter_context(
                            patch(
                                "scripts.select_policy_v3._export_selected_onnx",
                                side_effect=fake_export,
                            )
                        )
                        stack.enter_context(
                            patch(
                                "scripts.select_policy_v3._onnx_session",
                                return_value=object(),
                            )
                        )
                        stack.enter_context(
                            patch(
                                "scripts.select_policy_v3._selected_onnx_parity",
                                side_effect=fake_parity,
                            )
                        )

                    with ExitStack() as stack:
                        enter_shared_patches(stack)
                        stack.enter_context(
                            patch(
                                "scripts.select_policy_v3._publish_exact_file",
                                side_effect=crash_after_boundary,
                            )
                        )
                        if published_before_outage < 3:
                            with self.assertRaises(SimulatedOutage):
                                _materialize_selected_deployment(selected, verification)
                        else:
                            _materialize_selected_deployment(selected, verification)

                    with ExitStack() as stack:
                        enter_shared_patches(stack)
                        deployment = _materialize_selected_deployment(
                            selected, verification
                        )
                    for path in destinations:
                        self.assertTrue(path.is_file())
                    self.assertEqual(
                        deployment["onnx"]["sha256"],
                        select_v3._sha256_file(selected_onnx),
                    )

    def test_selected_publication_recovery_rejects_public_byte_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol = root / "training" / "v3" / "protocol.json"
            seal = root / "training" / "v3" / "source-seal.json"
            selected_onnx = root / "artifacts" / "selected.onnx"
            selected_parity = root / "training" / "v3" / "selected-parity.json"
            selected_manifest = root / "artifacts" / "selected-manifest.json"
            checkpoint = root / "checkpoints" / "selected.zip"
            protocol.parent.mkdir(parents=True)
            checkpoint.parent.mkdir(parents=True)
            protocol.write_bytes(b"protocol\n")
            seal.write_bytes(b"seal\n")
            checkpoint.write_bytes(b"checkpoint\n")
            selected = {
                "path": checkpoint.relative_to(root).as_posix(),
                "sha256": "a" * 64,
                "completed_transitions": 92_160,
            }
            verification = {
                "scientific_source_sha256": "b" * 64,
                "observation_order_sha256": "c" * 64,
                "action_order_sha256": "d" * 64,
            }

            def fake_export(_model: object, path: Path) -> None:
                path.write_bytes(b"deterministic-selected-onnx\n")

            def fake_parity(
                _model: object,
                _session: object,
                chosen: dict[str, object],
                path: Path,
            ) -> dict[str, object]:
                return {
                    "schema_version": 1,
                    "status": "passed",
                    "passed": True,
                    "checkpoint_sha256": chosen["sha256"],
                    "onnx_sha256": select_v3._sha256_file(path),
                }

            with (
                patch.multiple(
                    select_v3,
                    ROOT=root,
                    PROTOCOL_PATH=protocol,
                    SOURCE_SEAL_PATH=seal,
                    SELECTED_ONNX_PATH=selected_onnx,
                    SELECTED_PARITY_PATH=selected_parity,
                    SELECTED_MANIFEST_PATH=selected_manifest,
                ),
                patch("scripts.select_policy_v3.PPO.load", return_value=object()),
                patch(
                    "scripts.select_policy_v3._export_selected_onnx",
                    side_effect=fake_export,
                ),
                patch("scripts.select_policy_v3._onnx_session", return_value=object()),
                patch(
                    "scripts.select_policy_v3._selected_onnx_parity",
                    side_effect=fake_parity,
                ),
            ):
                _materialize_selected_deployment(selected, verification)
                selected_parity.write_bytes(b"drifted\n")
                with self.assertRaises(SelectionError):
                    _materialize_selected_deployment(selected, verification)

    def test_selection_receipt_materializes_engine_and_outcome_bindings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {
                "PROTOCOL_PATH": root / "protocol.json",
                "SOURCE_SEAL_PATH": root / "source-seal.json",
                "CONFIG_PATH": root / "config.json",
                "TRAINING_RECEIPT_PATH": root / "training-receipt.json",
                "TRAINING_USE_PATH": root / "training-use.json",
                "TRAINING_TERMINAL_PATH": root / "training-terminal.json",
                "LEDGER_PATH": root / "checkpoint-ledger.jsonl",
            }
            for name, path in paths.items():
                path.write_text(f"{name}\n", encoding="utf-8")
            report = {
                "path": "checkpoints/selected.zip",
                "sha256": "a" * 64,
                "completed_transitions": 92_160,
                "candidate_solved_count": 25,
                "candidate_mean_resilience_auc": 0.75,
            }
            verification = {
                "scientific_source_sha256": "b" * 64,
                "observation_order_sha256": "c" * 64,
                "action_order_sha256": "d" * 64,
            }
            with (
                patch.multiple(select_v3, ROOT=root, **paths),
                patch(
                    "scripts.select_policy_v3.verify_documents",
                    return_value=verification,
                ),
                patch("scripts.select_policy_v3.verify_training_authorization"),
                patch(
                    "scripts.select_policy_v3._validate_training_evidence",
                    return_value=(
                        {"campaign_provenance_sha256": "e" * 64},
                        {},
                        {},
                    ),
                ),
                patch(
                    "scripts.select_policy_v3._checkpoint_candidates",
                    return_value=[report],
                ),
                patch(
                    "scripts.select_policy_v3._evaluate_candidate",
                    side_effect=lambda value: value,
                ),
                patch(
                    "scripts.select_policy_v3._materialize_selected_deployment",
                    return_value={"deployment": "fixture"},
                ),
            ):
                receipt = select_v3.build_selection_receipt()
            self.assertEqual(
                receipt["engine_spec_sha256"], select_v3.ENGINE_V3_SPEC_SHA256
            )
            self.assertEqual(
                receipt["outcome_definition_sha256"],
                select_v3.SOLVED_DEFINITION_V3_SHA256,
            )

    def test_bc_receipt_allows_only_exact_deterministic_pre_stage_replay(self) -> None:
        campaign_identity = "a" * 64
        base = {
            "schema_version": 1,
            "status": "behavior_cloning_complete",
            "campaign_provenance_sha256": campaign_identity,
            "deterministic_replay_permitted_before_first_stage": True,
            "curriculum": {"full_dataset_mse": 0.25},
        }
        receipt = {**base, "receipt_semantic_sha256": canonical_hash(base)}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "behavior-cloning-receipt.json"
            path.write_text(json.dumps(receipt), encoding="utf-8")
            self.assertEqual(_load_curriculum_receipt(path, campaign_identity), receipt)
            receipt["curriculum"] = {"full_dataset_mse": 0.5}
            path.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                _load_curriculum_receipt(path, campaign_identity)

    def test_interrupted_publication_keeps_staging_and_rejects_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staged = root / "staging" / "model.bin"
            destination = root / "release" / "model.bin"
            staged.parent.mkdir()
            staged.write_bytes(b"registered-staged-bytes")
            publish_new(staged, destination)
            self.assertEqual(destination.read_bytes(), staged.read_bytes())
            self.assertTrue(staged.is_file())
            destination.write_bytes(b"drifted-release-bytes")
            self.assertEqual(staged.read_bytes(), b"registered-staged-bytes")
            with self.assertRaises(RuntimeError):
                publish_new(staged, destination)

    def test_stage_checkpoint_round_trip_uses_registered_boundaries(self) -> None:
        class FakeModel:
            def __init__(self, timesteps: int) -> None:
                self.num_timesteps = timesteps

            def save(self, path: Path) -> None:
                # SB3 writes an explicitly suffixed path exactly as supplied.
                # This catches regressions where a ``.tmp`` basename is passed
                # while the durability code expects a distinct ``.tmp.zip``.
                path.write_bytes(f"checkpoint:{self.num_timesteps}".encode())

        config = {
            "policy_seed": 37017,
            "production_stages": {"count": 7, "transitions_per_stage": 92160},
        }
        provenance = {"campaign": "stage-test"}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoints = root / "internal" / "training_runs" / "v3" / "checkpoints"
            ledger = root / "internal" / "training_runs" / "v3" / "ledger.jsonl"
            with patch.object(train_v3, "ROOT", root):
                first = _commit_completed_stage(
                    FakeModel(92160),
                    config,
                    provenance,
                    1,
                    checkpoints,
                    ledger,
                    "0" * 64,
                )
                _commit_completed_stage(
                    FakeModel(184320),
                    config,
                    provenance,
                    2,
                    checkpoints,
                    ledger,
                    first["record_sha256"],
                )
                loaded = _load_completed_stage_records(
                    config, provenance, checkpoints, ledger
                )
            self.assertEqual([row["num_timesteps"] for row in loaded], [92160, 184320])
            self.assertEqual(
                loaded[1]["previous_record_sha256"], first["record_sha256"]
            )

    def test_partial_six_file_release_completes_only_from_exact_staging(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage_root = root / "internal" / "training_runs" / "v3" / "staging" / "id"
            artifacts = root / "artifacts"
            training = root / "training" / "v3"
            run_root = root / "internal" / "training_runs" / "v3"
            paths = {
                "ROOT": root,
                "SB3_PATH": artifacts / "model.zip",
                "ONNX_PATH": artifacts / "model.onnx",
                "METADATA_PATH": artifacts / "metadata.json",
                "MANIFEST_PATH": artifacts / "manifest.json",
                "RECEIPT_PATH": training / "training-receipt.json",
                "PARITY_PATH": training / "parity.json",
                "TRAINING_USE_PATH": training / "use.json",
                "TERMINAL_MARKER_PATH": training / "terminal.json",
                "LEDGER_PATH": run_root / "ledger.jsonl",
            }
            campaign_provenance = {"campaign": "publication-test"}
            campaign_identity = canonical_hash(campaign_provenance)
            config = {
                "total_transitions": 645120,
                "production_stages": {"count": 7},
            }
            stage_root.mkdir(parents=True)
            training.mkdir(parents=True)
            run_root.mkdir(parents=True, exist_ok=True)
            paths["TRAINING_USE_PATH"].write_bytes(b"use-receipt\n")
            paths["LEDGER_PATH"].write_bytes(b"checkpoint-ledger\n")
            staged_by_key = {
                "sb3": stage_root / paths["SB3_PATH"].name,
                "onnx": stage_root / paths["ONNX_PATH"].name,
                "metadata": stage_root / paths["METADATA_PATH"].name,
                "manifest": stage_root / paths["MANIFEST_PATH"].name,
                "parity": stage_root / paths["PARITY_PATH"].name,
            }
            for key, staged in staged_by_key.items():
                staged.write_bytes(f"staged-{key}".encode())

            def binding(path: Path, staged: Path) -> dict[str, object]:
                return {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": train_v3.sha256_file(staged),
                }

            terminal_checkpoint = {
                **binding(paths["SB3_PATH"], staged_by_key["sb3"]),
                "bytes": staged_by_key["sb3"].stat().st_size,
            }
            receipt_base = {
                "schema_version": 1,
                "status": "completed",
                "execution_class": "sealed_authorized_campaign",
                "campaign_provenance": campaign_provenance,
                "campaign_provenance_sha256": campaign_identity,
                "completed_transitions": 645120,
                "completed_stage_count": 7,
                "checkpoint_ledger_sha256": train_v3.sha256_file(paths["LEDGER_PATH"]),
                "checkpoint_ledger_terminal_record_sha256": "e" * 64,
                "terminal_checkpoint": terminal_checkpoint,
                "artifacts": {
                    "training_onnx": binding(paths["ONNX_PATH"], staged_by_key["onnx"]),
                    "metadata": binding(
                        paths["METADATA_PATH"], staged_by_key["metadata"]
                    ),
                    "manifest": binding(
                        paths["MANIFEST_PATH"], staged_by_key["manifest"]
                    ),
                    "parity": binding(paths["PARITY_PATH"], staged_by_key["parity"]),
                },
                "manifest_sha256": train_v3.sha256_file(staged_by_key["manifest"]),
                "parity_sha256": train_v3.sha256_file(staged_by_key["parity"]),
            }
            receipt = {
                **receipt_base,
                "receipt_semantic_sha256": canonical_hash(receipt_base),
            }
            staged_receipt = stage_root / paths["RECEIPT_PATH"].name
            staged_receipt.write_text(json.dumps(receipt), encoding="utf-8")
            with patch.multiple(train_v3, **paths):
                publish_new(staged_by_key["sb3"], paths["SB3_PATH"])
                recovered = _publish_sealed_release(
                    stage_root, campaign_identity, campaign_provenance, config
                )
            self.assertEqual(recovered, receipt)
            for staged, destination in (
                (staged_by_key["sb3"], paths["SB3_PATH"]),
                (staged_by_key["onnx"], paths["ONNX_PATH"]),
                (staged_by_key["metadata"], paths["METADATA_PATH"]),
                (staged_by_key["manifest"], paths["MANIFEST_PATH"]),
                (staged_by_key["parity"], paths["PARITY_PATH"]),
                (staged_receipt, paths["RECEIPT_PATH"]),
            ):
                self.assertTrue(staged.is_file())
                self.assertEqual(staged.read_bytes(), destination.read_bytes())
            self.assertTrue(paths["TERMINAL_MARKER_PATH"].is_file())


if __name__ == "__main__":
    unittest.main()
