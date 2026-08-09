"""Run the single-use V3 final 40-case benchmark with durable exact-replay rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.scenarios_v3 import FINAL_FAMILIES_V3, FINAL_SEEDS_V3  # noqa: E402
from backend.app.simulator_core import canonical_hash  # noqa: E402
from backend.app.simulator_v3 import (  # noqa: E402
    ACTION_ORDER_V3,
    ENGINE_V3_SPEC_SHA256,
    OBSERVATION_ORDER_V3,
    SOLVED_DEFINITION_V3_SHA256,
    generate_disaster_tape_v3,
    rollout_candidate_v3,
    rollout_heuristic_v3,
)
from scripts.v3_protocol import (  # noqa: E402
    FINAL_TERMINAL_PATH,
    PROTOCOL_PATH,
    SELECTED_POLICY_ID,
    SELECTED_MANIFEST_PATH,
    SELECTED_ONNX_PATH,
    SELECTED_PARITY_PATH,
    SOURCE_SEAL_PATH,
    verify_documents,
    verify_final_authorization,
)

SELECTION_RECEIPT_PATH = ROOT / "training" / "v3" / "checkpoint-selection-receipt.json"
AUTHORIZATION_PATH = ROOT / "training" / "v3" / "final-authorization.json"
USE_RECEIPT_PATH = ROOT / "training" / "v3" / "final-use-receipt.json"
LEDGER_PATH = ROOT / "internal" / "training_runs" / "v3" / "final-ledger.jsonl"
RESULT_PATH = ROOT / "benchmarks" / "v3" / "final-40.json"
LOCK_PATH = ROOT / "internal" / "training_runs" / "v3" / "final.lock"
LEDGER_RECOVERY_ROOT = (
    ROOT / "internal" / "training_runs" / "v3" / "final-ledger-recovery"
)


class FinalEvaluationError(RuntimeError):
    """Raised when the frozen final-evaluation contract cannot be honored."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fsync_parent(path: Path) -> None:
    try:
        descriptor = os.open(path.parent, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


@contextmanager
def _exclusive_final_lock(identity_sha256: str):
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    handle = LOCK_PATH.open("a+b")
    acquired = False
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
            os.fsync(handle.fileno())
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except OSError as exc:
            raise FinalEvaluationError(
                "another final evaluator owns the exclusive lock"
            ) from exc
        owner = {
            "schema_version": 1,
            "status": "final_evaluation_lock",
            "pid": os.getpid(),
            "identity_sha256": identity_sha256,
        }
        payload = (json.dumps(owner, sort_keys=True) + "\n").encode("utf-8")
        handle.seek(0)
        handle.truncate()
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        yield
    finally:
        try:
            if acquired:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise FinalEvaluationError(f"{label} is missing or invalid: {path}") from exc
    if not isinstance(value, dict):
        raise FinalEvaluationError(f"{label} root must be an object")
    return value


def _write_new_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FinalEvaluationError(f"write-once artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp.", dir=path.parent
    )
    temporary = Path(temporary_name)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FinalEvaluationError(
                f"write-once artifact appeared concurrently: {path}"
            )
        os.replace(temporary, path)
        _fsync_parent(path)
    finally:
        temporary.unlink(missing_ok=True)


def _replace_durable_bytes(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.repair.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_parent(path)
    finally:
        temporary.unlink(missing_ok=True)


def _wilson_interval(successes: int, total: int) -> list[float]:
    if total <= 0:
        raise FinalEvaluationError("Wilson interval requires at least one case")
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [round(max(0.0, center - margin), 8), round(min(1.0, center + margin), 8)]


def _expected_identity() -> dict[str, Any]:
    verification = verify_documents()
    verified_authorization = verify_final_authorization()
    selection = _load_object(SELECTION_RECEIPT_PATH, "V3 selection receipt")
    authorization = _load_object(AUTHORIZATION_PATH, "V3 final authorization")
    if authorization != verified_authorization:
        raise FinalEvaluationError("final authorization verification drifted")
    selected = selection.get("selected_checkpoint")
    if not isinstance(selected, dict):
        raise FinalEvaluationError("selected checkpoint binding is missing")
    deployment = selection.get("selected_deployment")
    if not isinstance(deployment, dict):
        raise FinalEvaluationError("selected deployment binding is missing")
    if selection.get("protocol_sha256") != _sha256_file(PROTOCOL_PATH):
        raise FinalEvaluationError("selection receipt protocol binding drifted")
    if (
        selection.get("scientific_source_sha256")
        != verification["scientific_source_sha256"]
    ):
        raise FinalEvaluationError("selection receipt source binding drifted")
    if authorization.get("status") != "final_evaluation_authorized":
        raise FinalEvaluationError("final evaluation is not authorized")
    if authorization.get("selected_policy_id") != SELECTED_POLICY_ID:
        raise FinalEvaluationError("final authorization policy identity drifted")
    if authorization.get("selection_receipt_sha256") != _sha256_file(
        SELECTION_RECEIPT_PATH
    ):
        raise FinalEvaluationError("final authorization selection binding drifted")
    if authorization.get("selected_checkpoint_sha256") != selected["sha256"]:
        raise FinalEvaluationError("final authorization checkpoint binding drifted")
    if (
        authorization.get("selected_onnx_sha256") != _sha256_file(SELECTED_ONNX_PATH)
        or authorization.get("selected_parity_sha256")
        != _sha256_file(SELECTED_PARITY_PATH)
        or authorization.get("selected_manifest_sha256")
        != _sha256_file(SELECTED_MANIFEST_PATH)
        or authorization.get("candidate_runtime")
        != "onnxruntime_cpu_selected_deployment_only"
    ):
        raise FinalEvaluationError("final authorization deployment binding drifted")
    return {
        "selected_policy_id": SELECTED_POLICY_ID,
        "protocol_sha256": _sha256_file(PROTOCOL_PATH),
        "source_seal_sha256": _sha256_file(SOURCE_SEAL_PATH),
        "scientific_source_sha256": verification["scientific_source_sha256"],
        "selection_receipt_sha256": _sha256_file(SELECTION_RECEIPT_PATH),
        "authorization_sha256": _sha256_file(AUTHORIZATION_PATH),
        "checkpoint_sha256": selected["sha256"],
        "completed_transitions": selected["completed_transitions"],
        "onnx_path": deployment["onnx"]["path"],
        "onnx_sha256": deployment["onnx"]["sha256"],
        "parity_sha256": deployment["parity"]["sha256"],
        "manifest_sha256": deployment["manifest"]["sha256"],
        "candidate_runtime": "onnxruntime_cpu_selected_deployment_only",
        "engine_spec_sha256": ENGINE_V3_SPEC_SHA256,
        "outcome_definition_sha256": SOLVED_DEFINITION_V3_SHA256,
    }


def _expected_case_order() -> list[dict[str, Any]]:
    return [
        {
            "row_id": f"{family.id}:{seed}",
            "family_id": family.id,
            "case_seed": seed,
            "tape_seed": family.tape_seed(seed),
        }
        for family in FINAL_FAMILIES_V3
        for seed in FINAL_SEEDS_V3
    ]


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_summary(summary: Any, label: str) -> None:
    expected_keys = {
        "solved",
        "status",
        "reason_codes",
        "resilience_auc",
        "critical_service_days",
        "hard_violation_count",
        "max_conservation_residual",
        "trajectory_sha256",
    }
    if not isinstance(summary, dict) or set(summary) != expected_keys:
        raise FinalEvaluationError(f"{label} summary schema drifted")
    numeric_values = (
        summary["resilience_auc"],
        summary["critical_service_days"],
        summary["max_conservation_residual"],
    )
    solved = summary["solved"]
    if (
        not isinstance(solved, bool)
        or summary["status"] != ("solved" if solved else "failed")
        or not isinstance(summary["reason_codes"], list)
        or not all(isinstance(item, str) for item in summary["reason_codes"])
        or not all(
            isinstance(value, (int, float))
            and math.isfinite(float(value))
            and not isinstance(value, bool)
            for value in numeric_values
        )
        or not 0.0 <= float(summary["resilience_auc"]) <= 1.0
        or not isinstance(summary["critical_service_days"], int)
        or isinstance(summary["critical_service_days"], bool)
        or summary["critical_service_days"] < 0
        or not isinstance(summary["hard_violation_count"], int)
        or isinstance(summary["hard_violation_count"], bool)
        or summary["hard_violation_count"] < 0
        or float(summary["max_conservation_residual"]) < 0.0
        or not _is_sha256(summary["trajectory_sha256"])
    ):
        raise FinalEvaluationError(f"{label} summary values drifted")


def _validate_ordered_row(
    record: dict[str, Any], expected: dict[str, Any], identity_sha256: str
) -> None:
    expected_keys = {
        "row_id",
        "identity_sha256",
        "family_id",
        "case_seed",
        "tape_seed",
        "scenario_sha256",
        "shock_schedule_sha256",
        "candidate",
        "baseline",
        "absolute_outcome_pair",
        "secondary_rauc_candidate_minus_baseline",
        "deterministic_replay_mismatches",
        "previous_record_sha256",
        "record_sha256",
    }
    if set(record) != expected_keys:
        raise FinalEvaluationError("final ledger row schema drifted")
    for field, value in expected.items():
        if record.get(field) != value:
            raise FinalEvaluationError(
                "final ledger is not the exact ordered split prefix"
            )
    if record.get("identity_sha256") != identity_sha256:
        raise FinalEvaluationError("final ledger belongs to another frozen identity")
    if not _is_sha256(record.get("scenario_sha256")) or not _is_sha256(
        record.get("shock_schedule_sha256")
    ):
        raise FinalEvaluationError("final ledger scenario binding is invalid")
    _validate_summary(record.get("candidate"), "candidate")
    _validate_summary(record.get("baseline"), "baseline")
    candidate_solved = record["candidate"]["solved"]
    baseline_solved = record["baseline"]["solved"]
    expected_pair = (
        "both_solved"
        if candidate_solved and baseline_solved
        else "ppo_only"
        if candidate_solved
        else "heuristic_only"
        if baseline_solved
        else "neither"
    )
    if record.get("absolute_outcome_pair") != expected_pair:
        raise FinalEvaluationError("final ledger paired outcome drifted")
    expected_delta = round(
        record["candidate"]["resilience_auc"] - record["baseline"]["resilience_auc"],
        8,
    )
    recorded_delta = record.get("secondary_rauc_candidate_minus_baseline")
    if (
        not isinstance(recorded_delta, (int, float))
        or isinstance(recorded_delta, bool)
        or not math.isfinite(float(recorded_delta))
        or recorded_delta != expected_delta
    ):
        raise FinalEvaluationError("final ledger secondary metric drifted")
    replay_mismatches = record.get("deterministic_replay_mismatches")
    if (
        not isinstance(replay_mismatches, int)
        or isinstance(replay_mismatches, bool)
        or replay_mismatches < 0
    ):
        raise FinalEvaluationError("final ledger replay count is invalid")


def _load_durable_rows(identity: dict[str, Any]) -> list[dict[str, Any]]:
    if not LEDGER_PATH.exists():
        return []
    try:
        ledger_payload = LEDGER_PATH.read_bytes()
    except OSError as exc:
        raise FinalEvaluationError("final ledger is unreadable") from exc
    torn_tail = b""
    prefix_payload = ledger_payload
    if ledger_payload and not ledger_payload.endswith(b"\n"):
        prefix_end = ledger_payload.rfind(b"\n") + 1
        prefix_payload = ledger_payload[:prefix_end]
        torn_tail = ledger_payload[prefix_end:]
    try:
        raw_lines = prefix_payload.decode("utf-8").splitlines()
    except UnicodeError as exc:
        raise FinalEvaluationError("final ledger prefix is not UTF-8") from exc
    expected_order = _expected_case_order()
    if len(raw_lines) + int(bool(torn_tail)) > len(expected_order) or any(
        not line for line in raw_lines
    ):
        raise FinalEvaluationError("final ledger length or blank-line contract drifted")
    rows: list[dict[str, Any]] = []
    previous_hash = "0" * 64
    identity_sha256 = canonical_hash(identity)

    def validated_record(line: str, line_number: int) -> dict[str, Any]:
        nonlocal previous_hash
        expected = expected_order[line_number - 1]
        try:
            record = json.loads(line)
        except ValueError as exc:
            raise FinalEvaluationError(
                f"final ledger line {line_number} is invalid"
            ) from exc
        if not isinstance(record, dict):
            raise FinalEvaluationError("final ledger record root must be an object")
        claimed_hash = record.get("record_sha256")
        unsigned = {
            key: value for key, value in record.items() if key != "record_sha256"
        }
        if record.get("previous_record_sha256") != previous_hash:
            raise FinalEvaluationError("final ledger hash chain is broken")
        if claimed_hash != canonical_hash(unsigned):
            raise FinalEvaluationError("final ledger record hash drifted")
        _validate_ordered_row(record, expected, identity_sha256)
        family_index = (line_number - 1) // len(FINAL_SEEDS_V3)
        seed_index = (line_number - 1) % len(FINAL_SEEDS_V3)
        family = FINAL_FAMILIES_V3[family_index]
        seed = FINAL_SEEDS_V3[seed_index]
        scenario = family.build(seed)
        schedule = generate_disaster_tape_v3(scenario, family.tape_seed(seed))
        if record.get("scenario_sha256") != canonical_hash(
            scenario.model_dump(mode="json")
        ) or record.get("shock_schedule_sha256") != canonical_hash(
            [item.__dict__ for item in schedule]
        ):
            raise FinalEvaluationError("final ledger scenario or tape binding drifted")
        previous_hash = str(claimed_hash)
        return record

    for line_number, line in enumerate(raw_lines, start=1):
        rows.append(validated_record(line, line_number))

    if torn_tail:
        action = "preserve_tail_digest_then_truncate_to_last_complete_record"
        repaired_payload = prefix_payload
        recovered_record_sha256: str | None = None
        try:
            tail_text = torn_tail.decode("utf-8")
            recovered = validated_record(tail_text, len(rows) + 1)
        except (UnicodeError, ValueError, FinalEvaluationError):
            pass
        else:
            rows.append(recovered)
            recovered_record_sha256 = recovered["record_sha256"]
            action = "append_missing_newline"
            repaired_payload = ledger_payload + b"\n"
        recovery_base = {
            "schema_version": 1,
            "status": "final_ledger_torn_eof_reconciled",
            "identity_sha256": identity_sha256,
            "scope": "last_non_newline_tail_only",
            "pre_recovery_ledger_sha256": hashlib.sha256(ledger_payload).hexdigest(),
            "prefix_record_count": len(raw_lines),
            "tail_bytes": len(torn_tail),
            "tail_sha256": hashlib.sha256(torn_tail).hexdigest(),
            "action": action,
            "recovered_record_sha256": recovered_record_sha256,
            "post_recovery_ledger_sha256": hashlib.sha256(repaired_payload).hexdigest(),
        }
        recovery = {
            **recovery_base,
            "receipt_semantic_sha256": canonical_hash(recovery_base),
        }
        recovery_path = LEDGER_RECOVERY_ROOT / f"{recovery_base['tail_sha256']}.json"
        if recovery_path.exists():
            if _load_object(recovery_path, "V3 final-ledger recovery receipt") != (
                recovery
            ):
                raise FinalEvaluationError("final-ledger recovery receipt drifted")
        else:
            _write_new_json(recovery_path, recovery)
        _replace_durable_bytes(LEDGER_PATH, repaired_payload)
    return rows


def _append_row(row: dict[str, Any], previous_hash: str) -> dict[str, Any]:
    record = {**row, "previous_record_sha256": previous_hash}
    record["record_sha256"] = canonical_hash(record)
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER_PATH.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_parent(LEDGER_PATH)
    return record


def _summary(result: dict[str, Any]) -> dict[str, Any]:
    outcome = result["absolute_outcome"]
    return {
        "solved": bool(outcome["solved"]),
        "status": outcome["status"],
        "reason_codes": outcome["reason_codes"],
        "resilience_auc": result["rauc"],
        "critical_service_days": result["critical_service_days"],
        "hard_violation_count": result["hard_violation_count"],
        "max_conservation_residual": result["max_logistics_conservation_residual"],
        "trajectory_sha256": result["trajectory_sha256"],
    }


def _validate_selected_session(onnx_path: Path, session: ort.InferenceSession) -> None:
    """Independently enforce the selected deployment's registered interface."""

    model = onnx.load_model_from_string(onnx_path.read_bytes())
    opsets = [
        {"domain": item.domain or "", "version": int(item.version)}
        for item in model.opset_import
    ]
    graph_inputs = list(model.graph.input)
    graph_outputs = list(model.graph.output)

    def graph_shape(value_info: Any) -> list[str | int | None]:
        dimensions: list[str | int | None] = []
        for dimension in value_info.type.tensor_type.shape.dim:
            if dimension.HasField("dim_value"):
                dimensions.append(int(dimension.dim_value))
            elif dimension.HasField("dim_param"):
                dimensions.append(str(dimension.dim_param))
            else:
                dimensions.append(None)
        return dimensions

    runtime_inputs = session.get_inputs()
    runtime_outputs = session.get_outputs()
    expected_input_shape = ["batch", len(OBSERVATION_ORDER_V3)]
    expected_output_shape = ["batch", len(ACTION_ORDER_V3)]
    if (
        session.get_providers() != ["CPUExecutionProvider"]
        or opsets != [{"domain": "", "version": 17}]
        or len(graph_inputs) != 1
        or len(graph_outputs) != 1
        or graph_inputs[0].name != "observation"
        or graph_outputs[0].name != "action"
        or graph_inputs[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT
        or graph_outputs[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT
        or graph_shape(graph_inputs[0]) != expected_input_shape
        or graph_shape(graph_outputs[0]) != expected_output_shape
        or len(runtime_inputs) != 1
        or len(runtime_outputs) != 1
        or runtime_inputs[0].name != "observation"
        or runtime_outputs[0].name != "action"
        or runtime_inputs[0].type != "tensor(float)"
        or runtime_outputs[0].type != "tensor(float)"
        or list(runtime_inputs[0].shape) != expected_input_shape
        or list(runtime_outputs[0].shape) != expected_output_shape
    ):
        raise FinalEvaluationError("selected ONNX provider or interface drifted")


def _run_case(
    family: Any,
    seed: int,
    session: ort.InferenceSession,
    identity: dict[str, Any],
) -> dict[str, Any]:
    scenario = family.build(seed)
    tape_seed = family.tape_seed(seed)
    schedule = generate_disaster_tape_v3(scenario, tape_seed)

    def action_provider(observation: np.ndarray) -> np.ndarray:
        output = session.run(
            ["action"],
            {
                "observation": np.asarray(observation, dtype=np.float32).reshape(
                    1, len(OBSERVATION_ORDER_V3)
                )
            },
        )[0]
        action = np.asarray(output, dtype=np.float64)
        if action.shape != (1, len(ACTION_ORDER_V3)) or not np.all(np.isfinite(action)):
            raise FinalEvaluationError("selected ONNX emitted an invalid action")
        return action[0]

    candidate_first = rollout_candidate_v3(
        scenario, tape_seed, action_provider, schedule
    )
    candidate_second = rollout_candidate_v3(
        scenario, tape_seed, action_provider, schedule
    )
    baseline_first = rollout_heuristic_v3(scenario, tape_seed, schedule)
    baseline_second = rollout_heuristic_v3(scenario, tape_seed, schedule)
    replay_mismatches = int(
        candidate_first["trajectory_sha256"] != candidate_second["trajectory_sha256"]
    ) + int(baseline_first["trajectory_sha256"] != baseline_second["trajectory_sha256"])
    candidate = _summary(candidate_first)
    baseline = _summary(baseline_first)
    if candidate["solved"] and baseline["solved"]:
        pair = "both_solved"
    elif candidate["solved"]:
        pair = "ppo_only"
    elif baseline["solved"]:
        pair = "heuristic_only"
    else:
        pair = "neither"
    return {
        "row_id": f"{family.id}:{seed}",
        "identity_sha256": canonical_hash(identity),
        "family_id": family.id,
        "case_seed": seed,
        "tape_seed": tape_seed,
        "scenario_sha256": canonical_hash(scenario.model_dump(mode="json")),
        "shock_schedule_sha256": canonical_hash([item.__dict__ for item in schedule]),
        "candidate": candidate,
        "baseline": baseline,
        "absolute_outcome_pair": pair,
        "secondary_rauc_candidate_minus_baseline": round(
            candidate["resilience_auc"] - baseline["resilience_auc"], 8
        ),
        "deterministic_replay_mismatches": replay_mismatches,
    }


def _aggregate(identity: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 40 or len({row["row_id"] for row in rows}) != 40:
        raise FinalEvaluationError("final benchmark requires exactly 40 unique cases")
    expected_order = _expected_case_order()
    identity_sha256 = canonical_hash(identity)
    if identity.get("selected_policy_id") != SELECTED_POLICY_ID:
        raise FinalEvaluationError("final benchmark policy identity drifted")
    for row, expected in zip(rows, expected_order):
        _validate_ordered_row(row, expected, identity_sha256)
    candidate_solved = sum(row["candidate"]["solved"] for row in rows)
    baseline_solved = sum(row["baseline"]["solved"] for row in rows)
    pair_counts = {
        key: sum(row["absolute_outcome_pair"] == key for row in rows)
        for key in ("both_solved", "ppo_only", "heuristic_only", "neither")
    }
    replay_mismatches = sum(row["deterministic_replay_mismatches"] for row in rows)
    candidate_violations = sum(row["candidate"]["hard_violation_count"] for row in rows)
    baseline_violations = sum(row["baseline"]["hard_violation_count"] for row in rows)
    maximum_residual = max(
        max(
            row["candidate"]["max_conservation_residual"],
            row["baseline"]["max_conservation_residual"],
        )
        for row in rows
    )
    if (
        replay_mismatches
        or candidate_violations
        or baseline_violations
        or maximum_residual > 1e-6
    ):
        raise FinalEvaluationError(
            "final benchmark failed its replay or safety contract"
        )
    candidate_rauc = [row["candidate"]["resilience_auc"] for row in rows]
    baseline_rauc = [row["baseline"]["resilience_auc"] for row in rows]
    candidate_wins = sum(
        left > right + 1e-9 for left, right in zip(candidate_rauc, baseline_rauc)
    )
    baseline_wins = sum(
        right > left + 1e-9 for left, right in zip(candidate_rauc, baseline_rauc)
    )
    ties = 40 - candidate_wins - baseline_wins
    return {
        "schema_version": 1,
        "artifact_type": "city_recovery_v3_final_benchmark",
        "status": "complete",
        "primary_metric": "independent_absolute_disasters_solved",
        "synthetic_case_count": 40,
        "candidate": {
            "id": SELECTED_POLICY_ID,
            "solved_count": candidate_solved,
            "failed_count": 40 - candidate_solved,
            "solve_rate": candidate_solved / 40.0,
            "solve_rate_wilson_ci95": _wilson_interval(candidate_solved, 40),
            "mean_resilience_auc": round(fmean(candidate_rauc), 10),
        },
        "baseline": {
            "id": "reactive-public-state-heuristic-v3",
            "version": "3.0.0",
            "solved_count": baseline_solved,
            "failed_count": 40 - baseline_solved,
            "solve_rate": baseline_solved / 40.0,
            "solve_rate_wilson_ci95": _wilson_interval(baseline_solved, 40),
            "mean_resilience_auc": round(fmean(baseline_rauc), 10),
        },
        "paired_absolute_outcomes": pair_counts,
        "secondary_head_to_head_resilience_auc": {
            "candidate_wins": candidate_wins,
            "baseline_wins": baseline_wins,
            "ties": ties,
            "candidate_mean_minus_baseline_mean": round(
                fmean(candidate_rauc) - fmean(baseline_rauc), 10
            ),
        },
        "invariants": {
            "all_rows_present": True,
            "unique_rows": True,
            "same_tapes": True,
            "deterministic_replay_mismatches": replay_mismatches,
            "candidate_hard_violations": candidate_violations,
            "baseline_hard_violations": baseline_violations,
            "maximum_conservation_residual": maximum_residual,
        },
        "identity": identity,
        "rows_sha256": canonical_hash(rows),
        "ledger_sha256": _sha256_file(LEDGER_PATH),
    }


def _expected_final_terminal(
    identity: dict[str, Any], rows: list[dict[str, Any]], result: dict[str, Any]
) -> dict[str, Any]:
    """Return the exact terminal marker for a fully durable final invocation."""

    if len(rows) != 40 or not LEDGER_PATH.is_file():
        raise FinalEvaluationError("final terminal requires exactly 40 durable rows")
    if not RESULT_PATH.is_file() or not USE_RECEIPT_PATH.is_file():
        raise FinalEvaluationError("final terminal requires result and use receipt")
    if result != _aggregate(identity, rows):
        raise FinalEvaluationError("final result does not match its durable ledger")
    terminal_record_sha256 = rows[-1].get("record_sha256")
    if not _is_sha256(terminal_record_sha256):
        raise FinalEvaluationError("final ledger terminal record hash is invalid")
    base = {
        "schema_version": 1,
        "status": "final_evaluation_complete",
        "selected_policy_id": SELECTED_POLICY_ID,
        "identity": identity,
        "identity_sha256": canonical_hash(identity),
        "protocol_sha256": identity["protocol_sha256"],
        "source_seal_sha256": identity["source_seal_sha256"],
        "scientific_source_sha256": identity["scientific_source_sha256"],
        "selection_receipt_sha256": identity["selection_receipt_sha256"],
        "final_authorization_sha256": identity["authorization_sha256"],
        "selected_checkpoint_sha256": identity["checkpoint_sha256"],
        "selected_onnx_sha256": identity["onnx_sha256"],
        "selected_parity_sha256": identity["parity_sha256"],
        "selected_manifest_sha256": identity["manifest_sha256"],
        "case_count": 40,
        "iteration_order": "family_order_then_ascending_seed",
        "result": {
            "path": str(RESULT_PATH.relative_to(ROOT)).replace("\\", "/"),
            "sha256": _sha256_file(RESULT_PATH),
        },
        "final_use_receipt": {
            "path": str(USE_RECEIPT_PATH.relative_to(ROOT)).replace("\\", "/"),
            "sha256": _sha256_file(USE_RECEIPT_PATH),
        },
        "final_ledger": {
            "path": str(LEDGER_PATH.relative_to(ROOT)).replace("\\", "/"),
            "sha256": _sha256_file(LEDGER_PATH),
            "record_count": 40,
            "terminal_record_sha256": terminal_record_sha256,
        },
    }
    return {**base, "terminal_semantic_sha256": canonical_hash(base)}


def _publish_or_verify_final_terminal(
    identity: dict[str, Any], rows: list[dict[str, Any]], result: dict[str, Any]
) -> dict[str, Any]:
    expected = _expected_final_terminal(identity, rows, result)
    if FINAL_TERMINAL_PATH.exists():
        actual = _load_object(FINAL_TERMINAL_PATH, "V3 final terminal marker")
        if actual != expected:
            raise FinalEvaluationError("final terminal marker identity drifted")
        return actual
    _write_new_json(FINAL_TERMINAL_PATH, expected)
    return expected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-final-once",
        action="store_true",
        help="consume the sealed final suite or resume its one registered invocation",
    )
    args = parser.parse_args()
    if not args.run_final_once:
        raise FinalEvaluationError("explicit --run-final-once is required")
    identity = _expected_identity()
    identity_sha256 = canonical_hash(identity)
    with _exclusive_final_lock(identity_sha256):
        expected_use_receipt = {
            "schema_version": 1,
            "status": "single_final_invocation_started",
            "identity": identity,
            "identity_sha256": identity_sha256,
            "case_count": 40,
            "iteration_order": "family_order_then_ascending_seed",
            "durable_resume_only": True,
            "exclusive_lock_path": str(LOCK_PATH.relative_to(ROOT)).replace("\\", "/"),
        }
        if USE_RECEIPT_PATH.exists():
            if (
                _load_object(USE_RECEIPT_PATH, "V3 final-use receipt")
                != expected_use_receipt
            ):
                raise FinalEvaluationError(
                    "final-use receipt belongs to another identity"
                )
        else:
            if (
                LEDGER_PATH.exists()
                or RESULT_PATH.exists()
                or FINAL_TERMINAL_PATH.exists()
            ):
                raise FinalEvaluationError(
                    "final state exists without its write-once use receipt"
                )
            _write_new_json(USE_RECEIPT_PATH, expected_use_receipt)

        rows = _load_durable_rows(identity)
        if RESULT_PATH.exists():
            if len(rows) != 40:
                raise FinalEvaluationError(
                    "existing final result has an incomplete durable ledger"
                )
            result = _load_object(RESULT_PATH, "V3 final aggregate result")
            expected_result = _aggregate(identity, rows)
            if result != expected_result:
                raise FinalEvaluationError(
                    "existing final result does not match its durable ledger"
                )
            _publish_or_verify_final_terminal(identity, rows, result)
            print(json.dumps(result, sort_keys=True))
            return
        if FINAL_TERMINAL_PATH.exists():
            raise FinalEvaluationError("final terminal exists without its result")
        previous_hash = rows[-1]["record_sha256"] if rows else "0" * 64
        onnx_path = ROOT / identity["onnx_path"]
        if (
            onnx_path != SELECTED_ONNX_PATH
            or not onnx_path.is_file()
            or _sha256_file(onnx_path) != identity["onnx_sha256"]
        ):
            raise FinalEvaluationError("pinned selected ONNX bytes drifted")
        options = ort.SessionOptions()
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            onnx_path.read_bytes(), options, providers=["CPUExecutionProvider"]
        )
        _validate_selected_session(onnx_path, session)
        ordered_cases = [
            (family, seed) for family in FINAL_FAMILIES_V3 for seed in FINAL_SEEDS_V3
        ]
        for case_index, (family, seed) in enumerate(ordered_cases):
            if case_index < len(rows):
                continue
            row = _run_case(family, seed, session, identity)
            record = _append_row(row, previous_hash)
            _validate_ordered_row(
                record, _expected_case_order()[case_index], identity_sha256
            )
            rows.append(record)
            previous_hash = record["record_sha256"]
        result = _aggregate(identity, rows)
        _write_new_json(RESULT_PATH, result)
        _publish_or_verify_final_terminal(identity, rows, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
