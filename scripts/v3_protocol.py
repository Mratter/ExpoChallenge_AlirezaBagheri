"""Create and verify the frozen V3 scientific protocol and source seal.

This module never builds a scenario, runs the simulator, selects a checkpoint,
or authorizes training/final evaluation.  It binds the causal contracts and
fails closed when source, protocol, split, baseline, or selection rules drift.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.scenarios_v3 import (  # noqa: E402
    DEVELOPMENT_FAMILIES_V3,
    DEVELOPMENT_SEEDS_V3,
    FINAL_FAMILIES_V3,
    FINAL_SEEDS_V3,
    TRAINING_FAMILIES_V3,
    TRAINING_SEEDS_V3,
)
from backend.app.simulator_v3 import (  # noqa: E402
    ACTION_ORDER_V3,
    ENGINE_V3_ID,
    ENGINE_V3_SPEC_SHA256,
    ENGINE_V3_VERSION,
    OBSERVATION_ORDER_V3,
    RESULT_SCHEMA_V3,
    SOLVED_DEFINITION_V3,
    SOLVED_DEFINITION_V3_SHA256,
)

PROTOCOL_PATH = ROOT / "training" / "v3" / "protocol.json"
SOURCE_SEAL_PATH = ROOT / "training" / "v3" / "source-seal.json"
TRAINING_AUTHORIZATION_PATH = ROOT / "training" / "v3" / "training-authorization.json"
FINAL_AUTHORIZATION_PATH = ROOT / "training" / "v3" / "final-authorization.json"
FINAL_TERMINAL_PATH = ROOT / "training" / "v3" / "final-terminal.json"
SELECTION_RECEIPT_PATH = ROOT / "training" / "v3" / "checkpoint-selection-receipt.json"
SELECTED_ONNX_PATH = ROOT / "artifacts" / "city_recovery_ppo.v3.selected.onnx"
SELECTED_PARITY_PATH = ROOT / "training" / "v3" / "selected-onnx-parity.json"
SELECTED_MANIFEST_PATH = ROOT / "artifacts" / "model_manifest.v3.selected.json"
CONFIG_PATH = ROOT / "training" / "v3" / "config.json"
SEALING_TOOL_PATH = ROOT / "scripts" / "v3_protocol.py"

PROTOCOL_ID = "city-recovery-v3-preregistration-v1"
SELECTED_POLICY_ID = "city-recovery-sb3-ppo-v3-selected"
SCIENTIFIC_SOURCE_PATHS = (
    ".python-version",
    "requirements.txt",
    "backend/app/models.py",
    "backend/app/scenarios_v3.py",
    "backend/app/simulator_core.py",
    "backend/app/simulator_v2.py",
    "backend/app/simulator_v3.py",
    "model/ppo_v3.py",
    "scripts/train_policy_v3.py",
    "scripts/select_policy_v3.py",
    "scripts/evaluate_policy_v3.py",
    "training/v3/config.json",
    "training/v3/requirements-training.txt",
)
FINAL_STATE_PATHS = (
    "internal/training_runs/v3/final.lock",
    "training/v3/final-use-receipt.json",
    "training/v3/final-terminal.json",
    "benchmarks/v3/final-40.json",
    "internal/training_runs/v3/final-ledger.jsonl",
)


class ProtocolError(RuntimeError):
    """Raised when a frozen V3 governance binding is invalid."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def formatted_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ProtocolError(f"{label} is missing or invalid: {path}") from exc
    if not isinstance(value, dict):
        raise ProtocolError(f"{label} root must be an object")
    return value


def _function_source_sha256(relative_path: str, function_name: str) -> str:
    path = ROOT / relative_path
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    node = next(
        (
            item
            for item in tree.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            and item.name == function_name
        ),
        None,
    )
    if node is None or node.end_lineno is None:
        raise ProtocolError(
            f"bound function is missing: {relative_path}:{function_name}"
        )
    lines = text.replace("\r\n", "\n").replace("\r", "\n").splitlines(keepends=True)
    payload = "".join(lines[node.lineno - 1 : node.end_lineno]).rstrip("\n") + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def scientific_source_identity() -> dict[str, Any]:
    files = {
        relative: file_sha256(ROOT / relative) for relative in SCIENTIFIC_SOURCE_PATHS
    }
    return {
        "files_sha256": files,
        "semantic_sha256": canonical_hash(files),
    }


def _split_contract(
    split_id: str,
    families: tuple[Any, ...],
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    if not seeds:
        raise ProtocolError(f"{split_id} seeds are empty")
    if seeds != tuple(range(seeds[0], seeds[-1] + 1)):
        raise ProtocolError(f"{split_id} seeds must be one contiguous closed interval")
    return {
        "id": split_id,
        "family_count": len(families),
        "family_ids": [family.id for family in families],
        "seed_interval": {"first": seeds[0], "last": seeds[-1], "count": len(seeds)},
        "cartesian_case_count": len(families) * len(seeds),
        "iteration_order": "family_order_then_ascending_seed",
    }


def expected_protocol() -> dict[str, Any]:
    config = _load_object(CONFIG_PATH, "V3 training config")
    if config.get("environment") != "CityRecoveryEnv-v3":
        raise ProtocolError("V3 training environment binding drifted")
    if config.get("observation_count") != len(OBSERVATION_ORDER_V3):
        raise ProtocolError("V3 training observation count drifted")
    if config.get("action_count") != len(ACTION_ORDER_V3):
        raise ProtocolError("V3 training action count drifted")
    configured_transitions = int(config.get("total_transitions", 0))
    checkpoint_interval = int(config.get("checkpoint_interval_transitions", 0))
    if (
        configured_transitions <= 0
        or checkpoint_interval <= 0
        or configured_transitions % checkpoint_interval
    ):
        raise ProtocolError(
            "V3 training budget must be a positive multiple of its checkpoint interval"
        )
    stages = config.get("production_stages")
    vectorization = config.get("vectorization")
    rollout_transitions = int(config.get("n_steps_per_lane", 0)) * int(
        config.get("simulator_lanes", 0)
    )
    if (
        not isinstance(stages, dict)
        or stages.get("count") != 7
        or stages.get("rollouts_per_stage") != 30
        or stages.get("transitions_per_stage") != 30 * rollout_transitions
        or stages.get("episode_steps_per_lane") != 30
        or (stages["transitions_per_stage"] // int(config["simulator_lanes"]))
        % stages["episode_steps_per_lane"]
        != 0
        or stages.get("transitions_per_stage") != checkpoint_interval
        or stages.get("count") * checkpoint_interval != configured_transitions
        or stages.get("resume_at_stage_boundaries") is not True
        or stages.get("fresh_environment_per_stage") is not True
        or vectorization
        != {
            "backend": "SubprocVecEnv",
            "parent_torch_inter_op_threads": 1,
            "parent_torch_intra_op_threads": 12,
            "start_method": "spawn",
            "worker_blas_threads": 1,
        }
    ):
        raise ProtocolError("V3 production stage/resume contract drifted")
    source = scientific_source_identity()
    training = _split_contract("training", TRAINING_FAMILIES_V3, TRAINING_SEEDS_V3)
    development = _split_contract(
        "development", DEVELOPMENT_FAMILIES_V3, DEVELOPMENT_SEEDS_V3
    )
    final = _split_contract("final", FINAL_FAMILIES_V3, FINAL_SEEDS_V3)
    all_seeds = [
        set(TRAINING_SEEDS_V3),
        set(DEVELOPMENT_SEEDS_V3),
        set(FINAL_SEEDS_V3),
    ]
    if any(
        all_seeds[left] & all_seeds[right]
        for left in range(3)
        for right in range(left + 1, 3)
    ):
        raise ProtocolError("V3 split seed intervals overlap")
    if final["cartesian_case_count"] != 40:
        raise ProtocolError("the final V3 contract must contain exactly 40 cases")

    baseline_path = "backend/app/simulator_v3.py"
    baseline_function = "reactive_heuristic_action_v3"
    config_sha256 = file_sha256(CONFIG_PATH)
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "preregistered_separate_authorizations",
        "scientific_source": source,
        "bindings": {
            "environment": {
                "id": ENGINE_V3_ID,
                "version": ENGINE_V3_VERSION,
                "result_schema_version": RESULT_SCHEMA_V3,
                "spec_sha256": ENGINE_V3_SPEC_SHA256,
            },
            "observation": {
                "count": len(OBSERVATION_ORDER_V3),
                "order_sha256": canonical_hash(list(OBSERVATION_ORDER_V3)),
            },
            "action": {
                "count": len(ACTION_ORDER_V3),
                "order_sha256": canonical_hash(list(ACTION_ORDER_V3)),
            },
            "absolute_outcome": {
                "id": SOLVED_DEFINITION_V3["id"],
                "version": SOLVED_DEFINITION_V3["version"],
                "definition_sha256": SOLVED_DEFINITION_V3_SHA256,
            },
            "baseline": {
                "id": "reactive-public-state-heuristic-v3",
                "version": "3.0.0",
                "implementation_path": baseline_path,
                "implementation_function": baseline_function,
                "implementation_file_sha256": file_sha256(ROOT / baseline_path),
                "implementation_function_sha256": _function_source_sha256(
                    baseline_path, baseline_function
                ),
                "uses_same_observation_contract": True,
                "uses_same_action_contract": True,
                "future_tape_visible": False,
            },
            "training_config": {
                "path": "training/v3/config.json",
                "sha256": config_sha256,
                "policy_seed": config["policy_seed"],
                "configured_transitions": config["total_transitions"],
                "simulator_lanes": config["simulator_lanes"],
                "vectorization": config["vectorization"],
            },
        },
        "splits": {
            "training": training,
            "development": development,
            "final": final,
            "seed_intervals_pairwise_disjoint": True,
            "family_sets_role_separated": True,
        },
        "checkpoint_selection": {
            "selection_dataset": "development",
            "eligible_policy_seeds": [config["policy_seed"]],
            "campaign_completion_required_transitions": config["total_transitions"],
            "individual_checkpoint_minimum_transitions": config[
                "checkpoint_interval_transitions"
            ],
            "registered_stage_boundaries": [
                checkpoint_interval * stage_number
                for stage_number in range(1, stages["count"] + 1)
            ],
            "stage_count": stages["count"],
            "rollouts_per_stage": stages["rollouts_per_stage"],
            "episode_steps_per_lane": stages["episode_steps_per_lane"],
            "transitions_per_stage": stages["transitions_per_stage"],
            "stage_boundaries_are_exact": True,
            "fresh_environment_reset_per_stage": True,
            "outage_resume_from_latest_complete_stage_only": True,
            "earlier_registered_checkpoints_remain_eligible_after_campaign_completion": True,
            "eligible_sources": ["terminal_checkpoint", "checkpoint_ledger"],
            "mandatory_bindings": [
                "protocol_sha256",
                "source_seal_sha256",
                "engine_spec_sha256",
                "outcome_definition_sha256",
                "training_config_sha256",
                "checkpoint_sha256",
                "training_receipt_sha256",
                "training_use_receipt_sha256",
                "training_terminal_sha256",
                "checkpoint_ledger_sha256",
                "campaign_provenance_sha256",
            ],
            "mandatory_safety": {
                "candidate_hard_violations": 0,
                "candidate_max_conservation_residual_at_most": SOLVED_DEFINITION_V3[
                    "max_conservation_residual"
                ],
                "complete_development_case_count": development["cartesian_case_count"],
                "deterministic_replay_mismatches": 0,
            },
            "ranking": [
                {"metric": "candidate_solved_count", "direction": "maximize"},
                {"metric": "candidate_mean_resilience_auc", "direction": "maximize"},
                {"metric": "completed_transitions", "direction": "minimize"},
                {"metric": "checkpoint_sha256", "direction": "lexicographic_ascending"},
            ],
            "final_metrics_may_influence_selection": False,
            "receipt_path": "training/v3/checkpoint-selection-receipt.json",
            "receipt_write_once": True,
            "deployment_staging": {
                "root_template": (
                    "internal/training_runs/v3/selection-staging/"
                    "{selected_checkpoint_sha256}"
                ),
                "claim_name": "selection-claim.json",
                "stage_receipt_name": "selected-deployment-stage-receipt.json",
                "claim_write_once": True,
                "stage_receipt_write_once": True,
                "recovery": (
                    "recompute_exact_identity_and_artifact_bytes_then_publish_only_"
                    "missing_destinations"
                ),
            },
        },
        "final_evaluation": {
            "case_count": 40,
            "split_id": "final",
            "selected_policy_id": SELECTED_POLICY_ID,
            "suite_materialized_by_protocol_tool": False,
            "invocation_limit": 1,
            "single_use_scope": "selected_onnx_and_frozen_source_identity",
            "requires_paths": [
                "training/v3/checkpoint-selection-receipt.json",
                "training/v3/final-authorization.json",
            ],
            "use_receipt_path": "training/v3/final-use-receipt.json",
            "append_only_ledger_path": "internal/training_runs/v3/final-ledger.jsonl",
            "result_path": "benchmarks/v3/final-40.json",
            "terminal_marker": {
                "path": "training/v3/final-terminal.json",
                "schema_version": 1,
                "status": "final_evaluation_complete",
                "write_once": True,
                "recoverable_publication": (
                    "verify_identity_matching_result_and_ledger_then_publish_only_"
                    "missing_terminal"
                ),
                "required_bindings": [
                    "selected_policy_id",
                    "identity_sha256",
                    "protocol_sha256",
                    "source_seal_sha256",
                    "scientific_source_sha256",
                    "selection_receipt_sha256",
                    "final_authorization_sha256",
                    "selected_checkpoint_sha256",
                    "selected_onnx_sha256",
                    "selected_parity_sha256",
                    "selected_manifest_sha256",
                    "final_use_receipt_sha256",
                    "final_ledger_sha256",
                    "final_ledger_terminal_record_sha256",
                    "final_result_sha256",
                ],
            },
            "rows_write_once": True,
            "torn_eof_recovery": {
                "scope": "last_non_newline_tail_only",
                "receipt_root": "internal/training_runs/v3/final-ledger-recovery",
                "complete_valid_record": "append_missing_newline",
                "invalid_partial_record": (
                    "preserve_tail_digest_then_truncate_to_last_complete_record"
                ),
                "never_remove_complete_record": True,
            },
            "partial_resume": (
                "only identity-matching durable rows from the original invocation; "
                "never begin a second invocation"
            ),
            "source_or_checkpoint_change_requires_new_protocol": True,
            "failed_or_unfavorable_result_may_be_replaced": False,
            "candidate_runtime": "onnxruntime_cpu_selected_deployment_only",
            "selected_deployment_requirements": {
                "policy_id": SELECTED_POLICY_ID,
                "onnx_path": "artifacts/city_recovery_ppo.v3.selected.onnx",
                "parity_path": "training/v3/selected-onnx-parity.json",
                "manifest_path": "artifacts/model_manifest.v3.selected.json",
                "development_parity_required": True,
                "selection_receipt_hash_binding_required": True,
                "execution_providers": ["CPUExecutionProvider"],
                "opset": 17,
                "input": {
                    "name": "observation",
                    "type": "tensor(float)",
                    "shape": ["batch", len(OBSERVATION_ORDER_V3)],
                },
                "output": {
                    "name": "action",
                    "type": "tensor(float)",
                    "shape": ["batch", len(ACTION_ORDER_V3)],
                },
            },
            "exclusive_lock_path": "internal/training_runs/v3/final.lock",
            "ordered_rows_must_be_exact_final_split_prefix": True,
            "durable_writes_require_file_fsync": True,
        },
        "authorization": {
            "protocol_itself_authorizes_execution": False,
            "training_and_development_selection": {
                "receipt_path": "training/v3/training-authorization.json",
                "required_status": "training_and_development_selection_authorized",
                "write_once": True,
                "exactly_one_campaign": True,
                "stage_checkpoints_are_selection_and_resume_state": True,
                "latest_complete_stage_must_be_resumed": True,
                "timestep_override_permitted_after_seal": False,
                "atomic_use_receipt_path": "training/v3/training-use-receipt.json",
                "terminal_marker_path": "training/v3/training-terminal.json",
                "release_published_only_after_parity": True,
                "interrupted_release_resume": (
                    "publish_only_missing_files_after_exact_staging_hash_match"
                ),
                "behavior_cloning_pre_stage_one_resume": (
                    "deterministic_replay_must_exactly_match_write_once_receipt"
                ),
                "developmental_outputs_confined_to": "internal/developmental_runs/v3",
            },
            "final_evaluation": {
                "receipt_path": "training/v3/final-authorization.json",
                "required_status": "final_evaluation_authorized",
                "write_once": True,
                "may_be_created_only_after_checkpoint_selection": True,
                "selection_receipt_and_checkpoint_hashes_required": True,
            },
            "pre_protocol_artifacts_authorizing": False,
        },
    }


def expected_source_seal(protocol: dict[str, Any]) -> dict[str, Any]:
    base = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "frozen",
        "scientific_source": scientific_source_identity(),
        "protocol": {
            "path": "training/v3/protocol.json",
            "bytes": len(formatted_json_bytes(protocol)),
            "sha256": hashlib.sha256(formatted_json_bytes(protocol)).hexdigest(),
        },
        "sealing_tool": {
            "path": "scripts/v3_protocol.py",
            "sha256": file_sha256(SEALING_TOOL_PATH),
        },
    }
    return {**base, "seal_sha256": canonical_hash(base)}


def _assert_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ProtocolError(f"{label} drifted from the frozen V3 contract")


def verify_documents(*, require_final_unused: bool = False) -> dict[str, Any]:
    protocol = _load_object(PROTOCOL_PATH, "V3 protocol")
    source_seal = _load_object(SOURCE_SEAL_PATH, "V3 source seal")
    expected = expected_protocol()
    _assert_equal(protocol, expected, "V3 protocol")
    expected_seal = expected_source_seal(protocol)
    _assert_equal(source_seal, expected_seal, "V3 source seal")

    if require_final_unused:
        materialized = [
            relative for relative in FINAL_STATE_PATHS if (ROOT / relative).exists()
        ]
        if materialized:
            raise ProtocolError(
                "final suite is not unused; final-state paths exist: "
                + ", ".join(materialized)
            )

    return {
        "action_order_sha256": protocol["bindings"]["action"]["order_sha256"],
        "engine_spec_sha256": protocol["bindings"]["environment"]["spec_sha256"],
        "final_case_count": protocol["final_evaluation"]["case_count"],
        "final_unused": not any(
            (ROOT / relative).exists() for relative in FINAL_STATE_PATHS
        ),
        "observation_order_sha256": protocol["bindings"]["observation"]["order_sha256"],
        "outcome_definition_sha256": protocol["bindings"]["absolute_outcome"][
            "definition_sha256"
        ],
        "protocol_sha256": file_sha256(PROTOCOL_PATH),
        "seal_sha256": source_seal["seal_sha256"],
        "scientific_source_sha256": source_seal["scientific_source"]["semantic_sha256"],
        "status": "v3-protocol-verified",
    }


def expected_training_authorization() -> dict[str, Any]:
    """Return the exact post-seal authorization for one production campaign."""

    verification = verify_documents()
    config = _load_object(CONFIG_PATH, "V3 training config")
    base = {
        "schema_version": 1,
        "status": "training_and_development_selection_authorized",
        "rationale": "user_directed_v3_training_campaign",
        "protocol_sha256": file_sha256(PROTOCOL_PATH),
        "source_seal_sha256": file_sha256(SOURCE_SEAL_PATH),
        "scientific_source_sha256": verification["scientific_source_sha256"],
        "engine_spec_sha256": verification["engine_spec_sha256"],
        "outcome_definition_sha256": verification["outcome_definition_sha256"],
        "training_config_sha256": file_sha256(CONFIG_PATH),
        "policy_seed": config["policy_seed"],
        "configured_transitions": config["total_transitions"],
        "simulator_lanes": config["simulator_lanes"],
        "vectorization": config["vectorization"],
        "authorized_operations": [
            "one_exact_seven_stage_resumable_training_campaign",
            "one_write_once_development_checkpoint_selection",
        ],
        "timestep_override_permitted": False,
        "pre_protocol_probe_artifacts_authorized": False,
        "final_evaluation_authorized": False,
        "training_use_receipt_path": "training/v3/training-use-receipt.json",
        "training_terminal_marker_path": "training/v3/training-terminal.json",
        "immutable_release_required": True,
        "interrupted_release_resume": (
            "publish_only_missing_files_after_exact_staging_hash_match"
        ),
        "behavior_cloning_pre_stage_one_resume": (
            "deterministic_replay_must_exactly_match_write_once_receipt"
        ),
        "resume_from_latest_complete_stage_required": True,
        "stage_count": config["production_stages"]["count"],
        "transitions_per_stage": config["production_stages"]["transitions_per_stage"],
    }
    return {**base, "authorization_sha256": canonical_hash(base)}


def verify_training_authorization() -> dict[str, Any]:
    """Fail closed unless the write-once training authorization is exact."""

    actual = _load_object(TRAINING_AUTHORIZATION_PATH, "V3 training authorization")
    expected = expected_training_authorization()
    _assert_equal(actual, expected, "V3 training authorization")
    return actual


def _validated_selected_deployment(
    selection: dict[str, Any], verification: dict[str, Any]
) -> dict[str, Any]:
    receipt_unsigned = {
        key: value
        for key, value in selection.items()
        if key != "receipt_semantic_sha256"
    }
    if selection.get("receipt_semantic_sha256") != canonical_hash(receipt_unsigned):
        raise ProtocolError("selection receipt semantic hash drifted")
    expected_file_bindings = {
        "training_config_sha256": CONFIG_PATH,
        "training_receipt_sha256": ROOT / "training" / "v3" / "training-receipt.json",
        "training_use_receipt_sha256": (
            ROOT / "training" / "v3" / "training-use-receipt.json"
        ),
        "training_terminal_sha256": (
            ROOT / "training" / "v3" / "training-terminal.json"
        ),
        "checkpoint_ledger_sha256": (
            ROOT / "internal" / "training_runs" / "v3" / "checkpoint-ledger.jsonl"
        ),
    }
    for field, path in expected_file_bindings.items():
        if not path.is_file() or selection.get(field) != file_sha256(path):
            raise ProtocolError(f"selection receipt {field} binding drifted")
    training_receipt = _load_object(
        ROOT / "training" / "v3" / "training-receipt.json",
        "V3 training receipt",
    )
    if selection.get("campaign_provenance_sha256") != training_receipt.get(
        "campaign_provenance_sha256"
    ):
        raise ProtocolError("selection campaign provenance binding drifted")
    training_receipt_unsigned = {
        key: value
        for key, value in training_receipt.items()
        if key != "receipt_semantic_sha256"
    }
    if training_receipt.get("receipt_semantic_sha256") != canonical_hash(
        training_receipt_unsigned
    ):
        raise ProtocolError("training receipt semantic hash drifted")
    terminal_path = ROOT / "training" / "v3" / "training-terminal.json"
    terminal = _load_object(terminal_path, "V3 training terminal marker")
    terminal_unsigned = {
        key: value
        for key, value in terminal.items()
        if key != "terminal_semantic_sha256"
    }
    use_path = ROOT / "training" / "v3" / "training-use-receipt.json"
    ledger_path = ROOT / "internal" / "training_runs" / "v3" / "checkpoint-ledger.jsonl"
    release_files = terminal.get("release_files")
    expected_release_paths = {
        "artifacts/city_recovery_ppo.v3.zip",
        "artifacts/city_recovery_ppo.v3.onnx",
        "artifacts/city_recovery_ppo.v3.metadata.json",
        "artifacts/model_manifest.v3.json",
        "training/v3/onnx-parity.json",
        "training/v3/training-receipt.json",
    }
    if (
        terminal.get("terminal_semantic_sha256") != canonical_hash(terminal_unsigned)
        or terminal.get("status") != "sealed_training_release_complete"
        or terminal.get("campaign_provenance_sha256")
        != training_receipt.get("campaign_provenance_sha256")
        or terminal.get("configured_transitions")
        != training_receipt.get("config", {}).get("total_transitions")
        or terminal.get("actual_completed_transitions")
        != training_receipt.get("completed_transitions")
        or terminal.get("completed_stage_count")
        != training_receipt.get("completed_stage_count")
        or terminal.get("training_use_receipt_sha256") != file_sha256(use_path)
        or terminal.get("training_receipt_sha256")
        != file_sha256(ROOT / "training" / "v3" / "training-receipt.json")
        or terminal.get("checkpoint_ledger_sha256") != file_sha256(ledger_path)
        or not isinstance(release_files, dict)
        or set(release_files) != expected_release_paths
        or any(
            not (ROOT / relative).is_file() or file_sha256(ROOT / relative) != digest
            for relative, digest in release_files.items()
        )
    ):
        raise ProtocolError("training terminal provenance drifted")
    reports = selection.get("candidate_reports")
    selected = selection.get("selected_checkpoint")
    if not isinstance(reports, list) or not reports or not isinstance(selected, dict):
        raise ProtocolError("selection receipt candidate reports are missing")
    for report in reports:
        if (
            not isinstance(report, dict)
            or report.get("development_case_count") != 40
            or report.get("hard_violation_count") != 0
            or report.get("deterministic_replay_mismatches") != 0
            or float(report.get("max_conservation_residual", float("inf"))) > 1e-6
        ):
            raise ProtocolError("selection candidate safety evidence drifted")
    expected_selected = min(
        reports,
        key=lambda item: (
            -int(item["candidate_solved_count"]),
            -float(item["candidate_mean_resilience_auc"]),
            int(item["completed_transitions"]),
            str(item["sha256"]),
        ),
    )
    if selected != expected_selected:
        raise ProtocolError(
            "selection receipt winner does not follow registered ranking"
        )
    checkpoint_path = ROOT / str(selected.get("path", ""))
    checkpoint_sha256 = selected.get("sha256")
    if (
        not checkpoint_path.is_file()
        or not isinstance(checkpoint_sha256, str)
        or file_sha256(checkpoint_path) != checkpoint_sha256
    ):
        raise ProtocolError("selected V3 checkpoint bytes drifted")

    deployment = selection.get("selected_deployment")
    if not isinstance(deployment, dict):
        raise ProtocolError("selected deployment binding is missing")
    expected_deployment_paths = {
        "onnx": SELECTED_ONNX_PATH,
        "parity": SELECTED_PARITY_PATH,
        "manifest": SELECTED_MANIFEST_PATH,
    }
    for artifact_id, path in expected_deployment_paths.items():
        binding = deployment.get(artifact_id)
        if not isinstance(binding, dict):
            raise ProtocolError(f"selected {artifact_id} binding is missing")
        if binding.get("path") != str(path.relative_to(ROOT)).replace("\\", "/"):
            raise ProtocolError(f"selected {artifact_id} path drifted")
        if not path.is_file() or binding.get("sha256") != file_sha256(path):
            raise ProtocolError(f"selected {artifact_id} bytes drifted")
    checkpoint_binding = deployment.get("checkpoint")
    if not isinstance(checkpoint_binding, dict) or checkpoint_binding != {
        "path": selected["path"],
        "sha256": selected["sha256"],
        "completed_transitions": selected["completed_transitions"],
    }:
        raise ProtocolError("selected deployment checkpoint binding drifted")
    if deployment["onnx"].get("bytes") != SELECTED_ONNX_PATH.stat().st_size:
        raise ProtocolError("selected ONNX byte count drifted")

    parity = _load_object(SELECTED_PARITY_PATH, "selected V3 ONNX parity")
    expected_interface = {
        "execution_providers": ["CPUExecutionProvider"],
        "opset": 17,
        "input_name": "observation",
        "input_type": "tensor(float)",
        "input_shape": ["batch", len(OBSERVATION_ORDER_V3)],
        "output_name": "action",
        "output_type": "tensor(float)",
        "output_shape": ["batch", len(ACTION_ORDER_V3)],
        "observation_count": len(OBSERVATION_ORDER_V3),
        "action_count": len(ACTION_ORDER_V3),
    }
    if (
        parity.get("schema_version") != 1
        or parity.get("status") != "passed"
        or parity.get("passed") is not True
        or parity.get("checkpoint_sha256") != checkpoint_sha256
        or parity.get("onnx_sha256") != file_sha256(SELECTED_ONNX_PATH)
        or parity.get("interface") != expected_interface
        or parity.get("development_case_count") != 40
        or int(parity.get("action_sample_count", 0)) <= 0
        or float(parity.get("maximum_action_absolute_error", float("inf"))) > 1e-5
        or float(parity.get("maximum_rauc_absolute_error", float("inf"))) > 1e-6
        or parity.get("outcome_mismatches") != 0
        or parity.get("deterministic_replay_mismatches") != 0
        or parity.get("hard_violation_count") != 0
        or float(parity.get("maximum_conservation_residual", float("inf"))) > 1e-6
    ):
        raise ProtocolError("selected ONNX parity evidence drifted")
    manifest = _load_object(SELECTED_MANIFEST_PATH, "selected V3 manifest")
    bindings = manifest.get("bindings")
    files = manifest.get("files")
    policy = manifest.get("policy")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "selected_deployable"
        or not isinstance(bindings, dict)
        or not isinstance(files, dict)
        or policy
        != {
            "id": SELECTED_POLICY_ID,
            "algorithm": "PPO",
            "observation_count": len(OBSERVATION_ORDER_V3),
            "action_count": len(ACTION_ORDER_V3),
            "completed_transitions": selected["completed_transitions"],
        }
        or bindings.get("protocol_sha256") != file_sha256(PROTOCOL_PATH)
        or bindings.get("source_seal_sha256") != file_sha256(SOURCE_SEAL_PATH)
        or bindings.get("scientific_source_sha256")
        != verification["scientific_source_sha256"]
        or bindings.get("engine_spec_sha256") != verification["engine_spec_sha256"]
        or bindings.get("outcome_definition_sha256")
        != verification["outcome_definition_sha256"]
        or bindings.get("observation_order_sha256")
        != verification["observation_order_sha256"]
        or bindings.get("action_order_sha256") != verification["action_order_sha256"]
        or bindings.get("selected_checkpoint_sha256") != checkpoint_sha256
        or files.get("onnx")
        != {
            "path": str(SELECTED_ONNX_PATH.relative_to(ROOT)).replace("\\", "/"),
            "bytes": SELECTED_ONNX_PATH.stat().st_size,
            "sha256": file_sha256(SELECTED_ONNX_PATH),
            "input_name": "observation",
            "output_name": "action",
            "opset": 17,
            "execution_providers": ["CPUExecutionProvider"],
            "input_type": "tensor(float)",
            "input_shape": ["batch", len(OBSERVATION_ORDER_V3)],
            "output_type": "tensor(float)",
            "output_shape": ["batch", len(ACTION_ORDER_V3)],
        }
        or files.get("parity")
        != {
            "path": str(SELECTED_PARITY_PATH.relative_to(ROOT)).replace("\\", "/"),
            "sha256": file_sha256(SELECTED_PARITY_PATH),
        }
    ):
        raise ProtocolError("selected deployment manifest drifted")
    return deployment


def expected_final_authorization() -> dict[str, Any]:
    """Bind a separate final authorization to the selected checkpoint bytes."""

    verification = verify_documents()
    verify_training_authorization()
    selection = _load_object(SELECTION_RECEIPT_PATH, "V3 selection receipt")
    selected = selection.get("selected_checkpoint")
    if not isinstance(selected, dict):
        raise ProtocolError("V3 selection receipt has no selected checkpoint")
    checkpoint_path = ROOT / str(selected.get("path", ""))
    checkpoint_sha256 = selected.get("sha256")
    if (
        not checkpoint_path.is_file()
        or not isinstance(checkpoint_sha256, str)
        or file_sha256(checkpoint_path) != checkpoint_sha256
    ):
        raise ProtocolError("selected V3 checkpoint bytes drifted")
    if selection.get("protocol_sha256") != file_sha256(PROTOCOL_PATH):
        raise ProtocolError("selection receipt protocol binding drifted")
    if selection.get("source_seal_sha256") != file_sha256(SOURCE_SEAL_PATH):
        raise ProtocolError("selection receipt source-seal binding drifted")
    if (
        selection.get("scientific_source_sha256")
        != verification["scientific_source_sha256"]
    ):
        raise ProtocolError("selection receipt scientific-source binding drifted")
    if selection.get("engine_spec_sha256") != verification["engine_spec_sha256"]:
        raise ProtocolError("selection receipt engine binding drifted")
    if (
        selection.get("outcome_definition_sha256")
        != verification["outcome_definition_sha256"]
    ):
        raise ProtocolError("selection receipt outcome binding drifted")
    if selection.get("final_split_accessed") is not False:
        raise ProtocolError("selection receipt does not prove final-split isolation")
    if selection.get("status") != "checkpoint_selected_on_development_only":
        raise ProtocolError("selection receipt status is not authorizing")
    if selection.get("selection_dataset") != "development":
        raise ProtocolError("selection receipt did not use the development split")
    if selection.get("development_case_count") != 40:
        raise ProtocolError("selection receipt does not contain all development cases")
    if selection.get("ranking") != [
        "candidate_solved_count:descending",
        "candidate_mean_resilience_auc:descending",
        "completed_transitions:ascending",
        "checkpoint_sha256:ascending",
    ]:
        raise ProtocolError("selection receipt ranking drifted")
    deployment = _validated_selected_deployment(selection, verification)

    base = {
        "schema_version": 1,
        "status": "final_evaluation_authorized",
        "protocol_sha256": file_sha256(PROTOCOL_PATH),
        "source_seal_sha256": file_sha256(SOURCE_SEAL_PATH),
        "scientific_source_sha256": verification["scientific_source_sha256"],
        "engine_spec_sha256": verification["engine_spec_sha256"],
        "outcome_definition_sha256": verification["outcome_definition_sha256"],
        "selection_receipt_sha256": file_sha256(SELECTION_RECEIPT_PATH),
        "selected_checkpoint_sha256": checkpoint_sha256,
        "selected_onnx_sha256": deployment["onnx"]["sha256"],
        "selected_parity_sha256": deployment["parity"]["sha256"],
        "selected_manifest_sha256": deployment["manifest"]["sha256"],
        "selected_policy_id": SELECTED_POLICY_ID,
        "candidate_runtime": "onnxruntime_cpu_selected_deployment_only",
        "final_case_count": 40,
        "invocation_limit": 1,
        "durable_identity_matching_resume_only": True,
        "failed_or_unfavorable_result_may_be_replaced": False,
    }
    return {**base, "authorization_sha256": canonical_hash(base)}


def verify_final_authorization() -> dict[str, Any]:
    """Fail closed unless final authorization matches selection and source."""

    actual = _load_object(FINAL_AUTHORIZATION_PATH, "V3 final authorization")
    expected = expected_final_authorization()
    _assert_equal(actual, expected, "V3 final authorization")
    return actual


def _write_once(path: Path, value: Any, label: str) -> None:
    if path.exists():
        raise ProtocolError(f"{label} already exists; it is write-once")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise ProtocolError(f"stale temporary {label} exists: {temporary}")
    temporary.write_bytes(formatted_json_bytes(value))
    if path.exists():
        temporary.unlink(missing_ok=True)
        raise ProtocolError(f"{label} appeared during creation; refusing overwrite")
    temporary.replace(path)


def _write_new_documents() -> None:
    if PROTOCOL_PATH.exists() or SOURCE_SEAL_PATH.exists():
        raise ProtocolError(
            "protocol or source seal already exists; immutable governance documents are never replaced"
        )
    protocol = expected_protocol()
    seal = expected_source_seal(protocol)
    _write_once(PROTOCOL_PATH, protocol, "V3 protocol")
    _write_once(SOURCE_SEAL_PATH, seal, "V3 source seal")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-new", action="store_true")
    parser.add_argument("--authorize-training", action="store_true")
    parser.add_argument("--authorize-final", action="store_true")
    parser.add_argument("--require-final-unused", action="store_true")
    parser.add_argument(
        "--render",
        choices=("protocol", "seal", "both", "training-authorization"),
        help="print deterministic documents without writing them",
    )
    args = parser.parse_args()

    actions = sum(
        bool(value)
        for value in (
            args.write_new,
            args.authorize_training,
            args.authorize_final,
            args.render,
        )
    )
    if actions > 1:
        parser.error("choose only one write, authorization, or render action")

    if args.render:
        protocol = expected_protocol()
        seal = expected_source_seal(protocol)
        rendered: Any
        if args.render == "protocol":
            rendered = protocol
        elif args.render == "seal":
            rendered = seal
        elif args.render == "both":
            rendered = {"protocol": protocol, "source_seal": seal}
        else:
            rendered = expected_training_authorization()
        print(json.dumps(rendered, indent=2, sort_keys=True))
        return
    if args.write_new:
        _write_new_documents()
    elif args.authorize_training:
        _write_once(
            TRAINING_AUTHORIZATION_PATH,
            expected_training_authorization(),
            "V3 training authorization",
        )
    elif args.authorize_final:
        verify_documents(require_final_unused=True)
        _write_once(
            FINAL_AUTHORIZATION_PATH,
            expected_final_authorization(),
            "V3 final authorization",
        )
    print(
        json.dumps(
            verify_documents(require_final_unused=args.require_final_unused),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
