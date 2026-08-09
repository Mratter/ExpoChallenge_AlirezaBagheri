"""Fail-closed loader for the selected PPO-v3 deployment and final report.

The runtime never selects a checkpoint and never evaluates a benchmark case. It
only accepts the write-once selected ONNX deployment when its protocol, source,
parity, authorization, and aggregate final-report bindings all agree with the
active 73-observation/22-action simulator contract.
"""

from __future__ import annotations

import hashlib
import ast
import json
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from statistics import fmean
from typing import Any, Callable

import numpy as np
import onnx
import onnxruntime as ort

from backend.app.scenarios_v3 import (
    DEVELOPMENT_FAMILIES_V3,
    DEVELOPMENT_SEEDS_V3,
    FINAL_FAMILIES_V3,
    FINAL_SEEDS_V3,
    TRAINING_FAMILIES_V3,
    TRAINING_SEEDS_V3,
)
from backend.app.simulator_core import canonical_hash
from backend.app.simulator_v3 import (
    ACTION_ORDER_V3,
    ENGINE_V3_ID,
    ENGINE_V3_SPEC_SHA256,
    ENGINE_V3_VERSION,
    OBSERVATION_ORDER_V3,
    RESULT_SCHEMA_V3,
    SOLVED_DEFINITION_V3,
    SOLVED_DEFINITION_V3_SHA256,
)

ROOT = Path(__file__).resolve().parents[1]

POLICY_ID = "city-recovery-sb3-ppo-v3-selected"
POLICY_VERSION = "3.0.0"
POLICY_SCHEMA_VERSION = RESULT_SCHEMA_V3
POLICY_ARTIFACT_TYPE = "stable_baselines3_ppo"
MANIFEST_SCHEMA_VERSION = 1
ARTIFACT_LICENSE = "CC0-1.0"

SELECTED_ONNX_RELATIVE = Path("artifacts/city_recovery_ppo.v3.selected.onnx")
SELECTED_MANIFEST_RELATIVE = Path("artifacts/model_manifest.v3.selected.json")
SELECTED_PARITY_RELATIVE = Path("training/v3/selected-onnx-parity.json")
PROTOCOL_RELATIVE = Path("training/v3/protocol.json")
SOURCE_SEAL_RELATIVE = Path("training/v3/source-seal.json")
SELECTION_RECEIPT_RELATIVE = Path("training/v3/checkpoint-selection-receipt.json")
FINAL_AUTHORIZATION_RELATIVE = Path("training/v3/final-authorization.json")
CONFIG_RELATIVE = Path("training/v3/config.json")
TRAINING_AUTHORIZATION_RELATIVE = Path("training/v3/training-authorization.json")
TRAINING_RECEIPT_RELATIVE = Path("training/v3/training-receipt.json")
TRAINING_USE_RECEIPT_RELATIVE = Path("training/v3/training-use-receipt.json")
TRAINING_TERMINAL_RELATIVE = Path("training/v3/training-terminal.json")
TRAINING_LEDGER_RELATIVE = Path("internal/training_runs/v3/checkpoint-ledger.jsonl")
FINAL_REPORT_RELATIVE = Path("benchmarks/v3/final-40.json")
FINAL_USE_RECEIPT_RELATIVE = Path("training/v3/final-use-receipt.json")
FINAL_LEDGER_RELATIVE = Path("internal/training_runs/v3/final-ledger.jsonl")
FINAL_TERMINAL_RELATIVE = Path("training/v3/final-terminal.json")
FINAL_LOCK_RELATIVE = Path("internal/training_runs/v3/final.lock")
SEALING_TOOL_RELATIVE = Path("scripts/v3_protocol.py")

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

SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
SELECTED_ONNX_INTERFACE = {
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


class ArtifactError(RuntimeError):
    """Raised when the selected V3 release is absent, altered, or incompatible."""


@dataclass(frozen=True)
class PolicyBundle:
    metadata: dict[str, Any]
    session: ort.InferenceSession
    onnx_sha256: str
    sb3_sha256: str
    manifest_sha256: str
    manifest_schema_version: int
    parity_sha256: str
    benchmark: dict[str, Any]
    benchmark_sha256: str
    scientific_source_sha256: str


def _relative(path: Path) -> str:
    return path.as_posix()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path, label: str) -> str:
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError as exc:
        raise ArtifactError(f"{label} is missing or unreadable") from exc


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ArtifactError(f"{label} is missing or invalid") from exc
    if not isinstance(value, dict):
        raise ArtifactError(f"{label} root must be an object")
    return value, payload


def _safe_relative_file(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ArtifactError(f"{label} path is invalid")
    relative = Path(value)
    if relative.is_absolute():
        raise ArtifactError(f"{label} path must be repository-relative")
    resolved_root = root.resolve()
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ArtifactError(f"{label} path escapes the repository") from exc
    if not resolved.is_file():
        raise ArtifactError(f"{label} is missing")
    return resolved


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ArtifactError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _finite_number(value: Any, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ArtifactError(f"{label} is not a finite number")
    return float(value)


def _semantic_receipt(document: dict[str, Any], field: str, label: str) -> None:
    digest = _require_sha256(document.get(field), f"{label} semantic digest")
    unsigned = {key: value for key, value in document.items() if key != field}
    if digest != canonical_hash(unsigned):
        raise ArtifactError(f"{label} semantic digest drifted")


def _source_identity(root: Path) -> dict[str, Any]:
    files = {
        relative: _sha256_file(root / relative, f"scientific source {relative}")
        for relative in SCIENTIFIC_SOURCE_PATHS
    }
    return {"files_sha256": files, "semantic_sha256": canonical_hash(files)}


def _function_source_sha256(root: Path, relative_path: str, function_name: str) -> str:
    path = root / relative_path
    try:
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as exc:
        raise ArtifactError("V3 bound baseline source is unreadable") from exc
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
        raise ArtifactError("V3 bound baseline function is missing")
    lines = text.replace("\r\n", "\n").replace("\r", "\n").splitlines(keepends=True)
    payload = "".join(lines[node.lineno - 1 : node.end_lineno]).rstrip("\n") + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _split_contract(
    split_id: str, families: tuple[Any, ...], seeds: tuple[int, ...]
) -> dict[str, Any]:
    return {
        "id": split_id,
        "family_count": len(families),
        "family_ids": [family.id for family in families],
        "seed_interval": {
            "first": seeds[0],
            "last": seeds[-1],
            "count": len(seeds),
        },
        "cartesian_case_count": len(families) * len(seeds),
        "iteration_order": "family_order_then_ascending_seed",
    }


def _validate_protocol(root: Path) -> dict[str, str]:
    protocol_path = root / PROTOCOL_RELATIVE
    seal_path = root / SOURCE_SEAL_RELATIVE
    protocol, protocol_bytes = _read_json(protocol_path, "V3 protocol")
    seal, seal_bytes = _read_json(seal_path, "V3 source seal")
    source = _source_identity(root)
    _, config_bytes = _read_json(root / CONFIG_RELATIVE, "V3 training config")
    config = _validate_config(root)
    observation_order_sha256 = canonical_hash(list(OBSERVATION_ORDER_V3))
    action_order_sha256 = canonical_hash(list(ACTION_ORDER_V3))

    bindings = protocol.get("bindings")
    if (
        set(protocol)
        != {
            "schema_version",
            "protocol_id",
            "status",
            "scientific_source",
            "bindings",
            "splits",
            "checkpoint_selection",
            "final_evaluation",
            "authorization",
        }
        or protocol.get("schema_version") != 1
        or protocol.get("protocol_id") != "city-recovery-v3-preregistration-v1"
        or protocol.get("status") != "preregistered_separate_authorizations"
        or protocol.get("scientific_source") != source
        or not isinstance(bindings, dict)
    ):
        raise ArtifactError("V3 protocol identity drifted")
    environment = bindings.get("environment")
    observation = bindings.get("observation")
    action = bindings.get("action")
    outcome = bindings.get("absolute_outcome")
    baseline = bindings.get("baseline")
    training_config = bindings.get("training_config")
    if set(bindings) != {
        "environment",
        "observation",
        "action",
        "absolute_outcome",
        "baseline",
        "training_config",
    }:
        raise ArtifactError("V3 protocol binding schema drifted")
    if environment != {
        "id": ENGINE_V3_ID,
        "version": ENGINE_V3_VERSION,
        "result_schema_version": RESULT_SCHEMA_V3,
        "spec_sha256": ENGINE_V3_SPEC_SHA256,
    }:
        raise ArtifactError("V3 protocol environment binding drifted")
    if observation != {
        "count": len(OBSERVATION_ORDER_V3),
        "order_sha256": observation_order_sha256,
    }:
        raise ArtifactError("V3 protocol observation binding drifted")
    if action != {
        "count": len(ACTION_ORDER_V3),
        "order_sha256": action_order_sha256,
    }:
        raise ArtifactError("V3 protocol action binding drifted")
    if outcome != {
        "id": SOLVED_DEFINITION_V3["id"],
        "version": SOLVED_DEFINITION_V3["version"],
        "definition_sha256": SOLVED_DEFINITION_V3_SHA256,
    }:
        raise ArtifactError("V3 protocol outcome binding drifted")
    if (
        not isinstance(baseline, dict)
        or set(baseline)
        != {
            "id",
            "version",
            "implementation_path",
            "implementation_function",
            "implementation_file_sha256",
            "implementation_function_sha256",
            "uses_same_observation_contract",
            "uses_same_action_contract",
            "future_tape_visible",
        }
        or baseline.get("id") != "reactive-public-state-heuristic-v3"
        or baseline.get("version") != "3.0.0"
        or baseline.get("uses_same_observation_contract") is not True
        or baseline.get("uses_same_action_contract") is not True
        or baseline.get("future_tape_visible") is not False
    ):
        raise ArtifactError("V3 protocol baseline binding drifted")
    _require_sha256(
        baseline.get("implementation_function_sha256"),
        "V3 baseline function digest",
    )
    if (
        baseline.get("implementation_path") != "backend/app/simulator_v3.py"
        or baseline.get("implementation_function") != "reactive_heuristic_action_v3"
        or baseline.get("implementation_file_sha256")
        != source["files_sha256"]["backend/app/simulator_v3.py"]
        or baseline.get("implementation_function_sha256")
        != _function_source_sha256(
            root,
            "backend/app/simulator_v3.py",
            "reactive_heuristic_action_v3",
        )
        or training_config
        != {
            "path": _relative(CONFIG_RELATIVE),
            "sha256": _sha256_bytes(config_bytes),
            "policy_seed": config.get("policy_seed"),
            "configured_transitions": config.get("total_transitions"),
            "simulator_lanes": config.get("simulator_lanes"),
            "vectorization": config.get("vectorization"),
        }
    ):
        raise ArtifactError("V3 protocol implementation/config binding drifted")

    expected_splits = {
        "training": _split_contract(
            "training", TRAINING_FAMILIES_V3, TRAINING_SEEDS_V3
        ),
        "development": _split_contract(
            "development", DEVELOPMENT_FAMILIES_V3, DEVELOPMENT_SEEDS_V3
        ),
        "final": _split_contract("final", FINAL_FAMILIES_V3, FINAL_SEEDS_V3),
        "seed_intervals_pairwise_disjoint": True,
        "family_sets_role_separated": True,
    }
    if protocol.get("splits") != expected_splits:
        raise ArtifactError("V3 protocol split contract drifted")

    stages = config.get("production_stages")
    if not isinstance(stages, dict):
        raise ArtifactError("V3 production stage contract is missing")
    interval = config.get("checkpoint_interval_transitions")
    stage_count = stages.get("count")
    if not isinstance(interval, int) or not isinstance(stage_count, int):
        raise ArtifactError("V3 checkpoint stage contract drifted")
    expected_selection = {
        "selection_dataset": "development",
        "eligible_policy_seeds": [config["policy_seed"]],
        "campaign_completion_required_transitions": config["total_transitions"],
        "individual_checkpoint_minimum_transitions": interval,
        "registered_stage_boundaries": [
            interval * stage_number for stage_number in range(1, stage_count + 1)
        ],
        "stage_count": stage_count,
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
            "complete_development_case_count": 40,
            "deterministic_replay_mismatches": 0,
        },
        "ranking": [
            {"metric": "candidate_solved_count", "direction": "maximize"},
            {"metric": "candidate_mean_resilience_auc", "direction": "maximize"},
            {"metric": "completed_transitions", "direction": "minimize"},
            {"metric": "checkpoint_sha256", "direction": "lexicographic_ascending"},
        ],
        "final_metrics_may_influence_selection": False,
        "receipt_path": _relative(SELECTION_RECEIPT_RELATIVE),
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
    }
    if protocol.get("checkpoint_selection") != expected_selection:
        raise ArtifactError("V3 checkpoint-selection protocol drifted")

    final = protocol.get("final_evaluation")
    if not isinstance(final, dict):
        raise ArtifactError("V3 final-evaluation protocol is missing")
    deployment_requirements = final.get("selected_deployment_requirements")
    terminal_contract = final.get("terminal_marker")
    expected_terminal_bindings = [
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
    ]
    expected_deployment_requirements = {
        "policy_id": POLICY_ID,
        "onnx_path": _relative(SELECTED_ONNX_RELATIVE),
        "parity_path": _relative(SELECTED_PARITY_RELATIVE),
        "manifest_path": _relative(SELECTED_MANIFEST_RELATIVE),
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
    }
    if (
        set(final)
        != {
            "case_count",
            "split_id",
            "selected_policy_id",
            "suite_materialized_by_protocol_tool",
            "invocation_limit",
            "single_use_scope",
            "requires_paths",
            "use_receipt_path",
            "append_only_ledger_path",
            "result_path",
            "terminal_marker",
            "rows_write_once",
            "torn_eof_recovery",
            "partial_resume",
            "source_or_checkpoint_change_requires_new_protocol",
            "failed_or_unfavorable_result_may_be_replaced",
            "candidate_runtime",
            "selected_deployment_requirements",
            "exclusive_lock_path",
            "ordered_rows_must_be_exact_final_split_prefix",
            "durable_writes_require_file_fsync",
        }
        or final.get("split_id") != "final"
        or final.get("selected_policy_id") != POLICY_ID
        or final.get("case_count") != 40
        or final.get("suite_materialized_by_protocol_tool") is not False
        or final.get("invocation_limit") != 1
        or final.get("single_use_scope") != "selected_onnx_and_frozen_source_identity"
        or final.get("requires_paths")
        != [
            _relative(SELECTION_RECEIPT_RELATIVE),
            _relative(FINAL_AUTHORIZATION_RELATIVE),
        ]
        or final.get("use_receipt_path") != _relative(FINAL_USE_RECEIPT_RELATIVE)
        or final.get("append_only_ledger_path") != _relative(FINAL_LEDGER_RELATIVE)
        or final.get("result_path") != _relative(FINAL_REPORT_RELATIVE)
        or final.get("rows_write_once") is not True
        or final.get("torn_eof_recovery")
        != {
            "scope": "last_non_newline_tail_only",
            "receipt_root": "internal/training_runs/v3/final-ledger-recovery",
            "complete_valid_record": "append_missing_newline",
            "invalid_partial_record": (
                "preserve_tail_digest_then_truncate_to_last_complete_record"
            ),
            "never_remove_complete_record": True,
        }
        or final.get("partial_resume")
        != (
            "only identity-matching durable rows from the original invocation; "
            "never begin a second invocation"
        )
        or final.get("source_or_checkpoint_change_requires_new_protocol") is not True
        or final.get("failed_or_unfavorable_result_may_be_replaced") is not False
        or final.get("candidate_runtime") != "onnxruntime_cpu_selected_deployment_only"
        or final.get("exclusive_lock_path") != _relative(FINAL_LOCK_RELATIVE)
        or final.get("ordered_rows_must_be_exact_final_split_prefix") is not True
        or final.get("durable_writes_require_file_fsync") is not True
        or deployment_requirements != expected_deployment_requirements
        or terminal_contract
        != {
            "path": _relative(FINAL_TERMINAL_RELATIVE),
            "schema_version": 1,
            "status": "final_evaluation_complete",
            "write_once": True,
            "recoverable_publication": (
                "verify_identity_matching_result_and_ledger_then_publish_only_"
                "missing_terminal"
            ),
            "required_bindings": expected_terminal_bindings,
        }
    ):
        raise ArtifactError("V3 final-evaluation protocol binding drifted")

    authorization_contract = protocol.get("authorization")
    expected_training_authorization_contract = {
        "receipt_path": _relative(TRAINING_AUTHORIZATION_RELATIVE),
        "required_status": "training_and_development_selection_authorized",
        "write_once": True,
        "exactly_one_campaign": True,
        "stage_checkpoints_are_selection_and_resume_state": True,
        "latest_complete_stage_must_be_resumed": True,
        "timestep_override_permitted_after_seal": False,
        "atomic_use_receipt_path": _relative(TRAINING_USE_RECEIPT_RELATIVE),
        "terminal_marker_path": _relative(TRAINING_TERMINAL_RELATIVE),
        "release_published_only_after_parity": True,
        "interrupted_release_resume": (
            "publish_only_missing_files_after_exact_staging_hash_match"
        ),
        "behavior_cloning_pre_stage_one_resume": (
            "deterministic_replay_must_exactly_match_write_once_receipt"
        ),
        "developmental_outputs_confined_to": "internal/developmental_runs/v3",
    }
    if (
        not isinstance(authorization_contract, dict)
        or set(authorization_contract)
        != {
            "protocol_itself_authorizes_execution",
            "training_and_development_selection",
            "final_evaluation",
            "pre_protocol_artifacts_authorizing",
        }
        or authorization_contract.get("protocol_itself_authorizes_execution")
        is not False
        or authorization_contract.get("pre_protocol_artifacts_authorizing") is not False
        or authorization_contract.get("training_and_development_selection")
        != expected_training_authorization_contract
        or authorization_contract.get("final_evaluation")
        != {
            "receipt_path": _relative(FINAL_AUTHORIZATION_RELATIVE),
            "required_status": "final_evaluation_authorized",
            "write_once": True,
            "may_be_created_only_after_checkpoint_selection": True,
            "selection_receipt_and_checkpoint_hashes_required": True,
        }
    ):
        raise ArtifactError("V3 authorization protocol binding drifted")

    protocol_sha256 = _sha256_bytes(protocol_bytes)
    sealing_tool_sha256 = _sha256_file(root / SEALING_TOOL_RELATIVE, "V3 sealing tool")
    if (
        set(seal)
        != {
            "schema_version",
            "protocol_id",
            "status",
            "scientific_source",
            "protocol",
            "sealing_tool",
            "seal_sha256",
        }
        or seal.get("schema_version") != 1
        or seal.get("protocol_id") != protocol["protocol_id"]
        or seal.get("status") != "frozen"
        or seal.get("scientific_source") != source
        or seal.get("protocol")
        != {
            "path": _relative(PROTOCOL_RELATIVE),
            "bytes": len(protocol_bytes),
            "sha256": protocol_sha256,
        }
        or seal.get("sealing_tool")
        != {
            "path": _relative(SEALING_TOOL_RELATIVE),
            "sha256": sealing_tool_sha256,
        }
    ):
        raise ArtifactError("V3 source seal binding drifted")
    _semantic_receipt(seal, "seal_sha256", "V3 source seal")
    return {
        "protocol_sha256": protocol_sha256,
        "source_seal_sha256": _sha256_bytes(seal_bytes),
        "scientific_source_sha256": source["semantic_sha256"],
        "observation_order_sha256": observation_order_sha256,
        "action_order_sha256": action_order_sha256,
    }


def _validate_config(root: Path) -> dict[str, Any]:
    config, _ = _read_json(root / CONFIG_RELATIVE, "V3 training config")
    network = config.get("network")
    if (
        config.get("algorithm") != "Stable-Baselines3 PPO"
        or config.get("device") != "cpu"
        or config.get("environment") != ENGINE_V3_ID
        or config.get("observation_count") != len(OBSERVATION_ORDER_V3)
        or config.get("action_count") != len(ACTION_ORDER_V3)
        or not isinstance(config.get("expected_trainable_parameters"), int)
        or isinstance(config.get("expected_trainable_parameters"), bool)
        or config["expected_trainable_parameters"] != 322_733
        or not isinstance(config.get("total_transitions"), int)
        or isinstance(config.get("total_transitions"), bool)
        or config["total_transitions"] <= 0
        or not isinstance(config.get("policy_seed"), int)
        or isinstance(config.get("policy_seed"), bool)
        or not isinstance(config.get("simulator_lanes"), int)
        or isinstance(config.get("simulator_lanes"), bool)
        or config["simulator_lanes"] <= 0
        or network
        != {
            "activation": "SiLU",
            "actor": [384, 256, 128],
            "critic": [384, 256, 128],
            "orthogonal_initialization": True,
        }
    ):
        raise ArtifactError("V3 training configuration drifted")
    curriculum = config.get("behavior_cloning")
    stages = config.get("production_stages")
    vectorization = config.get("vectorization")
    n_steps = config.get("n_steps_per_lane")
    batch_size = config.get("batch_size")
    if (
        not isinstance(curriculum, dict)
        or curriculum.get("teacher") != "public-preparedness-curriculum-v3"
        or curriculum.get("teacher_version") != "1.0.0"
        or curriculum.get("training_split_only") is not True
        or not isinstance(curriculum.get("iterations"), int)
        or isinstance(curriculum.get("iterations"), bool)
        or curriculum["iterations"] < 1
        or curriculum.get("dagger_beta_schedule")
        != [1.0] + [0.0] * (curriculum["iterations"] - 1)
        or not isinstance(n_steps, int)
        or isinstance(n_steps, bool)
        or n_steps <= 0
        or not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size <= 0
    ):
        raise ArtifactError("V3 training curriculum/configuration drifted")
    rollout_batch = n_steps * config["simulator_lanes"]
    if (
        rollout_batch % batch_size != 0
        or not isinstance(stages, dict)
        or stages
        != {
            "count": 7,
            "rollouts_per_stage": 30,
            "transitions_per_stage": 30 * rollout_batch,
            "episode_steps_per_lane": 30,
            "resume_at_stage_boundaries": True,
            "fresh_environment_per_stage": True,
        }
        or config.get("checkpoint_interval_transitions")
        != stages["transitions_per_stage"]
        or config.get("total_transitions")
        != stages["count"] * stages["transitions_per_stage"]
        or vectorization
        != {
            "backend": "SubprocVecEnv",
            "parent_torch_inter_op_threads": 1,
            "parent_torch_intra_op_threads": 12,
            "start_method": "spawn",
            "worker_blas_threads": 1,
        }
    ):
        raise ArtifactError("V3 production stage/configuration drifted")
    return config


def _validate_training_chain(
    root: Path, config: dict[str, Any], bindings: dict[str, str]
) -> dict[str, Any]:
    training_authorization, training_authorization_bytes = _read_json(
        root / TRAINING_AUTHORIZATION_RELATIVE,
        "V3 training authorization",
    )
    training_authorization_base = {
        "schema_version": 1,
        "status": "training_and_development_selection_authorized",
        "rationale": "user_directed_v3_training_campaign",
        "protocol_sha256": bindings["protocol_sha256"],
        "source_seal_sha256": bindings["source_seal_sha256"],
        "scientific_source_sha256": bindings["scientific_source_sha256"],
        "engine_spec_sha256": ENGINE_V3_SPEC_SHA256,
        "outcome_definition_sha256": SOLVED_DEFINITION_V3_SHA256,
        "training_config_sha256": _sha256_bytes((root / CONFIG_RELATIVE).read_bytes()),
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
        "training_use_receipt_path": _relative(TRAINING_USE_RECEIPT_RELATIVE),
        "training_terminal_marker_path": _relative(TRAINING_TERMINAL_RELATIVE),
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
    expected_training_authorization = {
        **training_authorization_base,
        "authorization_sha256": canonical_hash(training_authorization_base),
    }
    if training_authorization != expected_training_authorization:
        raise ArtifactError("V3 training authorization contract drifted")
    receipt, receipt_bytes = _read_json(
        root / TRAINING_RECEIPT_RELATIVE, "V3 training receipt"
    )
    use_receipt, use_bytes = _read_json(
        root / TRAINING_USE_RECEIPT_RELATIVE, "V3 training-use receipt"
    )
    terminal, terminal_bytes = _read_json(
        root / TRAINING_TERMINAL_RELATIVE, "V3 training terminal marker"
    )
    ledger_path = root / TRAINING_LEDGER_RELATIVE
    try:
        ledger_bytes = ledger_path.read_bytes()
        ledger_lines = ledger_bytes.decode("utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ArtifactError("V3 training ledger is missing or unreadable") from exc
    _semantic_receipt(receipt, "receipt_semantic_sha256", "V3 training receipt")
    _semantic_receipt(terminal, "terminal_semantic_sha256", "V3 training terminal")
    provenance = receipt.get("campaign_provenance")
    provenance_sha256 = receipt.get("campaign_provenance_sha256")
    source_identity = _source_identity(root)
    expected_governance = {
        "protocol_sha256": bindings["protocol_sha256"],
        "source_seal_sha256": bindings["source_seal_sha256"],
        "training_authorization_sha256": _sha256_bytes(training_authorization_bytes),
        "scientific_source_sha256": bindings["scientific_source_sha256"],
    }
    expected_provenance_fields = {
        "schema_version",
        "execution_class",
        "governance",
        "scientific_source_sha256",
        "registered_config_sha256",
        "effective_config_sha256",
        "engine_spec_sha256",
        "outcome_definition_sha256",
        "policy_seed",
        "configured_transitions",
        "simulator_lanes",
        "checkpoint_interval_transitions",
        "production_stages",
        "vectorization",
        "runtime",
    }
    expected_receipt_fields = {
        "schema_version",
        "status",
        "execution_class",
        "governance",
        "campaign_provenance",
        "campaign_provenance_sha256",
        "runtime",
        "production_stage_contract",
        "completed_stage_count",
        "invocation_resumed_from_stage",
        "config",
        "source_identity",
        "engine_spec_sha256",
        "solved_definition_sha256",
        "completed_transitions",
        "trainable_parameters",
        "curriculum",
        "elapsed_seconds",
        "checkpoint_ledger_sha256",
        "checkpoint_ledger_record_count",
        "checkpoint_ledger_terminal_record_sha256",
        "terminal_checkpoint",
        "artifacts",
        "manifest_sha256",
        "parity_sha256",
        "receipt_semantic_sha256",
    }
    if (
        not isinstance(provenance, dict)
        or set(provenance) != expected_provenance_fields
        or provenance_sha256 != canonical_hash(provenance)
        or provenance.get("schema_version") != 1
        or provenance.get("execution_class") != "sealed_authorized_campaign"
        or provenance.get("governance") != expected_governance
        or provenance.get("scientific_source_sha256")
        != bindings["scientific_source_sha256"]
        or provenance.get("registered_config_sha256")
        != _sha256_bytes((root / CONFIG_RELATIVE).read_bytes())
        or provenance.get("effective_config_sha256") != canonical_hash(config)
        or provenance.get("engine_spec_sha256") != ENGINE_V3_SPEC_SHA256
        or provenance.get("outcome_definition_sha256") != SOLVED_DEFINITION_V3_SHA256
        or provenance.get("policy_seed") != config["policy_seed"]
        or provenance.get("configured_transitions") != config["total_transitions"]
        or provenance.get("simulator_lanes") != config["simulator_lanes"]
        or provenance.get("checkpoint_interval_transitions")
        != config["checkpoint_interval_transitions"]
        or provenance.get("production_stages") != config["production_stages"]
        or provenance.get("vectorization") != config["vectorization"]
        or not isinstance(provenance.get("runtime"), dict)
        or set(receipt) != expected_receipt_fields
        or receipt.get("schema_version") != 1
        or receipt.get("status") != "completed"
        or receipt.get("execution_class") != "sealed_authorized_campaign"
        or receipt.get("governance") != expected_governance
        or receipt.get("config") != config
        or receipt.get("source_identity") != source_identity
        or receipt.get("engine_spec_sha256") != ENGINE_V3_SPEC_SHA256
        or receipt.get("solved_definition_sha256") != SOLVED_DEFINITION_V3_SHA256
        or receipt.get("runtime") != provenance["runtime"]
        or receipt.get("production_stage_contract") != config["production_stages"]
        or not isinstance(receipt.get("invocation_resumed_from_stage"), int)
        or isinstance(receipt.get("invocation_resumed_from_stage"), bool)
        or not 0
        <= receipt["invocation_resumed_from_stage"]
        <= config["production_stages"]["count"]
        or not isinstance(receipt.get("trainable_parameters"), int)
        or receipt["trainable_parameters"] != config["expected_trainable_parameters"]
        or not isinstance(receipt.get("curriculum"), dict)
        or not isinstance(receipt.get("elapsed_seconds"), (int, float))
        or isinstance(receipt.get("elapsed_seconds"), bool)
        or not math.isfinite(float(receipt["elapsed_seconds"]))
        or float(receipt["elapsed_seconds"]) < 0.0
        or receipt.get("completed_transitions") != config["total_transitions"]
        or receipt.get("completed_stage_count") != config["production_stages"]["count"]
        or receipt.get("checkpoint_ledger_sha256") != _sha256_bytes(ledger_bytes)
        or receipt.get("checkpoint_ledger_record_count") != len(ledger_lines)
    ):
        raise ArtifactError("V3 training receipt provenance drifted")
    if (
        set(use_receipt)
        != {
            "schema_version",
            "status",
            "campaign_identity",
            "campaign_provenance",
            "campaign_provenance_sha256",
            "single_use",
            "stage_resume_only",
            "behavior_cloning_resume",
        }
        or use_receipt.get("schema_version") != 1
        or use_receipt.get("status") != "sealed_training_campaign_claimed"
        or use_receipt.get("single_use") is not True
        or use_receipt.get("stage_resume_only") is not True
        or use_receipt.get("campaign_identity") != provenance_sha256
        or use_receipt.get("campaign_provenance_sha256") != provenance_sha256
        or use_receipt.get("campaign_provenance") != provenance
        or use_receipt.get("behavior_cloning_resume")
        != "deterministic_replay_before_first_stage_only"
    ):
        raise ArtifactError("V3 training-use receipt provenance drifted")
    stage_count = config["production_stages"]["count"]
    interval = config["production_stages"]["transitions_per_stage"]
    if len(ledger_lines) != stage_count or any(not line for line in ledger_lines):
        raise ArtifactError("V3 training ledger stage count drifted")
    previous_hash = "0" * 64
    eligible_candidates: list[dict[str, Any]] = []
    for stage, line in enumerate(ledger_lines, start=1):
        try:
            record = json.loads(line)
        except ValueError as exc:
            raise ArtifactError("V3 training ledger JSON drifted") from exc
        if not isinstance(record, dict):
            raise ArtifactError("V3 training ledger record is invalid")
        if set(record) != {
            "schema_version",
            "record_type",
            "stage_number",
            "stage_start_transitions",
            "stage_transitions",
            "num_timesteps",
            "stage_seed",
            "fresh_environment_reset",
            "campaign_provenance",
            "campaign_provenance_sha256",
            "checkpoint",
            "sha256",
            "previous_record_sha256",
            "record_sha256",
        }:
            raise ArtifactError("V3 training ledger record schema drifted")
        claimed_hash = _require_sha256(
            record.get("record_sha256"), "V3 training ledger record digest"
        )
        unsigned = {
            key: value for key, value in record.items() if key != "record_sha256"
        }
        checkpoint = _safe_relative_file(
            root, record.get("checkpoint"), "V3 training checkpoint"
        )
        expected_checkpoint = (
            "internal/training_runs/v3/checkpoints/"
            f"stage-{stage:02d}-transition-{stage * interval:09d}.zip"
        )
        if (
            record.get("schema_version") != 1
            or record.get("record_type") != "v3_training_stage_checkpoint"
            or record.get("stage_number") != stage
            or record.get("stage_start_transitions") != (stage - 1) * interval
            or record.get("stage_transitions") != interval
            or record.get("num_timesteps") != stage * interval
            or record.get("stage_seed") != config["policy_seed"] + (stage - 1) * 10_007
            or record.get("fresh_environment_reset") is not True
            or record.get("campaign_provenance") != provenance
            or record.get("campaign_provenance_sha256") != provenance_sha256
            or record.get("checkpoint") != expected_checkpoint
            or record.get("previous_record_sha256") != previous_hash
            or claimed_hash != canonical_hash(unsigned)
            or record.get("sha256") != _sha256_bytes(checkpoint.read_bytes())
        ):
            raise ArtifactError("V3 training ledger provenance drifted")
        eligible_candidates.append(
            {
                "path": expected_checkpoint,
                "sha256": record["sha256"],
                "completed_transitions": record["num_timesteps"],
                "source": "checkpoint_ledger",
            }
        )
        previous_hash = claimed_hash
    if receipt.get("checkpoint_ledger_terminal_record_sha256") != previous_hash:
        raise ArtifactError("V3 training receipt terminal ledger binding drifted")
    curriculum = receipt.get("curriculum")
    if (
        curriculum.get("enabled") is not True
        or curriculum.get("teacher") != config["behavior_cloning"]["teacher"]
        or curriculum.get("teacher_version")
        != config["behavior_cloning"]["teacher_version"]
        or curriculum.get("training_split_only") is not True
        or curriculum.get("dagger_beta_schedule")
        != config["behavior_cloning"]["dagger_beta_schedule"]
    ):
        raise ArtifactError("V3 training curriculum evidence drifted")
    terminal_checkpoint = receipt.get("terminal_checkpoint")
    training_artifacts = receipt.get("artifacts")
    expected_training_artifacts = {
        "training_onnx": "artifacts/city_recovery_ppo.v3.onnx",
        "metadata": "artifacts/city_recovery_ppo.v3.metadata.json",
        "manifest": "artifacts/model_manifest.v3.json",
        "parity": "training/v3/onnx-parity.json",
    }
    if (
        not isinstance(terminal_checkpoint, dict)
        or set(terminal_checkpoint) != {"path", "bytes", "sha256"}
        or terminal_checkpoint.get("path") != "artifacts/city_recovery_ppo.v3.zip"
        or not isinstance(terminal_checkpoint.get("bytes"), int)
        or isinstance(terminal_checkpoint.get("bytes"), bool)
        or terminal_checkpoint["bytes"] <= 0
        or not isinstance(training_artifacts, dict)
        or set(training_artifacts) != set(expected_training_artifacts)
    ):
        raise ArtifactError("V3 training release artifact schema drifted")
    terminal_checkpoint_path = _safe_relative_file(
        root, terminal_checkpoint["path"], "V3 terminal training checkpoint"
    )
    if terminal_checkpoint[
        "bytes"
    ] != terminal_checkpoint_path.stat().st_size or terminal_checkpoint.get(
        "sha256"
    ) != _sha256_bytes(terminal_checkpoint_path.read_bytes()):
        raise ArtifactError("V3 terminal training checkpoint binding drifted")
    eligible_candidates.append(
        {
            "path": terminal_checkpoint["path"],
            "sha256": terminal_checkpoint["sha256"],
            "completed_transitions": receipt["completed_transitions"],
            "source": "terminal_checkpoint",
        }
    )
    unique_candidates: dict[tuple[str, int], dict[str, Any]] = {}
    for candidate in eligible_candidates:
        unique_candidates[(candidate["sha256"], candidate["completed_transitions"])] = (
            candidate
        )
    eligible_candidates = sorted(
        unique_candidates.values(),
        key=lambda item: (item["completed_transitions"], item["sha256"]),
    )
    for artifact_id, relative in expected_training_artifacts.items():
        binding = training_artifacts.get(artifact_id)
        artifact_path = _safe_relative_file(
            root,
            binding.get("path") if isinstance(binding, dict) else None,
            f"V3 training {artifact_id}",
        )
        if (
            set(binding) != {"path", "sha256"}
            or binding.get("path") != relative
            or binding.get("sha256") != _sha256_bytes(artifact_path.read_bytes())
        ):
            raise ArtifactError(f"V3 training {artifact_id} binding drifted")
    if (
        receipt.get("manifest_sha256") != training_artifacts["manifest"]["sha256"]
        or receipt.get("parity_sha256") != training_artifacts["parity"]["sha256"]
    ):
        raise ArtifactError("V3 training manifest/parity digest drifted")
    release_files = terminal.get("release_files")
    expected_release_paths = {
        "artifacts/city_recovery_ppo.v3.zip",
        "artifacts/city_recovery_ppo.v3.onnx",
        "artifacts/city_recovery_ppo.v3.metadata.json",
        "artifacts/model_manifest.v3.json",
        "training/v3/onnx-parity.json",
        _relative(TRAINING_RECEIPT_RELATIVE),
    }
    if (
        set(terminal)
        != {
            "schema_version",
            "status",
            "campaign_identity",
            "campaign_provenance_sha256",
            "configured_transitions",
            "actual_completed_transitions",
            "completed_stage_count",
            "training_use_receipt_sha256",
            "training_receipt_sha256",
            "checkpoint_ledger_sha256",
            "checkpoint_ledger_terminal_record_sha256",
            "release_files",
            "terminal_semantic_sha256",
        }
        or terminal.get("schema_version") != 1
        or terminal.get("status") != "sealed_training_release_complete"
        or terminal.get("campaign_identity") != provenance_sha256
        or terminal.get("campaign_provenance_sha256") != provenance_sha256
        or terminal.get("configured_transitions") != config["total_transitions"]
        or terminal.get("actual_completed_transitions") != config["total_transitions"]
        or terminal.get("completed_stage_count") != stage_count
        or terminal.get("training_use_receipt_sha256") != _sha256_bytes(use_bytes)
        or terminal.get("training_receipt_sha256") != _sha256_bytes(receipt_bytes)
        or terminal.get("checkpoint_ledger_sha256") != _sha256_bytes(ledger_bytes)
        or terminal.get("checkpoint_ledger_terminal_record_sha256") != previous_hash
        or not isinstance(release_files, dict)
        or set(release_files) != expected_release_paths
        or any(
            not (root / relative).is_file()
            or digest != _sha256_bytes((root / relative).read_bytes())
            for relative, digest in release_files.items()
        )
    ):
        raise ArtifactError("V3 training terminal provenance drifted")
    return {
        "authorization_sha256": _sha256_bytes(training_authorization_bytes),
        "receipt_sha256": _sha256_bytes(receipt_bytes),
        "use_receipt_sha256": _sha256_bytes(use_bytes),
        "terminal_sha256": _sha256_bytes(terminal_bytes),
        "ledger_sha256": _sha256_bytes(ledger_bytes),
        "campaign_provenance_sha256": provenance_sha256,
        "eligible_candidates": eligible_candidates,
    }


def _validate_selection_publication(
    root: Path,
    selected: dict[str, Any],
    bindings: dict[str, str],
) -> None:
    checkpoint_sha256 = _require_sha256(
        selected.get("sha256"), "selected checkpoint digest"
    )
    stage_root = (
        root
        / "internal"
        / "training_runs"
        / "v3"
        / "selection-staging"
        / checkpoint_sha256
    )
    claim_path = stage_root / "selection-claim.json"
    stage_receipt_path = stage_root / "selected-deployment-stage-receipt.json"
    claim, claim_bytes = _read_json(claim_path, "V3 selected-deployment staging claim")
    claim_base = {
        "schema_version": 1,
        "status": "selected_deployment_staging_claimed",
        "selected_checkpoint": selected,
        "protocol_sha256": bindings["protocol_sha256"],
        "source_seal_sha256": bindings["source_seal_sha256"],
        "scientific_source_sha256": bindings["scientific_source_sha256"],
        "engine_spec_sha256": ENGINE_V3_SPEC_SHA256,
        "outcome_definition_sha256": SOLVED_DEFINITION_V3_SHA256,
        "observation_order_sha256": bindings["observation_order_sha256"],
        "action_order_sha256": bindings["action_order_sha256"],
        "destinations": {
            "onnx": _relative(SELECTED_ONNX_RELATIVE),
            "parity": _relative(SELECTED_PARITY_RELATIVE),
            "manifest": _relative(SELECTED_MANIFEST_RELATIVE),
        },
    }
    expected_claim = {
        **claim_base,
        "claim_semantic_sha256": canonical_hash(claim_base),
    }
    if claim != expected_claim:
        raise ArtifactError("V3 selected-deployment staging claim drifted")
    stage_receipt, _ = _read_json(
        stage_receipt_path, "V3 selected-deployment staging receipt"
    )
    artifacts: dict[str, Any] = {}
    for artifact_id, relative in {
        "onnx": SELECTED_ONNX_RELATIVE,
        "parity": SELECTED_PARITY_RELATIVE,
        "manifest": SELECTED_MANIFEST_RELATIVE,
    }.items():
        public_path = _safe_relative_file(
            root, _relative(relative), f"selected V3 {artifact_id}"
        )
        staged_path = _safe_relative_file(
            root,
            _relative(stage_root.relative_to(root) / public_path.name),
            f"staged selected V3 {artifact_id}",
        )
        public_payload = public_path.read_bytes()
        staged_payload = staged_path.read_bytes()
        if staged_payload != public_payload:
            raise ArtifactError(f"staged selected V3 {artifact_id} bytes drifted")
        artifacts[artifact_id] = {
            "path": _relative(relative),
            "bytes": len(public_payload),
            "sha256": _sha256_bytes(public_payload),
        }
    stage_receipt_base = {
        "schema_version": 1,
        "status": "selected_deployment_staging_complete",
        "claim_sha256": _sha256_bytes(claim_bytes),
        "selected_checkpoint_sha256": checkpoint_sha256,
        "artifacts": artifacts,
    }
    expected_stage_receipt = {
        **stage_receipt_base,
        "receipt_semantic_sha256": canonical_hash(stage_receipt_base),
    }
    if stage_receipt != expected_stage_receipt:
        raise ArtifactError("V3 selected-deployment staging receipt drifted")


def _validate_deployment(root: Path, bindings: dict[str, str]) -> dict[str, Any]:
    manifest_path = root / SELECTED_MANIFEST_RELATIVE
    parity_path = root / SELECTED_PARITY_RELATIVE
    onnx_path = root / SELECTED_ONNX_RELATIVE
    selection_path = root / SELECTION_RECEIPT_RELATIVE
    authorization_path = root / FINAL_AUTHORIZATION_RELATIVE
    manifest, manifest_bytes = _read_json(manifest_path, "selected V3 manifest")
    parity, parity_bytes = _read_json(parity_path, "selected V3 parity")
    selection, selection_bytes = _read_json(selection_path, "V3 selection receipt")
    authorization, authorization_bytes = _read_json(
        authorization_path, "V3 final authorization"
    )
    config = _validate_config(root)
    training_chain = _validate_training_chain(root, config, bindings)

    policy = manifest.get("policy")
    manifest_bindings = manifest.get("bindings")
    files = manifest.get("files")
    if (
        set(manifest) != {"schema_version", "status", "policy", "bindings", "files"}
        or manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or manifest.get("status") != "selected_deployable"
        or not isinstance(policy, dict)
        or set(policy)
        != {
            "id",
            "algorithm",
            "observation_count",
            "action_count",
            "completed_transitions",
        }
        or policy.get("id") != POLICY_ID
        or policy.get("algorithm") != "PPO"
        or policy.get("observation_count") != len(OBSERVATION_ORDER_V3)
        or policy.get("action_count") != len(ACTION_ORDER_V3)
        or not isinstance(policy.get("completed_transitions"), int)
        or isinstance(policy.get("completed_transitions"), bool)
        or not 0 < policy["completed_transitions"] <= config["total_transitions"]
        or not isinstance(manifest_bindings, dict)
        or not isinstance(files, dict)
        or set(files) != {"onnx", "parity"}
    ):
        raise ArtifactError("selected V3 manifest policy contract drifted")
    expected_bindings = {
        "protocol_sha256": bindings["protocol_sha256"],
        "source_seal_sha256": bindings["source_seal_sha256"],
        "scientific_source_sha256": bindings["scientific_source_sha256"],
        "engine_spec_sha256": ENGINE_V3_SPEC_SHA256,
        "outcome_definition_sha256": SOLVED_DEFINITION_V3_SHA256,
        "observation_order_sha256": bindings["observation_order_sha256"],
        "action_order_sha256": bindings["action_order_sha256"],
    }
    if set(manifest_bindings) != {
        *expected_bindings,
        "selected_checkpoint_sha256",
    }:
        raise ArtifactError("selected V3 manifest binding fields drifted")
    for field, value in expected_bindings.items():
        if manifest_bindings.get(field) != value:
            raise ArtifactError(f"selected V3 manifest {field} drifted")
    checkpoint_sha256 = _require_sha256(
        manifest_bindings.get("selected_checkpoint_sha256"),
        "selected checkpoint digest",
    )

    try:
        onnx_payload = onnx_path.read_bytes()
    except OSError as exc:
        raise ArtifactError("selected V3 ONNX is missing or unreadable") from exc
    onnx_sha256 = _sha256_bytes(onnx_payload)
    onnx_binding = files.get("onnx")
    parity_binding = files.get("parity")
    if onnx_binding != {
        "path": _relative(SELECTED_ONNX_RELATIVE),
        "bytes": len(onnx_payload),
        "sha256": onnx_sha256,
        "input_name": "observation",
        "output_name": "action",
        "opset": 17,
        "execution_providers": ["CPUExecutionProvider"],
        "input_type": "tensor(float)",
        "input_shape": ["batch", len(OBSERVATION_ORDER_V3)],
        "output_type": "tensor(float)",
        "output_shape": ["batch", len(ACTION_ORDER_V3)],
    }:
        raise ArtifactError("selected V3 ONNX manifest binding drifted")
    parity_sha256 = _sha256_bytes(parity_bytes)
    if parity_binding != {
        "path": _relative(SELECTED_PARITY_RELATIVE),
        "sha256": parity_sha256,
    }:
        raise ArtifactError("selected V3 parity manifest binding drifted")
    parity_action_error = _finite_number(
        parity.get("maximum_action_absolute_error"),
        "selected V3 parity maximum action error",
    )
    parity_rauc_error = _finite_number(
        parity.get("maximum_rauc_absolute_error"),
        "selected V3 parity maximum RAUC error",
    )
    parity_conservation = _finite_number(
        parity.get("maximum_conservation_residual"),
        "selected V3 parity conservation residual",
    )
    if (
        set(parity)
        != {
            "schema_version",
            "status",
            "passed",
            "checkpoint_sha256",
            "onnx_sha256",
            "interface",
            "development_case_count",
            "action_sample_count",
            "maximum_action_absolute_error",
            "maximum_rauc_absolute_error",
            "outcome_mismatches",
            "deterministic_replay_mismatches",
            "hard_violation_count",
            "maximum_conservation_residual",
            "rows_sha256",
        }
        or parity.get("schema_version") != 1
        or parity.get("status") != "passed"
        or parity.get("passed") is not True
        or parity.get("checkpoint_sha256") != checkpoint_sha256
        or parity.get("onnx_sha256") != onnx_sha256
        or parity.get("interface") != SELECTED_ONNX_INTERFACE
        or parity.get("development_case_count") != 40
        or not isinstance(parity.get("action_sample_count"), int)
        or isinstance(parity.get("action_sample_count"), bool)
        or parity["action_sample_count"] <= 0
        or not 0.0 <= parity_action_error <= 1e-5
        or not 0.0 <= parity_rauc_error <= 1e-6
        or parity.get("outcome_mismatches") != 0
        or parity.get("deterministic_replay_mismatches") != 0
        or parity.get("hard_violation_count") != 0
        or not 0.0 <= parity_conservation <= 1e-6
    ):
        raise ArtifactError("selected V3 ONNX parity evidence drifted")
    _require_sha256(parity.get("rows_sha256"), "selected V3 parity rows digest")

    manifest_sha256 = _sha256_bytes(manifest_bytes)
    selection_sha256 = _sha256_bytes(selection_bytes)
    _semantic_receipt(selection, "receipt_semantic_sha256", "V3 selection receipt")
    selected_checkpoint = selection.get("selected_checkpoint")
    selected_deployment = selection.get("selected_deployment")
    candidate_reports = selection.get("candidate_reports")
    expected_ranking = [
        "candidate_solved_count:descending",
        "candidate_mean_resilience_auc:descending",
        "completed_transitions:ascending",
        "checkpoint_sha256:ascending",
    ]
    if (
        set(selection)
        != {
            "schema_version",
            "status",
            "protocol_sha256",
            "source_seal_sha256",
            "scientific_source_sha256",
            "engine_spec_sha256",
            "outcome_definition_sha256",
            "training_config_sha256",
            "training_receipt_sha256",
            "training_use_receipt_sha256",
            "training_terminal_sha256",
            "campaign_provenance_sha256",
            "checkpoint_ledger_sha256",
            "selection_dataset",
            "development_case_count",
            "ranking",
            "candidate_reports",
            "selected_checkpoint",
            "selected_deployment",
            "final_split_accessed",
            "receipt_semantic_sha256",
        }
        or selection.get("schema_version") != 1
        or selection.get("status") != "checkpoint_selected_on_development_only"
        or selection.get("selection_dataset") != "development"
        or selection.get("development_case_count") != 40
        or selection.get("ranking") != expected_ranking
        or selection.get("final_split_accessed") is not False
        or selection.get("protocol_sha256") != bindings["protocol_sha256"]
        or selection.get("source_seal_sha256") != bindings["source_seal_sha256"]
        or selection.get("scientific_source_sha256")
        != bindings["scientific_source_sha256"]
        or selection.get("engine_spec_sha256") != ENGINE_V3_SPEC_SHA256
        or selection.get("outcome_definition_sha256") != SOLVED_DEFINITION_V3_SHA256
        or selection.get("training_config_sha256")
        != _sha256_bytes((root / CONFIG_RELATIVE).read_bytes())
        or selection.get("training_receipt_sha256") != training_chain["receipt_sha256"]
        or selection.get("training_use_receipt_sha256")
        != training_chain["use_receipt_sha256"]
        or selection.get("training_terminal_sha256")
        != training_chain["terminal_sha256"]
        or selection.get("checkpoint_ledger_sha256") != training_chain["ledger_sha256"]
        or selection.get("campaign_provenance_sha256")
        != training_chain["campaign_provenance_sha256"]
        or not isinstance(candidate_reports, list)
        or len(candidate_reports) != len(training_chain["eligible_candidates"])
        or not isinstance(selected_checkpoint, dict)
        or not isinstance(selected_deployment, dict)
        or set(selected_deployment) != {"checkpoint", "onnx", "parity", "manifest"}
    ):
        raise ArtifactError("V3 selection receipt deployment binding drifted")
    report_keys = {
        "path",
        "sha256",
        "completed_transitions",
        "source",
        "development_case_count",
        "candidate_solved_count",
        "candidate_mean_resilience_auc",
        "candidate_critical_service_days",
        "hard_violation_count",
        "max_conservation_residual",
        "deterministic_replay_mismatches",
        "rows_sha256",
    }
    candidate_bindings = [
        {
            key: report.get(key)
            for key in ("path", "sha256", "completed_transitions", "source")
        }
        for report in candidate_reports
        if isinstance(report, dict)
    ]
    if candidate_bindings != training_chain["eligible_candidates"]:
        raise ArtifactError("V3 selection eligible candidate provenance drifted")
    for report in candidate_reports:
        if not isinstance(report, dict) or set(report) != report_keys:
            raise ArtifactError("V3 selection candidate report schema drifted")
        report_checkpoint = _safe_relative_file(
            root, report.get("path"), "V3 selection candidate checkpoint"
        )
        completed = report.get("completed_transitions")
        mean_rauc = _finite_number(
            report.get("candidate_mean_resilience_auc"),
            "V3 selection candidate mean AUC",
        )
        residual = _finite_number(
            report.get("max_conservation_residual"),
            "V3 selection candidate conservation residual",
        )
        if (
            report.get("sha256") != _sha256_bytes(report_checkpoint.read_bytes())
            or not isinstance(completed, int)
            or isinstance(completed, bool)
            or completed
            not in config["production_stages"].get(
                "registered_stage_boundaries",
                [
                    config["checkpoint_interval_transitions"] * stage
                    for stage in range(1, config["production_stages"]["count"] + 1)
                ],
            )
            or report.get("source") not in {"checkpoint_ledger", "terminal_checkpoint"}
            or report.get("development_case_count") != 40
            or not isinstance(report.get("candidate_solved_count"), int)
            or isinstance(report.get("candidate_solved_count"), bool)
            or not 0 <= report["candidate_solved_count"] <= 40
            or not 0.0 <= mean_rauc <= 1.0
            or not isinstance(report.get("candidate_critical_service_days"), int)
            or isinstance(report.get("candidate_critical_service_days"), bool)
            or report["candidate_critical_service_days"] < 0
            or report.get("hard_violation_count") != 0
            or not 0.0 <= residual <= 1e-6
            or report.get("deterministic_replay_mismatches") != 0
        ):
            raise ArtifactError("V3 selection candidate report evidence drifted")
        stage_number = completed // config["checkpoint_interval_transitions"]
        expected_checkpoint_path = (
            "artifacts/city_recovery_ppo.v3.zip"
            if report.get("source") == "terminal_checkpoint"
            else (
                "internal/training_runs/v3/checkpoints/"
                f"stage-{stage_number:02d}-transition-{completed:09d}.zip"
            )
        )
        if report.get("path") != expected_checkpoint_path or (
            report.get("source") == "terminal_checkpoint"
            and completed != config["total_transitions"]
        ):
            raise ArtifactError("V3 selection candidate path provenance drifted")
        _require_sha256(report.get("rows_sha256"), "V3 development rows digest")
    expected_boundaries = {
        config["checkpoint_interval_transitions"] * stage
        for stage in range(1, config["production_stages"]["count"] + 1)
    }
    if {report["completed_transitions"] for report in candidate_reports} != (
        expected_boundaries
    ):
        raise ArtifactError("V3 selection candidate stage coverage drifted")
    expected_selected = min(
        candidate_reports,
        key=lambda item: (
            -int(item["candidate_solved_count"]),
            -float(item["candidate_mean_resilience_auc"]),
            int(item["completed_transitions"]),
            str(item["sha256"]),
        ),
    )
    if selected_checkpoint != expected_selected:
        raise ArtifactError("V3 selection winner ranking drifted")
    if (
        selected_checkpoint.get("sha256") != checkpoint_sha256
        or selected_checkpoint.get("completed_transitions")
        != policy["completed_transitions"]
    ):
        raise ArtifactError("V3 selected checkpoint policy binding drifted")
    checkpoint_path = _safe_relative_file(
        root, selected_checkpoint.get("path"), "selected V3 checkpoint"
    )
    if _sha256_bytes(checkpoint_path.read_bytes()) != checkpoint_sha256:
        raise ArtifactError("selected V3 checkpoint bytes drifted")
    _validate_selection_publication(root, selected_checkpoint, bindings)
    expected_selected_files = {
        "onnx": {
            "path": _relative(SELECTED_ONNX_RELATIVE),
            "bytes": len(onnx_payload),
            "sha256": onnx_sha256,
        },
        "parity": {
            "path": _relative(SELECTED_PARITY_RELATIVE),
            "sha256": parity_sha256,
        },
        "manifest": {
            "path": _relative(SELECTED_MANIFEST_RELATIVE),
            "sha256": manifest_sha256,
        },
    }
    for artifact_id, expected in expected_selected_files.items():
        if selected_deployment.get(artifact_id) != expected:
            raise ArtifactError(f"selected V3 {artifact_id} deployment drifted")
    if selected_deployment.get("checkpoint") != {
        "path": selected_checkpoint.get("path"),
        "sha256": checkpoint_sha256,
        "completed_transitions": policy["completed_transitions"],
    }:
        raise ArtifactError("selected V3 checkpoint deployment drifted")

    authorization_sha256 = _sha256_bytes(authorization_bytes)
    _semantic_receipt(authorization, "authorization_sha256", "V3 final authorization")
    if (
        set(authorization)
        != {
            "schema_version",
            "status",
            "protocol_sha256",
            "source_seal_sha256",
            "scientific_source_sha256",
            "engine_spec_sha256",
            "outcome_definition_sha256",
            "selection_receipt_sha256",
            "selected_checkpoint_sha256",
            "selected_onnx_sha256",
            "selected_parity_sha256",
            "selected_manifest_sha256",
            "selected_policy_id",
            "candidate_runtime",
            "final_case_count",
            "invocation_limit",
            "durable_identity_matching_resume_only",
            "failed_or_unfavorable_result_may_be_replaced",
            "authorization_sha256",
        }
        or authorization.get("schema_version") != 1
        or authorization.get("status") != "final_evaluation_authorized"
        or authorization.get("selected_policy_id") != POLICY_ID
        or authorization.get("protocol_sha256") != bindings["protocol_sha256"]
        or authorization.get("source_seal_sha256") != bindings["source_seal_sha256"]
        or authorization.get("scientific_source_sha256")
        != bindings["scientific_source_sha256"]
        or authorization.get("engine_spec_sha256") != ENGINE_V3_SPEC_SHA256
        or authorization.get("outcome_definition_sha256") != SOLVED_DEFINITION_V3_SHA256
        or authorization.get("selection_receipt_sha256") != selection_sha256
        or authorization.get("selected_checkpoint_sha256") != checkpoint_sha256
        or authorization.get("selected_onnx_sha256") != onnx_sha256
        or authorization.get("selected_parity_sha256") != parity_sha256
        or authorization.get("selected_manifest_sha256") != manifest_sha256
        or authorization.get("candidate_runtime")
        != "onnxruntime_cpu_selected_deployment_only"
        or type(authorization.get("final_case_count")) is not int
        or authorization["final_case_count"] != 40
        or type(authorization.get("invocation_limit")) is not int
        or authorization["invocation_limit"] != 1
        or authorization.get("durable_identity_matching_resume_only") is not True
        or authorization.get("failed_or_unfavorable_result_may_be_replaced")
        is not False
    ):
        raise ArtifactError("V3 final authorization deployment binding drifted")

    return {
        "config": config,
        "policy": policy,
        "onnx_payload": onnx_payload,
        "onnx_sha256": onnx_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "manifest_sha256": manifest_sha256,
        "parity_sha256": parity_sha256,
        "selection_sha256": selection_sha256,
        "authorization_sha256": authorization_sha256,
    }


def _validate_planner_summary(
    value: Any, label: str, expected_id: str
) -> dict[str, Any]:
    expected_fields = {
        "id",
        "solved_count",
        "failed_count",
        "solve_rate",
        "solve_rate_wilson_ci95",
        "mean_resilience_auc",
    }
    if label == "baseline":
        expected_fields.add("version")
    if (
        not isinstance(value, dict)
        or set(value) != expected_fields
        or value.get("id") != expected_id
    ):
        raise ArtifactError(f"V3 final report {label} identity drifted")
    solved = value.get("solved_count")
    failed = value.get("failed_count")
    rate = value.get("solve_rate")
    interval = value.get("solve_rate_wilson_ci95")
    mean_rauc = value.get("mean_resilience_auc")
    if (
        not isinstance(solved, int)
        or isinstance(solved, bool)
        or not 0 <= solved <= 40
        or failed != 40 - solved
        or not isinstance(rate, (int, float))
        or isinstance(rate, bool)
        or not math.isclose(float(rate), solved / 40.0, abs_tol=1e-12)
        or not isinstance(interval, list)
        or len(interval) != 2
        or any(
            not isinstance(item, (int, float))
            or isinstance(item, bool)
            or not math.isfinite(float(item))
            or not 0.0 <= float(item) <= 1.0
            for item in interval
        )
        or float(interval[0]) > float(interval[1])
        or not isinstance(mean_rauc, (int, float))
        or isinstance(mean_rauc, bool)
        or not math.isfinite(float(mean_rauc))
        or not 0.0 <= float(mean_rauc) <= 1.0
    ):
        raise ArtifactError(f"V3 final report {label} aggregate drifted")
    return value


def _wilson_interval(successes: int, total: int) -> list[float]:
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


def _expected_final_order() -> list[dict[str, Any]]:
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


def _validate_final_row_summary(value: Any, label: str) -> dict[str, Any]:
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
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ArtifactError(f"V3 final ledger {label} summary schema drifted")
    solved = value["solved"]
    rauc = _finite_number(value["resilience_auc"], f"{label} resilience AUC")
    residual = _finite_number(
        value["max_conservation_residual"], f"{label} conservation residual"
    )
    if (
        not isinstance(solved, bool)
        or value["status"] != ("solved" if solved else "failed")
        or not isinstance(value["reason_codes"], list)
        or not all(isinstance(item, str) for item in value["reason_codes"])
        or not 0.0 <= rauc <= 1.0
        or not isinstance(value["critical_service_days"], int)
        or isinstance(value["critical_service_days"], bool)
        or value["critical_service_days"] < 0
        or not isinstance(value["hard_violation_count"], int)
        or isinstance(value["hard_violation_count"], bool)
        or value["hard_violation_count"] < 0
        or residual < 0.0
    ):
        raise ArtifactError(f"V3 final ledger {label} summary values drifted")
    _require_sha256(value["trajectory_sha256"], f"{label} trajectory digest")
    return value


def _load_final_ledger(
    root: Path, identity: dict[str, Any]
) -> tuple[list[dict[str, Any]], bytes]:
    ledger_path = root / FINAL_LEDGER_RELATIVE
    try:
        payload = ledger_path.read_bytes()
        lines = payload.decode("utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ArtifactError("V3 final ledger is missing or unreadable") from exc
    expected_order = _expected_final_order()
    if (
        not payload.endswith(b"\n")
        or len(lines) != 40
        or any(not line for line in lines)
    ):
        raise ArtifactError("V3 final ledger must contain exactly 40 durable rows")
    identity_sha256 = canonical_hash(identity)
    previous_hash = "0" * 64
    rows: list[dict[str, Any]] = []
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
    for index, (line, expected) in enumerate(zip(lines, expected_order), start=1):
        try:
            record = json.loads(line)
        except ValueError as exc:
            raise ArtifactError(f"V3 final ledger row {index} is invalid") from exc
        if not isinstance(record, dict) or set(record) != expected_keys:
            raise ArtifactError("V3 final ledger row schema drifted")
        if any(record.get(key) != value for key, value in expected.items()):
            raise ArtifactError("V3 final ledger order drifted")
        if record.get("identity_sha256") != identity_sha256:
            raise ArtifactError("V3 final ledger identity drifted")
        if record.get("previous_record_sha256") != previous_hash:
            raise ArtifactError("V3 final ledger hash chain is broken")
        claimed_hash = _require_sha256(
            record.get("record_sha256"), "V3 final ledger record digest"
        )
        unsigned = {
            key: value for key, value in record.items() if key != "record_sha256"
        }
        if claimed_hash != canonical_hash(unsigned):
            raise ArtifactError("V3 final ledger record digest drifted")
        _require_sha256(record["scenario_sha256"], "V3 final scenario digest")
        _require_sha256(record["shock_schedule_sha256"], "V3 final tape digest")
        candidate = _validate_final_row_summary(record["candidate"], "candidate")
        baseline = _validate_final_row_summary(record["baseline"], "baseline")
        expected_pair = (
            "both_solved"
            if candidate["solved"] and baseline["solved"]
            else "ppo_only"
            if candidate["solved"]
            else "heuristic_only"
            if baseline["solved"]
            else "neither"
        )
        delta = round(candidate["resilience_auc"] - baseline["resilience_auc"], 8)
        recorded_delta = record["secondary_rauc_candidate_minus_baseline"]
        replay_mismatches = record["deterministic_replay_mismatches"]
        if (
            record["absolute_outcome_pair"] != expected_pair
            or not isinstance(recorded_delta, (int, float))
            or isinstance(recorded_delta, bool)
            or not math.isfinite(float(recorded_delta))
            or recorded_delta != delta
            or not isinstance(replay_mismatches, int)
            or isinstance(replay_mismatches, bool)
            or replay_mismatches < 0
        ):
            raise ArtifactError("V3 final ledger comparison drifted")
        rows.append(record)
        previous_hash = claimed_hash
    return rows, payload


def _recomputed_final_sections(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_solved = sum(row["candidate"]["solved"] for row in rows)
    baseline_solved = sum(row["baseline"]["solved"] for row in rows)
    pair_counts = {
        key: sum(row["absolute_outcome_pair"] == key for row in rows)
        for key in ("both_solved", "ppo_only", "heuristic_only", "neither")
    }
    candidate_rauc = [float(row["candidate"]["resilience_auc"]) for row in rows]
    baseline_rauc = [float(row["baseline"]["resilience_auc"]) for row in rows]
    candidate_wins = sum(
        left > right + 1e-9 for left, right in zip(candidate_rauc, baseline_rauc)
    )
    baseline_wins = sum(
        right > left + 1e-9 for left, right in zip(candidate_rauc, baseline_rauc)
    )
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
    return {
        "candidate": {
            "id": POLICY_ID,
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
            "ties": 40 - candidate_wins - baseline_wins,
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
        "rows_sha256": canonical_hash(rows),
    }


def _validate_benchmark(
    root: Path, bindings: dict[str, str], deployment: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    report, report_bytes = _read_json(
        root / FINAL_REPORT_RELATIVE, "V3 aggregate final report"
    )
    if (
        set(report)
        != {
            "schema_version",
            "artifact_type",
            "status",
            "primary_metric",
            "synthetic_case_count",
            "candidate",
            "baseline",
            "paired_absolute_outcomes",
            "secondary_head_to_head_resilience_auc",
            "invariants",
            "identity",
            "rows_sha256",
            "ledger_sha256",
        }
        or report.get("schema_version") != 1
        or report.get("artifact_type") != "city_recovery_v3_final_benchmark"
        or report.get("status") != "complete"
        or report.get("primary_metric") != "independent_absolute_disasters_solved"
        or report.get("synthetic_case_count") != 40
    ):
        raise ArtifactError("V3 aggregate final report contract drifted")
    candidate = _validate_planner_summary(
        report.get("candidate"), "candidate", POLICY_ID
    )
    baseline = _validate_planner_summary(
        report.get("baseline"), "baseline", "reactive-public-state-heuristic-v3"
    )
    if baseline.get("version") != "3.0.0":
        raise ArtifactError("V3 final report baseline version drifted")

    pairs = report.get("paired_absolute_outcomes")
    pair_keys = ("both_solved", "ppo_only", "heuristic_only", "neither")
    if (
        not isinstance(pairs, dict)
        or set(pairs) != set(pair_keys)
        or any(
            not isinstance(pairs[key], int)
            or isinstance(pairs[key], bool)
            or pairs[key] < 0
            for key in pair_keys
        )
        or sum(pairs.values()) != 40
        or candidate["solved_count"] != pairs["both_solved"] + pairs["ppo_only"]
        or baseline["solved_count"] != pairs["both_solved"] + pairs["heuristic_only"]
    ):
        raise ArtifactError("V3 final report paired outcomes drifted")

    head_to_head = report.get("secondary_head_to_head_resilience_auc")
    if (
        not isinstance(head_to_head, dict)
        or set(head_to_head)
        != {
            "candidate_wins",
            "baseline_wins",
            "ties",
            "candidate_mean_minus_baseline_mean",
        }
        or any(
            not isinstance(head_to_head.get(key), int)
            or isinstance(head_to_head.get(key), bool)
            or head_to_head[key] < 0
            for key in ("candidate_wins", "baseline_wins", "ties")
        )
        or sum(head_to_head[key] for key in ("candidate_wins", "baseline_wins", "ties"))
        != 40
        or not isinstance(
            head_to_head.get("candidate_mean_minus_baseline_mean"), (int, float)
        )
        or isinstance(head_to_head.get("candidate_mean_minus_baseline_mean"), bool)
        or not math.isfinite(float(head_to_head["candidate_mean_minus_baseline_mean"]))
    ):
        raise ArtifactError("V3 final report secondary comparison drifted")

    invariants = report.get("invariants")
    if (
        not isinstance(invariants, dict)
        or set(invariants)
        != {
            "all_rows_present",
            "unique_rows",
            "same_tapes",
            "deterministic_replay_mismatches",
            "candidate_hard_violations",
            "baseline_hard_violations",
            "maximum_conservation_residual",
        }
        or invariants.get("all_rows_present") is not True
        or invariants.get("unique_rows") is not True
        or invariants.get("same_tapes") is not True
        or invariants.get("deterministic_replay_mismatches") != 0
        or invariants.get("candidate_hard_violations") != 0
        or invariants.get("baseline_hard_violations") != 0
        or not isinstance(invariants.get("maximum_conservation_residual"), (int, float))
        or isinstance(invariants.get("maximum_conservation_residual"), bool)
        or not math.isfinite(float(invariants["maximum_conservation_residual"]))
        or not 0.0 <= float(invariants["maximum_conservation_residual"]) <= 1e-6
    ):
        raise ArtifactError("V3 final report invariant evidence drifted")

    identity = report.get("identity")
    expected_identity = {
        "selected_policy_id": POLICY_ID,
        "protocol_sha256": bindings["protocol_sha256"],
        "source_seal_sha256": bindings["source_seal_sha256"],
        "scientific_source_sha256": bindings["scientific_source_sha256"],
        "selection_receipt_sha256": deployment["selection_sha256"],
        "authorization_sha256": deployment["authorization_sha256"],
        "checkpoint_sha256": deployment["checkpoint_sha256"],
        "completed_transitions": deployment["policy"]["completed_transitions"],
        "onnx_path": _relative(SELECTED_ONNX_RELATIVE),
        "onnx_sha256": deployment["onnx_sha256"],
        "parity_sha256": deployment["parity_sha256"],
        "manifest_sha256": deployment["manifest_sha256"],
        "candidate_runtime": "onnxruntime_cpu_selected_deployment_only",
        "engine_spec_sha256": ENGINE_V3_SPEC_SHA256,
        "outcome_definition_sha256": SOLVED_DEFINITION_V3_SHA256,
    }
    if identity != expected_identity:
        raise ArtifactError("V3 final report deployment identity drifted")
    rows, ledger_bytes = _load_final_ledger(root, expected_identity)
    recomputed = _recomputed_final_sections(rows)
    for field, value in recomputed.items():
        if report.get(field) != value:
            raise ArtifactError(f"V3 final report {field} does not match its ledger")
    ledger_sha256 = _sha256_bytes(ledger_bytes)
    if report.get("ledger_sha256") != ledger_sha256:
        raise ArtifactError("V3 final report ledger digest drifted")
    report_sha256 = _sha256_bytes(report_bytes)
    _validate_final_publication(
        root,
        expected_identity,
        rows,
        ledger_sha256,
        report_sha256,
    )
    return report, report_sha256


def _validate_final_publication(
    root: Path,
    identity: dict[str, Any],
    rows: list[dict[str, Any]],
    ledger_sha256: str,
    report_sha256: str,
) -> None:
    use_receipt, use_bytes = _read_json(
        root / FINAL_USE_RECEIPT_RELATIVE, "V3 final-use receipt"
    )
    identity_sha256 = canonical_hash(identity)
    expected_use_receipt = {
        "schema_version": 1,
        "status": "single_final_invocation_started",
        "identity": identity,
        "identity_sha256": identity_sha256,
        "case_count": 40,
        "iteration_order": "family_order_then_ascending_seed",
        "durable_resume_only": True,
        "exclusive_lock_path": _relative(FINAL_LOCK_RELATIVE),
    }
    if use_receipt != expected_use_receipt:
        raise ArtifactError("V3 final-use receipt contract drifted")
    terminal, _ = _read_json(root / FINAL_TERMINAL_RELATIVE, "V3 final terminal marker")
    terminal_record_sha256 = _require_sha256(
        rows[-1].get("record_sha256"), "V3 final ledger terminal record digest"
    )
    terminal_base = {
        "schema_version": 1,
        "status": "final_evaluation_complete",
        "selected_policy_id": POLICY_ID,
        "identity": identity,
        "identity_sha256": identity_sha256,
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
            "path": _relative(FINAL_REPORT_RELATIVE),
            "sha256": report_sha256,
        },
        "final_use_receipt": {
            "path": _relative(FINAL_USE_RECEIPT_RELATIVE),
            "sha256": _sha256_bytes(use_bytes),
        },
        "final_ledger": {
            "path": _relative(FINAL_LEDGER_RELATIVE),
            "sha256": ledger_sha256,
            "record_count": 40,
            "terminal_record_sha256": terminal_record_sha256,
        },
    }
    expected_terminal = {
        **terminal_base,
        "terminal_semantic_sha256": canonical_hash(terminal_base),
    }
    if terminal != expected_terminal:
        raise ArtifactError("V3 final terminal marker contract drifted")


def _create_session(payload: bytes) -> ort.InferenceSession:
    try:
        graph = onnx.load_model_from_string(payload)
        onnx.checker.check_model(graph)
        opsets = [
            {"domain": item.domain or "", "version": int(item.version)}
            for item in graph.opset_import
        ]
        if opsets != [{"domain": "", "version": 17}]:
            raise ArtifactError("selected V3 ONNX must use opset 17")
        options = ort.SessionOptions()
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            payload,
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
    except ArtifactError:
        raise
    except Exception as exc:
        raise ArtifactError(
            "selected V3 ONNX cannot be loaded by the CPU runtime"
        ) from exc
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise ArtifactError("selected V3 ONNX did not remain on CPUExecutionProvider")
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if (
        len(inputs) != 1
        or inputs[0].name != "observation"
        or inputs[0].type != "tensor(float)"
        or list(inputs[0].shape) != ["batch", len(OBSERVATION_ORDER_V3)]
    ):
        raise ArtifactError("selected V3 ONNX input schema is invalid")
    if (
        len(outputs) != 1
        or outputs[0].name != "action"
        or outputs[0].type != "tensor(float)"
        or list(outputs[0].shape) != ["batch", len(ACTION_ORDER_V3)]
    ):
        raise ArtifactError("selected V3 ONNX output schema is invalid")
    result = session.run(
        ["action"],
        {"observation": np.zeros((1, len(OBSERVATION_ORDER_V3)), dtype=np.float32)},
    )[0]
    if np.asarray(result).shape != (1, len(ACTION_ORDER_V3)) or not np.all(
        np.isfinite(result)
    ):
        raise ArtifactError(
            "selected V3 ONNX smoke inference returned an invalid action"
        )
    return session


def _load_release(
    root: Path,
    session_factory: Callable[[bytes], ort.InferenceSession] | None = None,
) -> PolicyBundle:
    bindings = _validate_protocol(root)
    deployment = _validate_deployment(root, bindings)
    benchmark, benchmark_sha256 = _validate_benchmark(root, bindings, deployment)
    session = (session_factory or _create_session)(deployment["onnx_payload"])
    config = deployment["config"]
    policy = deployment["policy"]
    metadata = {
        "id": POLICY_ID,
        "version": POLICY_VERSION,
        "schema_version": POLICY_SCHEMA_VERSION,
        "artifact_type": POLICY_ARTIFACT_TYPE,
        "license": ARTIFACT_LICENSE,
        "observation_order": list(OBSERVATION_ORDER_V3),
        "action_order": list(ACTION_ORDER_V3),
        "architecture": {
            "type": "separate_actor_critic_mlp",
            "actor": [
                len(OBSERVATION_ORDER_V3),
                *config["network"]["actor"],
                len(ACTION_ORDER_V3),
            ],
            "critic": [
                len(OBSERVATION_ORDER_V3),
                *config["network"]["critic"],
                1,
            ],
            "activation": config["network"]["activation"],
            "trainable_parameters": config["expected_trainable_parameters"],
        },
        "training": {
            "algorithm": "PPO",
            "library": "stable-baselines3",
            "device": "cpu",
            "environment": ENGINE_V3_ID,
            "requested_transitions": config["total_transitions"],
            "completed_transitions": policy["completed_transitions"],
            "seed": config["policy_seed"],
            "simulator_lanes": config["simulator_lanes"],
        },
        "export": {
            "format": "ONNX",
            "opset": 17,
            "input_name": "observation",
            "output_name": "action",
            "runtime_provider": "CPUExecutionProvider",
            "onnx_sha256": deployment["onnx_sha256"],
        },
        "environment": {
            "id": ENGINE_V3_ID,
            "version": ENGINE_V3_VERSION,
            "observation_count": len(OBSERVATION_ORDER_V3),
            "action_count": len(ACTION_ORDER_V3),
            "engine_spec_sha256": ENGINE_V3_SPEC_SHA256,
            "outcome_definition_sha256": SOLVED_DEFINITION_V3_SHA256,
        },
        "deployment": {
            "manifest_sha256": deployment["manifest_sha256"],
            "parity_sha256": deployment["parity_sha256"],
            "selected_checkpoint_sha256": deployment["checkpoint_sha256"],
            "scientific_source_sha256": bindings["scientific_source_sha256"],
            "benchmark_sha256": benchmark_sha256,
        },
    }
    return PolicyBundle(
        metadata=metadata,
        session=session,
        onnx_sha256=deployment["onnx_sha256"],
        sb3_sha256=deployment["checkpoint_sha256"],
        manifest_sha256=deployment["manifest_sha256"],
        manifest_schema_version=MANIFEST_SCHEMA_VERSION,
        parity_sha256=deployment["parity_sha256"],
        benchmark=benchmark,
        benchmark_sha256=benchmark_sha256,
        scientific_source_sha256=bindings["scientific_source_sha256"],
    )


@lru_cache(maxsize=1)
def load_policy() -> PolicyBundle:
    """Load the selected V3 release only when deployment and evidence are complete."""

    return _load_release(ROOT)


load_policy_v3 = load_policy
