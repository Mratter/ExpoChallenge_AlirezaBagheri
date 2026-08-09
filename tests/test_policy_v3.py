from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

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
from scripts import v3_protocol as protocol_contract
from model.ppo_v3 import (
    CONFIG_RELATIVE,
    FINAL_AUTHORIZATION_RELATIVE,
    FINAL_LEDGER_RELATIVE,
    FINAL_REPORT_RELATIVE,
    FINAL_TERMINAL_RELATIVE,
    FINAL_USE_RECEIPT_RELATIVE,
    POLICY_ID,
    PROTOCOL_RELATIVE,
    SCIENTIFIC_SOURCE_PATHS,
    SEALING_TOOL_RELATIVE,
    SELECTED_MANIFEST_RELATIVE,
    SELECTED_ONNX_RELATIVE,
    SELECTED_PARITY_RELATIVE,
    SELECTION_RECEIPT_RELATIVE,
    SOURCE_SEAL_RELATIVE,
    TRAINING_LEDGER_RELATIVE,
    TRAINING_AUTHORIZATION_RELATIVE,
    TRAINING_RECEIPT_RELATIVE,
    TRAINING_TERMINAL_RELATIVE,
    TRAINING_USE_RECEIPT_RELATIVE,
    ArtifactError,
    _expected_final_order,
    _load_release,
    _recomputed_final_sections,
)


def _formatted(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_json(root: Path, relative: Path, value: dict[str, Any]) -> bytes:
    payload = _formatted(value)
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return payload


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class _FakeSession:
    pass


def _release_fixture(root: Path) -> dict[str, Any]:
    config = {
        "environment": ENGINE_V3_ID,
        "observation_count": len(OBSERVATION_ORDER_V3),
        "action_count": len(ACTION_ORDER_V3),
        "expected_trainable_parameters": 322733,
        "total_transitions": 645120,
        "policy_seed": 37017,
        "simulator_lanes": 12,
        "network": {
            "activation": "SiLU",
            "actor": [384, 256, 128],
            "critic": [384, 256, 128],
        },
    }
    for relative in SCIENTIFIC_SOURCE_PATHS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative == "training/v3/config.json":
            path.write_bytes(_formatted(config))
        else:
            path.write_text(f"fixture:{relative}\n", encoding="utf-8")
    sealing_tool = root / SEALING_TOOL_RELATIVE
    sealing_tool.parent.mkdir(parents=True, exist_ok=True)
    sealing_tool.write_text("# fixture sealing tool\n", encoding="utf-8")

    source_files = {
        relative: _sha256((root / relative).read_bytes())
        for relative in SCIENTIFIC_SOURCE_PATHS
    }
    source = {
        "files_sha256": source_files,
        "semantic_sha256": canonical_hash(source_files),
    }
    protocol = {
        "schema_version": 1,
        "protocol_id": "city-recovery-v3-preregistration-v1",
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
                "uses_same_observation_contract": True,
                "uses_same_action_contract": True,
                "future_tape_visible": False,
            },
        },
    }
    protocol_payload = _write_json(root, PROTOCOL_RELATIVE, protocol)
    seal_base = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "status": "frozen",
        "scientific_source": source,
        "protocol": {
            "path": PROTOCOL_RELATIVE.as_posix(),
            "bytes": len(protocol_payload),
            "sha256": _sha256(protocol_payload),
        },
        "sealing_tool": {
            "path": SEALING_TOOL_RELATIVE.as_posix(),
            "sha256": _sha256(sealing_tool.read_bytes()),
        },
    }
    seal = {**seal_base, "seal_sha256": canonical_hash(seal_base)}
    seal_payload = _write_json(root, SOURCE_SEAL_RELATIVE, seal)
    protocol_sha256 = _sha256(protocol_payload)
    seal_sha256 = _sha256(seal_payload)

    checkpoint_sha256 = "c" * 64
    onnx_payload = b"fixture selected onnx"
    onnx_path = root / SELECTED_ONNX_RELATIVE
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_path.write_bytes(onnx_payload)
    onnx_sha256 = _sha256(onnx_payload)
    parity = {
        "schema_version": 1,
        "status": "passed",
        "passed": True,
        "checkpoint_sha256": checkpoint_sha256,
        "onnx_sha256": onnx_sha256,
        "interface": {
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
        },
        "development_case_count": 40,
        "action_sample_count": 1200,
        "maximum_action_absolute_error": 1e-7,
        "maximum_rauc_absolute_error": 1e-8,
        "outcome_mismatches": 0,
        "deterministic_replay_mismatches": 0,
        "hard_violation_count": 0,
        "maximum_conservation_residual": 2e-8,
    }
    parity_payload = _write_json(root, SELECTED_PARITY_RELATIVE, parity)
    parity_sha256 = _sha256(parity_payload)
    completed_transitions = 92160
    manifest = {
        "schema_version": 1,
        "status": "selected_deployable",
        "policy": {
            "id": "city-recovery-sb3-ppo-v3-selected",
            "algorithm": "PPO",
            "observation_count": len(OBSERVATION_ORDER_V3),
            "action_count": len(ACTION_ORDER_V3),
            "completed_transitions": completed_transitions,
        },
        "bindings": {
            "protocol_sha256": protocol_sha256,
            "source_seal_sha256": seal_sha256,
            "scientific_source_sha256": source["semantic_sha256"],
            "engine_spec_sha256": ENGINE_V3_SPEC_SHA256,
            "outcome_definition_sha256": SOLVED_DEFINITION_V3_SHA256,
            "observation_order_sha256": canonical_hash(list(OBSERVATION_ORDER_V3)),
            "action_order_sha256": canonical_hash(list(ACTION_ORDER_V3)),
            "selected_checkpoint_sha256": checkpoint_sha256,
        },
        "files": {
            "onnx": {
                "path": SELECTED_ONNX_RELATIVE.as_posix(),
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
            },
            "parity": {
                "path": SELECTED_PARITY_RELATIVE.as_posix(),
                "sha256": parity_sha256,
            },
        },
    }
    manifest_payload = _write_json(root, SELECTED_MANIFEST_RELATIVE, manifest)
    manifest_sha256 = _sha256(manifest_payload)
    checkpoint_path = "internal/training_runs/v3/checkpoints/stage-01.zip"
    selection_base = {
        "schema_version": 1,
        "status": "checkpoint_selected_on_development_only",
        "protocol_sha256": protocol_sha256,
        "source_seal_sha256": seal_sha256,
        "scientific_source_sha256": source["semantic_sha256"],
        "selection_dataset": "development",
        "development_case_count": 40,
        "selected_checkpoint": {
            "path": checkpoint_path,
            "sha256": checkpoint_sha256,
            "completed_transitions": completed_transitions,
        },
        "selected_deployment": {
            "checkpoint": {
                "path": checkpoint_path,
                "sha256": checkpoint_sha256,
                "completed_transitions": completed_transitions,
            },
            "onnx": {
                "path": SELECTED_ONNX_RELATIVE.as_posix(),
                "bytes": len(onnx_payload),
                "sha256": onnx_sha256,
            },
            "parity": {
                "path": SELECTED_PARITY_RELATIVE.as_posix(),
                "sha256": parity_sha256,
            },
            "manifest": {
                "path": SELECTED_MANIFEST_RELATIVE.as_posix(),
                "sha256": manifest_sha256,
            },
        },
        "final_split_accessed": False,
    }
    selection = {
        **selection_base,
        "receipt_semantic_sha256": canonical_hash(selection_base),
    }
    selection_payload = _write_json(root, SELECTION_RECEIPT_RELATIVE, selection)
    selection_sha256 = _sha256(selection_payload)
    authorization_base = {
        "schema_version": 1,
        "status": "final_evaluation_authorized",
        "protocol_sha256": protocol_sha256,
        "source_seal_sha256": seal_sha256,
        "scientific_source_sha256": source["semantic_sha256"],
        "selection_receipt_sha256": selection_sha256,
        "selected_checkpoint_sha256": checkpoint_sha256,
        "selected_onnx_sha256": onnx_sha256,
        "selected_parity_sha256": parity_sha256,
        "selected_manifest_sha256": manifest_sha256,
        "candidate_runtime": "onnxruntime_cpu_selected_deployment_only",
        "final_case_count": 40,
        "invocation_limit": 1,
    }
    authorization = {
        **authorization_base,
        "authorization_sha256": canonical_hash(authorization_base),
    }
    authorization_payload = _write_json(
        root, FINAL_AUTHORIZATION_RELATIVE, authorization
    )
    benchmark = {
        "schema_version": 1,
        "artifact_type": "city_recovery_v3_final_benchmark",
        "status": "complete",
        "primary_metric": "independent_absolute_disasters_solved",
        "synthetic_case_count": 40,
        "candidate": {
            "id": "city-recovery-sb3-ppo-v3-selected",
            "solved_count": 30,
            "failed_count": 10,
            "solve_rate": 0.75,
            "solve_rate_wilson_ci95": [0.5981, 0.8581],
            "mean_resilience_auc": 0.55,
        },
        "baseline": {
            "id": "reactive-public-state-heuristic-v3",
            "version": "3.0.0",
            "solved_count": 10,
            "failed_count": 30,
            "solve_rate": 0.25,
            "solve_rate_wilson_ci95": [0.1419, 0.4019],
            "mean_resilience_auc": 0.47,
        },
        "paired_absolute_outcomes": {
            "both_solved": 8,
            "ppo_only": 22,
            "heuristic_only": 2,
            "neither": 8,
        },
        "secondary_head_to_head_resilience_auc": {
            "candidate_wins": 29,
            "baseline_wins": 11,
            "ties": 0,
            "candidate_mean_minus_baseline_mean": 0.08,
        },
        "invariants": {
            "all_rows_present": True,
            "unique_rows": True,
            "same_tapes": True,
            "deterministic_replay_mismatches": 0,
            "candidate_hard_violations": 0,
            "baseline_hard_violations": 0,
            "maximum_conservation_residual": 2e-8,
        },
        "identity": {
            "protocol_sha256": protocol_sha256,
            "source_seal_sha256": seal_sha256,
            "scientific_source_sha256": source["semantic_sha256"],
            "selection_receipt_sha256": selection_sha256,
            "authorization_sha256": _sha256(authorization_payload),
            "checkpoint_sha256": checkpoint_sha256,
            "completed_transitions": completed_transitions,
            "onnx_path": SELECTED_ONNX_RELATIVE.as_posix(),
            "onnx_sha256": onnx_sha256,
            "parity_sha256": parity_sha256,
            "manifest_sha256": manifest_sha256,
            "candidate_runtime": "onnxruntime_cpu_selected_deployment_only",
            "engine_spec_sha256": ENGINE_V3_SPEC_SHA256,
            "outcome_definition_sha256": SOLVED_DEFINITION_V3_SHA256,
        },
        "rows_sha256": "d" * 64,
        "ledger_sha256": "e" * 64,
    }
    _write_json(root, FINAL_REPORT_RELATIVE, benchmark)
    return benchmark


def _sealed_release_fixture(root: Path) -> dict[str, Any]:
    active_root = Path(__file__).resolve().parents[1]
    for relative in (*SCIENTIFIC_SOURCE_PATHS, SEALING_TOOL_RELATIVE.as_posix()):
        source_path = active_root / relative
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source_path.read_bytes())

    with (
        patch.object(protocol_contract, "ROOT", root),
        patch.object(protocol_contract, "CONFIG_PATH", root / CONFIG_RELATIVE),
        patch.object(
            protocol_contract,
            "SEALING_TOOL_PATH",
            root / SEALING_TOOL_RELATIVE,
        ),
    ):
        protocol = protocol_contract.expected_protocol()
        protocol_payload = _write_json(root, PROTOCOL_RELATIVE, protocol)
        seal = protocol_contract.expected_source_seal(protocol)
    seal_payload = _write_json(root, SOURCE_SEAL_RELATIVE, seal)
    config = json.loads((root / CONFIG_RELATIVE).read_text(encoding="utf-8"))
    source = protocol["scientific_source"]
    protocol_sha256 = _sha256(protocol_payload)
    seal_sha256 = _sha256(seal_payload)
    config_sha256 = _sha256((root / CONFIG_RELATIVE).read_bytes())

    training_authorization_base = {
        "schema_version": 1,
        "status": "training_and_development_selection_authorized",
        "rationale": "user_directed_v3_training_campaign",
        "protocol_sha256": protocol_sha256,
        "source_seal_sha256": seal_sha256,
        "scientific_source_sha256": source["semantic_sha256"],
        "engine_spec_sha256": ENGINE_V3_SPEC_SHA256,
        "outcome_definition_sha256": SOLVED_DEFINITION_V3_SHA256,
        "training_config_sha256": config_sha256,
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
        "training_use_receipt_path": TRAINING_USE_RECEIPT_RELATIVE.as_posix(),
        "training_terminal_marker_path": TRAINING_TERMINAL_RELATIVE.as_posix(),
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
    training_authorization = {
        **training_authorization_base,
        "authorization_sha256": canonical_hash(training_authorization_base),
    }
    training_authorization_payload = _write_json(
        root, TRAINING_AUTHORIZATION_RELATIVE, training_authorization
    )

    runtime = {"fixture_runtime": "deterministic"}
    governance = {
        "protocol_sha256": protocol_sha256,
        "source_seal_sha256": seal_sha256,
        "training_authorization_sha256": _sha256(training_authorization_payload),
        "scientific_source_sha256": source["semantic_sha256"],
    }
    provenance = {
        "schema_version": 1,
        "execution_class": "sealed_authorized_campaign",
        "governance": governance,
        "scientific_source_sha256": source["semantic_sha256"],
        "registered_config_sha256": config_sha256,
        "effective_config_sha256": canonical_hash(config),
        "engine_spec_sha256": ENGINE_V3_SPEC_SHA256,
        "outcome_definition_sha256": SOLVED_DEFINITION_V3_SHA256,
        "policy_seed": config["policy_seed"],
        "configured_transitions": config["total_transitions"],
        "simulator_lanes": config["simulator_lanes"],
        "checkpoint_interval_transitions": config["checkpoint_interval_transitions"],
        "production_stages": config["production_stages"],
        "vectorization": config["vectorization"],
        "runtime": runtime,
    }
    provenance_sha256 = canonical_hash(provenance)
    training_use = {
        "schema_version": 1,
        "status": "sealed_training_campaign_claimed",
        "campaign_identity": provenance_sha256,
        "campaign_provenance": provenance,
        "campaign_provenance_sha256": provenance_sha256,
        "single_use": True,
        "stage_resume_only": True,
        "behavior_cloning_resume": "deterministic_replay_before_first_stage_only",
    }
    training_use_payload = _write_json(
        root, TRAINING_USE_RECEIPT_RELATIVE, training_use
    )

    interval = config["checkpoint_interval_transitions"]
    ledger_rows: list[dict[str, Any]] = []
    previous_hash = "0" * 64
    for stage in range(1, config["production_stages"]["count"] + 1):
        completed = stage * interval
        relative = Path(
            "internal/training_runs/v3/checkpoints/"
            f"stage-{stage:02d}-transition-{completed:09d}.zip"
        )
        payload = f"fixture checkpoint stage {stage}\n".encode()
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        row = {
            "schema_version": 1,
            "record_type": "v3_training_stage_checkpoint",
            "stage_number": stage,
            "stage_start_transitions": (stage - 1) * interval,
            "stage_transitions": interval,
            "num_timesteps": completed,
            "stage_seed": config["policy_seed"] + (stage - 1) * 10_007,
            "fresh_environment_reset": True,
            "campaign_provenance": provenance,
            "campaign_provenance_sha256": provenance_sha256,
            "checkpoint": relative.as_posix(),
            "sha256": _sha256(payload),
            "previous_record_sha256": previous_hash,
        }
        row["record_sha256"] = canonical_hash(row)
        ledger_rows.append(row)
        previous_hash = row["record_sha256"]
    training_ledger_payload = b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        for row in ledger_rows
    )
    training_ledger_path = root / TRAINING_LEDGER_RELATIVE
    training_ledger_path.parent.mkdir(parents=True, exist_ok=True)
    training_ledger_path.write_bytes(training_ledger_payload)

    release_payloads = {
        "artifacts/city_recovery_ppo.v3.zip": (
            root / ledger_rows[-1]["checkpoint"]
        ).read_bytes(),
        "artifacts/city_recovery_ppo.v3.onnx": b"fixture training onnx",
        "artifacts/city_recovery_ppo.v3.metadata.json": b"{}\n",
        "artifacts/model_manifest.v3.json": b"{}\n",
        "training/v3/onnx-parity.json": b"{}\n",
    }
    for relative, payload in release_payloads.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    curriculum = {
        "enabled": True,
        "teacher": config["behavior_cloning"]["teacher"],
        "teacher_version": config["behavior_cloning"]["teacher_version"],
        "training_split_only": True,
        "dagger_beta_schedule": config["behavior_cloning"]["dagger_beta_schedule"],
    }
    receipt_base = {
        "schema_version": 1,
        "status": "completed",
        "execution_class": "sealed_authorized_campaign",
        "governance": governance,
        "campaign_provenance": provenance,
        "campaign_provenance_sha256": provenance_sha256,
        "runtime": runtime,
        "production_stage_contract": config["production_stages"],
        "completed_stage_count": config["production_stages"]["count"],
        "invocation_resumed_from_stage": 0,
        "config": config,
        "source_identity": source,
        "engine_spec_sha256": ENGINE_V3_SPEC_SHA256,
        "solved_definition_sha256": SOLVED_DEFINITION_V3_SHA256,
        "completed_transitions": config["total_transitions"],
        "trainable_parameters": config["expected_trainable_parameters"],
        "curriculum": curriculum,
        "elapsed_seconds": 1.0,
        "checkpoint_ledger_sha256": _sha256(training_ledger_payload),
        "checkpoint_ledger_record_count": len(ledger_rows),
        "checkpoint_ledger_terminal_record_sha256": previous_hash,
        "terminal_checkpoint": {
            "path": "artifacts/city_recovery_ppo.v3.zip",
            "bytes": len(release_payloads["artifacts/city_recovery_ppo.v3.zip"]),
            "sha256": _sha256(release_payloads["artifacts/city_recovery_ppo.v3.zip"]),
        },
        "artifacts": {
            "training_onnx": {
                "path": "artifacts/city_recovery_ppo.v3.onnx",
                "sha256": _sha256(
                    release_payloads["artifacts/city_recovery_ppo.v3.onnx"]
                ),
            },
            "metadata": {
                "path": "artifacts/city_recovery_ppo.v3.metadata.json",
                "sha256": _sha256(
                    release_payloads["artifacts/city_recovery_ppo.v3.metadata.json"]
                ),
            },
            "manifest": {
                "path": "artifacts/model_manifest.v3.json",
                "sha256": _sha256(release_payloads["artifacts/model_manifest.v3.json"]),
            },
            "parity": {
                "path": "training/v3/onnx-parity.json",
                "sha256": _sha256(release_payloads["training/v3/onnx-parity.json"]),
            },
        },
        "manifest_sha256": _sha256(
            release_payloads["artifacts/model_manifest.v3.json"]
        ),
        "parity_sha256": _sha256(release_payloads["training/v3/onnx-parity.json"]),
    }
    training_receipt = {
        **receipt_base,
        "receipt_semantic_sha256": canonical_hash(receipt_base),
    }
    training_receipt_payload = _write_json(
        root, TRAINING_RECEIPT_RELATIVE, training_receipt
    )
    training_terminal_base = {
        "schema_version": 1,
        "status": "sealed_training_release_complete",
        "campaign_identity": provenance_sha256,
        "campaign_provenance_sha256": provenance_sha256,
        "configured_transitions": config["total_transitions"],
        "actual_completed_transitions": config["total_transitions"],
        "completed_stage_count": config["production_stages"]["count"],
        "training_use_receipt_sha256": _sha256(training_use_payload),
        "training_receipt_sha256": _sha256(training_receipt_payload),
        "checkpoint_ledger_sha256": _sha256(training_ledger_payload),
        "checkpoint_ledger_terminal_record_sha256": previous_hash,
        "release_files": {
            **{
                relative: _sha256(payload)
                for relative, payload in release_payloads.items()
            },
            TRAINING_RECEIPT_RELATIVE.as_posix(): _sha256(training_receipt_payload),
        },
    }
    training_terminal = {
        **training_terminal_base,
        "terminal_semantic_sha256": canonical_hash(training_terminal_base),
    }
    training_terminal_payload = _write_json(
        root, TRAINING_TERMINAL_RELATIVE, training_terminal
    )

    candidate_bindings = [
        {
            "path": row["checkpoint"],
            "sha256": row["sha256"],
            "completed_transitions": row["num_timesteps"],
            "source": "checkpoint_ledger",
        }
        for row in ledger_rows
    ]
    candidate_bindings.append(
        {
            "path": "artifacts/city_recovery_ppo.v3.zip",
            "sha256": _sha256(release_payloads["artifacts/city_recovery_ppo.v3.zip"]),
            "completed_transitions": config["total_transitions"],
            "source": "terminal_checkpoint",
        }
    )
    deduplicated = {
        (item["sha256"], item["completed_transitions"]): item
        for item in candidate_bindings
    }
    candidate_bindings = sorted(
        deduplicated.values(),
        key=lambda item: (item["completed_transitions"], item["sha256"]),
    )
    reports = []
    for index, binding in enumerate(candidate_bindings):
        reports.append(
            {
                **binding,
                "development_case_count": 40,
                "candidate_solved_count": 39 - index,
                "candidate_mean_resilience_auc": round(0.9 - index * 0.01, 10),
                "candidate_critical_service_days": index,
                "hard_violation_count": 0,
                "max_conservation_residual": 2e-8,
                "deterministic_replay_mismatches": 0,
                "rows_sha256": hashlib.sha256(
                    f"development rows {index}".encode()
                ).hexdigest(),
            }
        )
    selected = reports[0]
    checkpoint_sha256 = selected["sha256"]
    completed_transitions = selected["completed_transitions"]
    onnx_payload = b"fixture selected onnx"
    onnx_path = root / SELECTED_ONNX_RELATIVE
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_path.write_bytes(onnx_payload)
    onnx_sha256 = _sha256(onnx_payload)
    parity = {
        "schema_version": 1,
        "status": "passed",
        "passed": True,
        "checkpoint_sha256": checkpoint_sha256,
        "onnx_sha256": onnx_sha256,
        "interface": {
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
        },
        "development_case_count": 40,
        "action_sample_count": 1200,
        "maximum_action_absolute_error": 1e-7,
        "maximum_rauc_absolute_error": 1e-8,
        "outcome_mismatches": 0,
        "deterministic_replay_mismatches": 0,
        "hard_violation_count": 0,
        "maximum_conservation_residual": 2e-8,
        "rows_sha256": hashlib.sha256(b"parity rows").hexdigest(),
    }
    parity_payload = _write_json(root, SELECTED_PARITY_RELATIVE, parity)
    parity_sha256 = _sha256(parity_payload)
    manifest = {
        "schema_version": 1,
        "status": "selected_deployable",
        "policy": {
            "id": POLICY_ID,
            "algorithm": "PPO",
            "observation_count": len(OBSERVATION_ORDER_V3),
            "action_count": len(ACTION_ORDER_V3),
            "completed_transitions": completed_transitions,
        },
        "bindings": {
            "protocol_sha256": protocol_sha256,
            "source_seal_sha256": seal_sha256,
            "scientific_source_sha256": source["semantic_sha256"],
            "engine_spec_sha256": ENGINE_V3_SPEC_SHA256,
            "outcome_definition_sha256": SOLVED_DEFINITION_V3_SHA256,
            "observation_order_sha256": canonical_hash(list(OBSERVATION_ORDER_V3)),
            "action_order_sha256": canonical_hash(list(ACTION_ORDER_V3)),
            "selected_checkpoint_sha256": checkpoint_sha256,
        },
        "files": {
            "onnx": {
                "path": SELECTED_ONNX_RELATIVE.as_posix(),
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
            },
            "parity": {
                "path": SELECTED_PARITY_RELATIVE.as_posix(),
                "sha256": parity_sha256,
            },
        },
    }
    manifest_payload = _write_json(root, SELECTED_MANIFEST_RELATIVE, manifest)
    manifest_sha256 = _sha256(manifest_payload)

    stage_root = (
        root
        / "internal"
        / "training_runs"
        / "v3"
        / "selection-staging"
        / checkpoint_sha256
    )
    claim_base = {
        "schema_version": 1,
        "status": "selected_deployment_staging_claimed",
        "selected_checkpoint": selected,
        "protocol_sha256": protocol_sha256,
        "source_seal_sha256": seal_sha256,
        "scientific_source_sha256": source["semantic_sha256"],
        "engine_spec_sha256": ENGINE_V3_SPEC_SHA256,
        "outcome_definition_sha256": SOLVED_DEFINITION_V3_SHA256,
        "observation_order_sha256": canonical_hash(list(OBSERVATION_ORDER_V3)),
        "action_order_sha256": canonical_hash(list(ACTION_ORDER_V3)),
        "destinations": {
            "onnx": SELECTED_ONNX_RELATIVE.as_posix(),
            "parity": SELECTED_PARITY_RELATIVE.as_posix(),
            "manifest": SELECTED_MANIFEST_RELATIVE.as_posix(),
        },
    }
    claim = {**claim_base, "claim_semantic_sha256": canonical_hash(claim_base)}
    claim_payload = _write_json(stage_root, Path("selection-claim.json"), claim)
    staged_payloads = {
        "onnx": (SELECTED_ONNX_RELATIVE, onnx_payload),
        "parity": (SELECTED_PARITY_RELATIVE, parity_payload),
        "manifest": (SELECTED_MANIFEST_RELATIVE, manifest_payload),
    }
    for _, (relative, payload) in staged_payloads.items():
        (stage_root / relative.name).write_bytes(payload)
    stage_receipt_base = {
        "schema_version": 1,
        "status": "selected_deployment_staging_complete",
        "claim_sha256": _sha256(claim_payload),
        "selected_checkpoint_sha256": checkpoint_sha256,
        "artifacts": {
            artifact_id: {
                "path": relative.as_posix(),
                "bytes": len(payload),
                "sha256": _sha256(payload),
            }
            for artifact_id, (relative, payload) in staged_payloads.items()
        },
    }
    stage_receipt = {
        **stage_receipt_base,
        "receipt_semantic_sha256": canonical_hash(stage_receipt_base),
    }
    _write_json(
        stage_root,
        Path("selected-deployment-stage-receipt.json"),
        stage_receipt,
    )
    selected_deployment = {
        "checkpoint": {
            "path": selected["path"],
            "sha256": checkpoint_sha256,
            "completed_transitions": completed_transitions,
        },
        "onnx": {
            "path": SELECTED_ONNX_RELATIVE.as_posix(),
            "bytes": len(onnx_payload),
            "sha256": onnx_sha256,
        },
        "parity": {
            "path": SELECTED_PARITY_RELATIVE.as_posix(),
            "sha256": parity_sha256,
        },
        "manifest": {
            "path": SELECTED_MANIFEST_RELATIVE.as_posix(),
            "sha256": manifest_sha256,
        },
    }
    selection_base = {
        "schema_version": 1,
        "status": "checkpoint_selected_on_development_only",
        "protocol_sha256": protocol_sha256,
        "source_seal_sha256": seal_sha256,
        "scientific_source_sha256": source["semantic_sha256"],
        "engine_spec_sha256": ENGINE_V3_SPEC_SHA256,
        "outcome_definition_sha256": SOLVED_DEFINITION_V3_SHA256,
        "training_config_sha256": config_sha256,
        "training_receipt_sha256": _sha256(training_receipt_payload),
        "training_use_receipt_sha256": _sha256(training_use_payload),
        "training_terminal_sha256": _sha256(training_terminal_payload),
        "campaign_provenance_sha256": provenance_sha256,
        "checkpoint_ledger_sha256": _sha256(training_ledger_payload),
        "selection_dataset": "development",
        "development_case_count": 40,
        "ranking": [
            "candidate_solved_count:descending",
            "candidate_mean_resilience_auc:descending",
            "completed_transitions:ascending",
            "checkpoint_sha256:ascending",
        ],
        "candidate_reports": reports,
        "selected_checkpoint": selected,
        "selected_deployment": selected_deployment,
        "final_split_accessed": False,
    }
    selection = {
        **selection_base,
        "receipt_semantic_sha256": canonical_hash(selection_base),
    }
    selection_payload = _write_json(root, SELECTION_RECEIPT_RELATIVE, selection)
    selection_sha256 = _sha256(selection_payload)
    authorization_base = {
        "schema_version": 1,
        "status": "final_evaluation_authorized",
        "protocol_sha256": protocol_sha256,
        "source_seal_sha256": seal_sha256,
        "scientific_source_sha256": source["semantic_sha256"],
        "engine_spec_sha256": ENGINE_V3_SPEC_SHA256,
        "outcome_definition_sha256": SOLVED_DEFINITION_V3_SHA256,
        "selection_receipt_sha256": selection_sha256,
        "selected_checkpoint_sha256": checkpoint_sha256,
        "selected_onnx_sha256": onnx_sha256,
        "selected_parity_sha256": parity_sha256,
        "selected_manifest_sha256": manifest_sha256,
        "selected_policy_id": POLICY_ID,
        "candidate_runtime": "onnxruntime_cpu_selected_deployment_only",
        "final_case_count": 40,
        "invocation_limit": 1,
        "durable_identity_matching_resume_only": True,
        "failed_or_unfavorable_result_may_be_replaced": False,
    }
    authorization = {
        **authorization_base,
        "authorization_sha256": canonical_hash(authorization_base),
    }
    authorization_payload = _write_json(
        root, FINAL_AUTHORIZATION_RELATIVE, authorization
    )
    identity = {
        "selected_policy_id": POLICY_ID,
        "protocol_sha256": protocol_sha256,
        "source_seal_sha256": seal_sha256,
        "scientific_source_sha256": source["semantic_sha256"],
        "selection_receipt_sha256": selection_sha256,
        "authorization_sha256": _sha256(authorization_payload),
        "checkpoint_sha256": checkpoint_sha256,
        "completed_transitions": completed_transitions,
        "onnx_path": SELECTED_ONNX_RELATIVE.as_posix(),
        "onnx_sha256": onnx_sha256,
        "parity_sha256": parity_sha256,
        "manifest_sha256": manifest_sha256,
        "candidate_runtime": "onnxruntime_cpu_selected_deployment_only",
        "engine_spec_sha256": ENGINE_V3_SPEC_SHA256,
        "outcome_definition_sha256": SOLVED_DEFINITION_V3_SHA256,
    }
    final_rows: list[dict[str, Any]] = []
    previous_hash = "0" * 64
    for index, expected in enumerate(_expected_final_order()):
        candidate_solved = index < 30
        baseline_solved = index < 8 or 30 <= index < 32
        candidate_rauc, baseline_rauc = (0.6, 0.4) if index < 29 else (0.4, 0.6)

        def summary(label: str, solved: bool, rauc: float) -> dict[str, Any]:
            return {
                "solved": solved,
                "status": "solved" if solved else "failed",
                "reason_codes": [] if solved else ["fixture_failure"],
                "resilience_auc": rauc,
                "critical_service_days": index,
                "hard_violation_count": 0,
                "max_conservation_residual": 2e-8,
                "trajectory_sha256": hashlib.sha256(
                    f"{label}:{expected['row_id']}".encode()
                ).hexdigest(),
            }

        candidate = summary("candidate", candidate_solved, candidate_rauc)
        baseline = summary("baseline", baseline_solved, baseline_rauc)
        pair = (
            "both_solved"
            if candidate_solved and baseline_solved
            else "ppo_only"
            if candidate_solved
            else "heuristic_only"
            if baseline_solved
            else "neither"
        )
        row = {
            **expected,
            "identity_sha256": canonical_hash(identity),
            "scenario_sha256": hashlib.sha256(
                f"scenario:{expected['row_id']}".encode()
            ).hexdigest(),
            "shock_schedule_sha256": hashlib.sha256(
                f"tape:{expected['row_id']}".encode()
            ).hexdigest(),
            "candidate": candidate,
            "baseline": baseline,
            "absolute_outcome_pair": pair,
            "secondary_rauc_candidate_minus_baseline": round(
                candidate_rauc - baseline_rauc, 8
            ),
            "deterministic_replay_mismatches": 0,
            "previous_record_sha256": previous_hash,
        }
        row["record_sha256"] = canonical_hash(row)
        final_rows.append(row)
        previous_hash = row["record_sha256"]
    final_ledger_payload = b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        for row in final_rows
    )
    final_ledger_path = root / FINAL_LEDGER_RELATIVE
    final_ledger_path.parent.mkdir(parents=True, exist_ok=True)
    final_ledger_path.write_bytes(final_ledger_payload)
    benchmark = {
        "schema_version": 1,
        "artifact_type": "city_recovery_v3_final_benchmark",
        "status": "complete",
        "primary_metric": "independent_absolute_disasters_solved",
        "synthetic_case_count": 40,
        **_recomputed_final_sections(final_rows),
        "identity": identity,
        "ledger_sha256": _sha256(final_ledger_payload),
    }
    result_payload = _write_json(root, FINAL_REPORT_RELATIVE, benchmark)
    use_receipt = {
        "schema_version": 1,
        "status": "single_final_invocation_started",
        "identity": identity,
        "identity_sha256": canonical_hash(identity),
        "case_count": 40,
        "iteration_order": "family_order_then_ascending_seed",
        "durable_resume_only": True,
        "exclusive_lock_path": "internal/training_runs/v3/final.lock",
    }
    use_payload = _write_json(root, FINAL_USE_RECEIPT_RELATIVE, use_receipt)
    final_terminal_base = {
        "schema_version": 1,
        "status": "final_evaluation_complete",
        "selected_policy_id": POLICY_ID,
        "identity": identity,
        "identity_sha256": canonical_hash(identity),
        "protocol_sha256": protocol_sha256,
        "source_seal_sha256": seal_sha256,
        "scientific_source_sha256": source["semantic_sha256"],
        "selection_receipt_sha256": selection_sha256,
        "final_authorization_sha256": _sha256(authorization_payload),
        "selected_checkpoint_sha256": checkpoint_sha256,
        "selected_onnx_sha256": onnx_sha256,
        "selected_parity_sha256": parity_sha256,
        "selected_manifest_sha256": manifest_sha256,
        "case_count": 40,
        "iteration_order": "family_order_then_ascending_seed",
        "result": {
            "path": FINAL_REPORT_RELATIVE.as_posix(),
            "sha256": _sha256(result_payload),
        },
        "final_use_receipt": {
            "path": FINAL_USE_RECEIPT_RELATIVE.as_posix(),
            "sha256": _sha256(use_payload),
        },
        "final_ledger": {
            "path": FINAL_LEDGER_RELATIVE.as_posix(),
            "sha256": _sha256(final_ledger_payload),
            "record_count": 40,
            "terminal_record_sha256": previous_hash,
        },
    }
    final_terminal = {
        **final_terminal_base,
        "terminal_semantic_sha256": canonical_hash(final_terminal_base),
    }
    _write_json(root, FINAL_TERMINAL_RELATIVE, final_terminal)
    return benchmark


_release_fixture = _sealed_release_fixture


def test_release_is_absent_until_every_sealed_artifact_exists(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match="V3 protocol"):
        _load_release(tmp_path, lambda _: _FakeSession())  # type: ignore[arg-type]


def test_release_loads_only_cross_bound_73_by_22_deployment(tmp_path: Path) -> None:
    expected = _release_fixture(tmp_path)
    bundle = _load_release(tmp_path, lambda _: _FakeSession())  # type: ignore[arg-type]

    assert len(bundle.metadata["observation_order"]) == 73
    assert len(bundle.metadata["action_order"]) == 22
    assert bundle.metadata["id"] == POLICY_ID
    assert bundle.metadata["deployment"]["selected_checkpoint_sha256"] == (
        bundle.sb3_sha256
    )
    assert bundle.metadata["deployment"]["manifest_sha256"] == (bundle.manifest_sha256)
    assert bundle.metadata["deployment"]["parity_sha256"] == bundle.parity_sha256
    assert bundle.metadata["deployment"]["benchmark_sha256"] == (
        bundle.benchmark_sha256
    )
    assert bundle.benchmark["status"] == "complete"
    assert bundle.benchmark["candidate"]["id"] == POLICY_ID
    assert (
        bundle.benchmark["candidate"]["solved_count"]
        == expected["candidate"]["solved_count"]
    )
    assert (
        bundle.benchmark["baseline"]["solved_count"]
        == expected["baseline"]["solved_count"]
    )


def test_release_rejects_stale_action_contract(tmp_path: Path) -> None:
    _release_fixture(tmp_path)
    path = tmp_path / SELECTED_MANIFEST_RELATIVE
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["policy"]["action_count"] = 17
    path.write_bytes(_formatted(manifest))

    with pytest.raises(ArtifactError, match="policy contract"):
        _load_release(tmp_path, lambda _: _FakeSession())  # type: ignore[arg-type]


def test_release_rejects_selected_manifest_interface_drift(tmp_path: Path) -> None:
    _release_fixture(tmp_path)
    path = tmp_path / SELECTED_MANIFEST_RELATIVE
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["files"]["onnx"]["input_shape"] = ["batch", 72]
    path.write_bytes(_formatted(manifest))

    with pytest.raises(ArtifactError, match="ONNX manifest binding"):
        _load_release(tmp_path, lambda _: _FakeSession())  # type: ignore[arg-type]


def test_release_rejects_selected_parity_provider_drift(tmp_path: Path) -> None:
    _release_fixture(tmp_path)
    path = tmp_path / SELECTED_PARITY_RELATIVE
    parity = json.loads(path.read_text(encoding="utf-8"))
    parity["interface"]["execution_providers"] = ["CUDAExecutionProvider"]
    parity_payload = _formatted(parity)
    path.write_bytes(parity_payload)
    manifest_path = tmp_path / SELECTED_MANIFEST_RELATIVE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["parity"]["sha256"] = _sha256(parity_payload)
    manifest_path.write_bytes(_formatted(manifest))

    with pytest.raises(ArtifactError, match="parity evidence"):
        _load_release(tmp_path, lambda _: _FakeSession())  # type: ignore[arg-type]


def test_release_rejects_benchmark_count_not_supported_by_pairs(tmp_path: Path) -> None:
    _release_fixture(tmp_path)
    path = tmp_path / FINAL_REPORT_RELATIVE
    report = json.loads(path.read_text(encoding="utf-8"))
    report["candidate"]["solved_count"] = 31
    report["candidate"]["failed_count"] = 9
    report["candidate"]["solve_rate"] = 31 / 40
    path.write_bytes(_formatted(report))

    with pytest.raises(ArtifactError, match="paired outcomes"):
        _load_release(tmp_path, lambda _: _FakeSession())  # type: ignore[arg-type]


def test_release_rejects_missing_final_terminal(tmp_path: Path) -> None:
    _release_fixture(tmp_path)
    (tmp_path / FINAL_TERMINAL_RELATIVE).unlink()

    with pytest.raises(ArtifactError, match="final terminal"):
        _load_release(tmp_path, lambda _: _FakeSession())  # type: ignore[arg-type]


def test_release_rejects_final_ledger_hash_chain_tamper(tmp_path: Path) -> None:
    _release_fixture(tmp_path)
    path = tmp_path / FINAL_LEDGER_RELATIVE
    lines = path.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[4])
    row["candidate"]["resilience_auc"] = 0.123
    lines[4] = json.dumps(row, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ArtifactError, match="record digest"):
        _load_release(tmp_path, lambda _: _FakeSession())  # type: ignore[arg-type]


def test_release_rejects_selected_checkpoint_byte_tamper(tmp_path: Path) -> None:
    _release_fixture(tmp_path)
    selection = json.loads(
        (tmp_path / SELECTION_RECEIPT_RELATIVE).read_text(encoding="utf-8")
    )
    checkpoint = tmp_path / selection["selected_checkpoint"]["path"]
    checkpoint.write_bytes(checkpoint.read_bytes() + b"tamper")

    with pytest.raises(ArtifactError, match="training ledger provenance"):
        _load_release(tmp_path, lambda _: _FakeSession())  # type: ignore[arg-type]


def test_release_rejects_aggregate_not_recomputed_from_ledger(tmp_path: Path) -> None:
    _release_fixture(tmp_path)
    path = tmp_path / FINAL_REPORT_RELATIVE
    report = json.loads(path.read_text(encoding="utf-8"))
    report["candidate"]["mean_resilience_auc"] = 0.999
    path.write_bytes(_formatted(report))

    with pytest.raises(ArtifactError, match="candidate does not match its ledger"):
        _load_release(tmp_path, lambda _: _FakeSession())  # type: ignore[arg-type]


def test_release_rejects_selection_receipt_truncation(tmp_path: Path) -> None:
    _release_fixture(tmp_path)
    path = tmp_path / SELECTION_RECEIPT_RELATIVE
    selection = json.loads(path.read_text(encoding="utf-8"))
    selection.pop("candidate_reports")
    unsigned = {
        key: value
        for key, value in selection.items()
        if key != "receipt_semantic_sha256"
    }
    selection["receipt_semantic_sha256"] = canonical_hash(unsigned)
    path.write_bytes(_formatted(selection))

    with pytest.raises(ArtifactError, match="selection receipt deployment binding"):
        _load_release(tmp_path, lambda _: _FakeSession())  # type: ignore[arg-type]


def test_release_rejects_protocol_truncation(tmp_path: Path) -> None:
    _release_fixture(tmp_path)
    path = tmp_path / PROTOCOL_RELATIVE
    protocol = json.loads(path.read_text(encoding="utf-8"))
    protocol["final_evaluation"].pop("terminal_marker")
    path.write_bytes(_formatted(protocol))

    with pytest.raises(ArtifactError, match="final-evaluation protocol"):
        _load_release(tmp_path, lambda _: _FakeSession())  # type: ignore[arg-type]
