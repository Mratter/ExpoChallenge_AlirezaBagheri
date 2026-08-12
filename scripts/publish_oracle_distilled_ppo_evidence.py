#!/usr/bin/env python3
"""Publish portable, receipt-bound evidence for the DEV-only oracle-BC PPO study."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from statistics import fmean, pstdev, stdev
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.shared_evidence import (  # noqa: E402
    canonical_hash,
    file_sha256,
    load_json_object,
)

TOOL_ID = "publish_oracle_distilled_ppo_evidence.py"
SCHEMA_VERSION = "city-recovery-oracle-distilled-ppo-dev-evidence-v1"
REGISTERED_SEEDS = (37_017, 47_017, 57_017)
DIAGNOSTIC_MILESTONES = (0, 200_000, 500_000, 1_000_000, 2_000_000)
SELECTABLE_MILESTONES = (500_000, 1_000_000, 2_000_000)
FAMILY_IDS = (
    "v3_dev_river_flood",
    "v3_dev_industrial_outage",
    "v3_dev_logistics_strike",
    "v3_dev_seismic_cluster",
    "v3_dev_health_compound",
)
FAMILY_LABELS = {
    "v3_dev_river_flood": "River flood",
    "v3_dev_industrial_outage": "Industrial outage",
    "v3_dev_logistics_strike": "Logistics strike",
    "v3_dev_seismic_cluster": "Seismic cluster",
    "v3_dev_health_compound": "Health compound",
}
DEVELOPMENT_CASE_COUNT = 200
INCUMBENT_BEST_SOLVED = 178
INCUMBENT_ENDPOINT_MEAN = 171.4
PROMOTION_BEST_SOLVED = 183
PROMOTION_ENDPOINT_SOLVED = 172
PROMOTION_ENDPOINT_SEED_COUNT = 2
EXPECTED_ORACLE_RECEIPT_SHA256 = (
    "e7777e53f20b886bbb82b167e0303b20ee0de32dcf9b87f50d175a0b71c5dc89"
)
EXPECTED_STUDENT_RECEIPT_SHA256 = (
    "76025a6376db6905b1d96d08122a14bccc7639040921768a79e4c83debabec84"
)
EXPECTED_BC_ACTOR_SHA256 = (
    "73b7a9097386f0ae772981056aa331216ac97c5877c1a57e322f06cc95e43601"
)
EXPECTED_OBSERVATION_RMS_SHA256 = (
    "cb7b9a46369a0c225c3a6254433f6ef37e52b822ef44598fa4311b64e63a4ba4"
)
DEFAULT_OUTPUT = (
    ROOT
    / "internal"
    / "developmental_runs"
    / "v4"
    / "oracle-distilled-ppo-study-200.json"
)


class PublicationError(RuntimeError):
    """Raised when external study evidence is incomplete or inconsistent."""


def _load(path: Path, label: str) -> dict[str, Any]:
    return load_json_object(path, label, error_type=PublicationError)


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise PublicationError(f"{label} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PublicationError(f"{label} must be finite") from exc
    if not math.isfinite(result):
        raise PublicationError(f"{label} must be finite")
    return result


def _write_canonical_json(path: Path, value: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != encoded:
        raise PublicationError(f"refusing to overwrite different evidence: {path}")
    path.write_bytes(encoded)


def _relative_external_path(path: Path, study_root: Path) -> str:
    try:
        return path.resolve().relative_to(study_root.resolve()).as_posix()
    except ValueError as exc:
        raise PublicationError("study file escaped the external root") from exc


def _family_id(row: Mapping[str, Any]) -> str:
    row_id = row.get("row_id")
    if not isinstance(row_id, str) or ":" not in row_id:
        raise PublicationError("development row identity is malformed")
    family_id = row_id.split(":", 1)[0]
    if family_id not in FAMILY_IDS:
        raise PublicationError(f"unexpected development family: {family_id}")
    return family_id


def _validate_development(
    value: Any,
    *,
    expected_milestone: int,
    expected_total_transitions: int,
    expected_identity: list[tuple[str, int, int, str]] | None,
) -> tuple[dict[str, Any], list[tuple[str, int, int, str]]]:
    if not isinstance(value, dict):
        raise PublicationError("development result is missing")
    rows = value.get("rows")
    if not isinstance(rows, list) or len(rows) != DEVELOPMENT_CASE_COUNT:
        raise PublicationError("development result must contain 200 rows")
    identity: list[tuple[str, int, int, str]] = []
    solved = 0
    hard_violations = 0
    max_residual = 0.0
    resilience_values: list[float] = []
    tail_values: list[float] = []
    failure_reasons: Counter[str] = Counter()
    family_counts = {family_id: 0 for family_id in FAMILY_IDS}
    portable_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise PublicationError("development row is not an object")
        family_id = _family_id(row)
        case_identity = (
            str(row.get("row_id")),
            int(row.get("case_seed", -1)),
            int(row.get("tape_seed", -1)),
            str(row.get("tape_sha256", "")),
        )
        identity.append(case_identity)
        row_solved = row.get("solved")
        row_hard = row.get("hard_violation_count")
        row_residual = _finite(
            row.get("max_conservation_residual"), "row conservation residual"
        )
        row_resilience = _finite(row.get("resilience_auc"), "row resilience AUC")
        row_tail = _finite(row.get("minimum_tail_margin"), "row tail margin")
        reasons = row.get("reason_codes")
        if (
            not isinstance(row_solved, bool)
            or not isinstance(row_hard, int)
            or isinstance(row_hard, bool)
            or row_hard < 0
            or row_residual < 0.0
            or not isinstance(reasons, list)
            or any(not isinstance(reason, str) or not reason for reason in reasons)
        ):
            raise PublicationError("development row fields are malformed")
        solved += int(row_solved)
        family_counts[family_id] += int(row_solved)
        hard_violations += row_hard
        max_residual = max(max_residual, row_residual)
        resilience_values.append(row_resilience)
        tail_values.append(row_tail)
        if not row_solved:
            failure_reasons.update(reasons)
        portable_rows.append(
            {
                "case_seed": case_identity[1],
                "family_id": family_id,
                "hard_violation_count": row_hard,
                "maximum_conservation_residual": row_residual,
                "minimum_tail_margin": row_tail,
                "reason_codes": list(reasons),
                "resilience_auc": row_resilience,
                "row_id": case_identity[0],
                "solved": row_solved,
                "tape_seed": case_identity[2],
                "tape_sha256": case_identity[3],
            }
        )
    if expected_identity is not None and identity != expected_identity:
        raise PublicationError("development roster or tape identity drifted")
    if len(set(identity)) != DEVELOPMENT_CASE_COUNT:
        raise PublicationError("development row identities are not unique")
    if (
        value.get("active_actor_critic_transitions") != expected_milestone
        or value.get("total_environment_transitions") != expected_total_transitions
        or value.get("case_count") != DEVELOPMENT_CASE_COUNT
        or value.get("solved_count") != solved
        or not math.isclose(
            _finite(value.get("solve_rate"), "solve rate"),
            solved / DEVELOPMENT_CASE_COUNT,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or value.get("hard_violation_count") != hard_violations
        or _finite(value.get("maximum_conservation_residual"), "maximum residual")
        != max_residual
        or round(fmean(resilience_values), 10)
        != _finite(value.get("mean_resilience_auc"), "mean resilience AUC")
        or round(fmean(tail_values), 10)
        != _finite(value.get("mean_minimum_tail_margin"), "mean tail margin")
        or dict(sorted(failure_reasons.items()))
        != value.get("failure_reason_code_histogram")
        or hard_violations != 0
        or max_residual != 0.0
    ):
        raise PublicationError("development aggregate disagrees with its rows")
    return (
        {
            "active_actor_critic_transitions": expected_milestone,
            "case_count": DEVELOPMENT_CASE_COUNT,
            "failure_reason_code_histogram": dict(sorted(failure_reasons.items())),
            "hard_violation_count": 0,
            "maximum_conservation_residual": 0.0,
            "mean_minimum_tail_margin": round(fmean(tail_values), 10),
            "mean_resilience_auc": round(fmean(resilience_values), 10),
            "per_family_solved_count": family_counts,
            "portable_rows_sha256": canonical_hash(portable_rows),
            "rows": portable_rows,
            "solve_rate": solved / DEVELOPMENT_CASE_COUNT,
            "solved_count": solved,
            "source_rows_sha256": canonical_hash(rows),
            "total_environment_transitions": expected_total_transitions,
        },
        identity,
    )


def _validate_bundle(
    study_root: Path,
    reference: Mapping[str, Any],
    *,
    seed: int,
    milestone: int,
) -> dict[str, Any]:
    bundle = study_root / f"seed-{seed}" / "checkpoints" / f"ppo-{milestone}"
    manifest_path = bundle / "manifest.json"
    manifest = _load(manifest_path, "checkpoint manifest")
    checkpoint = manifest.get("checkpoint")
    normalization = manifest.get("normalization")
    training = manifest.get("training")
    if not isinstance(checkpoint, dict) or not isinstance(normalization, dict) or not isinstance(training, dict):
        raise PublicationError("checkpoint manifest is incomplete")
    model_path = bundle / "model.zip"
    normalization_path = bundle / "normalization.npz"
    model_file = checkpoint.get("file")
    normalization_file = normalization.get("file")
    if not isinstance(model_file, dict) or not isinstance(normalization_file, dict):
        raise PublicationError("checkpoint manifest file records are missing")
    expected_id = f"seed-{seed}-ppo-{milestone}"
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "city-recovery-ppo-checkpoint"
        or manifest.get("publication", {}).get("complete") is not True
        or checkpoint.get("id") != expected_id
        or checkpoint.get("active_actor_critic_transitions") != milestone
        or training.get("seed") != seed
        or training.get("milestone") != milestone
        or training.get("config_sha256") != reference.get("training_config_sha256")
        or normalization.get("observation_rms_sha256")
        != EXPECTED_OBSERVATION_RMS_SHA256
        or checkpoint.get("policy_state_sha256")
        != reference.get("policy_state_sha256")
        or checkpoint.get("actor_state_sha256")
        != reference.get("actor_state_sha256")
        or checkpoint.get("optimizer_state_sha256")
        != reference.get("optimizer_state_sha256")
        or normalization.get("return_rms_sha256")
        != reference.get("return_rms_sha256")
        or file_sha256(manifest_path) != reference.get("bundle_manifest_sha256")
        or file_sha256(model_path) != model_file.get("sha256")
        or model_file.get("sha256") != reference.get("checkpoint_sha256")
        or file_sha256(normalization_path) != normalization_file.get("sha256")
        or normalization_file.get("sha256")
        != reference.get("normalization_file_sha256")
    ):
        raise PublicationError(f"checkpoint bundle drifted: {expected_id}")
    return {
        "checkpoint_id": expected_id,
        "manifest_sha256": reference["bundle_manifest_sha256"],
        "model_sha256": reference["checkpoint_sha256"],
        "normalization_sha256": reference["normalization_file_sha256"],
        "observation_rms_sha256": EXPECTED_OBSERVATION_RMS_SHA256,
        "return_rms_sha256": reference["return_rms_sha256"],
        "selection_evaluation_export_supported": manifest.get("resume", {}).get(
            "selection_evaluation_export_supported"
        ),
    }


def _upstream_evidence(summary: Mapping[str, Any]) -> dict[str, Any]:
    student = summary.get("approved_student_reference")
    if not isinstance(student, dict):
        raise PublicationError("approved student reference is missing")
    student_receipt_ref = student.get("student_receipt")
    checkpoint = student.get("checkpoint")
    if not isinstance(student_receipt_ref, dict) or not isinstance(checkpoint, dict):
        raise PublicationError("approved student identity is incomplete")
    oracle_path = Path(r"E:\city-recovery-training-oracle-v4-attempt-01\training\receipt.json")
    student_path = Path(str(student_receipt_ref.get("path", "")))
    oracle = _load(oracle_path, "training oracle receipt")
    bc = _load(student_path, "oracle BC student receipt")
    if (
        file_sha256(oracle_path) != EXPECTED_ORACLE_RECEIPT_SHA256
        or student.get("dataset_receipt_sha256") != EXPECTED_ORACLE_RECEIPT_SHA256
        or oracle.get("status") != "complete_training_oracle_demonstrations"
        or oracle.get("case_count") != 192
        or oracle.get("demonstration_row_count") != 5_760
        or oracle.get("observation_count") != 73
        or oracle.get("action_count") != 22
        or oracle.get("oracle", {}).get("aggregate", {}).get("solved_count") != 187
        or any(
            oracle.get("invariants", {}).get(name) is not True
            for name in (
                "action_dimension_exactly_22",
                "all_conservation_residuals_exactly_zero",
                "all_hard_violation_counts_zero",
                "case_count_exactly_192",
                "demonstration_rows_exactly_5760",
                "observation_dimension_exactly_73",
                "row_ids_unique",
            )
        )
        or any(
            oracle.get("invariants", {}).get(name) is not False
            for name in (
                "development_split_used",
                "final_split_used",
                "learned_policy_loaded_or_run",
            )
        )
        or file_sha256(student_path) != EXPECTED_STUDENT_RECEIPT_SHA256
        or student_receipt_ref.get("sha256") != EXPECTED_STUDENT_RECEIPT_SHA256
        or bc.get("status")
        != "complete_eligible_for_separately_authorized_3_seed_ppo"
        or bc.get("development", {}).get("solved_count") != 157
        or bc.get("development_evaluation_count") != 1
        or bc.get("final_split_imported_or_used") is not False
        or bc.get("ppo_started") is not False
        or bc.get("catastrophic_gate", {}).get("passed") is not True
        or checkpoint.get("actor_state_sha256") != EXPECTED_BC_ACTOR_SHA256
        or checkpoint.get("observation_rms_sha256")
        != EXPECTED_OBSERVATION_RMS_SHA256
    ):
        raise PublicationError("upstream oracle or BC evidence drifted")
    return {
        "oracle_training_dataset": {
            "action_count": 22,
            "case_count": 192,
            "dataset_index_sha256": oracle["dataset_index_sha256"],
            "demonstration_row_count": 5_760,
            "hard_violation_count": 0,
            "maximum_conservation_residual": 0.0,
            "observation_count": 73,
            "receipt_sha256": EXPECTED_ORACLE_RECEIPT_SHA256,
            "rows_sha256": oracle["rows_sha256"],
            "solved_count": 187,
            "split": "train",
        },
        "oracle_bc_student": {
            "actor_state_sha256": EXPECTED_BC_ACTOR_SHA256,
            "checkpoint_manifest_sha256": checkpoint["manifest_sha256"],
            "checkpoint_model_sha256": checkpoint["model_sha256"],
            "checkpoint_normalization_sha256": checkpoint["normalization_sha256"],
            "development_rows_sha256": bc["development_rows_sha256"],
            "development_solved_count": 157,
            "heldout_hand_rule_mae": student["heldout_fit"][
                "hand_rule_action_mean_absolute_error"
            ],
            "heldout_hand_rule_mse": student["heldout_fit"][
                "hand_rule_action_mse"
            ],
            "heldout_oracle_mae": student["heldout_fit"][
                "oracle_action_mean_absolute_error"
            ],
            "heldout_oracle_mse": student["heldout_fit"]["oracle_action_mse"],
            "method": student["method"],
            "observation_rms_sha256": EXPECTED_OBSERVATION_RMS_SHA256,
            "receipt_sha256": EXPECTED_STUDENT_RECEIPT_SHA256,
            "student_contract_sha256": student["student_contract_sha256"],
        },
    }


def build_portable_receipt(study_root: Path) -> dict[str, Any]:
    """Validate the complete external study and return portable evidence."""

    study_root = study_root.resolve()
    protocol_path = study_root / "protocol.json"
    summary_path = study_root / "distilled-ppo-study-summary.json"
    protocol = _load(protocol_path, "distilled PPO protocol")
    summary = _load(summary_path, "distilled PPO summary")
    contract = protocol.get("contract")
    if not isinstance(contract, dict):
        raise PublicationError("study contract is missing")
    if (
        protocol.get("contract_sha256") != canonical_hash(contract)
        or summary.get("contract_sha256") != protocol.get("contract_sha256")
        or summary.get("status") != "complete_not_promoted"
        or summary.get("tool") != "run_distilled_ppo_study.py"
        or summary.get("split") != "dev"
        or summary.get("development_case_count") != DEVELOPMENT_CASE_COUNT
        or summary.get("final_split_imported_or_used") is not False
        or contract.get("registered_policy_seeds") != list(REGISTERED_SEEDS)
        or contract.get("scope", {}).get("final_split_imported_or_used") is not False
        or contract.get("selection", {}).get("resilience_auc_used_for_selection")
        is not False
        or contract.get("promotion_rule", {}).get("final_evaluation_authorized")
        is not False
        or contract.get("git_commit") != "b3bc503922c70d7dea010ff5b9f7bf276f457e3d"
        or contract.get("source_identity_sha256")
        != canonical_hash(contract.get("source_identity"))
        or summary.get("source_identity_sha256")
        != contract.get("source_identity_sha256")
    ):
        raise PublicationError("study protocol or summary contract drifted")

    upstream = _upstream_evidence(summary)
    summary_candidates = summary.get("ranking", {}).get("candidates")
    if not isinstance(summary_candidates, list) or len(summary_candidates) != 9:
        raise PublicationError("study summary must rank nine candidates")
    summary_by_key = {
        (int(row["policy_seed"]), int(row["active_actor_critic_transitions"])): row
        for row in summary_candidates
    }
    expected_identity: list[tuple[str, int, int, str]] | None = None
    seed_rows: list[dict[str, Any]] = []
    selectable: list[dict[str, Any]] = []
    receipt_hashes: dict[int, str] = {}
    initial_actors: set[str] = set()
    initial_rms: set[str] = set()
    fresh_critics: set[str] = set()

    for seed in REGISTERED_SEEDS:
        receipt_path = study_root / f"seed-{seed}" / "training-receipt.json"
        receipt = _load(receipt_path, f"seed {seed} training receipt")
        receipt_hash = file_sha256(receipt_path)
        receipt_hashes[seed] = receipt_hash
        summary_receipt = next(
            (row for row in summary.get("receipts", []) if row.get("seed") == seed),
            None,
        )
        initialization = receipt.get("initialization")
        behavior = receipt.get("behavior_cloning")
        normalization = receipt.get("normalization")
        warmup = receipt.get("critic_warmup")
        transitions = receipt.get("transition_counts")
        config = receipt.get("config")
        if not all(
            isinstance(value, dict)
            for value in (summary_receipt, initialization, behavior, normalization, warmup, transitions, config)
        ):
            raise PublicationError(f"seed {seed} receipt is incomplete")
        config_hash = canonical_hash(config)
        initial_actor = str(initialization.get("actor_sha256"))
        observation_rms = str(initialization.get("observation_rms_sha256"))
        fresh_critic = str(behavior.get("fresh_critic_state_sha256"))
        if (
            receipt.get("status") != "complete"
            or receipt.get("training_split") != "train"
            or receipt.get("evaluation_split") != "dev"
            or receipt.get("final_split_used") is not False
            or receipt.get("development_case_count") != DEVELOPMENT_CASE_COUNT
            or receipt.get("flow")
            != [
                "behavior_cloning_and_dagger",
                "actor_frozen_critic_warmup",
                "ppo_actor_critic_training",
                "deterministic_development_evaluation",
                "create_new_receipt",
            ]
            or receipt_hash != summary_receipt.get("sha256")
            or config_hash != summary_receipt.get("config_sha256")
            or config_hash
            != contract.get("registered_training_config_sha256_by_seed", {}).get(
                str(seed)
            )
            or initial_actor != EXPECTED_BC_ACTOR_SHA256
            or observation_rms != EXPECTED_OBSERVATION_RMS_SHA256
            or fresh_critic
            != contract.get("registered_fresh_critic_state_sha256_by_seed", {}).get(
                str(seed)
            )
            or behavior.get("dagger_iterations") != 0
            or behavior.get("interactive_relabelling") is not False
            or behavior.get("method")
            != "approved_external_single_pass_behavior_cloning"
            or behavior.get("legacy_bc_or_dagger_dataset_collected_by_ppo_worker")
            is not False
            or normalization.get("observation_rms_frozen") is not True
            or warmup.get("actor_parameters_byte_identical") is not True
            or warmup.get("actor_sha256_before") != initial_actor
            or warmup.get("actor_sha256_after") != initial_actor
            or warmup.get("completed_transitions") != 50_000
            or transitions.get("active_actor_critic") != 2_000_000
            or transitions.get("critic_warmup") != 50_000
            or transitions.get("total_environment") != 2_050_000
        ):
            raise PublicationError(f"seed {seed} initialization or training contract drifted")
        initial_actors.add(initial_actor)
        initial_rms.add(observation_rms)
        fresh_critics.add(fresh_critic)

        curve_specs = (
            ("bc_initialization", "bc_initialization", 0),
            ("post_critic_warmup", "post_critic_warmup", 0),
            ("ppo_200000_transitions", "ppo_200000", 200_000),
            ("ppo_500000_transitions", "ppo_500000", 500_000),
            ("ppo_1000000_transitions", "ppo_1000000", 1_000_000),
            ("ppo_2000000_transitions", "ppo_2000000", 2_000_000),
        )
        curve: list[dict[str, Any]] = []
        for source_key, phase, milestone in curve_specs:
            total = 0 if phase == "bc_initialization" else 50_000 + milestone
            result, identity = _validate_development(
                receipt.get("development_curve", {}).get(source_key),
                expected_milestone=milestone,
                expected_total_transitions=total,
                expected_identity=expected_identity,
            )
            expected_identity = identity
            curve.append({"phase": phase, **result})
            if milestone in SELECTABLE_MILESTONES:
                reference = summary_by_key.get((seed, milestone))
                if not isinstance(reference, dict):
                    raise PublicationError("ranked candidate is missing")
                development = reference.get("development")
                if not isinstance(development, dict) or any(
                    development.get(key) != result.get(key)
                    for key in (
                        "active_actor_critic_transitions",
                        "case_count",
                        "failure_reason_code_histogram",
                        "hard_violation_count",
                        "maximum_conservation_residual",
                        "mean_minimum_tail_margin",
                        "mean_resilience_auc",
                        "solve_rate",
                        "solved_count",
                        "total_environment_transitions",
                    )
                ) or development.get("rows_sha256") != result.get(
                    "source_rows_sha256"
                ):
                    raise PublicationError("summary candidate disagrees with receipt rows")
                bundle = _validate_bundle(
                    study_root, reference, seed=seed, milestone=milestone
                )
                selectable.append(
                    {
                        "active_actor_critic_transitions": milestone,
                        "actor_state_sha256": reference["actor_state_sha256"],
                        "bundle": bundle,
                        "development": result,
                        "id": reference["id"],
                        "optimizer_state_sha256": reference[
                            "optimizer_state_sha256"
                        ],
                        "policy_seed": seed,
                        "policy_state_sha256": reference["policy_state_sha256"],
                        "return_rms_sha256": reference["return_rms_sha256"],
                        "training_config_sha256": config_hash,
                        "training_receipt_sha256": receipt_hash,
                    }
                )
        seed_rows.append(
            {
                "critic_warmup_transitions": 50_000,
                "development_curve": curve,
                "fresh_critic_state_sha256": fresh_critic,
                "initial_actor_state_sha256": initial_actor,
                "observation_rms_sha256": observation_rms,
                "policy_seed": seed,
                "training_config_sha256": config_hash,
                "training_receipt": {
                    "path_within_external_study": _relative_external_path(
                        receipt_path, study_root
                    ),
                    "sha256": receipt_hash,
                    "size_bytes": receipt_path.stat().st_size,
                },
            }
        )
    if (
        initial_actors != {EXPECTED_BC_ACTOR_SHA256}
        or initial_rms != {EXPECTED_OBSERVATION_RMS_SHA256}
        or len(fresh_critics) != len(REGISTERED_SEEDS)
    ):
        raise PublicationError("cross-seed initialization invariants drifted")

    ranked = sorted(
        selectable,
        key=lambda row: (
            -int(row["development"]["solved_count"]),
            int(row["active_actor_critic_transitions"]),
            int(row["policy_seed"]),
        ),
    )
    summary_rank = [row["id"] for row in summary_candidates]
    if [row["id"] for row in ranked] != summary_rank:
        raise PublicationError("candidate ranking drifted")
    endpoints = sorted(
        (
            row
            for row in ranked
            if row["active_actor_critic_transitions"] == 2_000_000
        ),
        key=lambda row: int(row["policy_seed"]),
    )
    endpoint_counts = [int(row["development"]["solved_count"]) for row in endpoints]
    endpoint_per_family = {
        family_id: {
            "mean_solved_count": fmean(
                row["development"]["per_family_solved_count"][family_id]
                for row in endpoints
            ),
            "solved_counts_by_seed": {
                str(row["policy_seed"]): row["development"][
                    "per_family_solved_count"
                ][family_id]
                for row in endpoints
            },
        }
        for family_id in FAMILY_IDS
    }
    endpoint_summary = {
        "mean_delta_vs_incumbent": fmean(endpoint_counts) - INCUMBENT_ENDPOINT_MEAN,
        "mean_solved_count": fmean(endpoint_counts),
        "per_family": endpoint_per_family,
        "population_std_solved_count": pstdev(endpoint_counts),
        "sample_std_solved_count": stdev(endpoint_counts),
        "seed_count_at_or_above_172": sum(
            count >= PROMOTION_ENDPOINT_SOLVED for count in endpoint_counts
        ),
        "solved_counts_by_seed": {
            str(row["policy_seed"]): row["development"]["solved_count"]
            for row in endpoints
        },
    }
    best = ranked[0]
    best_pass = best["development"]["solved_count"] >= PROMOTION_BEST_SOLVED
    mean_pass = endpoint_summary["mean_solved_count"] > INCUMBENT_ENDPOINT_MEAN
    consistency_pass = (
        endpoint_summary["seed_count_at_or_above_172"]
        >= PROMOTION_ENDPOINT_SEED_COUNT
    )
    promotion = {
        "all_conditions_required": True,
        "conditions": {
            "best_checkpoint_at_least_183_of_200_dev": {
                "observed": best["development"]["solved_count"],
                "passed": best_pass,
                "threshold": PROMOTION_BEST_SOLVED,
            },
            "three_seed_2m_mean_above_incumbent_171_4": {
                "observed": endpoint_summary["mean_solved_count"],
                "passed": mean_pass,
                "threshold_exclusive": INCUMBENT_ENDPOINT_MEAN,
            },
            "at_least_two_of_three_2m_endpoints_at_or_above_172": {
                "observed": endpoint_summary["seed_count_at_or_above_172"],
                "passed": consistency_pass,
                "threshold": PROMOTION_ENDPOINT_SEED_COUNT,
            },
        },
        "decision": "complete_not_promoted",
        "final_evaluation_run_or_authorized": False,
        "passed": best_pass and mean_pass and consistency_pass,
        "resilience_auc_used": False,
    }
    if promotion != summary.get("promotion") or promotion["passed"]:
        raise PublicationError("promotion decision drifted")
    source_files = {
        path: contract["source_identity"][path]
        for path in sorted(contract["source_identity"])
    }
    return {
        "best_checkpoint": best,
        "candidate_count": len(ranked),
        "created_at_utc": summary["created_at_utc"],
        "cross_seed_initialization": {
            "distilled_actor_state_sha256": EXPECTED_BC_ACTOR_SHA256,
            "fresh_critic_state_sha256_by_seed": {
                str(row["policy_seed"]): row["fresh_critic_state_sha256"]
                for row in seed_rows
            },
            "observation_rms_frozen": True,
            "observation_rms_sha256": EXPECTED_OBSERVATION_RMS_SHA256,
        },
        "development_case_count": DEVELOPMENT_CASE_COUNT,
        "endpoint_summary": endpoint_summary,
        "external_evidence": {
            "attempt_id": "city-recovery-distilled-ppo-v4-attempt-02",
            "protocol": {
                "sha256": file_sha256(protocol_path),
                "size_bytes": protocol_path.stat().st_size,
            },
            "summary": {
                "sha256": file_sha256(summary_path),
                "size_bytes": summary_path.stat().st_size,
            },
            "training_receipts_sha256_by_seed": {
                str(seed): receipt_hashes[seed] for seed in REGISTERED_SEEDS
            },
        },
        "final_split_imported_or_used": False,
        "incumbent": {
            "best_of_20_development_solved_count": INCUMBENT_BEST_SOLVED,
            "five_seed_2m_endpoints": summary["incumbent_reference"][
                "five_seed_2m_endpoints"
            ],
            "selection_receipt": {
                "path": "internal/developmental_runs/v4/checkpoint-selection-200.json",
                "sha256": summary["incumbent_reference"]["selection_receipt"][
                    "sha256"
                ],
            },
            "training_summary": {
                "path": "internal/developmental_runs/v4/training-study-200-summary.json",
                "sha256": summary["incumbent_reference"]["training_summary"][
                    "sha256"
                ],
            },
        },
        "comparison": {
            "best_of_registered_challenger_vs_incumbent_best_of_20": {
                "challenger": best["development"]["solved_count"],
                "delta": (
                    best["development"]["solved_count"] - INCUMBENT_BEST_SOLVED
                ),
                "incumbent": INCUMBENT_BEST_SOLVED,
            },
            "challenger_three_seed_mean_vs_incumbent_five_seed_mean": {
                "challenger": endpoint_summary["mean_solved_count"],
                "delta": endpoint_summary["mean_delta_vs_incumbent"],
                "fairer_seed_level_comparison": True,
                "incumbent": INCUMBENT_ENDPOINT_MEAN,
            },
            "decisive_framing": "preregistered_conjunctive_promotion_rule",
        },
        "invariants": {
            "all_development_hard_violation_counts_zero": True,
            "all_development_maximum_conservation_residuals_zero": True,
            "all_nine_selectable_bundles_hash_verified": True,
            "canonical_development_roster_identical_across_all_evaluations": True,
            "distilled_actor_identical_across_seeds": True,
            "final_split_not_imported_or_used": True,
            "fresh_critic_unique_per_policy_seed": True,
            "observation_rms_identical_and_frozen_across_seeds": True,
        },
        "method_disclosure": {
            "dagger_iterations": 0,
            "distribution_shift_resolved": False,
            "interactive_relabelling": False,
            "legacy_generic_trainer_flow_label": "behavior_cloning_and_dagger",
            "legacy_generic_trainer_flow_label_is_nonoperative": True,
            "legacy_demonstrations_recollected_by_ppo_workers": False,
            "method": "single_pass_offline_oracle_behavior_cloning_then_critic_warmup_then_ppo",
            "null_scope": summary["null_scope"],
            "operative_zero_dagger_fields_are_authoritative": True,
        },
        "promotion": promotion,
        "ranking": {
            "candidates": ranked,
            "primary_metric": "development_solved_count",
            "resilience_auc_used_for_selection": False,
            "tie_breakers": ["earlier_transition_count", "lower_policy_seed"],
        },
        "registered_policy_seeds": list(REGISTERED_SEEDS),
        "schema_version": SCHEMA_VERSION,
        "seed_runs": seed_rows,
        "source_contract": {
            "git_commit": contract["git_commit"],
            "source_files": source_files,
            "source_identity_sha256": contract["source_identity_sha256"],
            "study_contract_sha256": protocol["contract_sha256"],
            "torch_runtime": contract["torch_runtime"],
            "torch_runtime_sha256": contract["torch_runtime_sha256"],
        },
        "split": "dev",
        "status": "complete_not_promoted",
        "tool": TOOL_ID,
        "training": contract["training"],
        "upstream_evidence": upstream,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    receipt = build_portable_receipt(args.study_root)
    _write_canonical_json(args.output.resolve(), receipt)
    print(
        "published oracle-distilled PPO evidence: "
        f"best={receipt['best_checkpoint']['development']['solved_count']}/200, "
        f"endpoint_mean={receipt['endpoint_summary']['mean_solved_count']:.1f}, "
        f"promoted={receipt['promotion']['passed']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
