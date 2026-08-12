#!/usr/bin/env python3
"""Publish portable, receipt-bound evidence for DEV-only family reweighting."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import fmean, pstdev, stdev
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.shared_evidence import canonical_hash, file_sha256  # noqa: E402
from scripts import moderate_family_training as study  # noqa: E402
from scripts.publish_oracle_distilled_ppo_evidence import (  # noqa: E402
    PublicationError,
    _load,
    _relative_external_path,
    _validate_development,
    _write_canonical_json,
)

TOOL_ID = "publish_moderate_family_evidence.py"
SCHEMA_VERSION = "city-recovery-moderate-family-dev-evidence-v1"
EXPECTED_DIFFICULTY_SHA256 = (
    "27d4b675273ebdfabc7ec5f6546a2d4c75ec5774e024c9d0c57484f800e4e5d4"
)
EXPECTED_PROTOCOL_SHA256 = (
    "4cc902fdee9e090df0be6042ccb5f2953eadde9693867e553f26b61ca8c65ad7"
)
EXPECTED_SUMMARY_SHA256 = (
    "935a0069d3c1eb53885e4ff5843ec5545eef4277a4a73ff6376a9948ea64e8a0"
)
EXPECTED_TRAINING_SUMMARY_SHA256 = (
    "7c39de680d74e22a4429a940f08473f9572dd50e9757ec94e006124e05a2925d"
)
EXPECTED_SELECTION_SHA256 = (
    "65fefa91903e6e7539ead5e1a957528454a9c01e8084ace56fa5047738e73e00"
)
SHIPPED_ARTIFACT_SHA256 = (
    "a9f5e9b41be57d7cd34623725a5ab4067aa75fbab16dc666cecc3c0a06c26483"
)
REGISTERED_SEEDS = (37_017, 47_017, 57_017)
DIAGNOSTIC_MILESTONES = (200_000, 500_000, 1_000_000, 2_000_000)
SELECTABLE_MILESTONES = (500_000, 1_000_000, 2_000_000)
FAMILY_IDS = (
    "v3_dev_river_flood",
    "v3_dev_industrial_outage",
    "v3_dev_logistics_strike",
    "v3_dev_seismic_cluster",
    "v3_dev_health_compound",
)
INCUMBENT_ENDPOINTS = {37_017: 172, 47_017: 171, 57_017: 171}
INCUMBENT_FIVE_SEED_ENDPOINTS = [172, 171, 171, 174, 169]
INCUMBENT_BEST = 178
INCUMBENT_MEAN = 171.4
BASELINE_STUDY_ROOT = Path(r"E:\city-recovery-v4-study-200-attempt-01")
DEFAULT_OUTPUT = (
    ROOT
    / "internal"
    / "developmental_runs"
    / "v4"
    / "moderate-family-study-200.json"
)


def _portable_training_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Project the validated source config without retaining a machine-local path."""

    sampler = config.get("training_family_sampler")
    if not isinstance(sampler, dict):
        raise PublicationError("moderate-family sampler config is missing")
    evidence_path = sampler.get("selection_evidence_path")
    if (
        not isinstance(evidence_path, str)
        or Path(evidence_path).name
        != "city-recovery-moderate-family-v4-difficulty-attempt-01.json"
    ):
        raise PublicationError("moderate-family selection evidence path drifted")
    portable_sampler = {
        key: value for key, value in sampler.items() if key != "selection_evidence_path"
    }
    portable_sampler["selection_evidence"] = {
        "portable_receipt_section": "difficulty_evidence",
        "source_receipt_sha256": EXPECTED_DIFFICULTY_SHA256,
    }
    return {
        **config,
        "training_family_sampler": portable_sampler,
    }


def _portable_bundle(
    study_root: Path,
    *,
    seed: int,
    milestone: int,
    reference: Mapping[str, Any],
    expected_config: Mapping[str, Any],
) -> dict[str, Any]:
    bundle = study_root / f"seed-{seed}" / "checkpoints" / f"ppo-{milestone}"
    manifest_path = bundle / "manifest.json"
    model_path = bundle / "model.zip"
    normalization_path = bundle / "normalization.npz"
    manifest = _load(manifest_path, "moderate-family checkpoint manifest")
    checkpoint = manifest.get("checkpoint")
    normalization = manifest.get("normalization")
    training = manifest.get("training")
    if not all(isinstance(value, dict) for value in (checkpoint, normalization, training)):
        raise PublicationError("moderate-family checkpoint manifest is incomplete")
    model_file = checkpoint.get("file")
    normalization_file = normalization.get("file")
    if not isinstance(model_file, dict) or not isinstance(normalization_file, dict):
        raise PublicationError("moderate-family checkpoint file records are missing")
    expected_id = f"seed-{seed}-ppo-{milestone}"
    manifest_sha = file_sha256(manifest_path)
    model_sha = file_sha256(model_path)
    normalization_sha = file_sha256(normalization_path)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "city-recovery-ppo-checkpoint"
        or manifest.get("publication", {}).get("complete") is not True
        or checkpoint.get("id") != expected_id
        or checkpoint.get("active_actor_critic_transitions") != milestone
        or training.get("seed") != seed
        or training.get("milestone") != milestone
        or training.get("config") != expected_config
        or training.get("config_sha256") != canonical_hash(expected_config)
        or reference.get("checkpoint_id") != expected_id
        or reference.get("active_actor_critic_transitions") != milestone
        or reference.get("manifest_sha256") != manifest_sha
        or reference.get("model_sha256") != model_sha
        or reference.get("normalization_sha256") != normalization_sha
        or reference.get("policy_state_sha256") != checkpoint.get("policy_state_sha256")
        or reference.get("actor_state_sha256") != checkpoint.get("actor_state_sha256")
        or reference.get("optimizer_state_sha256")
        != checkpoint.get("optimizer_state_sha256")
        or reference.get("obs_rms_sha256")
        != normalization.get("observation_rms_sha256")
        or reference.get("ret_rms_sha256")
        != normalization.get("return_rms_sha256")
        or model_file.get("sha256") != model_sha
        or normalization_file.get("sha256") != normalization_sha
    ):
        raise PublicationError(
            f"moderate-family checkpoint drifted: seed {seed}/{milestone}"
        )
    return {
        "actor_state_sha256": checkpoint["actor_state_sha256"],
        "checkpoint_id": expected_id,
        "manifest_sha256": manifest_sha,
        "model_sha256": model_sha,
        "normalization_sha256": normalization_sha,
        "observation_rms_sha256": normalization["observation_rms_sha256"],
        "optimizer_state_sha256": checkpoint["optimizer_state_sha256"],
        "policy_state_sha256": checkpoint["policy_state_sha256"],
        "return_rms_sha256": normalization["return_rms_sha256"],
        "selection_evaluation_export_supported": manifest.get("resume", {}).get(
            "selection_evaluation_export_supported"
        ),
        "training_config_sha256": canonical_hash(expected_config),
    }


def _difficulty_evidence(path: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    if file_sha256(path) != EXPECTED_DIFFICULTY_SHA256:
        raise PublicationError("training-family difficulty receipt hash drifted")
    try:
        receipt = study.validate_difficulty_receipt(path)
    except study.ModerateStudyError as exc:
        raise PublicationError("training-family difficulty receipt is invalid") from exc
    if (
        protocol.get("difficulty_receipt_sha256") != EXPECTED_DIFFICULTY_SHA256
        or receipt.get("split") != "train"
        or receipt.get("case_count") != 192
        or receipt.get("selection_policy", {}).get("artifact_sha256")
        != SHIPPED_ARTIFACT_SHA256
        or receipt.get("access_contract", {}).get("development_split_used") is not False
        or receipt.get("access_contract", {}).get("final_split_used") is not False
        or receipt.get("ranking", {}).get("development_evidence_used") is not False
        or any(receipt.get("invariants", {}).get(name) is not True for name in (
            "all_conservation_residuals_exactly_zero",
            "all_hard_violation_counts_zero",
            "exact_training_roster_6x32",
            "finite_critic_warmup_prefixes_balanced_within_one_episode",
            "hardest_families_have_exactly_2x_weight",
            "shipped_policy_alone_selected_family_weights",
            "tuned_rule_did_not_select_family_weights",
        ))
        or receipt.get("invariants", {}).get(
            "development_evidence_used_for_weight_selection"
        ) is not False
        or receipt.get("invariants", {}).get("final_split_used") is not False
    ):
        raise PublicationError("training-family difficulty contract drifted")
    rows = receipt["rows"]
    tuned = receipt["contextual_tuned_rule"]
    if (
        sum(int(row["solved"]) for row in rows) != 186
        or tuned.get("reported") is not True
        or sum(int(row["solved"]) for row in tuned["rows"]) != 180
    ):
        raise PublicationError("training-family difficulty totals drifted")
    source_files = {
        relative: receipt["source_identity"][relative]
        for relative in sorted(receipt["source_identity"])
    }
    return {
        "access_contract": receipt["access_contract"],
        "case_count": 192,
        "contextual_tuned_rule": {
            "per_family": tuned["per_family"],
            "ranked_family_ids": tuned["ranked_family_ids"],
            "reason": tuned["reason"],
            "reported": True,
            "rows": tuned["rows"],
            "rows_sha256": canonical_hash(tuned["rows"]),
            "solved_count": 180,
        },
        "ordered_case_contract_sha256": receipt["ordered_case_contract_sha256"],
        "per_family": receipt["per_family"],
        "ranking": receipt["ranking"],
        "receipt_sha256": EXPECTED_DIFFICULTY_SHA256,
        "rows": rows,
        "rows_sha256": canonical_hash(rows),
        "sampler": receipt["sampler"],
        "selection_policy": {
            "artifact_sha256": SHIPPED_ARTIFACT_SHA256,
            "id": "shipped_v4_onnx",
            "runtime": "onnxruntime_CPUExecutionProvider",
        },
        "shipped_policy_solved_count": 186,
        "source_contract": {
            "source_files": source_files,
            "source_identity_sha256": canonical_hash(source_files),
        },
        "split": "train",
    }


def _matched_incumbent_evidence(
    expected_identity: list[tuple[str, int, int, str]],
    challenger_config_by_seed: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    summary_path = (
        ROOT / "internal" / "developmental_runs" / "v4" / "training-study-200-summary.json"
    )
    selection_path = (
        ROOT / "internal" / "developmental_runs" / "v4" / "checkpoint-selection-200.json"
    )
    if (
        file_sha256(summary_path) != EXPECTED_TRAINING_SUMMARY_SHA256
        or file_sha256(selection_path) != EXPECTED_SELECTION_SHA256
    ):
        raise PublicationError("incumbent portable evidence drifted")
    summary = _load(summary_path, "incumbent training summary")
    baseline = summary.get("baseline")
    if not isinstance(baseline, dict):
        raise PublicationError("incumbent baseline is missing")
    baseline_endpoints = baseline.get("endpoints")
    if not isinstance(baseline_endpoints, list) or len(baseline_endpoints) != 5:
        raise PublicationError("incumbent endpoint roster is incomplete")
    all_endpoint_by_seed = {int(row["seed"]): row for row in baseline_endpoints}
    matched_curves: list[dict[str, Any]] = []
    five_seed_endpoints: list[dict[str, Any]] = []
    five_seed_identity: list[tuple[str, int, int, str]] | None = expected_identity
    for seed in (37_017, 47_017, 57_017, 67_017, 77_017):
        source = (
            BASELINE_STUDY_ROOT
            / "adopted_v3_equivalent_2m"
            / f"seed-{seed}"
            / "training-receipt.json"
        )
        reference = all_endpoint_by_seed.get(seed)
        if not isinstance(reference, dict) or file_sha256(source) != reference.get(
            "receipt_sha256"
        ):
            raise PublicationError(f"matched incumbent seed {seed} receipt drifted")
        receipt = _load(source, f"matched incumbent seed {seed} receipt")
        config = receipt.get("config")
        transitions = receipt.get("transition_counts")
        if not isinstance(config, dict) or not isinstance(transitions, dict):
            raise PublicationError("incumbent training receipt is incomplete")
        if seed in REGISTERED_SEEDS:
            challenger_config = challenger_config_by_seed.get(seed)
            if not isinstance(challenger_config, Mapping):
                raise PublicationError("challenger config roster is incomplete")
            comparable_challenger = {
                key: value
                for key, value in challenger_config.items()
                if key != "training_family_sampler"
            }
            if comparable_challenger != config:
                raise PublicationError(
                    f"matched seed {seed} differs beyond the family sampler"
                )
        curve: list[dict[str, Any]] = []
        milestones = (
            SELECTABLE_MILESTONES if seed in REGISTERED_SEEDS else (2_000_000,)
        )
        for milestone in milestones:
            development = receipt.get("development_curve", {}).get(
                f"ppo_{milestone}_transitions"
            )
            total = int(transitions["critic_warmup"]) + milestone
            portable, identity = _validate_development(
                development,
                expected_milestone=milestone,
                expected_total_transitions=total,
                expected_identity=five_seed_identity,
            )
            five_seed_identity = identity
            if identity != expected_identity:
                raise PublicationError("matched incumbent development roster drifted")
            curve.append({"phase": f"ppo_{milestone}", **portable})
        endpoint = curve[-1]
        if endpoint["solved_count"] != int(reference["solved_count"]):
            raise PublicationError("matched incumbent development evidence drifted")
        five_seed_endpoints.append(
            {
                "development": endpoint,
                "policy_seed": seed,
                "training_receipt_sha256": reference["receipt_sha256"],
            }
        )
        if seed in REGISTERED_SEEDS:
            if endpoint["solved_count"] != INCUMBENT_ENDPOINTS[seed]:
                raise PublicationError("matched incumbent endpoint count drifted")
            matched_curves.append(
                {
                    "development_curve": curve,
                    "policy_seed": seed,
                    "training_config": config,
                    "training_config_sha256": canonical_hash(config),
                    "training_receipt_sha256": reference["receipt_sha256"],
                }
            )
    return {
        "best_of_20_development_solved_count": INCUMBENT_BEST,
        "five_seed_2m_endpoints": {
            "mean": INCUMBENT_MEAN,
            "population_std": pstdev(INCUMBENT_FIVE_SEED_ENDPOINTS),
            "sample_std": stdev(INCUMBENT_FIVE_SEED_ENDPOINTS),
            "solved_counts": INCUMBENT_FIVE_SEED_ENDPOINTS,
        },
        "five_seed_2m_rows": five_seed_endpoints,
        "matched_seed_curves": matched_curves,
        "selection_receipt": {
            "path": "internal/developmental_runs/v4/checkpoint-selection-200.json",
            "sha256": EXPECTED_SELECTION_SHA256,
        },
        "training_summary": {
            "path": "internal/developmental_runs/v4/training-study-200-summary.json",
            "sha256": EXPECTED_TRAINING_SUMMARY_SHA256,
        },
    }


def build_portable_receipt(study_root: Path, difficulty_path: Path) -> dict[str, Any]:
    """Validate the full external study and build complete portable evidence."""

    study_root = study_root.resolve()
    difficulty_path = difficulty_path.resolve()
    protocol_path = study_root / "protocol.json"
    summary_path = study_root / "study-summary.json"
    protocol = _load(protocol_path, "moderate-family protocol")
    summary = _load(summary_path, "moderate-family summary")
    if (
        file_sha256(protocol_path) != EXPECTED_PROTOCOL_SHA256
        or file_sha256(summary_path) != EXPECTED_SUMMARY_SHA256
        or protocol.get("schema_version") != study.STUDY_SCHEMA
        or protocol.get("tool") != study.TOOL_ID
        or protocol.get("phase") != "preregistered_protocol"
        or protocol.get("policy_seeds") != list(REGISTERED_SEEDS)
        or protocol.get("active_transitions_per_seed") != 2_000_000
        or protocol.get("curve_milestones") != list(SELECTABLE_MILESTONES)
        or protocol.get("selection_metric") != "development_solved_count"
        or protocol.get("final_evaluation_authorized") is not False
        or summary.get("schema_version") != study.STUDY_SCHEMA
        or summary.get("decision") != "retain_shipped_policy"
        or summary.get("training_split") != "weighted_train"
        or summary.get("evaluation_split") != "dev"
        or summary.get("development_case_count") != 200
        or summary.get("final_split_imported_or_used") is not False
        or summary.get("final_evaluation_authorized") is not False
    ):
        raise PublicationError("moderate-family protocol or summary drifted")
    source_identity = protocol.get("source_identity")
    if not isinstance(source_identity, dict):
        raise PublicationError("moderate-family source identity is missing")
    for relative, expected_sha in source_identity.items():
        if file_sha256(ROOT / relative) != expected_sha:
            raise PublicationError(f"moderate-family source drifted: {relative}")
    if set(source_identity) != set(study.SOURCE_PATHS):
        raise PublicationError("moderate-family source identity is incomplete")
    difficulty = _difficulty_evidence(difficulty_path, protocol)
    try:
        rebuilt = study.build_study_summary(
            difficulty_path=difficulty_path,
            study_root=study_root,
        )
    except study.ModerateStudyError as exc:
        raise PublicationError("moderate-family study failed revalidation") from exc
    if rebuilt != summary:
        raise PublicationError("moderate-family summary does not rebuild exactly")

    summary_by_key = {
        (int(row["policy_seed"]), int(row["active_actor_critic_transitions"])): row
        for row in summary["selection"]["ranked_candidates"]
    }
    summary_receipt_by_seed = {
        int(row["policy_seed"]): row for row in summary["seed_rows"]
    }
    expected_identity: list[tuple[str, int, int, str]] | None = None
    candidates: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    challenger_config_by_seed: dict[int, dict[str, Any]] = {}
    training_hashes: dict[str, str] = {}
    bundle_count = 0
    curve_specs = (
        ("bc_initialization", "bc_initialization", 0),
        ("post_critic_warmup", "post_critic_warmup", 0),
        ("ppo_200000_transitions", "ppo_200000", 200_000),
        ("ppo_500000_transitions", "ppo_500000", 500_000),
        ("ppo_1000000_transitions", "ppo_1000000", 1_000_000),
        ("ppo_2000000_transitions", "ppo_2000000", 2_000_000),
    )
    for seed in REGISTERED_SEEDS:
        receipt_path = study_root / f"seed-{seed}" / "training-receipt.json"
        receipt = _load(receipt_path, f"moderate-family seed {seed} receipt")
        receipt_sha = file_sha256(receipt_path)
        training_hashes[str(seed)] = receipt_sha
        summary_seed = summary_receipt_by_seed.get(seed)
        config = receipt.get("config")
        initialization = receipt.get("initialization")
        behavior = receipt.get("behavior_cloning")
        normalization = receipt.get("normalization")
        warmup = receipt.get("critic_warmup")
        transitions = receipt.get("transition_counts")
        checks = receipt.get("checks")
        if not all(
            isinstance(value, dict)
            for value in (
                summary_seed,
                config,
                initialization,
                behavior,
                normalization,
                warmup,
                transitions,
                checks,
            )
        ):
            raise PublicationError("moderate-family training receipt is incomplete")
        warmup_transitions = int(warmup.get("completed_transitions", -1))
        if (
            receipt_sha != summary_seed.get("training_receipt_sha256")
            or receipt.get("status") != "complete"
            or receipt.get("training_split") != "train"
            or receipt.get("evaluation_split") != "dev"
            or receipt.get("final_split_used") is not False
            or config.get("policy_seed") != seed
            or config.get("active_actor_critic_transitions") != 2_000_000
            or config.get("learning_rate") != 7.5e-5
            or config.get("reward_profile") != "v3_equivalent"
            or config.get("preparedness_alignment_coefficient") != 10.0
            or config.get("vec_normalize") is not True
            or config.get("freeze_observation_rms") is not True
            or config.get("training_family_sampler", {}).get("family_weights")
            != difficulty["sampler"]["family_weights"]
            or behavior.get("observation_count") != 30_720
            or normalization.get("observation_rms_frozen") is not True
            or normalization.get("observation_rms_sha256")
            != behavior.get("observation_rms_sha256")
            or warmup.get("actor_parameters_byte_identical") is not True
            or warmup.get("actor_sha256_before") != warmup.get("actor_sha256_after")
            or not 50_000 <= warmup_transitions <= 100_000
            or warmup_transitions % 5_000 != 0
            or transitions.get("critic_warmup") != warmup_transitions
            or transitions.get("active_actor_critic") != 2_000_000
            or transitions.get("total_environment")
            != warmup_transitions + 2_000_000
            or any(value is not True for value in checks.values())
        ):
            raise PublicationError(f"moderate-family training contract drifted: {seed}")
        challenger_config_by_seed[seed] = config
        curve: list[dict[str, Any]] = []
        for source_key, phase, milestone in curve_specs:
            total = 0 if phase == "bc_initialization" else warmup_transitions + milestone
            development, identity = _validate_development(
                receipt.get("development_curve", {}).get(source_key),
                expected_milestone=milestone,
                expected_total_transitions=total,
                expected_identity=expected_identity,
            )
            expected_identity = identity
            bundle = None
            if milestone in DIAGNOSTIC_MILESTONES:
                reference = receipt.get("checkpoint_bundles", {}).get(str(milestone))
                if not isinstance(reference, dict):
                    raise PublicationError("moderate-family checkpoint reference missing")
                bundle = _portable_bundle(
                    study_root,
                    seed=seed,
                    milestone=milestone,
                    reference=reference,
                    expected_config=config,
                )
                bundle_count += 1
            summary_curve = {
                "active_actor_critic_transitions": development[
                    "active_actor_critic_transitions"
                ],
                "case_count": development["case_count"],
                "failure_reason_code_histogram": development[
                    "failure_reason_code_histogram"
                ],
                "hard_violation_count": 0,
                "maximum_conservation_residual": 0.0,
                "mean_minimum_tail_margin": development["mean_minimum_tail_margin"],
                "mean_resilience_auc": development["mean_resilience_auc"],
                "per_family_solved_count": development["per_family_solved_count"],
                "phase": phase,
                "portable_rows_sha256": development["portable_rows_sha256"],
                "solve_rate": development["solve_rate"],
                "solved_count": development["solved_count"],
                "source_rows_sha256": development["source_rows_sha256"],
                "total_environment_transitions": development[
                    "total_environment_transitions"
                ],
            }
            if bundle is not None:
                summary_curve["bundle"] = bundle
            curve.append(summary_curve)
            if milestone in SELECTABLE_MILESTONES:
                summary_candidate = summary_by_key.get((seed, milestone))
                if not isinstance(summary_candidate, dict):
                    raise PublicationError("moderate-family ranked candidate missing")
                if (
                    summary_candidate.get("development_rows_sha256")
                    != development["source_rows_sha256"]
                    or summary_candidate.get("solved_count")
                    != development["solved_count"]
                    or summary_candidate.get("checkpoint", {}).get(
                        "manifest_sha256"
                    )
                    != bundle["manifest_sha256"]
                ):
                    raise PublicationError("moderate-family candidate summary drifted")
                candidates.append(
                    {
                        "active_actor_critic_transitions": milestone,
                        "bundle": bundle,
                        "development": development,
                        "id": f"seed-{seed}-ppo-{milestone}",
                        "policy_seed": seed,
                        "training_config_sha256": canonical_hash(config),
                        "training_receipt_sha256": receipt_sha,
                    }
                )
        runs.append(
            {
                "bc_initialization": {
                    "actor_state_sha256": initialization["actor_sha256"],
                    "dataset_sha256": behavior["dataset_sha256"],
                    "observation_rms_sha256": normalization[
                        "observation_rms_sha256"
                    ],
                    "policy_state_sha256": initialization["policy_sha256"],
                },
                "critic_warmup_transitions": warmup_transitions,
                "development_curve": curve,
                "policy_seed": seed,
                "training_config_sha256": canonical_hash(config),
                "training_config": _portable_training_config(config),
                "portable_training_config_sha256": canonical_hash(
                    _portable_training_config(config)
                ),
                "training_receipt": {
                    "path_within_external_study": _relative_external_path(
                        receipt_path, study_root
                    ),
                    "sha256": receipt_sha,
                    "size_bytes": receipt_path.stat().st_size,
                },
            }
        )
    if expected_identity is None or bundle_count != 12:
        raise PublicationError("moderate-family study is incomplete")

    ranked = sorted(
        candidates,
        key=lambda row: (
            -int(row["development"]["solved_count"]),
            int(row["active_actor_critic_transitions"]),
            int(row["policy_seed"]),
        ),
    )
    if [row["id"] for row in ranked] != [
        row["checkpoint"]["checkpoint_id"]
        for row in summary["selection"]["ranked_candidates"]
    ]:
        raise PublicationError("moderate-family candidate ranking drifted")
    endpoints = sorted(
        (
            row
            for row in ranked
            if row["active_actor_critic_transitions"] == 2_000_000
        ),
        key=lambda row: int(row["policy_seed"]),
    )
    endpoint_counts = [int(row["development"]["solved_count"]) for row in endpoints]
    endpoint_summary = {
        "mean_solved_count": fmean(endpoint_counts),
        "per_family": {
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
        },
        "population_std_solved_count": pstdev(endpoint_counts),
        "sample_std_solved_count": stdev(endpoint_counts),
        "seed_count_at_or_above_172": sum(count >= 172 for count in endpoint_counts),
        "solved_counts_by_seed": {
            str(row["policy_seed"]): row["development"]["solved_count"]
            for row in endpoints
        },
    }
    promotion = study.promotion_gate(
        int(ranked[0]["development"]["solved_count"]), endpoint_counts
    )
    if promotion != summary["promotion_gate"] or promotion["passed"]:
        raise PublicationError("moderate-family promotion decision drifted")
    incumbent = _matched_incumbent_evidence(
        expected_identity, challenger_config_by_seed
    )
    incumbent_by_seed = {
        int(row["policy_seed"]): row
        for row in incumbent["five_seed_2m_rows"]
        if int(row["policy_seed"]) in REGISTERED_SEEDS
    }
    incumbent_curve_by_seed = {
        int(row["policy_seed"]): {
            int(point["active_actor_critic_transitions"]): point
            for point in row["development_curve"]
        }
        for row in incumbent["matched_seed_curves"]
    }
    challenger_by_key = {
        (int(row["policy_seed"]), int(row["active_actor_critic_transitions"])): row
        for row in ranked
    }
    matched_curve_comparison: dict[str, Any] = {}
    for milestone in SELECTABLE_MILESTONES:
        challenger_counts = [
            int(challenger_by_key[(seed, milestone)]["development"]["solved_count"])
            for seed in REGISTERED_SEEDS
        ]
        baseline_counts = [
            int(incumbent_curve_by_seed[seed][milestone]["solved_count"])
            for seed in REGISTERED_SEEDS
        ]
        matched_curve_comparison[str(milestone)] = {
            "challenger_mean": fmean(challenger_counts),
            "challenger_solved_counts": challenger_counts,
            "incumbent_mean": fmean(baseline_counts),
            "incumbent_solved_counts": baseline_counts,
            "mean_delta": fmean(challenger_counts) - fmean(baseline_counts),
        }
    per_family_comparison: dict[str, Any] = {}
    pooled_pairing = {
        "both_solved": 0,
        "challenger_only": 0,
        "incumbent_only": 0,
        "neither_solved": 0,
    }
    for family_id in FAMILY_IDS:
        challenger_counts = [
            int(row["development"]["per_family_solved_count"][family_id])
            for row in endpoints
        ]
        baseline_counts = [
            int(
                incumbent_by_seed[seed]["development"]["per_family_solved_count"][
                    family_id
                ]
            )
            for seed in REGISTERED_SEEDS
        ]
        family_pairing = {
            "both_solved": 0,
            "challenger_only": 0,
            "incumbent_only": 0,
            "neither_solved": 0,
        }
        for challenger in endpoints:
            seed = int(challenger["policy_seed"])
            baseline_rows = incumbent_by_seed[seed]["development"]["rows"]
            for challenger_row, baseline_row in zip(
                challenger["development"]["rows"], baseline_rows, strict=True
            ):
                if challenger_row["family_id"] != family_id:
                    continue
                if challenger_row["row_id"] != baseline_row["row_id"]:
                    raise PublicationError("matched endpoint row identity drifted")
                challenger_solved = bool(challenger_row["solved"])
                baseline_solved = bool(baseline_row["solved"])
                if challenger_solved and baseline_solved:
                    key = "both_solved"
                elif challenger_solved:
                    key = "challenger_only"
                elif baseline_solved:
                    key = "incumbent_only"
                else:
                    key = "neither_solved"
                family_pairing[key] += 1
                pooled_pairing[key] += 1
        per_family_comparison[family_id] = {
            "challenger_counts": challenger_counts,
            "challenger_mean": fmean(challenger_counts),
            "matched_incumbent_counts": baseline_counts,
            "matched_incumbent_mean": fmean(baseline_counts),
            "matched_mean_delta": fmean(challenger_counts) - fmean(baseline_counts),
            "pooled_case_pairing": family_pairing,
        }
    if pooled_pairing != {
        "both_solved": 504,
        "challenger_only": 13,
        "incumbent_only": 10,
        "neither_solved": 73,
    }:
        raise PublicationError("matched endpoint case pairing drifted")

    selection = _load(
        ROOT
        / "internal"
        / "developmental_runs"
        / "v4"
        / "checkpoint-selection-200.json",
        "incumbent checkpoint selection",
    )
    shipped = selection.get("selected_checkpoint")
    if not isinstance(shipped, dict):
        raise PublicationError("shipped checkpoint selection is missing")
    expected_shipped_receipt_path = (
        BASELINE_STUDY_ROOT
        / "adopted_v3_equivalent_2m"
        / "seed-67017"
        / "training-receipt.json"
    ).resolve()
    shipped_receipt_path = Path(str(shipped.get("training_receipt_path", ""))).resolve()
    if (
        shipped.get("id") != "seed-67017-ppo-1000000"
        or shipped.get("policy_seed") != 67_017
        or shipped.get("active_actor_critic_transitions") != 1_000_000
        or shipped_receipt_path != expected_shipped_receipt_path
        or shipped.get("training_receipt_sha256")
        != "37bc81cd677ae86458c207a99758c1c295b411906e7fe1ce2ea22d26bb22398f"
        or file_sha256(shipped_receipt_path)
        != shipped.get("training_receipt_sha256")
    ):
        raise PublicationError("shipped selected receipt identity drifted")
    shipped_receipt = _load(
        shipped_receipt_path,
        "shipped selected training receipt",
    )
    shipped_source = shipped_receipt.get("development_curve", {}).get(
        f"ppo_{shipped['active_actor_critic_transitions']}_transitions"
    )
    shipped_total = shipped_receipt.get("transition_counts", {}).get(
        "critic_warmup"
    ) + shipped["active_actor_critic_transitions"]
    shipped_portable, shipped_identity = _validate_development(
        shipped_source,
        expected_milestone=shipped["active_actor_critic_transitions"],
        expected_total_transitions=shipped_total,
        expected_identity=expected_identity,
    )
    if shipped_identity != expected_identity or shipped_portable["solved_count"] != 178:
        raise PublicationError("shipped selected development rows drifted")
    incumbent["shipped_selected_checkpoint"] = {
        "development": shipped_portable,
        "id": shipped["id"],
        "policy_seed": shipped["policy_seed"],
        "training_receipt_sha256": shipped["training_receipt_sha256"],
        "transitions": shipped["active_actor_critic_transitions"],
    }
    selected_pairing = {
        "both_solved": 0,
        "challenger_only": 0,
        "shipped_only": 0,
        "neither_solved": 0,
    }
    selected_family_discordance = {
        family_id: {"challenger_only": 0, "shipped_only": 0}
        for family_id in FAMILY_IDS
    }
    for challenger_row, shipped_row in zip(
        ranked[0]["development"]["rows"], shipped_portable["rows"], strict=True
    ):
        if challenger_row["row_id"] != shipped_row["row_id"]:
            raise PublicationError("selected comparison row identity drifted")
        challenger_solved = bool(challenger_row["solved"])
        shipped_solved = bool(shipped_row["solved"])
        if challenger_solved and shipped_solved:
            selected_pairing["both_solved"] += 1
        elif challenger_solved:
            selected_pairing["challenger_only"] += 1
            selected_family_discordance[challenger_row["family_id"]][
                "challenger_only"
            ] += 1
        elif shipped_solved:
            selected_pairing["shipped_only"] += 1
            selected_family_discordance[challenger_row["family_id"]][
                "shipped_only"
            ] += 1
        else:
            selected_pairing["neither_solved"] += 1
    if selected_pairing != {
        "both_solved": 175,
        "challenger_only": 1,
        "shipped_only": 3,
        "neither_solved": 21,
    }:
        raise PublicationError("selected checkpoint case pairing drifted")
    selected_family_counts = {
        family_id: {
            "challenger": ranked[0]["development"]["per_family_solved_count"][
                family_id
            ],
            "shipped": shipped_portable["per_family_solved_count"][family_id],
            "delta": (
                ranked[0]["development"]["per_family_solved_count"][family_id]
                - shipped_portable["per_family_solved_count"][family_id]
            ),
        }
        for family_id in FAMILY_IDS
    }
    five_seed_family_comparison = {}
    for family_id in FAMILY_IDS:
        baseline_counts = [
            row["development"]["per_family_solved_count"][family_id]
            for row in incumbent["five_seed_2m_rows"]
        ]
        challenger_counts = [
            row["development"]["per_family_solved_count"][family_id]
            for row in endpoints
        ]
        five_seed_family_comparison[family_id] = {
            "challenger_three_seed_mean": fmean(challenger_counts),
            "incumbent_five_seed_mean": fmean(baseline_counts),
            "mean_delta": fmean(challenger_counts) - fmean(baseline_counts),
        }
    source_files = {path: source_identity[path] for path in sorted(source_identity)}
    publication_source_files = {
        path: file_sha256(ROOT / path)
        for path in (
            "backend/app/shared_evidence.py",
            "scripts/moderate_family_training.py",
            "scripts/publish_moderate_family_evidence.py",
            "scripts/publish_oracle_distilled_ppo_evidence.py",
            "scripts/training_artifacts.py",
        )
    }
    return {
        "best_checkpoint": ranked[0],
        "candidate_count": 9,
        "comparison": {
            "best_registered_challenger_vs_incumbent_best_of_20": {
                "challenger": ranked[0]["development"]["solved_count"],
                "delta": ranked[0]["development"]["solved_count"] - INCUMBENT_BEST,
                "incumbent": INCUMBENT_BEST,
                "selection_candidate_counts": {"challenger": 9, "incumbent": 20},
                "selection_asymmetric": True,
            },
            "best_registered_challenger_vs_shipped_selected_case_pairing": {
                "family_solved_counts": selected_family_counts,
                "family_discordance": selected_family_discordance,
                **selected_pairing,
            },
            "challenger_three_seed_mean_vs_incumbent_five_seed_mean": {
                "challenger": endpoint_summary["mean_solved_count"],
                "delta": endpoint_summary["mean_solved_count"] - INCUMBENT_MEAN,
                "incumbent": INCUMBENT_MEAN,
            },
            "matched_same_seed_2m": {
                "challenger_solved_counts": endpoint_counts,
                "deltas": [
                    endpoint_counts[index] - INCUMBENT_ENDPOINTS[seed]
                    for index, seed in enumerate(REGISTERED_SEEDS)
                ],
                "incumbent_solved_counts": [
                    INCUMBENT_ENDPOINTS[seed] for seed in REGISTERED_SEEDS
                ],
                "mean_delta": fmean(endpoint_counts)
                - fmean(INCUMBENT_ENDPOINTS.values()),
                "fairer_matched_seed_comparison": True,
                "policy_seeds": list(REGISTERED_SEEDS),
                "pooled_case_pairing": pooled_pairing,
            },
            "matched_same_seed_curve": matched_curve_comparison,
            "per_family_matched_same_seed_2m": per_family_comparison,
            "per_family_vs_incumbent_five_seed_2m": five_seed_family_comparison,
        },
        "development_case_count": 200,
        "difficulty_evidence": difficulty,
        "endpoint_summary": endpoint_summary,
        "external_evidence": {
            "attempt_id": "city-recovery-moderate-family-v4-attempt-01",
            "difficulty_receipt": {
                "sha256": EXPECTED_DIFFICULTY_SHA256,
                "size_bytes": difficulty_path.stat().st_size,
            },
            "protocol": {
                "sha256": EXPECTED_PROTOCOL_SHA256,
                "size_bytes": protocol_path.stat().st_size,
            },
            "summary": {
                "sha256": EXPECTED_SUMMARY_SHA256,
                "size_bytes": summary_path.stat().st_size,
            },
            "training_receipts_sha256_by_seed": training_hashes,
        },
        "final_split_imported_or_used": False,
        "incumbent": incumbent,
        "invariants": {
            "all_12_durable_bundles_hash_verified": True,
            "all_development_hard_violation_counts_zero": True,
            "all_development_maximum_conservation_residuals_zero": True,
            "canonical_development_roster_identical_across_all_evaluations": True,
            "difficulty_ranking_uses_train_only": True,
            "final_split_not_imported_or_used": True,
            "matched_configs_identical_except_training_family_sampler": True,
            "nine_selectable_candidates_have_exact_portable_rows": True,
            "shipped_selected_receipt_hash_verified": True,
            "three_training_receipts_hash_verified": True,
        },
        "interpretation_scope": {
            "difficulty_selection_basis": "shipped_policy_train_family_solve_count",
            "duplicated_existing_training_support": True,
            "imitation_observation_count_incumbent": 23_040,
            "imitation_observation_count_treatment": 30_720,
            "imitation_observation_exposure_multiplier": 4 / 3,
            "observation_rms_count_incumbent": 23_040.0001,
            "observation_rms_count_treatment": 30_720.0001,
            "pure_fixed_volume_importance_reweighting": False,
            "failure_redistribution_possible": True,
            "new_scenarios_or_families_added": False,
            "per_family_development_deltas_are_descriptive_not_causal": True,
            "treatment": "two_hardest_of_six_train_families_weighted_2x",
        },
        "promotion": promotion,
        "publication_source_contract": {
            "source_files": publication_source_files,
            "source_identity_sha256": canonical_hash(publication_source_files),
        },
        "ranking": {
            "candidates": ranked,
            "primary_metric": "development_solved_count",
            "resilience_auc_used_for_selection": False,
            "tie_breakers": ["earlier_transition_count", "lower_policy_seed"],
        },
        "registered_policy_seeds": list(REGISTERED_SEEDS),
        "schema_version": SCHEMA_VERSION,
        "source_contract": {
            "source_files": source_files,
            "source_identity_sha256": canonical_hash(source_files),
        },
        "split": "dev",
        "status": "complete_not_promoted",
        "study_runs": runs,
        "tool": TOOL_ID,
        "training": {
            "active_actor_critic_transitions": 2_000_000,
            "family_weights": difficulty["sampler"]["family_weights"],
            "hardest_training_family_ids": difficulty["sampler"][
                "hardest_family_ids"
            ],
            "optimizer": {
                "batch_size": 500,
                "critic_warmup_max_transitions": 100_000,
                "critic_warmup_min_transitions": 50_000,
                "ent_coef": 0.003,
                "lanes": 20,
                "learning_rate": 7.5e-5,
                "n_steps_per_lane": 250,
                "target_kl": 0.02,
            },
            "preparedness_alignment_coefficient": 10.0,
            "reward_profile": "v3_equivalent",
            "weighted_cycle_case_count": 256,
            "canonical_unique_case_count": 192,
            "imitation_observation_count": 30_720,
            "observation_rms_count": 30_720.0001,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--difficulty-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    receipt = build_portable_receipt(args.study_root, args.difficulty_receipt)
    _write_canonical_json(args.output.resolve(), receipt)
    print(
        "published moderate-family evidence: "
        f"best={receipt['best_checkpoint']['development']['solved_count']}/200, "
        f"endpoint_mean={receipt['endpoint_summary']['mean_solved_count']:.3f}, "
        f"promoted={receipt['promotion']['passed']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
