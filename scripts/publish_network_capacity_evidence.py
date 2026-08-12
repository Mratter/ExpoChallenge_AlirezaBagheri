#!/usr/bin/env python3
"""Publish portable, receipt-bound evidence for the DEV-only capacity study."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from statistics import fmean, pstdev, stdev
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.shared_evidence import canonical_hash, file_sha256  # noqa: E402
from scripts import run_large_architecture_study as study  # noqa: E402
from scripts.publish_oracle_distilled_ppo_evidence import (  # noqa: E402
    PublicationError,
    _load,
    _relative_external_path,
    _validate_development,
    _write_canonical_json,
)

TOOL_ID = "publish_network_capacity_evidence.py"
SCHEMA_VERSION = "city-recovery-network-capacity-dev-evidence-v1"
EXPECTED_PROTOCOL_SHA256 = (
    "b80b9d99e629109b88e92d8dc9d1d0ec1c754ce70b2477225f61d97e9496a49a"
)
EXPECTED_SUMMARY_SHA256 = (
    "86e336a1cdd8c7584f7796030e121a8dbf110af8caa035c8fc65cc38778d3ddb"
)
EXPECTED_GIT_COMMIT = "420ebeb8d1b2d2973d1b31bf8efc7690630afe2d"
REGISTERED_SEEDS = (37_017, 47_017, 57_017)
ARM_IDS = ("large_lr_7_5e_5", "large_lr_3e_5")
LEARNING_RATES = {"large_lr_7_5e_5": 7.5e-5, "large_lr_3e_5": 3e-5}
SELECTABLE_MILESTONES = (500_000, 1_000_000, 2_000_000)
FAMILY_IDS = (
    "v3_dev_river_flood",
    "v3_dev_industrial_outage",
    "v3_dev_logistics_strike",
    "v3_dev_seismic_cluster",
    "v3_dev_health_compound",
)
INCUMBENT_BEST = 178
INCUMBENT_MEAN = 171.4
INCUMBENT_ENDPOINTS = [172, 171, 171, 174, 169]
DEFAULT_OUTPUT = (
    ROOT
    / "internal"
    / "developmental_runs"
    / "v4"
    / "network-capacity-study-200.json"
)


def _candidate_id(arm_id: str, seed: int, milestone: int) -> str:
    return f"{arm_id}:seed-{seed}-ppo-{milestone}"


def _portable_bundle(
    study_root: Path,
    arm_id: str,
    seed: int,
    milestone: int,
    reference: Mapping[str, Any],
    summary_candidate: Mapping[str, Any],
) -> dict[str, Any]:
    bundle = study_root / arm_id / f"seed-{seed}" / "checkpoints" / f"ppo-{milestone}"
    manifest_path = bundle / "manifest.json"
    model_path = bundle / "model.zip"
    normalization_path = bundle / "normalization.npz"
    manifest = _load(manifest_path, "capacity checkpoint manifest")
    checkpoint = manifest.get("checkpoint")
    normalization = manifest.get("normalization")
    training = manifest.get("training")
    if not all(isinstance(value, dict) for value in (checkpoint, normalization, training)):
        raise PublicationError("capacity checkpoint manifest is incomplete")
    model_file = checkpoint.get("file")
    normalization_file = normalization.get("file")
    if not isinstance(model_file, dict) or not isinstance(normalization_file, dict):
        raise PublicationError("capacity checkpoint file records are missing")
    expected_checkpoint_id = f"seed-{seed}-ppo-{milestone}"
    expected_manifest_sha = file_sha256(manifest_path)
    expected_model_sha = file_sha256(model_path)
    expected_normalization_sha = file_sha256(normalization_path)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "city-recovery-ppo-checkpoint"
        or manifest.get("publication", {}).get("complete") is not True
        or checkpoint.get("id") != expected_checkpoint_id
        or checkpoint.get("active_actor_critic_transitions") != milestone
        or training.get("seed") != seed
        or training.get("milestone") != milestone
        or reference.get("checkpoint_id") != expected_checkpoint_id
        or reference.get("active_actor_critic_transitions") != milestone
        or reference.get("manifest_sha256") != expected_manifest_sha
        or reference.get("model_sha256") != expected_model_sha
        or reference.get("normalization_sha256") != expected_normalization_sha
        or model_file.get("sha256") != expected_model_sha
        or normalization_file.get("sha256") != expected_normalization_sha
        or reference.get("policy_state_sha256") != checkpoint.get("policy_state_sha256")
        or reference.get("actor_state_sha256") != checkpoint.get("actor_state_sha256")
        or reference.get("optimizer_state_sha256")
        != checkpoint.get("optimizer_state_sha256")
        or reference.get("obs_rms_sha256")
        != normalization.get("observation_rms_sha256")
        or reference.get("ret_rms_sha256")
        != normalization.get("return_rms_sha256")
        or summary_candidate.get("bundle_manifest_sha256") != expected_manifest_sha
        or summary_candidate.get("checkpoint_sha256") != expected_model_sha
        or summary_candidate.get("normalization_file_sha256")
        != expected_normalization_sha
        or summary_candidate.get("observation_rms_sha256")
        != normalization.get("observation_rms_sha256")
        or training.get("config_sha256") != canonical_hash(training.get("config"))
    ):
        raise PublicationError(
            f"capacity checkpoint bundle drifted: {arm_id}/{seed}/{milestone}"
        )
    return {
        "actor_state_sha256": checkpoint["actor_state_sha256"],
        "checkpoint_id": expected_checkpoint_id,
        "manifest_sha256": expected_manifest_sha,
        "model_sha256": expected_model_sha,
        "normalization_sha256": expected_normalization_sha,
        "observation_rms_sha256": normalization["observation_rms_sha256"],
        "optimizer_state_sha256": checkpoint["optimizer_state_sha256"],
        "policy_state_sha256": checkpoint["policy_state_sha256"],
        "return_rms_sha256": normalization["return_rms_sha256"],
        "selection_evaluation_export_supported": manifest.get("resume", {}).get(
            "selection_evaluation_export_supported"
        ),
    }


def _paired_config(config: Mapping[str, Any]) -> dict[str, Any]:
    comparable = copy.deepcopy(dict(config))
    comparable.pop("learning_rate", None)
    architecture = comparable.get("architecture_experiment")
    if not isinstance(architecture, dict):
        raise PublicationError("architecture experiment config is missing")
    architecture.pop("arm_id", None)
    return comparable


def _endpoint_summary(candidates: Sequence[Mapping[str, Any]], arm_id: str) -> dict[str, Any]:
    endpoints = sorted(
        (
            row
            for row in candidates
            if row["arm_id"] == arm_id
            and row["active_actor_critic_transitions"] == 2_000_000
        ),
        key=lambda row: int(row["policy_seed"]),
    )
    counts = [int(row["development"]["solved_count"]) for row in endpoints]
    return {
        "arm_id": arm_id,
        "learning_rate": LEARNING_RATES[arm_id],
        "mean_solved_count": fmean(counts),
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
        "population_std_solved_count": pstdev(counts),
        "sample_std_solved_count": stdev(counts),
        "seed_count_at_or_above_172": sum(count >= 172 for count in counts),
        "solved_counts_by_seed": {
            str(row["policy_seed"]): row["development"]["solved_count"]
            for row in endpoints
        },
    }


def build_portable_receipt(study_root: Path) -> dict[str, Any]:
    """Validate six external runs and return complete portable DEV evidence."""

    study_root = study_root.resolve()
    protocol_path = study_root / "protocol.json"
    summary_path = study_root / "architecture-study-summary.json"
    protocol = _load(protocol_path, "network-capacity protocol")
    summary = _load(summary_path, "network-capacity summary")
    contract = protocol.get("contract")
    if not isinstance(contract, dict):
        raise PublicationError("network-capacity study contract is missing")
    source_identity = contract.get("source_identity")
    if not isinstance(source_identity, dict):
        raise PublicationError("network-capacity source identity is missing")
    if (
        file_sha256(protocol_path) != EXPECTED_PROTOCOL_SHA256
        or file_sha256(summary_path) != EXPECTED_SUMMARY_SHA256
        or protocol.get("contract_sha256") != canonical_hash(contract)
        or summary.get("contract_sha256") != protocol.get("contract_sha256")
        or contract.get("git_commit") != EXPECTED_GIT_COMMIT
        or contract.get("tool") != "run_large_architecture_study.py"
        or contract.get("registered_policy_seeds") != list(REGISTERED_SEEDS)
        or contract.get("scope", {}).get("final_split_imported_or_used") is not False
        or summary.get("status") != "complete_not_promoted"
        or summary.get("split") != "dev"
        or summary.get("development_case_count") != 200
        or summary.get("candidate_count") != 18
        or summary.get("final_split_imported_or_used") is not False
    ):
        raise PublicationError("network-capacity protocol or summary drifted")
    for relative_path, expected_sha in source_identity.items():
        if file_sha256(ROOT / relative_path) != expected_sha:
            raise PublicationError(f"registered source drifted: {relative_path}")

    # The preregistration runner performs its complete independent receipt and
    # bundle validation.  Compare the rebuilt result byte-for-field after
    # preserving the external creation timestamp.
    try:
        recomputed = study.build_summary(study_root, contract)
    except study.ArchitectureStudyError as exc:
        raise PublicationError("external capacity study failed revalidation") from exc
    recomputed["created_at_utc"] = summary.get("created_at_utc")
    if recomputed != summary:
        raise PublicationError("external capacity summary does not rebuild exactly")

    summary_candidates = summary.get("ranking", {}).get("candidates")
    if not isinstance(summary_candidates, list) or len(summary_candidates) != 18:
        raise PublicationError("capacity summary must rank 18 candidates")
    summary_by_key = {
        (
            str(row["arm_id"]),
            int(row["policy_seed"]),
            int(row["active_actor_critic_transitions"]),
        ): row
        for row in summary_candidates
    }
    summary_receipts = {
        (str(row["arm_id"]), int(row["seed"])): row for row in summary["receipts"]
    }
    expected_identity: list[tuple[str, int, int, str]] | None = None
    candidates: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    receipt_payloads: dict[tuple[str, int], dict[str, Any]] = {}
    receipt_hashes: dict[str, str] = {}

    curve_specs = (
        ("bc_initialization", "bc_initialization", 0),
        ("post_critic_warmup", "post_critic_warmup", 0),
        ("ppo_200000_transitions", "ppo_200000", 200_000),
        ("ppo_500000_transitions", "ppo_500000", 500_000),
        ("ppo_1000000_transitions", "ppo_1000000", 1_000_000),
        ("ppo_2000000_transitions", "ppo_2000000", 2_000_000),
    )
    for arm_id in ARM_IDS:
        for seed in REGISTERED_SEEDS:
            receipt_path = study_root / arm_id / f"seed-{seed}" / "training-receipt.json"
            receipt = _load(receipt_path, f"{arm_id} seed {seed} receipt")
            receipt_payloads[(arm_id, seed)] = receipt
            receipt_sha = file_sha256(receipt_path)
            receipt_hashes[f"{arm_id}/seed-{seed}"] = receipt_sha
            summary_receipt = summary_receipts.get((arm_id, seed))
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
                    summary_receipt,
                    config,
                    initialization,
                    behavior,
                    normalization,
                    warmup,
                    transitions,
                    checks,
                )
            ):
                raise PublicationError("capacity training receipt is incomplete")
            architecture = config.get("architecture_experiment")
            if (
                receipt.get("status") != "complete"
                or receipt.get("training_split") != "train"
                or receipt.get("evaluation_split") != "dev"
                or receipt.get("final_split_used") is not False
                or receipt_sha != summary_receipt.get("sha256")
                or canonical_hash(config) != summary_receipt.get("config_sha256")
                or config.get("policy_seed") != seed
                or config.get("learning_rate") != LEARNING_RATES[arm_id]
                or not isinstance(architecture, dict)
                or architecture.get("arm_id") != arm_id
                or architecture.get("actor_hidden_layers") != [768, 512, 256]
                or architecture.get("critic_hidden_layers") != [768, 512, 256]
                or architecture.get("activation") != "SiLU"
                or architecture.get("parameter_counts")
                != {"actor": 587_564, "critic": 582_145, "total_policy": 1_169_709}
                or config.get("active_actor_critic_transitions") != 2_000_000
                or config.get("reward_profile") != "v3_equivalent"
                or config.get("vec_normalize") is not True
                or config.get("freeze_observation_rms") is not True
                or normalization.get("observation_rms_frozen") is not True
                or warmup.get("actor_parameters_byte_identical") is not True
                or warmup.get("actor_sha256_before") != initialization.get("actor_sha256")
                or warmup.get("actor_sha256_after") != initialization.get("actor_sha256")
                or not isinstance(warmup.get("completed_transitions"), int)
                or not 50_000 <= warmup["completed_transitions"] <= 100_000
                or warmup["completed_transitions"] % 5_000 != 0
                or transitions.get("active_actor_critic") != 2_000_000
                or transitions.get("critic_warmup") != warmup.get("completed_transitions")
                or transitions.get("total_environment")
                != 2_000_000 + warmup.get("completed_transitions")
                or any(value is not True for value in checks.values())
            ):
                raise PublicationError(f"capacity training contract drifted: {arm_id}/{seed}")

            curve: list[dict[str, Any]] = []
            for source_key, phase, milestone in curve_specs:
                total = 0 if phase == "bc_initialization" else warmup["completed_transitions"] + milestone
                development, identity = _validate_development(
                    receipt.get("development_curve", {}).get(source_key),
                    expected_milestone=milestone,
                    expected_total_transitions=total,
                    expected_identity=expected_identity,
                )
                expected_identity = identity
                curve.append({"phase": phase, **development})
                if milestone in SELECTABLE_MILESTONES:
                    summary_candidate = summary_by_key.get((arm_id, seed, milestone))
                    reference = receipt.get("checkpoint_bundles", {}).get(str(milestone))
                    if not isinstance(summary_candidate, dict) or not isinstance(reference, dict):
                        raise PublicationError("registered capacity candidate is missing")
                    summary_dev = summary_candidate.get("development")
                    if not isinstance(summary_dev, dict) or summary_dev.get("rows_sha256") != development["source_rows_sha256"]:
                        raise PublicationError("capacity summary row hash drifted")
                    for field in (
                        "case_count",
                        "failure_reason_code_histogram",
                        "hard_violation_count",
                        "maximum_conservation_residual",
                        "mean_minimum_tail_margin",
                        "mean_resilience_auc",
                        "solve_rate",
                        "solved_count",
                    ):
                        if summary_dev.get(field) != development.get(field):
                            raise PublicationError("capacity summary aggregate drifted")
                    bundle = _portable_bundle(
                        study_root, arm_id, seed, milestone, reference, summary_candidate
                    )
                    candidates.append(
                        {
                            "active_actor_critic_transitions": milestone,
                            "arm_id": arm_id,
                            "bundle": bundle,
                            "development": development,
                            "id": _candidate_id(arm_id, seed, milestone),
                            "learning_rate": LEARNING_RATES[arm_id],
                            "policy_seed": seed,
                            "training_config_sha256": canonical_hash(config),
                            "training_receipt_sha256": receipt_sha,
                        }
                    )
            runs.append(
                {
                    "arm_id": arm_id,
                    "bc_initialization_identity": summary_receipt["bc_initialization_identity"],
                    "critic_warmup_transitions": warmup["completed_transitions"],
                    "development_curve": curve,
                    "learning_rate": LEARNING_RATES[arm_id],
                    "policy_seed": seed,
                    "training_config_sha256": canonical_hash(config),
                    "training_receipt": {
                        "path_within_external_study": _relative_external_path(receipt_path, study_root),
                        "sha256": receipt_sha,
                        "size_bytes": receipt_path.stat().st_size,
                    },
                }
            )

    paired_checks: list[dict[str, Any]] = []
    for seed in REGISTERED_SEEDS:
        high = receipt_payloads[(ARM_IDS[0], seed)]
        low = receipt_payloads[(ARM_IDS[1], seed)]
        high_identity = summary_receipts[(ARM_IDS[0], seed)]["bc_initialization_identity"]
        low_identity = summary_receipts[(ARM_IDS[1], seed)]["bc_initialization_identity"]
        if high_identity != low_identity or _paired_config(high["config"]) != _paired_config(low["config"]):
            raise PublicationError(f"paired learning-rate contract drifted: seed {seed}")
        paired_checks.append(
            {
                "bc_initialization_identity": high_identity,
                "only_registered_config_difference_is_learning_rate": True,
                "policy_seed": seed,
            }
        )
    if paired_checks != [
        {
            "policy_seed": row["seed"],
            "bc_initialization_identity": {
                key: row[key]
                for key in ("actor_sha256", "dataset_sha256", "observation_rms_sha256", "policy_sha256")
            },
            "only_registered_config_difference_is_learning_rate": row["only_registered_config_difference_is_learning_rate"],
        }
        for row in summary["paired_learning_rate_checks"]
    ]:
        raise PublicationError("paired learning-rate summary drifted")

    arm_order = {arm_id: index for index, arm_id in enumerate(ARM_IDS)}
    ranked = sorted(
        candidates,
        key=lambda row: (
            -int(row["development"]["solved_count"]),
            int(row["active_actor_critic_transitions"]),
            arm_order[str(row["arm_id"])],
            int(row["policy_seed"]),
        ),
    )
    if [
        (row["arm_id"], row["policy_seed"], row["active_actor_critic_transitions"])
        for row in ranked
    ] != [
        (row["arm_id"], row["policy_seed"], row["active_actor_critic_transitions"])
        for row in summary_candidates
    ]:
        raise PublicationError("capacity candidate ranking drifted")

    endpoint_summaries = {
        arm_id: _endpoint_summary(ranked, arm_id) for arm_id in ARM_IDS
    }
    selected_arm_id = str(ranked[0]["arm_id"])
    selected_endpoint = endpoint_summaries[selected_arm_id]
    best_solved = int(ranked[0]["development"]["solved_count"])
    promotion = {
        "all_conditions_required": True,
        "conditions": {
            "selected_arm_at_least_two_seed_endpoints_at_or_above_172": {
                "observed": selected_endpoint["seed_count_at_or_above_172"],
                "passed": selected_endpoint["seed_count_at_or_above_172"] >= 2,
                "threshold": 2,
            },
            "selected_arm_three_seed_2m_mean_above_171_4": {
                "observed": selected_endpoint["mean_solved_count"],
                "passed": selected_endpoint["mean_solved_count"] > INCUMBENT_MEAN,
                "threshold_exclusive": INCUMBENT_MEAN,
            },
            "selected_checkpoint_at_least_183": {
                "observed": best_solved,
                "passed": best_solved >= 183,
                "threshold": 183,
            },
        },
        "decision": "complete_not_promoted",
        "final_evaluation_run_or_authorized": False,
        "resilience_auc_used": False,
    }
    promotion["passed"] = all(
        condition["passed"] for condition in promotion["conditions"].values()
    )
    promotion["decision"] = (
        "promote" if promotion["passed"] else "complete_not_promoted"
    )
    if promotion["passed"]:
        raise PublicationError("unexpected capacity-study promotion result")
    source_files = {path: source_identity[path] for path in sorted(source_identity)}
    curve_analysis: dict[str, Any] = {}
    for arm_id in ARM_IDS:
        deltas: dict[str, int] = {}
        one_million: list[int] = []
        two_million: list[int] = []
        for run in runs:
            if run["arm_id"] != arm_id:
                continue
            by_phase = {row["phase"]: row for row in run["development_curve"]}
            before = int(by_phase["ppo_1000000"]["solved_count"])
            after = int(by_phase["ppo_2000000"]["solved_count"])
            deltas[str(run["policy_seed"])] = after - before
            one_million.append(before)
            two_million.append(after)
        curve_analysis[arm_id] = {
            "all_three_seeds_increased_from_1m_to_2m": all(delta > 0 for delta in deltas.values()),
            "mean_1m_solved_count": fmean(one_million),
            "mean_2m_solved_count": fmean(two_million),
            "mean_delta_1m_to_2m": fmean(two_million) - fmean(one_million),
            "per_seed_delta_1m_to_2m": deltas,
        }

    baseline = summary["baseline_reference"]
    if (
        file_sha256(
            ROOT / "internal/developmental_runs/v4/training-study-200-summary.json"
        )
        != baseline["canonical_summary"]["sha256"]
        or file_sha256(
            ROOT / "internal/developmental_runs/v4/checkpoint-selection-200.json"
        )
        != baseline["shipped_selected_checkpoint"]["selection_receipt_sha256"]
        or baseline["five_seed_2m_endpoints"]
        != {
            "mean": INCUMBENT_MEAN,
            "population_std": pstdev(INCUMBENT_ENDPOINTS),
            "sample_std": stdev(INCUMBENT_ENDPOINTS),
            "solved_counts": INCUMBENT_ENDPOINTS,
        }
        or baseline["shipped_selected_checkpoint"]["candidate_count"] != 20
        or baseline["shipped_selected_checkpoint"]["solved_count"]
        != INCUMBENT_BEST
    ):
        raise PublicationError("incumbent development evidence drifted")
    baseline_curves: list[dict[str, Any]] = []
    for source in baseline["curves"]:
        seed = int(source["seed"])
        points = [
            {
                "active_actor_critic_transitions": int(point["active_actor_critic_transitions"]),
                "case_count": int(point["case_count"]),
                "failure_reason_code_histogram": point["failure_reason_code_histogram"],
                "hard_violation_count": int(point["hard_violation_count"]),
                "maximum_conservation_residual": float(point["maximum_conservation_residual"]),
                "mean_minimum_tail_margin": float(point["mean_minimum_tail_margin"]),
                "mean_resilience_auc": float(point["mean_resilience_auc"]),
                "ordered_case_identity_sha256": point["ordered_case_identity_sha256"],
                "solve_rate": float(point["solve_rate"]),
                "solved_count": int(point["solved_count"]),
                "source_rows_sha256": point["rows_sha256"],
            }
            for point in source["curve"]
        ]
        if [point["active_actor_critic_transitions"] for point in points] != list(
            SELECTABLE_MILESTONES
        ):
            raise PublicationError("incumbent development curve drifted")
        baseline_curves.append(
            {
                "bc_initialization_identity": source["bc_initialization_identity"],
                "curve": points,
                "policy_seed": seed,
                "training_config_sha256": source["config_sha256"],
                "training_receipt_sha256": source["receipt_sha256"],
            }
        )
    baseline_curve_means = {
        str(milestone): fmean(
            next(
                point["solved_count"]
                for point in curve["curve"]
                if point["active_actor_critic_transitions"] == milestone
            )
            for curve in baseline_curves
        )
        for milestone in SELECTABLE_MILESTONES
    }
    if baseline_curve_means != {
        "500000": 168.2,
        "1000000": 171.4,
        "2000000": 171.4,
    }:
        raise PublicationError("incumbent curve means drifted")
    publication_source_files = {
        path: file_sha256(ROOT / path)
        for path in (
            "backend/app/shared_evidence.py",
            "scripts/publish_network_capacity_evidence.py",
            "scripts/publish_oracle_distilled_ppo_evidence.py",
        )
    }
    return {
        "architecture": {
            **contract["architecture"],
            "activation": "SiLU",
            "actor_parameter_semantics": {
                "deterministic_mean_path": 587_542,
                "learned_log_standard_deviation": 22,
                "total_actor": 587_564,
            },
            "incumbent_parameter_counts": {
                "actor": 162_732,
                "actor_deterministic_mean_path": 162_710,
                "critic": 160_001,
                "learned_log_standard_deviation": 22,
                "total_policy": 322_733,
            },
            "public_interface": {"action_count": 22, "observation_count": 73},
            "smaller_optional_arm_run": False,
        },
        "arm_endpoint_summaries": endpoint_summaries,
        "best_checkpoint": ranked[0],
        "candidate_count": 18,
        "comparison": {
            "best_of_registered_challenger_vs_incumbent_best_of_20": {
                "challenger": best_solved,
                "delta": best_solved - INCUMBENT_BEST,
                "incumbent": INCUMBENT_BEST,
            },
            "decisive_framing": "preregistered_conjunctive_promotion_rule",
            "large_lr_3e_5_vs_incumbent_same_seed_2m": {
                "challenger_solved_counts": [178, 176, 175],
                "deltas": [6, 5, 4],
                "incumbent_solved_counts": [172, 171, 171],
                "policy_seeds": [37_017, 47_017, 57_017],
            },
            "selected_arm_three_seed_mean_vs_incumbent_five_seed_mean": {
                "challenger": selected_endpoint["mean_solved_count"],
                "delta": selected_endpoint["mean_solved_count"] - INCUMBENT_MEAN,
                "fairer_seed_level_comparison": True,
                "incumbent": INCUMBENT_MEAN,
            },
        },
        "created_at_utc": summary["created_at_utc"],
        "curve_analysis": curve_analysis,
        "development_case_count": 200,
        "external_evidence": {
            "attempt_id": "city-recovery-large-architecture-v4-attempt-01",
            "protocol": {"sha256": EXPECTED_PROTOCOL_SHA256, "size_bytes": protocol_path.stat().st_size},
            "summary": {"sha256": EXPECTED_SUMMARY_SHA256, "size_bytes": summary_path.stat().st_size},
            "training_receipts_sha256": receipt_hashes,
        },
        "final_split_imported_or_used": False,
        "incumbent": {
            "best_of_20_development_solved_count": INCUMBENT_BEST,
            "five_seed_2m_endpoints": {
                "mean": INCUMBENT_MEAN,
                "population_std": pstdev(INCUMBENT_ENDPOINTS),
                "sample_std": stdev(INCUMBENT_ENDPOINTS),
                "solved_counts": INCUMBENT_ENDPOINTS,
            },
            "five_seed_curves": baseline_curves,
            "five_seed_mean_solved_count_by_milestone": baseline_curve_means,
            "selection_receipt": {
                "path": "internal/developmental_runs/v4/checkpoint-selection-200.json",
                "sha256": baseline["shipped_selected_checkpoint"]["selection_receipt_sha256"],
            },
            "training_summary": {
                "path": "internal/developmental_runs/v4/training-study-200-summary.json",
                "sha256": baseline["canonical_summary"]["sha256"],
            },
        },
        "invariants": {
            "all_18_selectable_bundles_hash_verified": True,
            "all_development_hard_violation_counts_zero": True,
            "all_development_maximum_conservation_residuals_zero": True,
            "canonical_development_roster_identical_across_all_evaluations": True,
            "final_split_not_imported_or_used": True,
            "paired_learning_rate_initialization_and_config_verified": True,
            "six_training_receipts_hash_verified": True,
        },
        "null_scope": contract["null_scope"],
        "paired_learning_rate_checks": paired_checks,
        "promotion": promotion,
        "publication_source_contract": {
            "source_files": publication_source_files,
            "source_identity_sha256": canonical_hash(publication_source_files),
        },
        "ranking": {
            "candidates": ranked,
            "primary_metric": "development_solved_count",
            "resilience_auc_used_for_selection": False,
            "tie_breakers": [
                "earlier_active_actor_critic_transitions",
                "registered_arm_order",
                "lower_policy_seed",
            ],
        },
        "registered_arms": contract["registered_arms"],
        "registered_policy_seeds": list(REGISTERED_SEEDS),
        "schema_version": SCHEMA_VERSION,
        "source_contract": {
            "git_commit": contract["git_commit"],
            "source_files": source_files,
            "source_identity_sha256": canonical_hash(source_files),
            "study_contract_sha256": protocol["contract_sha256"],
        },
        "split": "dev",
        "status": "complete_not_promoted",
        "study_runs": runs,
        "tool": TOOL_ID,
        "training": contract["training"],
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
    best = receipt["best_checkpoint"]
    print(
        "published network-capacity evidence: "
        f"best={best['development']['solved_count']}/200, "
        f"selected_arm={best['arm_id']}, promoted={receipt['promotion']['passed']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
