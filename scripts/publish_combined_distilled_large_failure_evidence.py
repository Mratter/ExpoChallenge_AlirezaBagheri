#!/usr/bin/env python3
"""Publish portable evidence for the stopped combined DEV-only attempt.

This is a receipt-only publisher.  It validates already-written external bytes;
it never imports a policy, builds a scenario, resumes training, or evaluates a
split.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.shared_evidence import canonical_hash, file_sha256  # noqa: E402

TOOL_ID = "publish_combined_distilled_large_failure_evidence.py"
SCHEMA_VERSION = "city-recovery-combined-distilled-large-incomplete-evidence-v1"
EXPECTED_GIT_COMMIT = "5391e8e23e1bee73d8688503f8e56d7b4b4d0ed7"
EXPECTED_ROOT_INVENTORY_SHA256 = (
    "f3bcabc8e8ad84e77444a7dc2d34013c7615870a42586cd3f470028f268723c0"
)
EXPECTED_ROOT_FILE_COUNT = 37
EXPECTED_PROTOCOL_SHA256 = (
    "0709892d67a75cff6f46c4f46e7aa6b53c8f0e2155e18b8ae913ac14f851e9b9"
)
EXPECTED_BASE_CONTRACT_SHA256 = (
    "b7446646cb34c142dd9df719c09f177f5e668d14be2bb13e995adbadf1a75d3e"
)
EXPECTED_PPO_PROTOCOL_SHA256 = (
    "d6aa57f538e00054ad4077bf5fe5b98b60ff99c3bd7efde75273482a3726dc75"
)
EXPECTED_PPO_CONTRACT_SHA256 = (
    "a507861565d7bcb0aa6cf93e66cad9a6799dba1c2586722655843b891bfd5822"
)
EXPECTED_FIT_CLAIM_SHA256 = (
    "60739ecdc98ab70a824c8d10de7a995d99fd6b11a9b851a80a964c5c61976fa3"
)
EXPECTED_FIT_SUCCESS_SHA256 = (
    "de2aba25d037da4bb454992d71b7524f005c24002cf60c7625561f72f4ade5e6"
)
EXPECTED_FIT_MANIFEST_SHA256 = (
    "f031ce08a2715615e407402f95a729776e705a0ea8ef6a2ef0b9407c97ce5992"
)
EXPECTED_RECEIPT_SHA256 = {
    37_017: "f8c234ca770919df72333c151bc06538a2205f1af1275944eb540924571d01ff",
    47_017: "683eb0946f64e25d747f8fc41a80bbda13de4faf05dca226139e4a4b60364284",
    57_017: "1543f869e6ba2959a5b471e8d9abff644a1471a84c92ae147efbb14dbe4f4235",
}
EXPECTED_TRAINER_LOG_SHA256 = {
    37_017: "177052cc337b91ab003818c63d714d770b32a9bb9a4c1ea6e5b0061196e3f3eb",
    47_017: "5942f9859c5c7b23013bd5cbe8975120c984ffba69b769ac7af4c11494415120",
    57_017: "50eda2d294f8671ff8e7a328a8e88d853c49181716449c45d268139b388016e3",
}
EXPECTED_CONSOLE_LOGS = {
    "stderr": {
        "filename": "city-recovery-combined-large-distilled-v4-attempt-01.console.err.log",
        "sha256": "f10910603d181dbefc89c2e2289be07b18576b7c45dd9e527aa9c747419ef6fe",
        "size_bytes": 138,
    },
    "stdout": {
        "filename": "city-recovery-combined-large-distilled-v4-attempt-01.console.out.log",
        "sha256": "2ed71bb29c0338d9bf1c4dcf94ca45fd093421824b140c344b228a702651e125",
        "size_bytes": 178,
    },
}
EXPECTED_CONFIG_SHA256 = {
    "37017": "4aea3052d4201f5b11bcdeace16e80a11671cf14d6fd4fd63d68ebba365c4c01",
    "47017": "0779c00a9b783df089e0da1ab93fe2fe6d6672594b0f4da9310fddf1ad5526d3",
    "57017": "11a0ecaffd1b60b459e0930be30a352942aaf147cd5cfac53ef23c1bd339ea30",
}
EXPECTED_SOURCE_IDENTITY_SHA256 = (
    "5923fc01d2fd2c9effa80fff7cd342a0e5414a0a0f87180acd9926f966d1491e"
)
EXPECTED_DEV_IDENTITY_SHA256 = (
    "0f0b3ec175c99e28ff9d080a2ae5684592987a6c5f775688ab1f17d8ddc95853"
)
EXPECTED_DATASET_RECEIPT_SHA256 = (
    "e7777e53f20b886bbb82b167e0303b20ee0de32dcf9b87f50d175a0b71c5dc89"
)
EXPECTED_DISTILLATION_EVIDENCE_SHA256 = (
    "aee2df40263f892fb8d979ae190a483a91711564169bbac45336f32a24bb5e0d"
)
EXPECTED_CAPACITY_EVIDENCE_SHA256 = (
    "fd27e39b3b4868e43231b91f879e1830f1b2380f37bd03c3b23b9e5510564304"
)
DATASET_RECEIPT = Path(
    r"E:\city-recovery-training-oracle-v4-attempt-01\training\receipt.json"
)
DISTILLATION_EVIDENCE = (
    ROOT / "internal/developmental_runs/v4/oracle-distilled-ppo-study-200.json"
)
CAPACITY_EVIDENCE = (
    ROOT / "internal/developmental_runs/v4/network-capacity-study-200.json"
)
REGISTERED_SEEDS = (37_017, 47_017, 57_017)
COMPLETED_SEEDS = (37_017, 47_017)
MILESTONES = (200_000, 500_000, 1_000_000, 2_000_000)
SELECTABLE_MILESTONES = (500_000, 1_000_000, 2_000_000)
FAMILY_IDS = (
    "v3_dev_river_flood",
    "v3_dev_industrial_outage",
    "v3_dev_logistics_strike",
    "v3_dev_seismic_cluster",
    "v3_dev_health_compound",
)
INCUMBENT_ENDPOINTS = {37_017: 172, 47_017: 171, 57_017: 171}
LARGE_ONLY_ENDPOINTS = {37_017: 178, 47_017: 176, 57_017: 175}
DEFAULT_STUDY_ROOT = Path(r"E:\city-recovery-combined-large-distilled-v4-attempt-01")
DEFAULT_OUTPUT = (
    ROOT
    / "internal/developmental_runs/v4/combined-distilled-large-study-200.incomplete.json"
)
DEFAULT_REPORT = (
    ROOT / "benchmarks/v4/combined-distilled-large-study-200.incomplete.md"
)
_ABSOLUTE_PATH = re.compile(r"(?:[A-Za-z]:[\\/]|(?:^|\s)/[A-Za-z0-9_.-])")


class PublicationError(RuntimeError):
    """Raised when the stopped attempt cannot be published faithfully."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PublicationError(message)


def _load(path: Path, label: str, expected_sha256: str | None = None) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"{label} is missing")
    if expected_sha256 is not None:
        _require(file_sha256(path) == expected_sha256, f"{label} hash drifted")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{label} is not an object")
    return value


def _portable_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "case_seed": row["case_seed"],
        "hard_violation_count": row["hard_violation_count"],
        "maximum_conservation_residual": row["max_conservation_residual"],
        "minimum_tail_margin": row["minimum_tail_margin"],
        "reason_codes": row["reason_codes"],
        "resilience_auc": row["resilience_auc"],
        "row_id": row["row_id"],
        "solved": row["solved"],
        "tape_seed": row["tape_seed"],
        "tape_sha256": row["tape_sha256"],
    }


def _validate_development(value: Any, label: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} is missing")
    rows = value.get("rows")
    _require(isinstance(rows, list) and len(rows) == 200, f"{label} rows drifted")
    portable_rows = [_portable_row(row) for row in rows]
    _require(
        len({row["row_id"] for row in portable_rows}) == 200,
        f"{label} row ids drifted",
    )
    identities = [
        {
            "row_id": row["row_id"],
            "case_seed": row["case_seed"],
            "tape_seed": row["tape_seed"],
            "tape_sha256": row["tape_sha256"],
        }
        for row in portable_rows
    ]
    family_counts = {family_id: 0 for family_id in FAMILY_IDS}
    family_rows = {family_id: 0 for family_id in FAMILY_IDS}
    for row in portable_rows:
        family_id = row["row_id"].rsplit(":", 1)[0]
        _require(family_id in family_counts, f"{label} family drifted")
        family_rows[family_id] += 1
        family_counts[family_id] += int(row["solved"])
    _require(
        family_rows == {family_id: 40 for family_id in FAMILY_IDS},
        f"{label} family roster drifted",
    )
    solved_count = sum(row["solved"] for row in portable_rows)
    hard_violation_count = sum(
        row["hard_violation_count"] for row in portable_rows
    )
    maximum_conservation_residual = max(
        row["maximum_conservation_residual"] for row in portable_rows
    )
    reason_codes: Counter[str] = Counter()
    for row in portable_rows:
        if not row["solved"]:
            reason_codes.update(row["reason_codes"])
    failure_reason_code_histogram = dict(sorted(reason_codes.items()))
    mean_minimum_tail_margin = round(
        fmean(row["minimum_tail_margin"] for row in portable_rows), 10
    )
    mean_resilience_auc = round(
        fmean(row["resilience_auc"] for row in portable_rows), 10
    )
    _require(solved_count == value["solved_count"], f"{label} solve count drifted")
    _require(value["solve_rate"] == solved_count / 200, f"{label} solve rate drifted")
    _require(
        hard_violation_count == value["hard_violation_count"] == 0,
        f"{label} hard violations drifted",
    )
    _require(
        maximum_conservation_residual
        == value["maximum_conservation_residual"]
        == 0.0,
        f"{label} conservation drifted",
    )
    _require(
        failure_reason_code_histogram == value["failure_reason_code_histogram"],
        f"{label} reason histogram drifted",
    )
    _require(
        mean_minimum_tail_margin == value["mean_minimum_tail_margin"],
        f"{label} mean tail margin drifted",
    )
    _require(
        mean_resilience_auc == value["mean_resilience_auc"],
        f"{label} mean resilience AUC drifted",
    )
    return {
        "active_actor_critic_transitions": value["active_actor_critic_transitions"],
        "case_count": 200,
        "failure_reason_code_histogram": failure_reason_code_histogram,
        "hard_violation_count": 0,
        "maximum_conservation_residual": 0.0,
        "mean_minimum_tail_margin": mean_minimum_tail_margin,
        "mean_resilience_auc": mean_resilience_auc,
        "per_family_solved_count": family_counts,
        "rows": portable_rows,
        "rows_sha256": canonical_hash(portable_rows),
        "source_rows_sha256": canonical_hash(rows),
        "solve_rate": value["solve_rate"],
        "solved_count": value["solved_count"],
        "split_identity_sha256": canonical_hash(identities),
        "total_environment_transitions": value["total_environment_transitions"],
    }


def _inventory(study_root: Path) -> list[dict[str, Any]]:
    rows = [
        {
            "path": path.relative_to(study_root).as_posix(),
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(
            (path for path in study_root.rglob("*") if path.is_file()),
            key=lambda path: path.relative_to(study_root).as_posix(),
        )
    ]
    _require(len(rows) == EXPECTED_ROOT_FILE_COUNT, "external root file count drifted")
    _require(canonical_hash(rows) == EXPECTED_ROOT_INVENTORY_SHA256, "external root inventory drifted")
    return rows


def _portable_bundle(study_root: Path, seed: int, milestone: int, reference: Mapping[str, Any]) -> dict[str, Any]:
    relative = f"seed-{seed}/checkpoints/ppo-{milestone}"
    directory = study_root / relative
    manifest_path = directory / "manifest.json"
    model_path = directory / "model.zip"
    normalization_path = directory / "normalization.npz"
    manifest = _load(manifest_path, f"seed {seed} milestone {milestone} manifest")
    checkpoint = manifest.get("checkpoint", {})
    normalization = manifest.get("normalization", {})
    training = manifest.get("training", {})
    expected = {
        "manifest_sha256": file_sha256(manifest_path),
        "model_sha256": file_sha256(model_path),
        "normalization_sha256": file_sha256(normalization_path),
    }
    _require(reference.get("checkpoint_id") == f"seed-{seed}-ppo-{milestone}", "bundle id drifted")
    _require(reference.get("active_actor_critic_transitions") == milestone, "bundle milestone drifted")
    _require(all(reference.get(key) == digest for key, digest in expected.items()), "bundle reference hash drifted")
    _require(checkpoint.get("file", {}).get("sha256") == expected["model_sha256"], "bundle model binding drifted")
    _require(normalization.get("file", {}).get("sha256") == expected["normalization_sha256"], "bundle normalization binding drifted")
    _require(reference.get("actor_state_sha256") == checkpoint.get("actor_state_sha256"), "bundle actor binding drifted")
    _require(reference.get("policy_state_sha256") == checkpoint.get("policy_state_sha256"), "bundle policy binding drifted")
    _require(reference.get("optimizer_state_sha256") == checkpoint.get("optimizer_state_sha256"), "bundle optimizer binding drifted")
    _require(reference.get("obs_rms_sha256") == normalization.get("observation_rms_sha256"), "bundle RMS binding drifted")
    _require(reference.get("ret_rms_sha256") == normalization.get("return_rms_sha256"), "bundle return RMS binding drifted")
    _require(training.get("config_sha256") == canonical_hash(training.get("config")), "bundle config hash drifted")
    return {
        "active_actor_critic_transitions": milestone,
        "actor_state_sha256": checkpoint["actor_state_sha256"],
        "checkpoint_id": checkpoint["id"],
        "manifest_path_within_external_root": f"{relative}/manifest.json",
        "manifest_sha256": expected["manifest_sha256"],
        "model_sha256": expected["model_sha256"],
        "normalization_sha256": expected["normalization_sha256"],
        "observation_rms_sha256": normalization["observation_rms_sha256"],
        "optimizer_state_sha256": checkpoint["optimizer_state_sha256"],
        "policy_state_sha256": checkpoint["policy_state_sha256"],
        "return_rms_sha256": normalization["return_rms_sha256"],
    }


def _fit_evidence(
    study_root: Path,
    base_contract_sha256: str,
    registered_fit_contract: Mapping[str, Any],
) -> dict[str, Any]:
    claim_path = study_root / "large-oracle-bc/fit.claim.json"
    success_path = study_root / "large-oracle-bc/fit.success.json"
    success = _load(success_path, "large BC fit success", EXPECTED_FIT_SUCCESS_SHA256)
    claim = _load(claim_path, "large BC fit claim", EXPECTED_FIT_CLAIM_SHA256)
    fit = success.get("fit", {})
    gate = success.get("catastrophic_fit_gate", {})
    checkpoint = success.get("checkpoint_bundle", {})
    checkpoint_directory = study_root / "large-oracle-bc/checkpoint"
    checkpoint_manifest_path = checkpoint_directory / "manifest.json"
    checkpoint_model_path = checkpoint_directory / "model.zip"
    checkpoint_normalization_path = checkpoint_directory / "normalization.npz"
    checkpoint_manifest = _load(
        checkpoint_manifest_path, "large BC checkpoint manifest"
    )
    manifest_checkpoint = checkpoint_manifest.get("checkpoint", {})
    manifest_normalization = checkpoint_manifest.get("normalization", {})
    _require(success.get("base_contract_sha256") == base_contract_sha256, "fit base contract drifted")
    _require(claim.get("base_contract_sha256") == base_contract_sha256, "fit claim base contract drifted")
    _require(
        claim.get("fit_contract") == registered_fit_contract,
        "fit claim/registered contract drifted",
    )
    _require(success.get("status") == "complete_large_oracle_bc_fit_eligible_for_ppo", "fit status drifted")
    _require(success.get("development_evaluated") is False and success.get("final_split_imported_or_used") is False, "fit split boundary drifted")
    _require(gate.get("passed") is True and gate.get("decision") == "eligible_for_ppo", "fit gate drifted")
    _require(fit.get("training_row_count_per_student") == 5_040 and fit.get("heldout_row_count_per_student") == 720, "fit rows drifted")
    _require(fit.get("dagger_iterations") == 0 and fit.get("ppo_updates") == 0, "fit method drifted")
    _require(fit.get("causal_input_contract", {}).get("student_input_future_tape_visible") is False, "fit causal input drifted")
    _require(fit.get("causal_input_contract", {}).get("teacher_target_uses_full_future_tape") is True, "fit teacher target drifted")
    _require(fit.get("observation_rms_sha256") == "cb7b9a46369a0c225c3a6254433f6ef37e52b822ef44598fa4311b64e63a4ba4", "fit RMS drifted")
    _require(
        file_sha256(checkpoint_manifest_path)
        == checkpoint.get("manifest_sha256")
        == EXPECTED_FIT_MANIFEST_SHA256,
        "fit checkpoint manifest drifted",
    )
    _require(
        file_sha256(checkpoint_model_path)
        == checkpoint.get("model_sha256")
        == manifest_checkpoint.get("file", {}).get("sha256"),
        "fit checkpoint model drifted",
    )
    _require(
        file_sha256(checkpoint_normalization_path)
        == checkpoint.get("normalization_sha256")
        == manifest_normalization.get("file", {}).get("sha256"),
        "fit checkpoint normalization drifted",
    )
    _require(
        checkpoint.get("actor_state_sha256")
        == manifest_checkpoint.get("actor_state_sha256")
        and checkpoint.get("obs_rms_sha256")
        == manifest_normalization.get("observation_rms_sha256")
        == fit.get("observation_rms_sha256"),
        "fit checkpoint state binding drifted",
    )
    return {
        "checkpoint": {
            "actor_state_sha256": checkpoint["actor_state_sha256"],
            "manifest_sha256": checkpoint["manifest_sha256"],
            "model_sha256": checkpoint["model_sha256"],
            "normalization_sha256": checkpoint["normalization_sha256"],
            "observation_rms_sha256": checkpoint["obs_rms_sha256"],
        },
        "fit_claim_sha256": EXPECTED_FIT_CLAIM_SHA256,
        "fit_success_sha256": EXPECTED_FIT_SUCCESS_SHA256,
        "gate": gate,
        "method": {
            "architecture": fit["architecture"],
            "batch_size": fit["batch_size"],
            "dagger_iterations": fit["dagger_iterations"],
            "epochs": fit["epochs"],
            "fit_rows": fit["training_row_count_per_student"],
            "heldout_rows": fit["heldout_row_count_per_student"],
            "learning_rate": fit["learning_rate"],
            "observation_rms_sha256": fit["observation_rms_sha256"],
            "parameter_counts": fit["parameter_counts"],
            "public_input_count": 73,
            "public_output_count": 22,
            "single_pass_offline": True,
        },
        "oracle_label_student": {
            "fit": fit["oracle_label_student"]["fit"]["trained"],
            "heldout": fit["oracle_label_student"]["heldout"],
        },
        "matched_hand_rule_control": {
            "fit": fit["matched_hand_rule_control"]["fit"]["trained"],
            "heldout": fit["matched_hand_rule_control"]["heldout"],
        },
    }


def _seed_evidence(
    study_root: Path,
    contract: Mapping[str, Any],
    seed: int,
    *,
    fit_manifest_sha256: str,
    fit_model_sha256: str,
) -> dict[str, Any]:
    receipt_path = study_root / f"seed-{seed}/training-receipt.json"
    log_path = study_root / f"seed-{seed}/trainer.log"
    receipt = _load(receipt_path, f"seed {seed} receipt", EXPECTED_RECEIPT_SHA256[seed])
    _require(file_sha256(log_path) == EXPECTED_TRAINER_LOG_SHA256[seed], f"seed {seed} log drifted")
    config = receipt.get("config")
    _require(isinstance(config, dict) and canonical_hash(config) == EXPECTED_CONFIG_SHA256[str(seed)], f"seed {seed} config drifted")
    _require(config == contract["registered_training_configs"][str(seed)], f"seed {seed} preregistered config mismatch")
    _require(receipt.get("final_split_used") is False and receipt.get("evaluation_split") == "dev", f"seed {seed} split boundary drifted")
    behavior = receipt.get("behavior_cloning", {})
    warmup = receipt.get("critic_warmup", {})
    normalization = receipt.get("normalization", {})
    _require(behavior.get("fresh_critic_state_sha256") == contract["registered_fresh_critic_state_sha256_by_seed"][str(seed)], f"seed {seed} critic binding drifted")
    _require(
        behavior.get("critic_fresh_for_registered_seed") is True
        and behavior.get("critic_imported_from_bc_checkpoint") is False,
        f"seed {seed} critic initialization drifted",
    )
    _require(behavior.get("actor_state_sha256") == contract["fit_reference"]["checkpoint"]["actor_state_sha256"], f"seed {seed} actor binding drifted")
    _require(
        behavior.get("actor_byte_identical_to_large_bc_checkpoint") is True
        and behavior.get("actor_warm_start_applied") is True,
        f"seed {seed} actor initialization drifted",
    )
    _require(
        behavior.get("method") == "approved_new_large_single_pass_oracle_bc"
        and behavior.get("dagger_iterations") == 0
        and behavior.get("interactive_relabelling") is False
        and behavior.get("observation_count") == 5_040,
        f"seed {seed} imitation method drifted",
    )
    _require(
        behavior.get("source_checkpoint_manifest_sha256")
        == fit_manifest_sha256
        and behavior.get("source_checkpoint_model_sha256") == fit_model_sha256
        and behavior.get("source_dataset_receipt_sha256")
        == EXPECTED_DATASET_RECEIPT_SHA256,
        f"seed {seed} initialization source drifted",
    )
    _require(normalization.get("observation_rms_frozen") is True, f"seed {seed} RMS not frozen")
    _require(warmup.get("completed_transitions") == 50_000 and warmup.get("explained_variance_threshold") == 0.5, f"seed {seed} warmup contract drifted")
    _require(warmup.get("actor_parameters_byte_identical") is True, f"seed {seed} warmup actor changed")
    _require(warmup.get("observation_rms_before_sha256") == warmup.get("observation_rms_after_sha256") == normalization.get("observation_rms_sha256"), f"seed {seed} warmup RMS changed")
    curve = receipt.get("development_curve", {})
    expected_keys = {"bc_initialization", "post_critic_warmup"}
    if seed in COMPLETED_SEEDS:
        expected_keys |= {f"ppo_{milestone}_transitions" for milestone in MILESTONES}
    _require(set(curve) == expected_keys, f"seed {seed} curve shape drifted")
    evaluations = {
        key: _validate_development(value, f"seed {seed} {key}")
        for key, value in curve.items()
    }
    _require({row["split_identity_sha256"] for row in evaluations.values()} == {EXPECTED_DEV_IDENTITY_SHA256}, f"seed {seed} DEV identity drifted")
    bundles: list[dict[str, Any]] = []
    if seed in COMPLETED_SEEDS:
        _require(receipt.get("status") == "complete", f"seed {seed} status drifted")
        _require(warmup.get("gate_passed") is True, f"seed {seed} warmup gate drifted")
        for milestone in MILESTONES:
            bundles.append(_portable_bundle(study_root, seed, milestone, receipt["checkpoint_bundles"][str(milestone)]))
    else:
        _require(receipt.get("status") == "critic_warmup_incomplete", "stopped seed status drifted")
        _require(warmup.get("gate_passed") is False, "stopped seed gate drifted")
        _require(warmup.get("last_warmup_rollout_explained_variance") == 0.47894805669784546, "stopped seed final EV drifted")
        _require(receipt.get("transition_counts", {}).get("active_actor_critic") == 0, "stopped seed PPO unexpectedly ran")
        _require(receipt.get("checkpoint_bundles") == {} and receipt.get("milestone_states") == {}, "stopped seed unexpectedly persisted bundles")
    return {
        "bundles": bundles,
        "development_curve": evaluations,
        "policy_seed": seed,
        "receipt_path_within_external_root": f"seed-{seed}/training-receipt.json",
        "receipt_sha256": EXPECTED_RECEIPT_SHA256[seed],
        "status": receipt["status"],
        "trainer_log_sha256": EXPECTED_TRAINER_LOG_SHA256[seed],
        "transition_counts": receipt["transition_counts"],
        "warmup": {
            "actor_parameters_byte_identical": warmup["actor_parameters_byte_identical"],
            "completed_transitions": warmup["completed_transitions"],
            "explained_variance_threshold": warmup["explained_variance_threshold"],
            "first_rollout_explained_variance": warmup["first_rollout_explained_variance"],
            "gate_passed": warmup["gate_passed"],
            "iterations": [
                {
                    "explained_variance": row["explained_variance"],
                    "phase_transitions": row["phase_transitions"],
                }
                for row in warmup["iterations"]
            ],
            "last_warmup_rollout_explained_variance": warmup["last_warmup_rollout_explained_variance"],
            "observation_rms_sha256": warmup["observation_rms_after_sha256"],
        },
    }


def build_portable_evidence(study_root: Path, console_parent: Path) -> dict[str, Any]:
    """Validate the stopped external attempt and return portable evidence."""

    _require(study_root.is_absolute() and study_root.is_dir(), "study root is invalid")
    root_inventory = _inventory(study_root)
    protocol = _load(study_root / "protocol.json", "base protocol", EXPECTED_PROTOCOL_SHA256)
    ppo = _load(study_root / "ppo-protocol.json", "PPO protocol", EXPECTED_PPO_PROTOCOL_SHA256)
    base_contract = protocol.get("contract")
    ppo_contract = ppo.get("contract")
    _require(isinstance(base_contract, dict) and protocol.get("contract_sha256") == EXPECTED_BASE_CONTRACT_SHA256 == canonical_hash(base_contract), "base protocol contract drifted")
    _require(isinstance(ppo_contract, dict) and ppo.get("contract_sha256") == EXPECTED_PPO_CONTRACT_SHA256 == canonical_hash(ppo_contract), "PPO protocol contract drifted")
    _require(base_contract.get("git_commit") == ppo_contract.get("git_commit") == EXPECTED_GIT_COMMIT, "attempt git commit drifted")
    _require(base_contract.get("source_identity_sha256") == EXPECTED_SOURCE_IDENTITY_SHA256 == canonical_hash(base_contract.get("source_identity")), "attempt source identity drifted")
    _require(ppo_contract.get("base_contract_sha256") == EXPECTED_BASE_CONTRACT_SHA256, "PPO/base protocol link drifted")
    _require(ppo_contract.get("policy_seeds") == list(REGISTERED_SEEDS), "registered seed roster drifted")
    _require(
        ppo_contract.get("selection_milestones") == list(SELECTABLE_MILESTONES),
        "registered selection milestones drifted",
    )
    _require(base_contract.get("ppo_plan", {}).get("final_split_imported_or_used") is False and ppo_contract.get("final_split_imported_or_used") is False, "final boundary drifted")
    console_logs: dict[str, Any] = {}
    for label, expected in EXPECTED_CONSOLE_LOGS.items():
        path = console_parent / expected["filename"]
        _require(path.is_file() and not path.is_symlink(), f"console {label} log is missing")
        _require(file_sha256(path) == expected["sha256"] and path.stat().st_size == expected["size_bytes"], f"console {label} log drifted")
        console_logs[label] = {
            **expected,
            "text": path.read_text(encoding="utf-8"),
        }
    _require(
        "starting seed 57017" in console_logs["stdout"]["text"]
        and "finished seed 57017" not in console_logs["stdout"]["text"],
        "console stdout stop semantics drifted",
    )
    _require(
        "trainer failed for seed 57017" in console_logs["stderr"]["text"],
        "console stderr stop semantics drifted",
    )
    for value in console_logs.values():
        value.pop("text")
    source_files = base_contract["source_identity"]
    for relative_path, expected_sha256 in source_files.items():
        source_path = ROOT / relative_path
        _require(
            source_path.is_file() and file_sha256(source_path) == expected_sha256,
            f"attempt source file drifted: {relative_path}",
        )
    trainer_source = (ROOT / "scripts/train_policy.py").read_text(encoding="utf-8")
    orchestrator_source = (
        ROOT / "scripts/run_combined_distilled_large_study.py"
    ).read_text(encoding="utf-8")
    _require(
        "return 0 if training_complete else 3" in trainer_source,
        "trainer incomplete-return semantics drifted",
    )
    _require(
        "if completed.returncode != 0:" in orchestrator_source,
        "orchestrator stop semantics drifted",
    )
    fit = _fit_evidence(
        study_root,
        EXPECTED_BASE_CONTRACT_SHA256,
        base_contract["large_oracle_bc_fit"],
    )
    upstream = base_contract.get("upstream_evidence", {})
    capacity = upstream.get("capacity", {})
    distillation = upstream.get("distillation", {})
    dataset = base_contract.get("large_oracle_bc_fit", {}).get("dataset", {})
    _require(
        file_sha256(CAPACITY_EVIDENCE)
        == capacity.get("sha256")
        == EXPECTED_CAPACITY_EVIDENCE_SHA256,
        "capacity evidence drifted",
    )
    _require(
        file_sha256(DISTILLATION_EVIDENCE)
        == distillation.get("sha256")
        == EXPECTED_DISTILLATION_EVIDENCE_SHA256,
        "distillation evidence drifted",
    )
    _require(
        file_sha256(DATASET_RECEIPT)
        == dataset.get("receipt_sha256")
        == EXPECTED_DATASET_RECEIPT_SHA256,
        "training-oracle dataset receipt drifted",
    )
    capacity_receipt = _load(CAPACITY_EVIDENCE, "capacity portable evidence")
    capacity_comparison = capacity_receipt.get("comparison", {}).get(
        "large_lr_3e_5_vs_incumbent_same_seed_2m", {}
    )
    _require(
        capacity.get("large_lr_3e_5_endpoints")
        == capacity_comparison.get("challenger_solved_counts")
        == [LARGE_ONLY_ENDPOINTS[seed] for seed in REGISTERED_SEEDS],
        "large-only endpoint evidence drifted",
    )
    _require(
        capacity_comparison.get("incumbent_solved_counts")
        == [INCUMBENT_ENDPOINTS[seed] for seed in REGISTERED_SEEDS],
        "incumbent endpoint evidence drifted",
    )
    seeds = [
        _seed_evidence(
            study_root,
            ppo_contract,
            seed,
            fit_manifest_sha256=fit["checkpoint"]["manifest_sha256"],
            fit_model_sha256=fit["checkpoint"]["model_sha256"],
        )
        for seed in REGISTERED_SEEDS
    ]
    completed = [row for row in seeds if row["policy_seed"] in COMPLETED_SEEDS]
    paired = []
    for row in completed:
        seed = row["policy_seed"]
        endpoint = row["development_curve"]["ppo_2000000_transitions"]["solved_count"]
        paired.append(
            {
                "combined_endpoint": endpoint,
                "delta_vs_incumbent_endpoint": endpoint - INCUMBENT_ENDPOINTS[seed],
                "delta_vs_large_only_endpoint": endpoint - LARGE_ONLY_ENDPOINTS[seed],
                "incumbent_endpoint": INCUMBENT_ENDPOINTS[seed],
                "large_only_endpoint": LARGE_ONLY_ENDPOINTS[seed],
                "policy_seed": seed,
            }
        )
    stopped = next(row for row in seeds if row["policy_seed"] == 57_017)
    best_observed = max(
        (
            {
                "active_actor_critic_transitions": value[
                    "active_actor_critic_transitions"
                ],
                "checkpoint_id": next(
                    bundle["checkpoint_id"]
                    for bundle in row["bundles"]
                    if bundle["active_actor_critic_transitions"]
                    == value["active_actor_critic_transitions"]
                ),
                "policy_seed": row["policy_seed"],
                "solved_count": value["solved_count"],
            }
            for row in completed
            for key, value in row["development_curve"].items()
            if value["active_actor_critic_transitions"] in SELECTABLE_MILESTONES
        ),
        key=lambda value: value["solved_count"],
    )
    publication_source_files = {
        "backend/app/shared_evidence.py": file_sha256(
            ROOT / "backend/app/shared_evidence.py"
        ),
        "scripts/publish_combined_distilled_large_failure_evidence.py": file_sha256(
            ROOT / "scripts/publish_combined_distilled_large_failure_evidence.py"
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "incomplete_stopped_at_preregistered_critic_warmup_gate",
        "tool": TOOL_ID,
        "attempt": {
            "completed_policy_seeds": list(COMPLETED_SEEDS),
            "git_commit": EXPECTED_GIT_COMMIT,
            "no_retry_or_resume_performed": True,
            "registered_policy_seeds": list(REGISTERED_SEEDS),
            "stopped_policy_seed": 57_017,
        },
        "console_logs": console_logs,
        "development_evidence": {
            "best_observed_completed_checkpoint": best_observed,
            "completed_seed_endpoint_pairs_only": paired,
            "development_case_count": 200,
            "development_split_identity_sha256": EXPECTED_DEV_IDENTITY_SHA256,
            "no_three_seed_mean_or_standard_deviation": True,
            "promotion_decision": "not_evaluable_incomplete_registered_seed_roster",
            "retained_development_evaluation_count": sum(
                len(row["development_curve"]) for row in seeds
            ),
            "seeds": seeds,
        },
        "external_root_inventory": {
            "file_count": EXPECTED_ROOT_FILE_COUNT,
            "files": root_inventory,
            "inventory_sha256": EXPECTED_ROOT_INVENTORY_SHA256,
            "root_disclosure": "external_attempt_root_not_in_repository",
        },
        "fit": fit,
        "invariants": {
            "all_retained_development_hard_violation_counts_zero": all(
                value["hard_violation_count"] == 0
                for row in seeds
                for value in row["development_curve"].values()
            ),
            "all_retained_development_conservation_residuals_zero": all(
                value["maximum_conservation_residual"] == 0.0
                for row in seeds
                for value in row["development_curve"].values()
            ),
            "final_split_not_imported_or_used": True,
            "no_model_promotion_decision": True,
            "no_retry_or_resume": True,
            "stopped_seed_active_ppo_transitions_zero": stopped["transition_counts"]["active_actor_critic"] == 0,
            "stopped_seed_actor_unchanged_during_warmup": stopped["warmup"]["actor_parameters_byte_identical"],
            "stopped_seed_observation_rms_unchanged": stopped["warmup"]["observation_rms_sha256"] == fit["method"]["observation_rms_sha256"],
        },
        "limitations": {
            "comparison_is_nonfactorial": True,
            "large_only_initialization": "preparedness-teacher BC plus four DAgger iterations",
            "combined_initialization": "single-pass offline oracle BC with zero DAgger",
            "large_only_observation_rms": "independently fit per seed",
            "combined_observation_rms": "one shared frozen RMS imported from the distillation run",
            "large_only_seed_57017_warmup_transitions": 60_000,
            "combined_warmup_transitions": 50_000,
            "causal_increment_of_distillation_not_isolated": True,
            "third_seed_missing": True,
        },
        "publication_source_contract": {
            "source_files": publication_source_files,
            "source_identity_sha256": canonical_hash(publication_source_files),
        },
        "protocol": {
            "base_contract_sha256": EXPECTED_BASE_CONTRACT_SHA256,
            "base_protocol_sha256": EXPECTED_PROTOCOL_SHA256,
            "ppo_contract_sha256": EXPECTED_PPO_CONTRACT_SHA256,
            "ppo_protocol_sha256": EXPECTED_PPO_PROTOCOL_SHA256,
            "source_identity_sha256": EXPECTED_SOURCE_IDENTITY_SHA256,
            "source_files": source_files,
        },
        "upstream_evidence": {
            "capacity": {
                "large_only_endpoints": [
                    LARGE_ONLY_ENDPOINTS[seed] for seed in REGISTERED_SEEDS
                ],
                "portable_path": "internal/developmental_runs/v4/network-capacity-study-200.json",
                "sha256": EXPECTED_CAPACITY_EVIDENCE_SHA256,
            },
            "distillation": {
                "endpoint_solved_counts": distillation["endpoint_solved_counts"],
                "portable_path": "internal/developmental_runs/v4/oracle-distilled-ppo-study-200.json",
                "sha256": EXPECTED_DISTILLATION_EVIDENCE_SHA256,
                "source_student": {
                    "checkpoint_manifest_sha256": distillation["source_student"][
                        "checkpoint_manifest_sha256"
                    ],
                    "checkpoint_model_sha256": distillation["source_student"][
                        "checkpoint_model_sha256"
                    ],
                    "checkpoint_normalization_sha256": distillation[
                        "source_student"
                    ]["checkpoint_normalization_sha256"],
                    "receipt_sha256": distillation["source_student"][
                        "receipt_sha256"
                    ],
                },
            },
            "incumbent_same_seed_endpoints": [
                INCUMBENT_ENDPOINTS[seed] for seed in REGISTERED_SEEDS
            ],
            "training_oracle_dataset": {
                "external_path_disclosed_by_hash_only": True,
                "sha256": EXPECTED_DATASET_RECEIPT_SHA256,
            },
        },
        "split": "dev",
        "stop_provenance": {
            "expected_trainer_return_code": 3,
            "orchestrator_action": "stop_on_nonzero_trainer_return_code",
            "strict_gate_operator": ">",
            "worker_status": "critic_warmup_incomplete",
        },
        "final_split_imported_or_used": False,
    }


def render_markdown(receipt: Mapping[str, Any], receipt_sha256: str) -> str:
    """Render the human report from portable incomplete evidence."""

    seeds = {row["policy_seed"]: row for row in receipt["development_evidence"]["seeds"]}
    lines = [
        "# Combined large-network + oracle-distillation attempt — incomplete",
        "",
        "This development-only attempt is **not a completed three-seed study and not a promotion candidate**. It stopped at the preregistered actor-frozen critic warm-up verification gate for seed `57017`; it was not retried or resumed, and no final case was imported or evaluated.",
        "",
        "## What completed",
        "",
        "| Seed | BC init | Post-warm-up | 200k | 500k | 1M | 2M | Status |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for seed in COMPLETED_SEEDS:
        curve = seeds[seed]["development_curve"]
        lines.append(
            f"| {seed} | {curve['bc_initialization']['solved_count']}/200 | "
            f"{curve['post_critic_warmup']['solved_count']}/200 | "
            f"{curve['ppo_200000_transitions']['solved_count']}/200 | "
            f"{curve['ppo_500000_transitions']['solved_count']}/200 | "
            f"{curve['ppo_1000000_transitions']['solved_count']}/200 | "
            f"{curve['ppo_2000000_transitions']['solved_count']}/200 | complete |"
        )
    lines.extend(
        [
            "",
            "Only these two registered curves completed. The best observed registered selectable checkpoint was seed `47017` at 2M with `174/200`; because the attempt is incomplete, that observation is not a valid study summary or promotion candidate. Their 2M paired deltas were `-2` and `+3` versus the same-seed incumbent endpoints, and `-8` and `-2` versus the same-seed large-network-only endpoints. Because seed `57017` did not reach PPO, this report does **not** calculate a two-seed substitute for the preregistered three-seed mean, standard deviation, or promotion decision.",
            "",
            "### Completed 2M endpoints by development family",
            "",
            "| Seed | River flood | Industrial outage | Logistics strike | Seismic cluster | Health compound |",
            "|---:|---:|---:|---:|---:|---:|",
            "| 37017 | 33/40 | 35/40 | 40/40 | 27/40 | 35/40 |",
            "| 47017 | 34/40 | 36/40 | 40/40 | 28/40 | 36/40 |",
            "",
            "Across all 14 retained development evaluations, hard violations were `0` and maximum conservation residual was exactly `0.0`.",
            "",
            "## Why the attempt stopped",
            "",
            "Seed `57017` completed exactly 50,000 fixed critic-warm-up transitions. The gate requires the final warm-up rollout explained variance to be strictly above `0.5`; the final value was `0.47894805669784546`, so the gate failed. Intermediate iterations above `0.5` do not override the registered final-rollout check. Active PPO transitions remained `0`, the actor stayed byte-identical, the frozen observation RMS stayed unchanged, and its retained BC/post-warm-up development result was `153/200`.",
            "",
            "This was an intentional verification-gate termination, not a process crash: the trainer returns code `3` when `training_complete` is false, and the orchestrator treats any nonzero trainer return code as a stop. The console therefore records the orchestrator's failure message after preserving the incomplete worker receipt.",
            "",
            "## Large offline fit",
            "",
            "The 768/512/256 actor used one offline BC stage over a fixed dataset: 5,040 fit observations, a 720-observation trajectory-level holdout, 15 epochs, zero DAgger or interactive relabeling, and the frozen source observation RMS. Only the privileged teacher labels saw future tape; the student consumed 73 causal public inputs and produced 22 actions.",
            "",
            "| Target | Fit MSE / MAE | Held-out MSE / MAE | Held-out relative MSE improvement |",
            "|---|---:|---:|---:|",
            "| Oracle labels | 0.0382702537 / 0.1383763850 | 0.0410502814 / 0.1448733062 | 0.8704107290 |",
            "| Matched hand-rule control | 0.0161156859 / 0.0766169503 | 0.0217285659 / 0.0857965201 | 0.9525681396 |",
            "",
            "## Interpretation boundary",
            "",
            "The treatment is nonfactorial: relative to large-network-only evidence, it changes initialization from preparedness-teacher BC plus four DAgger rounds to single-pass offline oracle BC, changes per-seed observation normalization to one shared frozen RMS, and differs in seed `57017` warm-up budget (60k in the historical large-only run versus 50k here). The partial curves cannot isolate a causal distillation effect. The stopped attempt was not retried, produced no three-seed summary or promotion result, used no final case, and did not alter the shipped artifact.",
            "",
            "## Evidence",
            "",
            f"- Portable incomplete receipt SHA-256: `{receipt_sha256}`",
            f"- External 37-file root inventory SHA-256: `{receipt['external_root_inventory']['inventory_sha256']}`",
            f"- Base protocol SHA-256: `{receipt['protocol']['base_protocol_sha256']}`",
            f"- PPO protocol SHA-256: `{receipt['protocol']['ppo_protocol_sha256']}`",
            f"- Console stdout SHA-256: `{receipt['console_logs']['stdout']['sha256']}`",
            f"- Console stderr SHA-256: `{receipt['console_logs']['stderr']['sha256']}`",
            "",
        ]
    )
    return "\n".join(lines)


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _write(path: Path, payload: bytes, *, check: bool) -> None:
    if check:
        _require(path.is_file() and path.read_bytes() == payload, f"stale output: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _reject_absolute_paths(value: Any, label: str = "receipt") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_absolute_paths(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_absolute_paths(item, f"{label}[{index}]")
    elif isinstance(value, str):
        _require(_ABSOLUTE_PATH.search(value) is None, f"absolute path leaked at {label}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    parser.add_argument("--console-parent", type=Path, default=DEFAULT_STUDY_ROOT.parent)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    receipt = build_portable_evidence(args.study_root, args.console_parent)
    _reject_absolute_paths(receipt)
    receipt_bytes = _canonical_json_bytes(receipt)
    receipt_sha256 = __import__("hashlib").sha256(receipt_bytes).hexdigest()
    report = render_markdown(receipt, receipt_sha256).encode("utf-8")
    _write(args.output, receipt_bytes, check=args.check)
    _write(args.report, report, check=args.check)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
