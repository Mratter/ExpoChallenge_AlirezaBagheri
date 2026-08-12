#!/usr/bin/env python3
"""Evaluate the five preregistered 2M-seed actors as a DEV-only ensemble.

Every member receives the same raw observation through its own frozen
VecNormalize transform. The deterministic, clipped 22-action vectors are then
averaged with equal weight. This utility exposes no split selector and imports
only the canonical development roster; it cannot evaluate the final split.
"""

from __future__ import annotations

import argparse
import platform
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Sequence

import gymnasium
import numpy as np
import stable_baselines3
import torch
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.city.scenarios import (  # noqa: E402
    DEVELOPMENT_FAMILIES,
    DEVELOPMENT_SEEDS,
)
from backend.app.shared_evidence import (  # noqa: E402
    canonical_hash,
    file_sha256,
    load_json_object,
    split_contract,
    wilson_interval,
)
from scripts.export_policy import (  # noqa: E402
    ACTION_COUNT,
    EXPECTED_DEVELOPMENT_CASES,
    ObservationNormalization,
    _rollout as rollout_development_case,
    development_cases,
    load_observation_normalization,
    sb3_actions,
    write_new_json,
)
from scripts.training_artifacts import (  # noqa: E402
    checkpoint_bundle_reference,
    load_checkpoint_bundle,
)

TOOL_ID = "evaluate_seed_ensemble.py"
SCHEMA_VERSION = "city-recovery-development-action-mean-ensemble-v1"
ARM_NAME = "adopted_v3_equivalent_2m"
REGISTERED_POLICY_SEEDS = (37017, 47017, 57017, 67017, 77017)
ACTIVE_TRANSITIONS = 2_000_000
MILESTONE_DIRECTORY = "ppo-2000000"
DEFAULT_RECEIPT = (
    ROOT
    / "internal"
    / "developmental_runs"
    / "v4"
    / "action-mean-ensemble-5x2m-dev-200.json"
)
SELECTION_RECEIPT = (
    ROOT / "internal" / "developmental_runs" / "v4" / "checkpoint-selection-200.json"
)
SHIPPED_PARITY_RECEIPT = (
    ROOT
    / "internal"
    / "developmental_runs"
    / "v4"
    / "city_recovery_ppo.v4.parity.json"
)
SHIPPED_PARITY_SHA256 = (
    "e3b487df8221db75d58dc68eccbc9df93af16cb0e9f17b5bc60cf50a5b42ba6c"
)
SHIPPED_PARITY_ROWS_SHA256 = (
    "ca9320566b86dfb7a02d2cb9232c7a28c80f08dbbd700dffc6d2af9af1c22d6b"
)


class EnsembleError(RuntimeError):
    """Raised when the DEV-only ensemble contract cannot be proven."""


@dataclass(frozen=True, slots=True)
class EnsembleMember:
    """One strongly verified PPO endpoint and its own observation transform."""

    seed: int
    model: PPO
    normalization: ObservationNormalization
    identity: dict[str, Any]


class ActionMeanActor:
    """Average five independently normalized deterministic actor actions."""

    def __init__(self, members: Sequence[EnsembleMember]) -> None:
        if tuple(member.seed for member in members) != REGISTERED_POLICY_SEEDS:
            raise EnsembleError(
                "ensemble members must use the five registered seeds in order"
            )
        self.members = tuple(members)
        self.call_count = 0
        self.component_count = 0
        self.maximum_member_span = 0.0
        self._member_span_sum = 0.0

    def __call__(self, raw_observations: np.ndarray) -> np.ndarray:
        member_actions = np.stack(
            [
                sb3_actions(member.model, member.normalization, raw_observations)
                for member in self.members
            ],
            axis=0,
        ).astype(np.float64)
        if member_actions.ndim != 3 or member_actions.shape[2] != ACTION_COUNT:
            raise EnsembleError("member action tensor contract drifted")
        if not np.all(np.isfinite(member_actions)) or np.any(
            np.abs(member_actions) > 1.0
        ):
            raise EnsembleError("member actions must be finite and bounded")
        spans = np.max(member_actions, axis=0) - np.min(member_actions, axis=0)
        self.call_count += int(member_actions.shape[1])
        self.component_count += int(spans.size)
        self.maximum_member_span = max(
            self.maximum_member_span,
            float(np.max(spans)),
        )
        self._member_span_sum += float(np.sum(spans, dtype=np.float64))
        averaged = np.mean(member_actions, axis=0, dtype=np.float64)
        return np.clip(averaged, -1.0, 1.0).astype(np.float32)

    def disagreement_receipt(self) -> dict[str, Any]:
        """Describe member dispersion over every evaluated action component."""

        return {
            "raw_observation_count": self.call_count,
            "action_component_count": self.component_count,
            "maximum_member_action_span": self.maximum_member_span,
            "mean_member_action_span": (
                self._member_span_sum / self.component_count
                if self.component_count
                else 0.0
            ),
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "gymnasium": gymnasium.__version__,
        "operating_system": platform.platform(),
    }


def _load_seed_sweep_summary(study_root: Path) -> dict[str, Any]:
    path = study_root / "seed-sweep-summary.json"
    summary = load_json_object(path, "seed sweep summary", error_type=EnsembleError)
    baseline = summary.get("baseline")
    rows = summary.get("rows")
    if (
        summary.get("tool") != "run_training_study.py"
        or summary.get("phase") != "seed_sweep"
        or summary.get("split") != "dev"
        or summary.get("final_split_used") is not False
        or summary.get("development_case_count") != EXPECTED_DEVELOPMENT_CASES
        or summary.get("registered_policy_seeds") != list(REGISTERED_POLICY_SEEDS)
        or not isinstance(baseline, dict)
        or baseline.get("name") != ARM_NAME
        or baseline.get("transitions") != ACTIVE_TRANSITIONS
        or baseline.get("reward_profile") != "v3_equivalent"
        or baseline.get("bc_warm_start") is not True
        or baseline.get("vec_normalize") is not True
        or not isinstance(rows, list)
        or [row.get("seed") for row in rows if isinstance(row, dict)]
        != list(REGISTERED_POLICY_SEEDS)
    ):
        raise EnsembleError(
            "seed sweep summary does not identify the registered DEV endpoints"
        )
    return {
        "path": _portable_path(path),
        "sha256": file_sha256(
            path, label="seed sweep summary", error_type=EnsembleError
        ),
        "endpoint_solved_counts": {
            str(row["seed"]): row["solved_count"] for row in rows
        },
    }


def _load_member(study_root: Path, seed: int) -> EnsembleMember:
    run_root = study_root / ARM_NAME / f"seed-{seed}"
    bundle_root = run_root / "checkpoints" / MILESTONE_DIRECTORY
    training_receipt_path = run_root / "training-receipt.json"
    loaded = load_checkpoint_bundle(bundle_root, device="cpu")
    reference = checkpoint_bundle_reference(loaded.bundle)
    training_receipt = load_json_object(
        training_receipt_path,
        f"seed {seed} training receipt",
        error_type=EnsembleError,
    )
    config = training_receipt.get("config")
    bundles = training_receipt.get("checkpoint_bundles")
    endpoint = training_receipt.get("development")
    manifest_training = loaded.manifest.get("training")
    if (
        training_receipt.get("status") != "complete"
        or training_receipt.get("training_split") != "train"
        or training_receipt.get("evaluation_split") != "dev"
        or training_receipt.get("final_split_used") is not False
        or not isinstance(config, dict)
        or config.get("policy_seed") != seed
        or config.get("active_actor_critic_transitions") != ACTIVE_TRANSITIONS
        or config.get("reward_profile") != "v3_equivalent"
        or config.get("bc_warm_start") is not True
        or config.get("vec_normalize") is not True
        or config.get("freeze_observation_rms") is not True
        or not isinstance(bundles, dict)
        or canonical_hash(bundles.get(str(ACTIVE_TRANSITIONS)))
        != canonical_hash(reference)
        or not isinstance(endpoint, dict)
        or endpoint.get("case_count") != EXPECTED_DEVELOPMENT_CASES
        or not isinstance(manifest_training, dict)
        or manifest_training.get("seed") != seed
        or manifest_training.get("milestone") != ACTIVE_TRANSITIONS
        or manifest_training.get("config") != config
    ):
        raise EnsembleError(f"seed {seed} training and bundle provenance disagree")

    normalization = load_observation_normalization(loaded.bundle.normalization_path)
    if normalization.state_sha256 != reference["obs_rms_sha256"]:
        raise EnsembleError(f"seed {seed} observation RMS identity mismatch")
    model = loaded.model
    model.policy.to("cpu")
    model.policy.set_training_mode(False)
    return EnsembleMember(
        seed=seed,
        model=model,
        normalization=normalization,
        identity={
            "seed": seed,
            "weight": 1.0 / len(REGISTERED_POLICY_SEEDS),
            "checkpoint_bundle": reference,
            "training_config_sha256": manifest_training["config_sha256"],
            "training_config": config,
            "training_receipt": {
                "path": _portable_path(training_receipt_path),
                "sha256": file_sha256(
                    training_receipt_path,
                    label=f"seed {seed} training receipt",
                    error_type=EnsembleError,
                ),
            },
            "individual_endpoint_development": {
                "case_count": endpoint["case_count"],
                "solved_count": endpoint["solved_count"],
                "solve_rate": endpoint["solve_rate"],
                "hard_violation_count": endpoint["hard_violation_count"],
                "maximum_conservation_residual": endpoint[
                    "maximum_conservation_residual"
                ],
            },
        },
    )


def load_ensemble_members(
    study_root: Path,
) -> tuple[list[EnsembleMember], dict[str, Any]]:
    """Strongly load the five registered 2M endpoints in seed order."""

    root = study_root.expanduser().resolve(strict=True)
    summary = _load_seed_sweep_summary(root)
    members = [_load_member(root, seed) for seed in REGISTERED_POLICY_SEEDS]
    common_configs = [
        {
            key: value
            for key, value in member.identity["training_config"].items()
            if key != "policy_seed"
        }
        for member in members
    ]
    if len({canonical_hash(config) for config in common_configs}) != 1:
        raise EnsembleError(
            "ensemble member training configs differ beyond policy seed"
        )
    summary["common_training_config_sha256_excluding_policy_seed"] = canonical_hash(
        common_configs[0]
    )
    return members, summary


def _load_selected_checkpoint_reference() -> dict[str, Any]:
    receipt = load_json_object(
        SELECTION_RECEIPT,
        "development checkpoint selection receipt",
        error_type=EnsembleError,
    )
    winner = receipt.get("winner")
    selected = receipt.get("selected_checkpoint")
    if (
        receipt.get("split") != "dev"
        or receipt.get("final_split_used") is not False
        or receipt.get("development_case_count") != EXPECTED_DEVELOPMENT_CASES
        or not isinstance(winner, dict)
        or winner.get("solved_count") != 178
        or not isinstance(selected, dict)
    ):
        raise EnsembleError("selected-checkpoint DEV reference contract drifted")
    parity = load_json_object(
        SHIPPED_PARITY_RECEIPT,
        "accepted shipped-policy development parity receipt",
        error_type=EnsembleError,
    )
    parity_block = parity.get("parity")
    parity_rows = (
        parity_block.get("rows") if isinstance(parity_block, dict) else None
    )
    if (
        file_sha256(
            SHIPPED_PARITY_RECEIPT,
            label="accepted shipped-policy development parity receipt",
            error_type=EnsembleError,
        )
        != SHIPPED_PARITY_SHA256
        or not isinstance(parity_block, dict)
        or parity_block.get("passed") is not True
        or parity.get("split") != "dev"
        or parity.get("final_split_used") is not False
        or parity.get("development_case_count") != EXPECTED_DEVELOPMENT_CASES
        or parity_block.get("onnx_solved_count") != 178
        or not isinstance(parity_rows, list)
        or len(parity_rows) != EXPECTED_DEVELOPMENT_CASES
        or parity_block.get("rows_sha256") != SHIPPED_PARITY_ROWS_SHA256
        or canonical_hash(parity_rows) != SHIPPED_PARITY_ROWS_SHA256
    ):
        raise EnsembleError("accepted shipped-policy DEV parity contract drifted")
    return {
        "receipt_path": _portable_path(SELECTION_RECEIPT),
        "receipt_sha256": file_sha256(
            SELECTION_RECEIPT,
            label="development checkpoint selection receipt",
            error_type=EnsembleError,
        ),
        "checkpoint_id": selected.get("id"),
        "solved_count": winner["solved_count"],
        "solve_rate": winner["solve_rate"],
        "parity_receipt_path": _portable_path(SHIPPED_PARITY_RECEIPT),
        "parity_receipt_sha256": SHIPPED_PARITY_SHA256,
        "parity_rows_sha256": SHIPPED_PARITY_ROWS_SHA256,
        "parity_rows": parity_rows,
    }


def evaluate_ensemble(members: Sequence[EnsembleMember]) -> dict[str, Any]:
    """Run one complete deterministic evaluation on the canonical DEV roster."""

    cases = development_cases()
    if len(cases) != EXPECTED_DEVELOPMENT_CASES:
        raise EnsembleError("canonical development roster must contain 200 cases")
    actor = ActionMeanActor(members)
    rows = [
        rollout_development_case(case, actor, label="five_seed_action_mean").row
        for case in cases
    ]
    expected_ids = [case.row_id for case in cases]
    if [row["row_id"] for row in rows] != expected_ids or len(set(expected_ids)) != len(
        expected_ids
    ):
        raise EnsembleError("development row order or identity drifted")
    solved_count = sum(bool(row["solved"]) for row in rows)
    hard_violations = sum(int(row["hard_violation_count"]) for row in rows)
    maximum_residual = max(float(row["maximum_conservation_residual"]) for row in rows)
    if hard_violations != 0 or maximum_residual != 0.0:
        raise EnsembleError("ensemble development rollout violated city invariants")
    reasons = Counter(
        reason for row in rows if not row["solved"] for reason in row["reason_codes"]
    )
    per_family = []
    for family in DEVELOPMENT_FAMILIES:
        family_rows = [row for row in rows if row["family_id"] == family.id]
        per_family.append(
            {
                "family_id": family.id,
                "case_count": len(family_rows),
                "solved_count": sum(bool(row["solved"]) for row in family_rows),
                "solve_rate": sum(bool(row["solved"]) for row in family_rows)
                / len(family_rows),
            }
        )
    return {
        "case_count": len(rows),
        "solved_count": solved_count,
        "solve_rate": solved_count / len(rows),
        "wilson_95": wilson_interval(solved_count, len(rows), digits=10),
        "mean_resilience_auc": round(
            fmean(float(row["resilience_auc"]) for row in rows), 10
        ),
        "hard_violation_count": hard_violations,
        "maximum_conservation_residual": maximum_residual,
        "failure_reason_code_histogram": dict(sorted(reasons.items())),
        "per_family": per_family,
        "member_disagreement": actor.disagreement_receipt(),
        "ordered_row_ids_sha256": canonical_hash(expected_ids),
        "ordered_tapes_sha256": canonical_hash([row["tape_sha256"] for row in rows]),
        "rows_sha256": canonical_hash(rows),
        "rows": rows,
    }


def build_receipt(
    members: Sequence[EnsembleMember],
    seed_sweep: dict[str, Any],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    """Bind ensemble results to member, roster, source, and comparison identities."""

    selected = _load_selected_checkpoint_reference()
    selected_rows = selected.pop("parity_rows")
    ensemble_rows = evaluation["rows"]
    if [row["row_id"] for row in ensemble_rows] != [
        row["row_id"] for row in selected_rows
    ]:
        raise EnsembleError("ensemble and selected policy DEV rows are not aligned")
    both = [
        left["row_id"]
        for left, right in zip(ensemble_rows, selected_rows, strict=True)
        if left["solved"] and right["onnx"]["solved"]
    ]
    ensemble_only = [
        left["row_id"]
        for left, right in zip(ensemble_rows, selected_rows, strict=True)
        if left["solved"] and not right["onnx"]["solved"]
    ]
    selected_only = [
        left["row_id"]
        for left, right in zip(ensemble_rows, selected_rows, strict=True)
        if not left["solved"] and right["onnx"]["solved"]
    ]
    neither = [
        left["row_id"]
        for left, right in zip(ensemble_rows, selected_rows, strict=True)
        if not left["solved"] and not right["onnx"]["solved"]
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL_ID,
        "status": "complete_development_experiment",
        "created_at_utc": _utc_now(),
        "split": "dev",
        "final_split_used": False,
        "split_contract": split_contract(
            "dev", DEVELOPMENT_FAMILIES, DEVELOPMENT_SEEDS
        ),
        "ensemble": {
            "name": "five_seed_2m_action_mean",
            "member_count": len(members),
            "registered_policy_seeds": list(REGISTERED_POLICY_SEEDS),
            "active_actor_critic_transitions_per_member": ACTIVE_TRANSITIONS,
            "combination": (
                "For each raw observation, apply each member's own frozen "
                "VecNormalize observation transform, obtain its deterministic "
                "clipped float32 action, accumulate the arithmetic mean in "
                "float64 with equal 0.2 weights, clip to [-1, 1], and cast to "
                "float32 before environment.step."
            ),
            "members": [member.identity for member in members],
            "members_identity_sha256": canonical_hash(
                [member.identity for member in members]
            ),
        },
        "source_seed_sweep": seed_sweep,
        "evaluation": evaluation,
        "comparison_to_selected_single_checkpoint": {
            **selected,
            "ensemble_minus_selected_solved_cases": (
                evaluation["solved_count"] - selected["solved_count"]
            ),
            "purpose": (
                "Development-only diagnostic against the 178/200 checkpoint "
                "selected from 20 development candidates."
            ),
            "matched_contingency": {
                "both_solved": len(both),
                "ensemble_only": len(ensemble_only),
                "selected_only": len(selected_only),
                "neither_solved": len(neither),
                "ensemble_only_row_ids": ensemble_only,
                "selected_only_row_ids": selected_only,
            },
        },
        "deployment_status": {
            "shipped_artifact_changed": False,
            "application_wiring_changed": False,
            "exported": False,
            "served": False,
            "statement": (
                "This DEV result is not interchangeable with the shipped single-actor "
                "ONNX policy. Deployment would require a self-contained five-actor "
                "export with all five frozen normalizers, SB3-to-ONNX action parity, "
                "full DEV outcome parity, a new manifest, and explicit app wiring."
            ),
        },
        "decision": {
            "candidate_kept": evaluation["solved_count"] > selected["solved_count"],
            "candidate_promoted": False,
            "configuration_search_performed": False,
            "candidate_count": 1,
            "reason": (
                "Keep as an exploratory development candidate only when the "
                "single preregistered equal-weight ensemble exceeds 178/200."
            ),
        },
        "runtime_versions": _runtime_versions(),
        "source_files": {
            path: file_sha256(
                ROOT / path,
                label=path,
                error_type=EnsembleError,
            )
            for path in (
                "scripts/evaluate_seed_ensemble.py",
                "scripts/export_policy.py",
                "scripts/training_artifacts.py",
                "backend/app/city/environment.py",
                "backend/app/city/outcome.py",
                "backend/app/city/scenarios.py",
            )
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study-root",
        type=Path,
        required=True,
        help="external root containing the preregistered five-seed study",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if DEFAULT_RECEIPT.exists():
        raise EnsembleError(f"refusing to overwrite DEV receipt: {DEFAULT_RECEIPT}")
    members, seed_sweep = load_ensemble_members(args.study_root)
    evaluation = evaluate_ensemble(members)
    receipt = build_receipt(members, seed_sweep, evaluation)
    write_new_json(DEFAULT_RECEIPT, receipt)
    print(
        "development ensemble: "
        f"{evaluation['solved_count']}/{evaluation['case_count']} "
        f"({evaluation['solve_rate']:.4f}); "
        f"hard violations={evaluation['hard_violation_count']}; "
        f"max conservation residual={evaluation['maximum_conservation_residual']:.1f}"
    )
    print(f"receipt: {_portable_path(DEFAULT_RECEIPT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
