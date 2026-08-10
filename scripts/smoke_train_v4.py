#!/usr/bin/env python3
"""Run nonauthorizing v4 PPO learning and matched reward gates.

Only authored training cases update either policy. Deterministic evaluation uses
the 40-case development split. This tool never imports final split constants,
authorizes evaluation, selects a checkpoint, or writes deployable artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from math import comb
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Sequence

# Apply native-library caps before NumPy, Torch, or SB3 is imported. Spawned
# workers import this module afresh on Windows.
for _thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import numpy as np  # noqa: E402
import torch  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.running_mean_std import RunningMeanStd  # noqa: E402
from stable_baselines3.common.utils import set_random_seed  # noqa: E402
from stable_baselines3.common.vec_env import (  # noqa: E402
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
    VecNormalize,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.scenarios_v3 import (  # noqa: E402
    DEVELOPMENT_FAMILIES_V3,
    DEVELOPMENT_SEEDS_V3,
    TRAINING_FAMILIES_V3,
    TRAINING_SEEDS_V3,
)
from backend.app.simulator_v3 import (  # noqa: E402
    _summarize_v3,
    generate_disaster_tape_v3,
    public_preparedness_curriculum_action_v3,
)
from backend.app.simulator_v4 import (  # noqa: E402
    ACTION_ORDER_V4,
    OBSERVATION_ORDER_V4,
    CityRecoveryEnvV4,
    CyclingScenarioEnvV4,
    REWARD_PROFILES_V4,
)

DEFAULT_TRANSITIONS = 200_000
DEFAULT_LANES = 20
DEFAULT_N_STEPS = 250
DEFAULT_BATCH_SIZE = 500
DEFAULT_POLICY_SEED = 37_017
PROFILE_ORDER = ("v3_equivalent", "risk_averse")
DAGGER_BETA_SCHEDULE = (1.0, 0.0, 0.0, 0.0)
CRITIC_EXPLAINED_VARIANCE_GATE = 0.5
CRITIC_WARMUP_MAX_TRANSITIONS = 100_000
ACTIVE_MILESTONES = (50_000, 100_000, 200_000)
STEP3E_ACTIVE_TRANSITIONS = 1_000_000
STEP3E_ACTIVE_MILESTONES = (200_000, 500_000, STEP3E_ACTIVE_TRANSITIONS)
STEP3E_CRITIC_WARMUP_TRANSITIONS = 50_000
STEP3E_CRITIC_WARMUP_MAX_TRANSITIONS = 100_000
STEP3E_LEARNING_RATE = 7.5e-5
STEP3E_TARGET_KL = 0.02
STEP3E_ENT_COEF = 0.003
STEP3E_BC_EPOCHS = 15
STEP3E_HEADROOM_RECEIPT = (
    ROOT / "internal" / "developmental_runs" / "v4" / "headroom-probe-v4-dev.json"
)
STEP3E_HEADROOM_SHA256 = (
    "f037c98d8fec483dfa6b5c9c1691842597a4163c7d1ee6f3e72618f987d671b9"
)
STEP3E_OPTIMIZER_RECEIPT = (
    ROOT
    / "internal"
    / "developmental_runs"
    / "v4"
    / "ppo-learning-gate-200k-seed-37017-attempt-06.json"
)
STEP3E_OPTIMIZER_SHA256 = (
    "9011d6254cea90a2b30174d24c0cc3180b536b79251797ea642475f184799751"
)
STEP3E_SUMMARY_RECEIPT = (
    ROOT
    / "internal"
    / "developmental_runs"
    / "v4"
    / "ppo-learning-gate-summary-seed-37017.json"
)
STEP3E_SUMMARY_SHA256 = (
    "f3069e9c8196145fafffab271cd8afc5baa8de3ccd738f8f0bb97c5e6ccdf975"
)
STEP3E_BC_DATASET_SHA256 = (
    "ff9ddb39e17a3aad58acdda752599d4b96cd0372c4682e728447af2151ab8bf4"
)
STEP3E_INITIAL_POLICY_SHA256 = (
    "999d9b4b3017d58e105021fbe4bf2723c0c3b559fa653368f25c239dd688e0bc"
)
STEP3E_INITIAL_ACTOR_SHA256 = (
    "c82456e8b69fffb0a2e771a80c8c434590374a1c7d057d5fd44c183a3fbd292e"
)
STEP3E_INITIAL_OBSERVATION_RMS_SHA256 = (
    "456c8fab41d53a8d1ecc23fdf461cc9df5642726cff0f84f5bb2f94643876835"
)
STEP3E_INITIAL_RETURN_RMS_SHA256 = (
    "3fd67bc5b3ce298e0b68d7d0b1010fbc4edd62fa59f35862dd4c0a970d6cbae9"
)
STEP3E_ATTEMPT06_200K_ROWS_SHA256 = (
    "d950d9037f929b5ef2a8517fe2f343d16e1d37044d40734480224e41ea82149f"
)
STEP3E_ATTEMPT06_BC_ROWS_SHA256 = (
    "5094dbcafc87fae9dbcc92916a09e503866f131af631ba083ca979fbb59203ed"
)
STEP3E_ATTEMPT06_POLICY_200K_SHA256 = (
    "86aac502ca13a31beff80a9bd8529b330b68c0970273f92eee2a7b6c4d5f0919"
)
STEP3E_ATTEMPT06_RETURN_RMS_200K_SHA256 = (
    "5a29ff72a6069e63b93cda625b37db91c847346373d3124b9f13495688f43ace"
)
STEP3E_TRAINING_ROSTER_AND_TAPES_SHA256 = (
    "fa088279d023d70b13871a98a68f12af016ede6f2e8bc4618ef7c223d75ce74e"
)
STEP3E_INCOMPLETE_ATTEMPT_RECEIPT = (
    ROOT
    / "internal"
    / "developmental_runs"
    / "v4"
    / "step3e-matched-reward-1m-seed-37017.json"
)
STEP3E_INCOMPLETE_ATTEMPT_SHA256 = (
    "7c7afbefadeb8f5d264c64ba5abd14c5c07902041c6fa86833cc60ea24bee514"
)
STEP3E_CONTESTED_ROW_IDS = (
    "v3_dev_industrial_outage:820000",
    "v3_dev_river_flood:820007",
    "v3_dev_seismic_cluster:820001",
    "v3_dev_seismic_cluster:820004",
)
PROTECTED_V3_DIRECTORY_ROOTS = (
    "artifacts",
    "training/v3",
    "internal/training_runs/v3",
)
PROTECTED_V3_EXTRA_FILES_SHA256 = {
    "artifacts/city_recovery_ppo.v3.metadata.json": (
        "a7a5a8a549f05febee0906dc45cc9d73109ab25d9f768167bf7f050c0494c895"
    ),
    "artifacts/city_recovery_ppo.v3.onnx": (
        "830dae7c6e0a758671019b8c30a29f24e24b085538c21ae3da8f3363a8fb9f38"
    ),
    "artifacts/city_recovery_ppo.v3.zip": (
        "93e2f72d898b5739422501256c5c390ae11ea22c94b126e712e9da7374b64308"
    ),
    "artifacts/model_manifest.v3.json": (
        "140b0177ab6abc2a5fc2ee6901a9577c33a0b5a3ab98e944b91cc0e9f0ed84cd"
    ),
}
STEP3E_SUPERVISOR_GATE_CORRECTION = (
    "The original compound Step 3d gate was not passed. The supervisor corrected "
    "two mis-specified criteria: requiring +2 solves after 200k transitions tested "
    "only 2.5% of the intended 8M budget, and treating every recorded approximate-"
    "KL maximum above 1.5 times target_kl as instability misread Stable-Baselines3 "
    "early-stop semantics. SB3 checks each minibatch before applying the offending "
    "update, then breaks; its logger reports the mean of the final partial epoch. "
    "A recorded maximum above the target is diagnostic, not a separate failure. "
    "Based on attempt 06's "
    "critic explained variance, byte-identical frozen actor, +1 solve, improved "
    "tail margin, target-KL early stops, and exact physics invariants, plus Step "
    "3.5's measured headroom, the supervisor declared corrected Step 3d passed "
    "and explicitly authorized this Step 3e comparison."
)
REPORTED_APPROX_KL_STABILITY_MULTIPLIER = 1.5
REPORTED_APPROX_KL_STABILITY_DEFINITION = (
    "maximum recorded per-iteration SB3 approximate KL must be <= "
    "1.5 * configured target_kl"
)


class SmokeError(RuntimeError):
    """Raised when the matched smoke contract cannot be honored."""


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    rendered = json.dumps(
        value,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(rendered).hexdigest()


def historical_step3_rows_hash(rows: Sequence[dict[str, Any]]) -> str:
    projected = [
        {
            key: value
            for key, value in row.items()
            if key
            not in {
                "minimum_tail_margin",
                "case_seed",
                "tape_seed",
                "tape_sha256",
            }
        }
        for row in rows
    ]
    return canonical_hash(projected)


def _receipt_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(resolved)


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SmokeError(f"cannot load {label}: {path}") from exc
    if not isinstance(value, dict):
        raise SmokeError(f"{label} must contain a JSON object")
    return value


def validate_step3e_provenance(
    headroom_path: Path,
    optimizer_path: Path,
    summary_path: Path = STEP3E_SUMMARY_RECEIPT,
    incomplete_attempt_path: Path = STEP3E_INCOMPLETE_ATTEMPT_RECEIPT,
) -> dict[str, Any]:
    """Bind Step 3e to the authorized dev-only headroom and attempt-06 evidence."""
    headroom_path = headroom_path.resolve()
    optimizer_path = optimizer_path.resolve()
    summary_path = summary_path.resolve()
    incomplete_attempt_path = incomplete_attempt_path.resolve()
    headroom_hash = file_sha256(headroom_path)
    optimizer_hash = file_sha256(optimizer_path)
    summary_hash = file_sha256(summary_path)
    incomplete_attempt_hash = file_sha256(incomplete_attempt_path)
    if headroom_hash != STEP3E_HEADROOM_SHA256:
        raise SmokeError("Step 3.5 headroom receipt hash mismatch")
    if optimizer_hash != STEP3E_OPTIMIZER_SHA256:
        raise SmokeError("Step 3d attempt-06 receipt hash mismatch")
    if summary_hash != STEP3E_SUMMARY_SHA256:
        raise SmokeError("Step 3d summary receipt hash mismatch")
    if incomplete_attempt_hash != STEP3E_INCOMPLETE_ATTEMPT_SHA256:
        raise SmokeError("incomplete Step 3e attempt receipt hash mismatch")

    summary = _load_json_object(summary_path, "Step 3d summary receipt")
    if (
        summary.get("status")
        != "developmental_ppo_learning_gate_failed_nonauthorizing"
        or summary.get("split") != "dev"
        or summary.get("final_split_used") is not False
        or summary.get("gate", {}).get("gate_passed") is not False
        or summary.get("matched_reward_comparison_run") is not False
        or summary.get("conclusion", {}).get("next_step_authorized") is not False
        or summary.get("conclusion", {}).get("reward_comparison_is_interpretable")
        is not False
    ):
        raise SmokeError("Step 3d summary history drifted")

    incomplete_attempt = _load_json_object(
        incomplete_attempt_path, "incomplete Step 3e attempt receipt"
    )
    incomplete_profiles = incomplete_attempt.get("profiles", {})
    if (
        incomplete_attempt.get("gate_mode") != "step3e"
        or incomplete_attempt.get("split") != "dev"
        or incomplete_attempt.get("final_split_used") is not False
        or incomplete_attempt.get("gate", {}).get(
            "matched_run_validity_passed"
        )
        is not False
        or incomplete_profiles.get("v3_equivalent", {}).get(
            "active_actor_critic_transitions"
        )
        != STEP3E_ACTIVE_TRANSITIONS
        or incomplete_profiles.get("risk_averse", {}).get(
            "active_actor_critic_transitions"
        )
        != 0
        or incomplete_profiles.get("risk_averse", {})
        .get("critic_warmup", {})
        .get("fresh_rollout_explained_variance_at_unfreeze", 1.0)
        >= CRITIC_EXPLAINED_VARIANCE_GATE
    ):
        raise SmokeError("incomplete Step 3e attempt evidence drifted")

    headroom = _load_json_object(headroom_path, "Step 3.5 headroom receipt")
    if (
        headroom.get("tool") != "headroom_probe_v4.py"
        or headroom.get("status")
        != "privileged_development_headroom_probe_nonauthorizing"
        or headroom.get("split") != "dev"
        or headroom.get("authorizing") is not False
        or headroom.get("authorizes_training") is not False
        or headroom.get("final_split_used") is not False
        or headroom.get("uses_final_split") is not False
        or headroom.get("selects_or_exports_policy") is not False
        or headroom.get("decision", {}).get("row") != "headroom"
        or headroom.get("decision", {}).get("step_3e_started") is not False
        or headroom.get("decision", {}).get("stop_for_direction") is not True
        or headroom.get("feasibility", {}).get(
            "achievable_lower_bound_solved_count"
        )
        != 37
        or headroom.get("feasibility", {}).get("contested_count") != 4
    ):
        raise SmokeError("Step 3.5 headroom authorization evidence drifted")

    contested_rows = [
        row
        for row in headroom.get("rows", [])
        if row.get("classification", {}).get("decision_partition") == "contested"
    ]
    contested_ids = tuple(sorted(str(row.get("row_id")) for row in contested_rows))
    if contested_ids != STEP3E_CONTESTED_ROW_IDS:
        raise SmokeError("Step 3.5 contested development cases drifted")
    if any(
        row.get("planners", {}).get("best_ppo", {}).get("solved") is not False
        or row.get("planners", {}).get("oracle", {}).get("solved") is not True
        for row in contested_rows
    ):
        raise SmokeError("Step 3.5 contested-case evidence is inconsistent")

    source_identity = headroom.get("source_identity", {})
    current_sources = {
        "headroom_probe_v4_sha256": file_sha256(
            ROOT / "scripts" / "headroom_probe_v4.py"
        ),
        "simulator_v4_sha256": file_sha256(
            ROOT / "backend" / "app" / "simulator_v4.py"
        ),
        "simulator_core_v4_sha256": file_sha256(
            ROOT / "backend" / "app" / "simulator_core_v4.py"
        ),
    }
    if source_identity != current_sources:
        raise SmokeError("Step 3.5 source identity no longer matches the worktree")

    optimizer = _load_json_object(optimizer_path, "Step 3d attempt-06 receipt")
    if (
        optimizer.get("tool") != "smoke_train_v4.py"
        or optimizer.get("split") != "dev"
        or optimizer.get("gate_mode") != "learning"
        or optimizer.get("authorizing") is not False
        or optimizer.get("authorizes_training") is not False
        or optimizer.get("final_split_used") is not False
        or optimizer.get("uses_final_split") is not False
        or optimizer.get("selects_or_exports_policy") is not False
        or optimizer.get("gate", {}).get("gate_passed") is not False
    ):
        raise SmokeError("Step 3d attempt-06 provenance contract drifted")
    config = optimizer.get("config", {})
    expected_config = {
        "active_actor_critic_transition_budget_per_profile": 200_000,
        "batch_size": DEFAULT_BATCH_SIZE,
        "clip_range": 0.15,
        "critic_warmup_explained_variance_threshold": (
            CRITIC_EXPLAINED_VARIANCE_GATE
        ),
        "critic_warmup_min_transitions_per_profile": (
            STEP3E_CRITIC_WARMUP_TRANSITIONS
        ),
        "ent_coef": STEP3E_ENT_COEF,
        "learning_rate": STEP3E_LEARNING_RATE,
        "log_std_init": -1.5,
        "n_epochs": 5,
        "n_steps_per_lane": DEFAULT_N_STEPS,
        "observation_rms_frozen_during_ppo": True,
        "policy_seed": DEFAULT_POLICY_SEED,
        "profiles": ["v3_equivalent"],
        "rollout_size": DEFAULT_LANES * DEFAULT_N_STEPS,
        "simulator_lanes": DEFAULT_LANES,
        "target_kl": STEP3E_TARGET_KL,
        "use_sde": False,
        "vec_normalize": True,
    }
    if any(config.get(key) != value for key, value in expected_config.items()):
        raise SmokeError("Step 3d attempt-06 optimizer configuration drifted")
    attempt_profile = optimizer.get("profiles", {}).get("v3_equivalent", {})
    attempt_gate = optimizer.get("gate", {})
    canonical_initialization = {
        "actor_sha256": STEP3E_INITIAL_ACTOR_SHA256,
        "observation_rms_sha256": STEP3E_INITIAL_OBSERVATION_RMS_SHA256,
        "policy_sha256": STEP3E_INITIAL_POLICY_SHA256,
        "return_rms_sha256": STEP3E_INITIAL_RETURN_RMS_SHA256,
    }
    attempt_rows_200k = attempt_profile.get("development_curve", {}).get(
        "active_actor_critic_200000_transitions", {}
    ).get("rows", [])
    if (
        optimizer.get("behavior_cloning", {}).get("epochs_per_iteration")
        != STEP3E_BC_EPOCHS
        or optimizer.get("behavior_cloning", {}).get("dataset_sha256")
        != STEP3E_BC_DATASET_SHA256
        or optimizer.get("behavior_cloning", {}).get("policy_state_sha256")
        != STEP3E_INITIAL_POLICY_SHA256
        or optimizer.get("behavior_cloning", {}).get("observation_rms_sha256")
        != STEP3E_INITIAL_OBSERVATION_RMS_SHA256
        or optimizer.get("initialization_match", {})
        .get("arms", {})
        .get("v3_equivalent")
        != canonical_initialization
        or attempt_profile.get("actor_sha256_after_warmup")
        != STEP3E_INITIAL_ACTOR_SHA256
        or attempt_profile.get("critic_warmup_environment_transitions")
        != STEP3E_CRITIC_WARMUP_TRANSITIONS
        or attempt_profile.get("active_actor_critic_transitions") != 200_000
        or attempt_gate.get("critic_explained_variance_passed") is not True
        or attempt_gate.get("initial_solved_count") != 32
        or attempt_gate.get("final_solved_count") != 33
        or attempt_gate.get("solve_gain") != 1
        or historical_step3_rows_hash(attempt_rows_200k)
        != STEP3E_ATTEMPT06_200K_ROWS_SHA256
    ):
        raise SmokeError("Step 3d attempt-06 evidence fields drifted")

    return {
        "supervisor_gate_correction": STEP3E_SUPERVISOR_GATE_CORRECTION,
        "original_compound_gate_passed": False,
        "corrected_step_3d_declared_passed": True,
        "headroom_receipt": {
            "path": _receipt_path(headroom_path),
            "sha256": headroom_hash,
            "achievable_lower_bound_solved_count": 37,
            "contested_count": 4,
            "contested_row_ids": list(contested_ids),
        },
        "optimizer_receipt": {
            "path": _receipt_path(optimizer_path),
            "sha256": optimizer_hash,
            "attempt": 6,
            "original_gate_passed": False,
            "critic_explained_variance_passed": True,
            "solve_gain_at_200k": 1,
            "reported_approx_kl_max": attempt_gate.get("reported_approx_kl_max"),
            "target_kl_early_stop_rows": attempt_gate.get("early_stop_rows"),
            "canonical_initialization_hashes": canonical_initialization,
            "development_200k_rows_sha256": STEP3E_ATTEMPT06_200K_ROWS_SHA256,
        },
        "original_step3_summary_receipt": {
            "path": _receipt_path(summary_path),
            "sha256": summary_hash,
            "original_gate_passed": False,
            "next_step_authorized": False,
            "matched_reward_comparison_run": False,
        },
        "incomplete_step3e_attempt": {
            "path": _receipt_path(incomplete_attempt_path),
            "sha256": incomplete_attempt_hash,
            "matched_run_validity_passed": False,
            "control_active_transitions": STEP3E_ACTIVE_TRANSITIONS,
            "risk_averse_active_transitions": 0,
            "failure_reason": (
                "risk-averse critic EV remained below 0.5 at the incorrectly "
                "fixed 50k warm-up ceiling; supervisor specified 50k minimum"
            ),
        },
        "validated_source_identity": current_sources,
        "protected_v3_expected_files_sha256": headroom.get("invariants", {}).get(
            "protected_v3_files_sha256"
        ),
    }


def protected_v3_snapshot(expected: dict[str, str]) -> dict[str, Any]:
    if not isinstance(expected, dict) or not expected:
        raise SmokeError("Step 3.5 protected-v3 hash map is missing")
    complete_expected = {**expected, **PROTECTED_V3_EXTRA_FILES_SHA256}
    expected_paths = set(complete_expected)
    current_directory_paths: set[str] = set()
    for relative_root in PROTECTED_V3_DIRECTORY_ROOTS:
        root = ROOT / relative_root
        if not root.is_dir():
            raise SmokeError(f"protected-v3 directory is missing: {relative_root}")
        current_directory_paths.update(
            str(path.relative_to(ROOT)).replace("\\", "/")
            for path in root.rglob("*")
            if path.is_file()
        )
    expected_directory_paths = {
        relative
        for relative in expected_paths
        if any(
            relative == root or relative.startswith(f"{root}/")
            for root in PROTECTED_V3_DIRECTORY_ROOTS
        )
    }
    if current_directory_paths != expected_directory_paths:
        raise SmokeError("protected-v3 directory membership changed")

    current: dict[str, str] = {}
    for relative, expected_hash in sorted(complete_expected.items()):
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            raise SmokeError("Step 3.5 protected-v3 hash map is malformed")
        path = (ROOT / relative).resolve()
        try:
            path.relative_to(ROOT.resolve())
        except ValueError as exc:
            raise SmokeError("protected-v3 path escapes the repository") from exc
        if not path.is_file():
            raise SmokeError(f"protected-v3 file is missing: {relative}")
        current[relative] = file_sha256(path)
    if current != complete_expected:
        raise SmokeError("protected-v3 files no longer match Step 3.5")
    rendered = json.dumps(
        current,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "file_count": len(current),
        "map_sha256": hashlib.sha256(rendered).hexdigest(),
        "files_sha256": current,
    }


class FreezableRunningMeanStd(RunningMeanStd):
    """Running moments that may be fixed while normalization stays enabled."""

    def __init__(self, *, shape: tuple[int, ...]) -> None:
        super().__init__(shape=shape)
        self.frozen = False

    def update(self, array: np.ndarray) -> None:
        if not self.frozen:
            super().update(array)


class InstrumentedPPO(PPO):
    """PPO that retains every rollout-update diagnostic in memory."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.training_iterations: list[dict[str, Any]] = []
        self.diagnostic_phase = "unassigned"
        self.diagnostic_phase_start = 0

    def set_diagnostic_phase(self, phase: str) -> None:
        self.diagnostic_phase = phase
        self.diagnostic_phase_start = int(self.num_timesteps)

    def train(self) -> None:
        updates_before = int(self._n_updates)
        super().train()
        logged = self.logger.name_to_value
        required = (
            "explained_variance",
            "approx_kl",
            "clip_fraction",
            "entropy_loss",
            "value_loss",
            "policy_gradient_loss",
        )
        metrics: dict[str, float] = {}
        for name in required:
            value = float(logged[f"train/{name}"])
            if not np.isfinite(value):
                raise SmokeError(f"non-finite PPO diagnostic: {name}")
            metrics[name] = value
        action_std = torch.exp(self.policy.log_std.detach()).cpu().numpy()
        if not np.all(np.isfinite(action_std)):
            raise SmokeError("non-finite PPO action standard deviation")
        epochs_completed = int(self._n_updates) - updates_before
        self.training_iterations.append(
            {
                "iteration": len(self.training_iterations) + 1,
                "phase": self.diagnostic_phase,
                "total_transitions": int(self.num_timesteps),
                "phase_transitions": int(self.num_timesteps)
                - self.diagnostic_phase_start,
                "epochs_completed": epochs_completed,
                "early_stop_detected_before_final_epoch": (
                    epochs_completed < self.n_epochs
                ),
                "target_kl_guard_enabled": self.target_kl is not None,
                **metrics,
                "action_std_mean": float(np.mean(action_std)),
                "action_std_min": float(np.min(action_std)),
                "action_std_max": float(np.max(action_std)),
                "action_std_by_dimension": [
                    float(value) for value in action_std.reshape(-1)
                ],
                "learning_rate": float(
                    self.policy.optimizer.param_groups[0]["lr"]
                ),
            }
        )


def training_scenarios() -> list[tuple[Any, int]]:
    if any(not family.id.startswith("v3_train_") for family in TRAINING_FAMILIES_V3):
        raise SmokeError("a non-training family entered the smoke training split")
    return [
        (family.build(seed), family.tape_seed(seed))
        for family in TRAINING_FAMILIES_V3
        for seed in TRAINING_SEEDS_V3
    ]


def disaster_tape_sha256(schedule: Sequence[Any]) -> str:
    return canonical_hash([asdict(shock) for shock in schedule])


def training_roster_and_tapes_contract() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for family in TRAINING_FAMILIES_V3:
        for case_seed in TRAINING_SEEDS_V3:
            scenario = family.build(case_seed)
            tape_seed = family.tape_seed(case_seed)
            schedule = generate_disaster_tape_v3(scenario, tape_seed)
            rows.append(
                {
                    "family_id": family.id,
                    "case_seed": case_seed,
                    "tape_seed": tape_seed,
                    "scenario": scenario.model_dump(mode="json"),
                    "tape_sha256": disaster_tape_sha256(schedule),
                }
            )
    return {"case_count": len(rows), "sha256": canonical_hash(rows)}


def step3e_source_identity() -> dict[str, str]:
    paths = {
        "smoke_train_v4_sha256": Path(__file__).resolve(),
        "simulator_v4_sha256": ROOT / "backend" / "app" / "simulator_v4.py",
        "simulator_core_v4_sha256": (
            ROOT / "backend" / "app" / "simulator_core_v4.py"
        ),
        "simulator_v3_sha256": ROOT / "backend" / "app" / "simulator_v3.py",
        "scenarios_v3_sha256": ROOT / "backend" / "app" / "scenarios_v3.py",
    }
    return {name: file_sha256(path) for name, path in paths.items()}


@dataclass(frozen=True)
class TrainingLaneFactory:
    lane: int
    lane_count: int
    reward_profile: str

    def __call__(self) -> CyclingScenarioEnvV4:
        for variable in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            if os.environ.get(variable) != "1":
                raise SmokeError(f"spawned worker thread cap drifted: {variable}")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        scenarios = training_scenarios()
        offset = self.lane % len(scenarios)
        rotated = scenarios[offset:] + scenarios[:offset]
        return CyclingScenarioEnvV4(
            rotated,
            collect_evidence=False,
            reward_profile=self.reward_profile,
        )


def spawn_environment(reward_profile: str, lanes: int, seed: int) -> SubprocVecEnv:
    factories: list[Callable[[], CyclingScenarioEnvV4]] = [
        TrainingLaneFactory(lane, lanes, reward_profile) for lane in range(lanes)
    ]
    environment = SubprocVecEnv(factories, start_method="spawn")
    environment.seed(seed)
    return environment


def build_model(
    environment: VecEnv,
    *,
    seed: int,
    n_steps: int,
    batch_size: int,
    learning_rate: float = 1e-4,
    target_kl: float = 0.02,
    ent_coef: float = 0.003,
) -> InstrumentedPPO:
    return InstrumentedPPO(
        "MlpPolicy",
        environment,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.15,
        ent_coef=ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=target_kl,
        use_sde=False,
        policy_kwargs={
            "activation_fn": torch.nn.SiLU,
            "net_arch": {"pi": [384, 256, 128], "vf": [384, 256, 128]},
            "ortho_init": True,
            "log_std_init": -1.5,
        },
        seed=seed,
        device="cpu",
        verbose=0,
    )


def behavior_cloning_dataset() -> tuple[np.ndarray, np.ndarray]:
    observations: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for scenario, tape_seed in training_scenarios():
        schedule = generate_disaster_tape_v3(scenario, tape_seed)
        environment = CityRecoveryEnvV4(
            scenario,
            tape_seed,
            schedule,
            collect_evidence=False,
            reward_profile="v3_equivalent",
        )
        observation, _ = environment.reset(seed=tape_seed)
        terminated = False
        while not terminated:
            action, evidence = public_preparedness_curriculum_action_v3(observation)
            if evidence.get("future_tape_visible") is not False:
                raise SmokeError("BC teacher exposed future tape information")
            observations.append(np.asarray(observation, dtype=np.float32).copy())
            targets.append(np.asarray(action, dtype=np.float32).copy())
            observation, _, terminated, _, _ = environment.step(action)
    observation_array = np.asarray(observations, dtype=np.float32)
    target_array = np.asarray(targets, dtype=np.float32)
    if (
        observation_array.shape != (5_760, len(OBSERVATION_ORDER_V4))
        or target_array.shape != (5_760, len(ACTION_ORDER_V4))
        or not np.all(np.isfinite(observation_array))
        or not np.all(np.isfinite(target_array))
    ):
        raise SmokeError("BC dataset contract drifted")
    return observation_array, target_array


def array_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def state_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def actor_state(model: PPO) -> dict[str, torch.Tensor]:
    state = {
        f"policy_net.{name}": parameter.detach().cpu().clone()
        for name, parameter in model.policy.mlp_extractor.policy_net.named_parameters()
    }
    state.update(
        {
            f"action_net.{name}": parameter.detach().cpu().clone()
            for name, parameter in model.policy.action_net.named_parameters()
        }
    )
    state["log_std"] = model.policy.log_std.detach().cpu().clone()
    return state


def freeze_actor_for_critic_warmup(model: PPO) -> int:
    for parameter in model.policy.parameters():
        parameter.requires_grad_(False)
    critic_parameters = [
        *model.policy.mlp_extractor.value_net.parameters(),
        *model.policy.value_net.parameters(),
    ]
    for parameter in critic_parameters:
        parameter.requires_grad_(True)
    trainable = [
        parameter
        for parameter in model.policy.parameters()
        if parameter.requires_grad
    ]
    if {id(parameter) for parameter in trainable} != {
        id(parameter) for parameter in critic_parameters
    }:
        raise SmokeError("critic warm-up trainable-parameter contract drifted")
    return sum(parameter.numel() for parameter in trainable)


def unfreeze_policy(model: PPO) -> None:
    for parameter in model.policy.parameters():
        parameter.requires_grad_(True)
    if not all(parameter.requires_grad for parameter in model.policy.parameters()):
        raise SmokeError("policy unfreeze contract drifted")


def rms_state(rms: RunningMeanStd) -> dict[str, Any]:
    return {
        "mean": np.asarray(rms.mean, dtype=np.float64).copy(),
        "var": np.asarray(rms.var, dtype=np.float64).copy(),
        "count": float(rms.count),
    }


def load_rms_state(rms: RunningMeanStd, state: dict[str, Any]) -> None:
    rms.mean = np.asarray(state["mean"], dtype=np.float64).copy()
    rms.var = np.asarray(state["var"], dtype=np.float64).copy()
    rms.count = float(state["count"])


def rms_digest(state: dict[str, Any]) -> str:
    return array_digest(
        np.asarray(state["mean"], dtype=np.float64),
        np.asarray(state["var"], dtype=np.float64),
        np.asarray([state["count"]], dtype=np.float64),
    )


def normalize_observations(
    observations: np.ndarray, state: dict[str, Any]
) -> np.ndarray:
    normalized = (
        np.asarray(observations, dtype=np.float64) - state["mean"]
    ) / np.sqrt(state["var"] + 1e-8)
    return np.clip(normalized, -10.0, 10.0).astype(np.float32)


def policy_rollout_dataset(
    model: PPO, observation_rms: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    observations: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for scenario, tape_seed in training_scenarios():
        environment = CityRecoveryEnvV4(
            scenario,
            tape_seed,
            collect_evidence=False,
            reward_profile="v3_equivalent",
        )
        observation, _ = environment.reset(seed=tape_seed)
        terminated = False
        while not terminated:
            teacher_action, evidence = public_preparedness_curriculum_action_v3(
                observation
            )
            if (
                evidence.get("teacher_id")
                != "public-preparedness-curriculum-v3"
                or evidence.get("teacher_version") != "1.0.0"
                or evidence.get("future_tape_visible") is not False
            ):
                raise SmokeError("BC teacher contract drifted")
            observations.append(np.asarray(observation, dtype=np.float32).copy())
            targets.append(np.asarray(teacher_action, dtype=np.float32).copy())
            normalized = normalize_observations(
                np.asarray(observation, dtype=np.float32).reshape(1, -1),
                observation_rms,
            )[0]
            rollout_action, _ = model.predict(normalized, deterministic=True)
            observation, _, terminated, _, _ = environment.step(rollout_action)
    observation_array = np.asarray(observations, dtype=np.float32)
    target_array = np.asarray(targets, dtype=np.float32)
    if (
        observation_array.shape != (5_760, len(OBSERVATION_ORDER_V4))
        or target_array.shape != (5_760, len(ACTION_ORDER_V4))
        or not np.all(np.isfinite(observation_array))
        or not np.all(np.isfinite(target_array))
    ):
        raise SmokeError("DAgger rollout dataset contract drifted")
    return observation_array, target_array


def clone_actor_state(
    observations: np.ndarray,
    targets: np.ndarray,
    *,
    seed: int,
    n_steps: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    target_kl: float,
    ent_coef: float,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
    scenario, tape_seed = training_scenarios()[0]
    environment = DummyVecEnv(
        [
            lambda: CityRecoveryEnvV4(
                scenario,
                tape_seed,
                collect_evidence=False,
                reward_profile="v3_equivalent",
            )
        ]
    )
    try:
        model = build_model(
            environment,
            seed=seed,
            n_steps=n_steps,
            # This one-lane model only supplies the policy container for BC;
            # choose its full one-lane rollout as the inert PPO batch size.
            batch_size=min(batch_size, n_steps),
            learning_rate=learning_rate,
            target_kl=target_kl,
            ent_coef=ent_coef,
        )
        actor_parameters = [
            *model.policy.mlp_extractor.policy_net.parameters(),
            *model.policy.action_net.parameters(),
        ]
        optimizer = torch.optim.Adam(actor_parameters, lr=1e-3)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed ^ 0xBC37017)
        observation_rms = RunningMeanStd(
            shape=(len(OBSERVATION_ORDER_V4),)
        )
        observation_batches: list[np.ndarray] = []
        target_batches: list[np.ndarray] = []
        next_observations = observations
        next_targets = targets
        iteration_reports: list[dict[str, Any]] = []
        final_loss = float("nan")
        for iteration, beta in enumerate(DAGGER_BETA_SCHEDULE):
            if iteration:
                next_observations, next_targets = policy_rollout_dataset(
                    model, rms_state(observation_rms)
                )
            observation_rms.update(next_observations)
            observation_batches.append(next_observations)
            target_batches.append(next_targets)
            cumulative_observations = np.concatenate(observation_batches)
            cumulative_targets = np.concatenate(target_batches)
            normalized_observations = normalize_observations(
                cumulative_observations, rms_state(observation_rms)
            )
            observation_tensor = torch.as_tensor(
                normalized_observations, dtype=torch.float32
            )
            target_tensor = torch.as_tensor(
                cumulative_targets, dtype=torch.float32
            )
            model.policy.train()
            for _ in range(epochs):
                permutation = torch.randperm(
                    observation_tensor.shape[0], generator=generator
                )
                for start in range(0, observation_tensor.shape[0], 512):
                    indices = permutation[start : start + 512]
                    distribution = model.policy.get_distribution(
                        observation_tensor[indices]
                    )
                    predicted = distribution.distribution.mean
                    loss = torch.nn.functional.mse_loss(
                        predicted, target_tensor[indices]
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor_parameters, 1.0)
                    optimizer.step()
                    final_loss = float(loss.detach().cpu())
            model.policy.eval()
            with torch.no_grad():
                predicted = model.policy.get_distribution(
                    observation_tensor
                ).distribution.mean
                full_loss = float(
                    torch.nn.functional.mse_loss(predicted, target_tensor).cpu()
                )
            iteration_reports.append(
                {
                    "iteration": iteration + 1,
                    "dagger_beta": beta,
                    "cumulative_observation_count": int(
                        observation_tensor.shape[0]
                    ),
                    "full_dataset_mse": full_loss,
                }
            )
        model.policy.eval()
        state = {
            name: value.detach().cpu().clone()
            for name, value in model.policy.state_dict().items()
        }
        observation_rms_state = rms_state(observation_rms)
        return state, observation_rms_state, {
            "teacher": "public-preparedness-curriculum-v3",
            "teacher_version": "1.0.0",
            "training_split_only": True,
            "dagger_beta_schedule": list(DAGGER_BETA_SCHEDULE),
            "iterations": len(DAGGER_BETA_SCHEDULE),
            "epochs_per_iteration": epochs,
            "batch_size": 512,
            "observation_count": int(cumulative_observations.shape[0]),
            "final_batch_mse": final_loss,
            "full_dataset_mse": full_loss,
            "iteration_reports": iteration_reports,
            "dataset_sha256": array_digest(
                cumulative_observations, cumulative_targets
            ),
            "policy_state_sha256": state_digest(state),
            "observation_normalization": True,
            "observation_rms_sha256": rms_digest(observation_rms_state),
            "observation_rms_count": observation_rms_state["count"],
            "value_head_initialization": (
                "orthogonal_random_unchanged_during_actor_only_bc"
            ),
        }
    finally:
        environment.close()


def evaluate_development(
    model: PPO, reward_profile: str, normalizer: VecNormalize
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for family in DEVELOPMENT_FAMILIES_V3:
        if not family.id.startswith("v3_dev_"):
            raise SmokeError("a non-development family entered smoke evaluation")
        for case_seed in DEVELOPMENT_SEEDS_V3:
            scenario = family.build(case_seed)
            tape_seed = family.tape_seed(case_seed)
            schedule = generate_disaster_tape_v3(scenario, tape_seed)
            environment = CityRecoveryEnvV4(
                scenario,
                tape_seed,
                schedule,
                collect_evidence=True,
                reward_profile=reward_profile,
            )
            observation, _ = environment.reset(seed=tape_seed)
            terminated = False
            while not terminated:
                normalized = normalizer.normalize_obs(
                    np.asarray(observation, dtype=np.float32).copy()
                )
                action, _ = model.predict(normalized, deterministic=True)
                observation, _, terminated, _, _ = environment.step(action)
            summary = _summarize_v3("v4_smoke", environment.trajectory, scenario)
            outcome = summary["absolute_outcome"]
            tail_minimum_services = np.asarray(
                outcome["tail_minimum_services"], dtype=np.float64
            )
            recovery_targets = np.asarray(
                outcome["recovery_targets"], dtype=np.float64
            )
            minimum_tail_margin = float(
                np.min(tail_minimum_services - recovery_targets)
            )
            resilience_auc = float(summary["rauc"])
            conservation_residual = float(
                summary["max_logistics_conservation_residual"]
            )
            if not (
                np.isfinite(resilience_auc)
                and np.isfinite(minimum_tail_margin)
                and np.isfinite(conservation_residual)
                and np.all(np.isfinite(tail_minimum_services))
                and np.all(np.isfinite(recovery_targets))
            ):
                raise SmokeError("non-finite development evaluation outcome")
            rows.append(
                {
                    "row_id": f"{family.id}:{case_seed}",
                    "case_seed": case_seed,
                    "tape_seed": tape_seed,
                    "tape_sha256": disaster_tape_sha256(schedule),
                    "solved": bool(outcome["solved"]),
                    "reason_codes": list(outcome["reason_codes"]),
                    "resilience_auc": resilience_auc,
                    "minimum_tail_margin": minimum_tail_margin,
                    "hard_violation_count": int(summary["hard_violation_count"]),
                    "max_conservation_residual": conservation_residual,
                    "target_met_by_service": list(outcome["target_met_by_service"]),
                    "tail_minimum_services": tail_minimum_services.tolist(),
                    "recovery_targets": recovery_targets.tolist(),
                }
            )
    if len(rows) != 40 or len({row["row_id"] for row in rows}) != 40:
        raise SmokeError("development evaluation must contain 40 unique rows")
    reasons = Counter(
        reason for row in rows if not row["solved"] for reason in row["reason_codes"]
    )
    return {
        "case_count": len(rows),
        "solved_count": sum(row["solved"] for row in rows),
        "solve_rate": sum(row["solved"] for row in rows) / len(rows),
        "mean_resilience_auc": round(fmean(row["resilience_auc"] for row in rows), 10),
        "mean_minimum_tail_margin": round(
            fmean(row["minimum_tail_margin"] for row in rows), 10
        ),
        "hard_violation_count": sum(row["hard_violation_count"] for row in rows),
        "maximum_conservation_residual": max(
            row["max_conservation_residual"] for row in rows
        ),
        "failure_reason_code_histogram": dict(sorted(reasons.items())),
        "rows": rows,
    }


def evaluate_development_frozen(
    model: PPO, reward_profile: str, normalizer: VecNormalize
) -> dict[str, Any]:
    previous_training = normalizer.training
    previous_norm_reward = normalizer.norm_reward
    normalizer.training = False
    normalizer.norm_reward = False
    try:
        return evaluate_development(model, reward_profile, normalizer)
    finally:
        normalizer.training = previous_training
        normalizer.norm_reward = previous_norm_reward


def development_curve_evaluation(
    evaluation: dict[str, Any],
    *,
    active_actor_critic_transitions: int,
    total_environment_transitions: int,
) -> dict[str, Any]:
    """Attach explicit transition accounting to one development evaluation."""
    return {
        "active_actor_critic_transitions": active_actor_critic_transitions,
        "total_environment_transitions": total_environment_transitions,
        **evaluation,
    }


def reported_approx_kl_summary(
    iterations: Sequence[dict[str, Any]], target_kl: float
) -> dict[str, Any]:
    """Summarize the recorded SB3 KL values against its target-KL guard."""
    values = [float(iteration["approx_kl"]) for iteration in iterations]
    stability_limit = REPORTED_APPROX_KL_STABILITY_MULTIPLIER * target_kl
    maximum = max(values) if values else None
    return {
        "reported_approx_kl_max": maximum,
        "reported_approx_kl_stability_limit": stability_limit,
        "reported_approx_kl_stability_multiplier": (
            REPORTED_APPROX_KL_STABILITY_MULTIPLIER
        ),
        "reported_approx_kl_stability_definition": (
            REPORTED_APPROX_KL_STABILITY_DEFINITION
        ),
        "reported_approx_kl_stable": (
            maximum is not None and maximum <= stability_limit
        ),
    }


def step3e_target_kl_diagnostics(
    iterations: Sequence[dict[str, Any]], target_kl: float
) -> dict[str, Any]:
    values = [float(iteration["approx_kl"]) for iteration in iterations]
    early_stops = [
        iteration
        for iteration in iterations
        if iteration["early_stop_detected_before_final_epoch"]
    ]
    return {
        "target_kl": target_kl,
        "reported_approx_kl_max_diagnostic": max(values) if values else None,
        "reported_approx_kl_rows_above_target": sum(
            value > target_kl for value in values
        ),
        "target_kl_guard_enabled_on_every_iteration": bool(iterations)
        and all(iteration["target_kl_guard_enabled"] for iteration in iterations),
        "early_stop_detected_before_final_epoch_count": len(early_stops),
        "early_stop_detection_count_is_lower_bound": True,
        "obsolete_max_kl_ceiling_applied": False,
        "supervisor_corrected_interpretation": (
            "SB3 checks each minibatch before applying the offending update and "
            "then breaks; the logger reports the mean of the final partial epoch. "
            "The retained early-stop flag cannot detect a break during epoch five, "
            "so its count is a lower bound, not an independent pass/fail ceiling."
        ),
    }


def early_stop_row_summary(
    iterations: Sequence[dict[str, Any]],
) -> dict[str, int]:
    """Count rollout-update rows that did or did not stop before epoch five."""
    early_stop_rows = sum(
        bool(iteration["early_stop_detected_before_final_epoch"])
        for iteration in iterations
    )
    return {
        "iteration_row_count": len(iterations),
        "early_stop_row_count": early_stop_rows,
        "full_epoch_row_count": len(iterations) - early_stop_rows,
    }


def paired_comparison(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_rows = left["rows"]
    right_rows = right["rows"]
    if [row["row_id"] for row in left_rows] != [
        row["row_id"] for row in right_rows
    ]:
        raise SmokeError("smoke evaluations are not paired on identical dev cases")
    if [row.get("tape_sha256") for row in left_rows] != [
        row.get("tape_sha256") for row in right_rows
    ]:
        raise SmokeError("smoke evaluations are not paired on identical tapes")
    both = sum(a["solved"] and b["solved"] for a, b in zip(left_rows, right_rows))
    left_only = sum(
        a["solved"] and not b["solved"] for a, b in zip(left_rows, right_rows)
    )
    right_only = sum(
        b["solved"] and not a["solved"] for a, b in zip(left_rows, right_rows)
    )
    discordant = left_only + right_only
    if discordant:
        tail = sum(
            comb(discordant, index)
            for index in range(min(left_only, right_only) + 1)
        ) / (2**discordant)
        mcnemar_p = min(1.0, 2.0 * tail)
    else:
        mcnemar_p = 1.0
    return {
        "both_solved": both,
        "v3_equivalent_only": left_only,
        "risk_averse_only": right_only,
        "neither": len(left_rows) - both - left_only - right_only,
        "discordant": discordant,
        "exact_two_sided_mcnemar_p": mcnemar_p,
    }


def contested_case_outcomes(
    evaluation: dict[str, Any], contested_row_ids: Sequence[str]
) -> dict[str, Any]:
    rows = {str(row["row_id"]): row for row in evaluation["rows"]}
    if len(rows) != len(evaluation["rows"]):
        raise SmokeError("development evaluation contains duplicate row ids")
    missing = [row_id for row_id in contested_row_ids if row_id not in rows]
    if missing:
        raise SmokeError(f"development evaluation missed contested rows: {missing}")
    outcomes = [
        {
            "row_id": row_id,
            "solved": bool(rows[row_id]["solved"]),
            "minimum_tail_margin": float(rows[row_id]["minimum_tail_margin"]),
            "resilience_auc": float(rows[row_id]["resilience_auc"]),
            "reason_codes": list(rows[row_id]["reason_codes"]),
        }
        for row_id in contested_row_ids
    ]
    converted = [row["row_id"] for row in outcomes if row["solved"]]
    return {
        "case_count": len(outcomes),
        "converted_from_step3_best_ppo_unsolved_count": len(converted),
        "converted_row_ids": converted,
        "outcomes": outcomes,
    }


def step3e_lexicographic_key(
    evaluation: dict[str, Any], contested_row_ids: Sequence[str]
) -> tuple[int, int, float, float]:
    contested = contested_case_outcomes(evaluation, contested_row_ids)
    return (
        int(evaluation["solved_count"]),
        int(contested["converted_from_step3_best_ppo_unsolved_count"]),
        float(evaluation["mean_minimum_tail_margin"]),
        float(evaluation["mean_resilience_auc"]),
    )


def step3e_carry_decision(
    control_final: dict[str, Any],
    risk_500k: dict[str, Any],
    risk_final: dict[str, Any],
    contested_row_ids: Sequence[str],
) -> dict[str, Any]:
    risk_ahead = int(risk_final["solved_count"]) > int(
        control_final["solved_count"]
    )
    key_500k = step3e_lexicographic_key(risk_500k, contested_row_ids)
    key_final = step3e_lexicographic_key(risk_final, contested_row_ids)
    still_rising = key_final > key_500k
    if risk_ahead and still_rising:
        decision = "carry_risk_averse"
        interpretation = (
            "risk-averse is ahead on final dev solves and is still improving "
            "lexicographically from 500k to 1M"
        )
    elif int(risk_final["solved_count"]) < int(control_final["solved_count"]):
        decision = "carry_v3_equivalent"
        interpretation = "risk-averse underperforms the control on final dev solves"
    elif not risk_ahead:
        decision = "carry_v3_equivalent"
        interpretation = (
            "indistinguishable on the preregistered final dev solve-count primary"
        )
    else:
        decision = "carry_v3_equivalent"
        interpretation = (
            "risk-averse leads at 1M but is not still rising; the comparison is "
            "inconclusive for carrying the treatment forward"
        )
    return {
        "preregistered_rule": (
            "carry risk_averse only when its 1M solved count is strictly greater "
            "than v3_equivalent and its lexicographic tuple (solves, contested "
            "solves, mean minimum tail margin, mean resilience AUC) strictly "
            "improves from 500k to 1M; otherwise carry v3_equivalent"
        ),
        "risk_averse_ahead_on_final_solved_count": risk_ahead,
        "risk_averse_500k_lexicographic_key": list(key_500k),
        "risk_averse_1m_lexicographic_key": list(key_final),
        "risk_averse_still_rising_500k_to_1m": still_rising,
        "decision": decision,
        "interpretation": interpretation,
    }


def all_evaluation_hard_violations_zero(result: dict[str, Any]) -> bool:
    return all(
        evaluation["hard_violation_count"] == 0
        for evaluation in result["development_curve"].values()
    )


def all_evaluation_conservation_residuals_zero(result: dict[str, Any]) -> bool:
    return all(
        evaluation["maximum_conservation_residual"] == 0.0
        for evaluation in result["development_curve"].values()
    )


def development_tape_contract_sha256(evaluation: dict[str, Any]) -> str:
    return canonical_hash(
        [
            {
                "row_id": row["row_id"],
                "case_seed": row["case_seed"],
                "tape_seed": row["tape_seed"],
                "tape_sha256": row["tape_sha256"],
            }
            for row in evaluation["rows"]
        ]
    )


def diagnostic_rows_valid(
    rows: Sequence[dict[str, Any]], expected_count: int
) -> bool:
    required = {
        "explained_variance",
        "approx_kl",
        "clip_fraction",
        "entropy_loss",
        "value_loss",
        "policy_gradient_loss",
        "action_std_mean",
        "action_std_min",
        "action_std_max",
        "action_std_by_dimension",
        "target_kl_guard_enabled",
    }
    return len(rows) == expected_count and all(
        required <= set(row) and len(row["action_std_by_dimension"]) == len(ACTION_ORDER_V4)
        for row in rows
    )


def step3e_return_rms_continuity_valid(
    result: dict[str, Any],
    initial_return_rms_sha256: str,
    *,
    warmup_transitions: int,
    active_transitions: int,
) -> bool:
    """Verify reward moments continued without resets inside one matched arm."""
    warmup = result["critic_warmup"]
    final = result["vecnormalize"]
    before_count = float(warmup["return_rms_before_count"])
    after_count = float(warmup["return_rms_after_count"])
    final_count = float(final["return_rms_count"])
    return (
        warmup["return_rms_before_sha256"] == initial_return_rms_sha256
        and before_count < after_count < final_count
        and np.isclose(
            after_count - before_count,
            warmup_transitions,
            rtol=0.0,
            atol=1e-6,
        )
        and np.isclose(
            final_count - after_count,
            active_transitions,
            rtol=0.0,
            atol=1e-6,
        )
    )


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise SmokeError(f"refusing to overwrite smoke result: {path}")
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    rendered = json.dumps(
        payload,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
    os.rename(temporary, path)


def validate_step3e_output_path(path: Path) -> Path:
    allowed = (ROOT / "internal" / "developmental_runs" / "v4").resolve()
    target = path.resolve()
    try:
        target.relative_to(allowed)
    except ValueError as exc:
        raise SmokeError(
            "Step 3e receipt must stay under internal/developmental_runs/v4"
        ) from exc
    if target.exists():
        raise SmokeError(f"refusing to overwrite Step 3e receipt: {target}")
    return target


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gate-mode",
        choices=("learning", "reward", "step3e"),
        default="learning",
        help=(
            "learning proves original-reward PPO beats BC; reward is locked "
            "behind that gate"
        ),
    )
    parser.add_argument("--transitions", type=int, default=DEFAULT_TRANSITIONS)
    parser.add_argument("--lanes", type=int, default=DEFAULT_LANES)
    parser.add_argument("--n-steps", type=int, default=DEFAULT_N_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--policy-seed", type=int, default=DEFAULT_POLICY_SEED)
    parser.add_argument("--bc-epochs", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--target-kl", type=float, default=0.02)
    parser.add_argument("--ent-coef", type=float, default=0.003)
    parser.add_argument(
        "--critic-warmup-min-transitions",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--critic-warmup-max-transitions",
        type=int,
        default=CRITIC_WARMUP_MAX_TRANSITIONS,
    )
    parser.add_argument(
        "--critic-ev-threshold",
        type=float,
        default=CRITIC_EXPLAINED_VARIANCE_GATE,
    )
    parser.add_argument(
        "--freeze-observation-rms",
        action="store_true",
        help=(
            "keep BC observation moments fixed while reward moments continue "
            "updating"
        ),
    )
    parser.add_argument(
        "--profile",
        action="append",
        choices=REWARD_PROFILES_V4,
        default=[],
        help="repeatable; defaults to the matched v3-equivalent and risk-averse pair",
    )
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument(
        "--headroom-receipt",
        type=Path,
        default=STEP3E_HEADROOM_RECEIPT,
        help="hash-pinned Step 3.5 development receipt used only by step3e",
    )
    parser.add_argument(
        "--optimizer-provenance-receipt",
        type=Path,
        default=STEP3E_OPTIMIZER_RECEIPT,
        help="hash-pinned attempt-06 development receipt used only by step3e",
    )
    parser.add_argument(
        "--step3-summary-receipt",
        type=Path,
        default=STEP3E_SUMMARY_RECEIPT,
        help="hash-pinned failed original Step 3d summary retained as history",
    )
    parser.add_argument(
        "--supervisor-step3e-authorization",
        action="store_true",
        help="required explicit acknowledgement of the corrected Step 3d gate",
    )
    parser.add_argument(
        "--minimum-solve-gain",
        type=int,
        default=3,
        help="required risk-averse solved-count gain for the smoke gate",
    )
    parser.add_argument(
        "--minimum-learning-solve-gain",
        type=int,
        default=2,
        help="required original-reward PPO gain over its measured BC start",
    )
    return parser.parse_args(argv)


def reset_policy_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_random_seed(seed, using_cuda=False)


def learning_milestones(
    total_transitions: int, rollout_size: int, gate_mode: str = "learning"
) -> list[int]:
    if gate_mode == "step3e":
        if total_transitions != STEP3E_ACTIVE_TRANSITIONS or not all(
            milestone % rollout_size == 0 for milestone in STEP3E_ACTIVE_MILESTONES
        ):
            raise SmokeError("Step 3e milestones require exact complete rollouts")
        return list(STEP3E_ACTIVE_MILESTONES)
    if total_transitions == ACTIVE_MILESTONES[-1] and all(
        milestone % rollout_size == 0 for milestone in ACTIVE_MILESTONES
    ):
        return list(ACTIVE_MILESTONES)
    return [total_transitions]


def validate_step3e_runtime_config(
    args: argparse.Namespace, rollout_size: int
) -> None:
    expected = {
        "transitions": STEP3E_ACTIVE_TRANSITIONS,
        "lanes": DEFAULT_LANES,
        "n_steps": DEFAULT_N_STEPS,
        "batch_size": DEFAULT_BATCH_SIZE,
        "policy_seed": DEFAULT_POLICY_SEED,
        "bc_epochs": STEP3E_BC_EPOCHS,
        "learning_rate": STEP3E_LEARNING_RATE,
        "target_kl": STEP3E_TARGET_KL,
        "ent_coef": STEP3E_ENT_COEF,
        "critic_warmup_min_transitions": STEP3E_CRITIC_WARMUP_TRANSITIONS,
        "critic_warmup_max_transitions": (
            STEP3E_CRITIC_WARMUP_MAX_TRANSITIONS
        ),
        "critic_ev_threshold": CRITIC_EXPLAINED_VARIANCE_GATE,
        "freeze_observation_rms": True,
    }
    mismatches = {
        key: {"expected": value, "actual": getattr(args, key)}
        for key, value in expected.items()
        if getattr(args, key) != value
    }
    if rollout_size != DEFAULT_LANES * DEFAULT_N_STEPS:
        mismatches["rollout_size"] = {
            "expected": DEFAULT_LANES * DEFAULT_N_STEPS,
            "actual": rollout_size,
        }
    if mismatches:
        raise SmokeError(
            "Step 3e must use the exact supervisor-adopted attempt-06 regime: "
            + json.dumps(mismatches, sort_keys=True)
        )
    if not args.supervisor_step3e_authorization:
        raise SmokeError(
            "Step 3e requires --supervisor-step3e-authorization"
        )
    if args.json_output is None:
        raise SmokeError("Step 3e requires a create-new --json-output receipt")


def _resolve_input_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.gate_mode == "learning":
        profiles = tuple(args.profile or ("v3_equivalent",))
        if profiles != ("v3_equivalent",):
            raise SmokeError(
                "the learning gate must use only the original v3 reward"
            )
    else:
        profiles = tuple(args.profile or PROFILE_ORDER)
        if profiles != PROFILE_ORDER:
            raise SmokeError(
                f"the {args.gate_mode} gate requires the registered matched "
                "profile order"
            )
    if len(set(profiles)) != len(profiles):
        raise SmokeError("reward profiles must be unique")
    rollout_size = args.lanes * args.n_steps
    step3e_provenance: dict[str, Any] | None = None
    step3e_output: Path | None = None
    protected_v3_before: dict[str, Any] | None = None
    step3e_sources_before: dict[str, str] | None = None
    training_contract: dict[str, Any] | None = None
    if args.gate_mode == "step3e":
        validate_step3e_runtime_config(args, rollout_size)
        step3e_output = validate_step3e_output_path(
            _resolve_input_path(args.json_output)
        )
        step3e_provenance = validate_step3e_provenance(
            _resolve_input_path(args.headroom_receipt),
            _resolve_input_path(args.optimizer_provenance_receipt),
            _resolve_input_path(args.step3_summary_receipt),
        )
        protected_v3_before = protected_v3_snapshot(
            step3e_provenance["protected_v3_expected_files_sha256"]
        )
        step3e_sources_before = step3e_source_identity()
        training_contract = training_roster_and_tapes_contract()
        if training_contract != {
            "case_count": 192,
            "sha256": STEP3E_TRAINING_ROSTER_AND_TAPES_SHA256,
        }:
            raise SmokeError("Step 3e training roster or disaster tapes drifted")
    elif args.supervisor_step3e_authorization:
        raise SmokeError(
            "--supervisor-step3e-authorization is valid only for gate-mode step3e"
        )
    if (
        args.transitions <= 0
        or args.lanes <= 0
        or args.n_steps <= 0
        or args.batch_size <= 0
        or args.transitions % rollout_size
        or rollout_size % args.batch_size
        or args.critic_warmup_min_transitions < 0
        or args.critic_warmup_min_transitions % rollout_size
        or args.critic_warmup_max_transitions < rollout_size
        or args.critic_warmup_max_transitions % rollout_size
        or args.critic_warmup_min_transitions
        > args.critic_warmup_max_transitions
    ):
        raise SmokeError(
            "training and critic warm-up must divide into complete lane "
            "rollouts and batch size must divide the rollout"
        )
    if (
        args.bc_epochs <= 0
        or args.minimum_solve_gain < 0
        or args.minimum_learning_solve_gain < 0
        or not np.isfinite(args.critic_ev_threshold)
        or not np.isfinite(args.learning_rate)
        or args.learning_rate <= 0.0
        or not np.isfinite(args.target_kl)
        or args.target_kl <= 0.0
        or not np.isfinite(args.ent_coef)
        or args.ent_coef < 0.0
    ):
        raise SmokeError("BC, solve-gain, or critic-gate arguments are invalid")

    torch.set_num_threads(min(12, os.cpu_count() or 1))
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    reset_policy_seed(args.policy_seed)

    started = time.perf_counter()
    observations, targets = behavior_cloning_dataset()
    initial_state, initial_observation_rms, bc_receipt = clone_actor_state(
        observations,
        targets,
        seed=args.policy_seed,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        epochs=args.bc_epochs,
        learning_rate=args.learning_rate,
        target_kl=args.target_kl,
        ent_coef=args.ent_coef,
    )
    if args.gate_mode == "step3e" and (
        bc_receipt["dataset_sha256"] != STEP3E_BC_DATASET_SHA256
        or bc_receipt["policy_state_sha256"] != STEP3E_INITIAL_POLICY_SHA256
        or bc_receipt["observation_rms_sha256"]
        != STEP3E_INITIAL_OBSERVATION_RMS_SHA256
    ):
        raise SmokeError("Step 3e behavior-cloning initialization drifted")
    results: dict[str, Any] = {}
    initial_hashes: dict[str, dict[str, str]] = {}
    for profile in profiles:
        raw_environment: SubprocVecEnv | None = None
        environment: VecNormalize | None = None
        try:
            reset_policy_seed(args.policy_seed)
            raw_environment = spawn_environment(
                profile, args.lanes, args.policy_seed
            )
            environment = VecNormalize(
                raw_environment,
                training=True,
                norm_obs=True,
                norm_reward=True,
                clip_obs=10.0,
                clip_reward=10.0,
                gamma=0.99,
                epsilon=1e-8,
            )
            if args.freeze_observation_rms:
                fixed_observation_rms = FreezableRunningMeanStd(
                    shape=(len(OBSERVATION_ORDER_V4),)
                )
                load_rms_state(
                    fixed_observation_rms, initial_observation_rms
                )
                fixed_observation_rms.frozen = True
                environment.obs_rms = fixed_observation_rms
            else:
                load_rms_state(environment.obs_rms, initial_observation_rms)
            model = build_model(
                environment,
                seed=args.policy_seed,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                target_kl=args.target_kl,
                ent_coef=args.ent_coef,
            )
            model.policy.load_state_dict(initial_state, strict=True)
            profile_initial_hashes = {
                "policy_sha256": state_digest(model.policy.state_dict()),
                "actor_sha256": state_digest(actor_state(model)),
                "observation_rms_sha256": rms_digest(
                    rms_state(environment.obs_rms)
                ),
                "return_rms_sha256": rms_digest(rms_state(environment.ret_rms)),
            }
            if (
                profile_initial_hashes["policy_sha256"]
                != bc_receipt["policy_state_sha256"]
                or profile_initial_hashes["observation_rms_sha256"]
                != bc_receipt["observation_rms_sha256"]
            ):
                raise SmokeError("matched policy initialization drifted")
            if args.gate_mode == "step3e" and profile_initial_hashes != {
                "policy_sha256": STEP3E_INITIAL_POLICY_SHA256,
                "actor_sha256": STEP3E_INITIAL_ACTOR_SHA256,
                "observation_rms_sha256": STEP3E_INITIAL_OBSERVATION_RMS_SHA256,
                "return_rms_sha256": STEP3E_INITIAL_RETURN_RMS_SHA256,
            }:
                raise SmokeError("Step 3e canonical initialization hash drifted")
            if initial_hashes and profile_initial_hashes != next(
                iter(initial_hashes.values())
            ):
                raise SmokeError("matched arm initialization hashes differ")
            if model.policy.optimizer.state:
                raise SmokeError("PPO optimizer must start with empty state")
            initial_hashes[profile] = profile_initial_hashes

            initial_evaluation = development_curve_evaluation(
                evaluate_development_frozen(model, profile, environment),
                active_actor_critic_transitions=0,
                total_environment_transitions=0,
            )
            development_curve: dict[str, dict[str, Any]] = {
                "bc_initialization": initial_evaluation
            }
            profile_started = time.perf_counter()

            actor_hash_before_warmup = state_digest(actor_state(model))
            observation_rms_before_warmup = rms_state(environment.obs_rms)
            return_rms_before_warmup = rms_state(environment.ret_rms)
            critic_parameter_count = freeze_actor_for_critic_warmup(model)
            model.set_diagnostic_phase("critic_warmup")
            warmup_completed = 0
            warmup_metric_start = len(model.training_iterations)
            explained_variance_before: float | None = None
            explained_variance_after = float("-inf")
            while warmup_completed < args.critic_warmup_max_transitions:
                model.learn(
                    total_timesteps=rollout_size,
                    reset_num_timesteps=warmup_completed == 0,
                    progress_bar=False,
                )
                warmup_completed += rollout_size
                if int(model.num_timesteps) != warmup_completed:
                    raise SmokeError(
                        "critic warm-up missed an exact rollout boundary"
                    )
                explained_variance_after = float(
                    model.training_iterations[-1]["explained_variance"]
                )
                if explained_variance_before is None:
                    explained_variance_before = explained_variance_after
                if (
                    warmup_completed
                    >= args.critic_warmup_min_transitions
                    and explained_variance_after > args.critic_ev_threshold
                ):
                    break
            critic_gate_passed = (
                explained_variance_after > args.critic_ev_threshold
            )
            actor_hash_after_warmup = state_digest(actor_state(model))
            observation_rms_after_warmup = rms_state(environment.obs_rms)
            return_rms_after_warmup = rms_state(environment.ret_rms)
            if actor_hash_after_warmup != actor_hash_before_warmup:
                raise SmokeError("actor changed during critic-only warm-up")
            development_curve["post_critic_warmup"] = (
                development_curve_evaluation(
                    evaluate_development_frozen(model, profile, environment),
                    active_actor_critic_transitions=0,
                    total_environment_transitions=warmup_completed,
                )
            )
            warmup_metrics = model.training_iterations[warmup_metric_start:]

            active_completed = 0
            active_metrics: list[dict[str, Any]] = []
            training_milestone_states: dict[str, dict[str, Any]] = {}
            if critic_gate_passed:
                unfreeze_policy(model)
                model.set_diagnostic_phase("actor_critic_training")
                active_metric_start = len(model.training_iterations)
                for milestone in learning_milestones(
                    args.transitions, rollout_size, args.gate_mode
                ):
                    model.learn(
                        total_timesteps=milestone - active_completed,
                        reset_num_timesteps=False,
                        progress_bar=False,
                    )
                    active_completed = milestone
                    expected_total = warmup_completed + active_completed
                    if int(model.num_timesteps) != expected_total:
                        raise SmokeError(
                            "active PPO missed an exact transition milestone"
                        )
                    development_curve[
                        f"active_actor_critic_{milestone}_transitions"
                    ] = development_curve_evaluation(
                        evaluate_development_frozen(
                            model, profile, environment
                        ),
                        active_actor_critic_transitions=milestone,
                        total_environment_transitions=expected_total,
                    )
                    training_milestone_states[str(milestone)] = {
                        "policy_sha256": state_digest(model.policy.state_dict()),
                        "actor_sha256": state_digest(actor_state(model)),
                        "observation_rms_sha256": rms_digest(
                            rms_state(environment.obs_rms)
                        ),
                        "return_rms_sha256": rms_digest(
                            rms_state(environment.ret_rms)
                        ),
                        "return_rms_count": rms_state(environment.ret_rms)[
                            "count"
                        ],
                    }
                active_metrics = model.training_iterations[active_metric_start:]

            profile_elapsed = time.perf_counter() - profile_started
            final_evaluation = development_curve[
                f"active_actor_critic_{args.transitions}_transitions"
                if active_completed == args.transitions
                else "post_critic_warmup"
            ]
            active_kl_summary = (
                step3e_target_kl_diagnostics(active_metrics, args.target_kl)
                if args.gate_mode == "step3e"
                else reported_approx_kl_summary(active_metrics, args.target_kl)
            )
            active_final_ev = (
                float(active_metrics[-1]["explained_variance"])
                if active_metrics
                else float("-inf")
            )
            approx_kl_stable = bool(
                active_kl_summary.get("reported_approx_kl_stable", False)
            )
            solve_gain = (
                final_evaluation["solved_count"]
                - initial_evaluation["solved_count"]
            )
            learning_gate_passed = (
                critic_gate_passed
                and active_completed == args.transitions
                and solve_gain >= args.minimum_learning_solve_gain
                and active_final_ev > args.critic_ev_threshold
                and approx_kl_stable
                and final_evaluation["hard_violation_count"] == 0
                and final_evaluation["maximum_conservation_residual"] == 0.0
            )
            results[profile] = {
                "reward_profile": profile,
                "elapsed_seconds": round(profile_elapsed, 3),
                "critic_warmup_environment_transitions": warmup_completed,
                "active_actor_critic_transitions": active_completed,
                "total_environment_transitions": int(model.num_timesteps),
                "training_fps": round(
                    (warmup_completed + active_completed) / profile_elapsed,
                    3,
                ),
                "policy_state_sha256": state_digest(model.policy.state_dict()),
                "actor_sha256_after_warmup": actor_hash_after_warmup,
                "critic_warmup": {
                    "actor_frozen": True,
                    "actor_parameters_byte_identical": True,
                    "actor_sha256_before": actor_hash_before_warmup,
                    "actor_sha256_after": actor_hash_after_warmup,
                    "critic_trainable_parameter_count": critic_parameter_count,
                    "maximum_transitions": (
                        args.critic_warmup_max_transitions
                    ),
                    "minimum_transitions": (
                        args.critic_warmup_min_transitions
                    ),
                    "completed_transitions": warmup_completed,
                    "explained_variance_threshold": args.critic_ev_threshold,
                    "pre_warmup_rollout_explained_variance": (
                        explained_variance_before
                    ),
                    "fresh_rollout_explained_variance_at_unfreeze": (
                        explained_variance_after
                    ),
                    "observation_rms_before_sha256": rms_digest(
                        observation_rms_before_warmup
                    ),
                    "observation_rms_before_count": (
                        observation_rms_before_warmup["count"]
                    ),
                    "observation_rms_after_sha256": rms_digest(
                        observation_rms_after_warmup
                    ),
                    "observation_rms_after_count": (
                        observation_rms_after_warmup["count"]
                    ),
                    "observation_rms_frozen": (
                        args.freeze_observation_rms
                    ),
                    "return_rms_before_sha256": rms_digest(
                        return_rms_before_warmup
                    ),
                    "return_rms_before_count": return_rms_before_warmup[
                        "count"
                    ],
                    "return_rms_after_sha256": rms_digest(
                        return_rms_after_warmup
                    ),
                    "return_rms_after_count": return_rms_after_warmup["count"],
                    "diagnostic_reward_units": "vecnormalize_normalized",
                    "gate_passed": critic_gate_passed,
                    "early_stop_rows": early_stop_row_summary(
                        warmup_metrics
                    ),
                    "iterations": warmup_metrics,
                },
                "active_training": {
                    "actor_unfrozen": critic_gate_passed,
                    "active_actor_critic_transitions": active_completed,
                    "total_environment_transitions": int(
                        model.num_timesteps
                    ),
                    **active_kl_summary,
                    "fresh_rollout_final_explained_variance": (
                        active_final_ev if active_metrics else None
                    ),
                    "diagnostic_reward_units": "vecnormalize_normalized",
                    "early_stop_rows": early_stop_row_summary(
                        active_metrics
                    ),
                    "iterations": active_metrics,
                },
                "vecnormalize": {
                    "norm_obs": True,
                    "norm_reward_during_training": True,
                    "norm_reward_during_evaluation": False,
                    "training_during_evaluation": False,
                    "observation_rms_sha256": rms_digest(
                        rms_state(environment.obs_rms)
                    ),
                    "return_rms_sha256": rms_digest(
                        rms_state(environment.ret_rms)
                    ),
                    "return_rms_count": rms_state(environment.ret_rms)["count"],
                },
                "development_curve": development_curve,
                "development": final_evaluation,
                "training_milestone_states": training_milestone_states,
                "training_roster_and_tapes": training_contract,
                "learning_gate": {
                    "initial_solved_count": initial_evaluation["solved_count"],
                    "final_solved_count": final_evaluation["solved_count"],
                    "solve_gain": solve_gain,
                    "minimum_solve_gain": args.minimum_learning_solve_gain,
                    "gate_passed": learning_gate_passed,
                    "gate_applied": args.gate_mode == "learning",
                },
            }
            if args.gate_mode == "step3e":
                assert step3e_provenance is not None
                contested_ids = step3e_provenance["headroom_receipt"][
                    "contested_row_ids"
                ]
                results[profile]["headroom_contested_cases"] = (
                    contested_case_outcomes(final_evaluation, contested_ids)
                )
                results[profile]["headroom_contested_curve"] = {
                    str(milestone): contested_case_outcomes(
                        development_curve[
                            f"active_actor_critic_{milestone}_transitions"
                        ],
                        contested_ids,
                    )
                    for milestone in STEP3E_ACTIVE_MILESTONES
                    if f"active_actor_critic_{milestone}_transitions"
                    in development_curve
                }
        finally:
            if environment is not None:
                environment.close()
            elif raw_environment is not None:
                raw_environment.close()

    step3e_invariants: dict[str, bool] | None = None
    if args.gate_mode == "step3e":
        expected_curve_keys = {
            "bc_initialization",
            "post_critic_warmup",
            *(
                f"active_actor_critic_{milestone}_transitions"
                for milestone in STEP3E_ACTIVE_MILESTONES
            ),
        }
        control = results["v3_equivalent"]
        treatment = results["risk_averse"]
        initial_values = list(initial_hashes.values())
        step3e_invariants = {
            "exact_profile_order": profiles == PROFILE_ORDER,
            "reward_profile_is_only_treatment": (
                control["training_roster_and_tapes"]
                == treatment["training_roster_and_tapes"]
                == training_contract
                and control["reward_profile"] == "v3_equivalent"
                and treatment["reward_profile"] == "risk_averse"
            ),
            "training_roster_and_tapes_match_both_arms": (
                control["training_roster_and_tapes"]
                == treatment["training_roster_and_tapes"]
                and control["training_roster_and_tapes"]["case_count"] == 192
            ),
            "development_tapes_match_every_curve_point": (
                len(
                    {
                        development_tape_contract_sha256(evaluation)
                        for result in results.values()
                        for evaluation in result["development_curve"].values()
                    }
                )
                == 1
            ),
            "all_initial_policy_actor_observation_and_return_hashes_match": (
                len(initial_values) == 2 and initial_values[0] == initial_values[1]
            ),
            "initial_development_evaluations_match_exactly": (
                control["development_curve"]["bc_initialization"]
                == treatment["development_curve"]["bc_initialization"]
            ),
            "bc_rows_match_attempt06_canonical_evidence": (
                historical_step3_rows_hash(
                    control["development_curve"]["bc_initialization"]["rows"]
                )
                == STEP3E_ATTEMPT06_BC_ROWS_SHA256
            ),
            "post_warmup_actor_hashes_match": (
                control["actor_sha256_after_warmup"]
                == treatment["actor_sha256_after_warmup"]
            ),
            "actor_byte_identical_within_each_warmup": all(
                result["critic_warmup"]["actor_parameters_byte_identical"]
                for result in results.values()
            ),
            "fixed_observation_rms_unchanged_within_each_arm": all(
                result["critic_warmup"]["observation_rms_before_sha256"]
                == result["critic_warmup"]["observation_rms_after_sha256"]
                == initial_hashes[profile]["observation_rms_sha256"]
                == result["vecnormalize"]["observation_rms_sha256"]
                for profile, result in results.items()
            ),
            "initial_return_rms_hashes_match": (
                initial_values[0]["return_rms_sha256"]
                == initial_values[1]["return_rms_sha256"]
            ),
            "return_rms_continuity_without_reset_both_arms": all(
                step3e_return_rms_continuity_valid(
                    result,
                    initial_hashes[profile]["return_rms_sha256"],
                    warmup_transitions=result[
                        "critic_warmup_environment_transitions"
                    ],
                    active_transitions=STEP3E_ACTIVE_TRANSITIONS,
                )
                for profile, result in results.items()
            ),
            "equal_active_budget_and_registered_adaptive_warmup_both_arms": all(
                result["active_actor_critic_transitions"]
                == STEP3E_ACTIVE_TRANSITIONS
                and STEP3E_CRITIC_WARMUP_TRANSITIONS
                <= result["critic_warmup_environment_transitions"]
                <= STEP3E_CRITIC_WARMUP_MAX_TRANSITIONS
                and result["critic_warmup_environment_transitions"]
                % rollout_size
                == 0
                and result["total_environment_transitions"]
                == result["critic_warmup_environment_transitions"]
                + STEP3E_ACTIVE_TRANSITIONS
                for result in results.values()
            ),
            "only_registered_evaluation_milestones": all(
                set(result["development_curve"]) == expected_curve_keys
                for result in results.values()
            ),
            "critic_explained_variance_gate_passed_both_arms": all(
                result["critic_warmup"]["gate_passed"]
                for result in results.values()
            ),
            "final_explained_variance_above_threshold_both_arms": all(
                result["active_training"][
                    "fresh_rollout_final_explained_variance"
                ]
                is not None
                and result["active_training"][
                    "fresh_rollout_final_explained_variance"
                ]
                > args.critic_ev_threshold
                for result in results.values()
            ),
            "target_kl_guard_enabled_both_arms": all(
                result["active_training"][
                    "target_kl_guard_enabled_on_every_iteration"
                ]
                for result in results.values()
            ),
            "obsolete_max_kl_ceiling_not_applied": all(
                result["active_training"]["obsolete_max_kl_ceiling_applied"]
                is False
                for result in results.values()
            ),
            "exact_diagnostic_row_counts_and_schema_both_arms": all(
                diagnostic_rows_valid(
                    result["critic_warmup"]["iterations"],
                    expected_count=(
                        result["critic_warmup_environment_transitions"]
                        // rollout_size
                    ),
                )
                and diagnostic_rows_valid(
                    result["active_training"]["iterations"], expected_count=200
                )
                for result in results.values()
            ),
            "all_curve_hard_violations_zero": all(
                all_evaluation_hard_violations_zero(result)
                for result in results.values()
            ),
            "all_curve_conservation_residuals_exactly_zero": all(
                all_evaluation_conservation_residuals_zero(result)
                for result in results.values()
            ),
            "four_headroom_contested_rows_present_both_arms": all(
                result["headroom_contested_cases"]["case_count"] == 4
                for result in results.values()
            ),
            "contested_outcomes_recorded_at_200k_500k_1m_both_arms": all(
                set(result["headroom_contested_curve"])
                == {str(milestone) for milestone in STEP3E_ACTIVE_MILESTONES}
                for result in results.values()
            ),
            "v3_control_200k_rows_match_attempt06": (
                historical_step3_rows_hash(
                    control["development_curve"].get(
                        "active_actor_critic_200000_transitions", {"rows": []}
                    )["rows"]
                )
                == STEP3E_ATTEMPT06_200K_ROWS_SHA256
            ),
            "v3_control_200k_internal_state_matches_attempt06": (
                control["training_milestone_states"].get("200000", {}).get(
                    "policy_sha256"
                )
                == STEP3E_ATTEMPT06_POLICY_200K_SHA256
                and control["training_milestone_states"]
                .get("200000", {})
                .get("return_rms_sha256")
                == STEP3E_ATTEMPT06_RETURN_RMS_200K_SHA256
            ),
            "development_only_no_final_split_used": True,
        }

    payload: dict[str, Any] = {
        "schema_version": 1,
        "tool": "smoke_train_v4.py",
        "status": (
            "supervisor_corrected_step3d_passed_step3e_matched_reward_nonauthorizing"
            if args.gate_mode == "step3e"
            else "developmental_ppo_gate_nonauthorizing"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorizing": False,
        "split": "dev",
        "final_split_used": False,
        "same_tapes": (
            step3e_invariants["training_roster_and_tapes_match_both_arms"]
            and step3e_invariants["development_tapes_match_every_curve_point"]
            if args.gate_mode == "step3e" and step3e_invariants is not None
            else True
        ),
        "gate_mode": args.gate_mode,
        "treatment": (
            "optimizer_learning_from_shared_bc"
            if args.gate_mode == "learning"
            else "reward_profile_only"
        ),
        "uses_training_split_for_learning": True,
        "uses_development_split_for_evaluation": True,
        "uses_final_split": False,
        "authorizes_training": False,
        "selects_or_exports_policy": False,
        "diagnostic_checkpoints_persisted": False,
        "resumable_from_receipt": False,
        "config": {
            "active_actor_critic_transition_budget_per_profile": (
                args.transitions
            ),
            "active_actor_critic_budget_excludes_critic_warmup": True,
            "critic_warmup_min_transitions_per_profile": (
                args.critic_warmup_min_transitions
            ),
            "critic_warmup_max_transitions_per_profile": (
                args.critic_warmup_max_transitions
            ),
            "critic_warmup_explained_variance_threshold": (
                args.critic_ev_threshold
            ),
            "critic_warmup_rule": (
                "same 50k minimum / EV>0.5 stop / 100k maximum algorithm in "
                "both arms; realized duration may differ only because reward_profile "
                "changes critic targets"
                if args.gate_mode == "step3e"
                else "adaptive explained-variance threshold"
            ),
            "minimum_total_environment_transitions_per_profile": (
                args.transitions + args.critic_warmup_min_transitions
            ),
            "maximum_total_environment_transitions_per_profile": (
                args.transitions + args.critic_warmup_max_transitions
            ),
            "simulator_lanes": args.lanes,
            "n_steps_per_lane": args.n_steps,
            "rollout_size": rollout_size,
            "batch_size": args.batch_size,
            "n_epochs": 5,
            "learning_rate": args.learning_rate,
            "learning_rate_schedule": "constant",
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.15,
            "ent_coef": args.ent_coef,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "log_std_init": -1.5,
            "target_kl": args.target_kl,
            **(
                {
                    "target_kl_semantics": (
                        "SB3 per-minibatch check before the offending update; "
                        "logger reports the final partial-epoch mean"
                    ),
                    "obsolete_max_kl_ceiling_applied": False,
                }
                if args.gate_mode == "step3e"
                else {
                    "reported_approx_kl_stability_limit": (
                        REPORTED_APPROX_KL_STABILITY_MULTIPLIER * args.target_kl
                    ),
                    "reported_approx_kl_stability_multiplier": (
                        REPORTED_APPROX_KL_STABILITY_MULTIPLIER
                    ),
                    "reported_approx_kl_stability_definition": (
                        REPORTED_APPROX_KL_STABILITY_DEFINITION
                    ),
                }
            ),
            "use_sde": False,
            "vec_normalize": True,
            "observation_rms_frozen_during_ppo": (
                args.freeze_observation_rms
            ),
            "return_rms_contract": (
                "identical initial state and uninterrupted per-arm updates; "
                "final hashes may differ only as a consequence of reward_profile"
                if args.gate_mode == "step3e"
                else "standard VecNormalize updates"
            ),
            "policy_seed": args.policy_seed,
            "profiles": list(profiles),
        },
        "actual_transition_counts_by_profile": {
            profile: {
                "critic_warmup_environment_transitions": result[
                    "critic_warmup_environment_transitions"
                ],
                "active_actor_critic_transitions": result[
                    "active_actor_critic_transitions"
                ],
                "total_environment_transitions": result[
                    "total_environment_transitions"
                ],
            }
            for profile, result in results.items()
        },
        "behavior_cloning": bc_receipt,
        "initialization_match": {
            "all_initialization_hashes_match": len(
                {
                    json.dumps(value, sort_keys=True)
                    for value in initial_hashes.values()
                }
            )
            <= 1,
            "optimizer_state_empty_at_start": True,
            "arms": initial_hashes,
        },
        "profiles": results,
        **(
            {
                "step3e_authorization_and_provenance": step3e_provenance,
                "step3e_matched_invariants": step3e_invariants,
                "source_identity": {
                    "before": step3e_sources_before,
                },
            }
            if args.gate_mode == "step3e"
            else {}
        ),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    if args.gate_mode == "learning":
        gate_passed = results["v3_equivalent"]["learning_gate"][
            "gate_passed"
        ]
        payload["gate"] = {
            **results["v3_equivalent"]["learning_gate"],
            "critic_explained_variance_passed": results["v3_equivalent"][
                "critic_warmup"
            ]["gate_passed"],
            **{
                key: results["v3_equivalent"]["active_training"][key]
                for key in (
                    "reported_approx_kl_max",
                    "reported_approx_kl_stability_limit",
                    "reported_approx_kl_stability_multiplier",
                    "reported_approx_kl_stability_definition",
                    "reported_approx_kl_stable",
                )
            },
            "early_stop_rows": results["v3_equivalent"][
                "active_training"
            ]["early_stop_rows"],
            "diagnosis_required": not gate_passed,
        }
    elif args.gate_mode == "reward":
        baseline = results["v3_equivalent"]["development"]
        risk_averse = results["risk_averse"]["development"]
        gain = risk_averse["solved_count"] - baseline["solved_count"]
        gate_passed = (
            gain >= args.minimum_solve_gain
            and baseline["hard_violation_count"] == 0
            and baseline["maximum_conservation_residual"] == 0.0
            and risk_averse["hard_violation_count"] == 0
            and risk_averse["maximum_conservation_residual"] == 0.0
        )
        payload["gate"] = {
            "risk_averse_solved_gain": gain,
            "risk_averse_mean_resilience_auc_gain": round(
                risk_averse["mean_resilience_auc"]
                - baseline["mean_resilience_auc"],
                10,
            ),
            "paired_outcomes": paired_comparison(baseline, risk_averse),
            "minimum_solve_gain": args.minimum_solve_gain,
            "gate_passed": gate_passed,
            "diagnosis_required": not gate_passed,
        }
    else:
        assert step3e_invariants is not None
        assert step3e_provenance is not None
        baseline = results["v3_equivalent"]["development"]
        risk_averse = results["risk_averse"]["development"]
        matched_contract_passed = all(step3e_invariants.values())
        risk_500k = results["risk_averse"]["development_curve"].get(
            "active_actor_critic_500000_transitions"
        )
        carry = (
            step3e_carry_decision(
                baseline,
                risk_500k,
                risk_averse,
                step3e_provenance["headroom_receipt"]["contested_row_ids"],
            )
            if matched_contract_passed and risk_500k is not None
            else {
                "decision": "inconclusive_invalid_matched_run",
                "interpretation": (
                    "one or more preregistered matching or physics invariants failed"
                ),
            }
        )
        gate_passed = matched_contract_passed
        control_contested = results["v3_equivalent"][
            "headroom_contested_cases"
        ]
        risk_contested = results["risk_averse"]["headroom_contested_cases"]
        payload["gate"] = {
            "original_compound_step3d_gate_passed": False,
            "corrected_step3d_declared_passed_by_supervisor": True,
            "step3e_comparison_interpretable": matched_contract_passed,
            "risk_averse_solved_gain": (
                risk_averse["solved_count"] - baseline["solved_count"]
            ),
            "risk_averse_contested_conversion_gain": (
                risk_contested["converted_from_step3_best_ppo_unsolved_count"]
                - control_contested[
                    "converted_from_step3_best_ppo_unsolved_count"
                ]
            ),
            "risk_averse_mean_minimum_tail_margin_gain": round(
                risk_averse["mean_minimum_tail_margin"]
                - baseline["mean_minimum_tail_margin"],
                10,
            ),
            "risk_averse_mean_resilience_auc_gain": round(
                risk_averse["mean_resilience_auc"]
                - baseline["mean_resilience_auc"],
                10,
            ),
            "paired_outcomes": paired_comparison(baseline, risk_averse),
            "per_arm_contested_outcomes": {
                "v3_equivalent": control_contested,
                "risk_averse": risk_contested,
            },
            "carry_decision": carry,
            "matched_run_validity_passed": matched_contract_passed,
            "scientific_outcome": carry["decision"],
            "target_kl_diagnostics_are_non_gating": True,
            # Retained as the process exit-status field. For Step 3e this means
            # the matched experiment is valid, not that risk_averse won.
            "gate_passed": gate_passed,
            "diagnosis_required": not gate_passed,
        }

    if args.gate_mode == "step3e":
        assert step3e_output is not None
        assert protected_v3_before is not None
        assert step3e_sources_before is not None
        step3e_sources_after = step3e_source_identity()
        if step3e_sources_after != step3e_sources_before:
            raise SmokeError("executing Step 3e sources changed during the run")
        payload["source_identity"] = {
            "unchanged": True,
            "before": step3e_sources_before,
            "after": step3e_sources_after,
        }
        protected_v3_after = protected_v3_snapshot(
            step3e_provenance["protected_v3_expected_files_sha256"]
        )
        if protected_v3_after != protected_v3_before:
            raise SmokeError("protected-v3 snapshot changed during Step 3e")
        payload["protected_v3"] = {
            "unchanged": True,
            "before": protected_v3_before,
            "after": protected_v3_after,
        }
        write_json_atomic(step3e_output, payload)
    elif args.json_output is not None:
        write_json_atomic(args.json_output.resolve(), payload)
    printable = json.loads(json.dumps(payload, allow_nan=False))
    for profile in printable.get("profiles", {}).values():
        profile["development"] = {
            key: value
            for key, value in profile["development"].items()
            if key != "rows"
        }
        profile["development_curve"] = {
            milestone: {
                key: value
                for key, value in evaluation.items()
                if key != "rows"
            }
            for milestone, evaluation in profile["development_curve"].items()
        }
        for section_name in ("critic_warmup", "active_training"):
            section = profile.get(section_name, {})
            iterations = section.pop("iterations", [])
            section["iterations_omitted_from_stdout"] = len(iterations)
    print(json.dumps(printable, allow_nan=False, indent=2, sort_keys=True))
    return 0 if gate_passed else 3


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    raise SystemExit(main())
