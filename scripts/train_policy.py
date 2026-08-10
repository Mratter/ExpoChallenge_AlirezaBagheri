#!/usr/bin/env python3
"""Train one canonical city-recovery policy and record development evidence.

The production flow is intentionally linear: prepare a BC/DAgger actor, warm
the critic while the actor is frozen, run PPO actor-critic updates, evaluate
deterministically on the development roster, and create a new receipt. Training
uses only authored training cases; this module never imports or evaluates the
single-use final split.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

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

from backend.app.city.environment import (  # noqa: E402
    ACTION_ORDER,
    OBSERVATION_ORDER,
    CityRecoveryEnv,
    CyclingScenarioEnv,
)
from backend.app.city.outcome import summarize_trajectory  # noqa: E402
from backend.app.city.physics import SERVICES  # noqa: E402
from backend.app.city.planners import preparedness_teacher_action  # noqa: E402
from backend.app.city.scenarios import (  # noqa: E402
    DEVELOPMENT_FAMILIES,
    DEVELOPMENT_SEEDS,
    TRAINING_FAMILIES,
    TRAINING_SEEDS,
    generate_disaster_tape,
)
from backend.app.shared_evidence import canonical_hash  # noqa: E402
from scripts.training_artifacts import (  # noqa: E402
    TrainingArtifactError,
    persist_checkpoint_bundle,
)

TOOL_ID = "train_policy.py"
DEFAULT_TRANSITIONS = 8_000_000
DEFAULT_LANES = 20
DEFAULT_N_STEPS = 250
DEFAULT_BATCH_SIZE = 500
DEFAULT_POLICY_SEED = 37_017
DEFAULT_BC_EPOCHS = 15
DEFAULT_LEARNING_RATE = 7.5e-5
DEFAULT_TARGET_KL = 0.02
DEFAULT_ENT_COEF = 0.003
DEFAULT_REWARD_PROFILE = "v3_equivalent"
DEFAULT_PREPAREDNESS_ALIGNMENT_COEFFICIENT = 10.0
DEFAULT_CRITIC_WARMUP_MIN_TRANSITIONS = 50_000
DEFAULT_CRITIC_WARMUP_MAX_TRANSITIONS = 100_000
CRITIC_EXPLAINED_VARIANCE_GATE = 0.5
EVALUATION_MILESTONES = (200_000, 500_000, 1_000_000)
DAGGER_BETA_SCHEDULE = (1.0, 0.0, 0.0, 0.0)
DIAGNOSTIC_NAMES = (
    "explained_variance",
    "approx_kl",
    "clip_fraction",
    "entropy_loss",
    "value_loss",
    "policy_gradient_loss",
)
DEVELOPMENT_CASE_COUNT = len(DEVELOPMENT_FAMILIES) * len(DEVELOPMENT_SEEDS)
CANONICAL_DEVELOPMENT_CASE_COUNT = 200


class TrainingError(RuntimeError):
    """Raised when the development-only training contract cannot be honored."""


class FreezableRunningMeanStd(RunningMeanStd):
    """Running moments that may be fixed while normalization stays enabled."""

    def __init__(self, *, shape: tuple[int, ...]) -> None:
        super().__init__(shape=shape)
        self.frozen = False

    def update(self, array: np.ndarray) -> None:
        if not self.frozen:
            super().update(array)


class InstrumentedPPO(PPO):
    """PPO that retains every rollout-update diagnostic for the receipt."""

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
        metrics: dict[str, float] = {}
        for name in DIAGNOSTIC_NAMES:
            value = float(logged[f"train/{name}"])
            if not np.isfinite(value):
                raise TrainingError(f"non-finite PPO diagnostic: {name}")
            metrics[name] = value

        action_std = torch.exp(self.policy.log_std.detach()).cpu().numpy()
        if not np.all(np.isfinite(action_std)):
            raise TrainingError("non-finite PPO action standard deviation")
        epochs_completed = int(self._n_updates) - updates_before
        self.training_iterations.append(
            {
                "iteration": len(self.training_iterations) + 1,
                "phase": self.diagnostic_phase,
                "total_transitions": int(self.num_timesteps),
                "phase_transitions": (
                    int(self.num_timesteps) - self.diagnostic_phase_start
                ),
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
    """Build the complete authored training roster and its tape seeds."""

    training_ids = {family.id for family in TRAINING_FAMILIES}
    development_ids = {family.id for family in DEVELOPMENT_FAMILIES}
    if training_ids & development_ids:
        raise TrainingError("training and development scenario families overlap")
    return [
        (family.build(seed), family.tape_seed(seed))
        for family in TRAINING_FAMILIES
        for seed in TRAINING_SEEDS
    ]


def disaster_tape_sha256(schedule: Sequence[Any]) -> str:
    """Hash one generated tape as data, not as a source-code seal."""

    return canonical_hash([asdict(shock) for shock in schedule])


def training_roster_and_tapes_contract() -> dict[str, Any]:
    """Describe the exact training examples consumed by this run."""

    rows: list[dict[str, Any]] = []
    for family in TRAINING_FAMILIES:
        for case_seed in TRAINING_SEEDS:
            scenario = family.build(case_seed)
            tape_seed = family.tape_seed(case_seed)
            schedule = generate_disaster_tape(scenario, tape_seed)
            rows.append(
                {
                    "family_id": family.id,
                    "case_seed": case_seed,
                    "tape_seed": tape_seed,
                    "scenario": scenario.model_dump(mode="json"),
                    "tape_sha256": disaster_tape_sha256(schedule),
                }
            )
    return {
        "case_count": len(rows),
        "contract_sha256": canonical_hash(rows),
    }


def runtime_versions() -> dict[str, str]:
    """Record the libraries and operating system used for training."""

    packages = (
        "numpy",
        "torch",
        "stable-baselines3",
        "gymnasium",
        "onnx",
        "onnxruntime",
    )
    return {
        "python": platform.python_version(),
        "operating_system": platform.platform(),
        **{name: importlib.metadata.version(name) for name in packages},
    }


@dataclass(frozen=True)
class TrainingLaneFactory:
    """Create one spawn-safe lane over a deterministic roster rotation."""

    lane: int
    reward_profile: str = DEFAULT_REWARD_PROFILE
    preparedness_alignment_coefficient: float = (
        DEFAULT_PREPAREDNESS_ALIGNMENT_COEFFICIENT
    )

    def __call__(self) -> CyclingScenarioEnv:
        for variable in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            if os.environ.get(variable) != "1":
                raise TrainingError(f"spawned worker thread cap drifted: {variable}")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        scenarios = training_scenarios()
        offset = self.lane % len(scenarios)
        rotated = scenarios[offset:] + scenarios[:offset]
        return CyclingScenarioEnv(
            rotated,
            collect_evidence=False,
            reward_profile=self.reward_profile,
            preparedness_alignment_coefficient=(
                self.preparedness_alignment_coefficient
            ),
        )


def spawn_environment(
    lanes: int,
    seed: int,
    *,
    reward_profile: str = DEFAULT_REWARD_PROFILE,
    preparedness_alignment_coefficient: float = (
        DEFAULT_PREPAREDNESS_ALIGNMENT_COEFFICIENT
    ),
) -> SubprocVecEnv:
    """Start deterministic subprocess lanes for on-policy training."""

    factories: list[Callable[[], CyclingScenarioEnv]] = [
        TrainingLaneFactory(
            lane,
            reward_profile=reward_profile,
            preparedness_alignment_coefficient=(
                preparedness_alignment_coefficient
            ),
        )
        for lane in range(lanes)
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
    learning_rate: float = DEFAULT_LEARNING_RATE,
    target_kl: float = DEFAULT_TARGET_KL,
    ent_coef: float = DEFAULT_ENT_COEF,
) -> InstrumentedPPO:
    """Construct PPO with the adopted intermediate optimizer regime."""

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


def _validate_imitation_dataset(
    observations: np.ndarray,
    targets: np.ndarray,
    *,
    label: str,
) -> None:
    scenarios = training_scenarios()
    expected_rows = sum(scenario.horizon_days for scenario, _ in scenarios)
    if (
        observations.shape != (expected_rows, len(OBSERVATION_ORDER))
        or targets.shape != (expected_rows, len(ACTION_ORDER))
        or len(SERVICES) != 5
        or not np.all(np.isfinite(observations))
        or not np.all(np.isfinite(targets))
    ):
        raise TrainingError(f"{label} dataset contract drifted")


def behavior_cloning_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Collect public-state teacher demonstrations on the training roster."""

    observations: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for scenario, tape_seed in training_scenarios():
        schedule = generate_disaster_tape(scenario, tape_seed)
        environment = CityRecoveryEnv(
            scenario,
            tape_seed,
            schedule,
            collect_evidence=False,
        )
        observation, _ = environment.reset(seed=tape_seed)
        terminated = False
        while not terminated:
            action, evidence = preparedness_teacher_action(observation)
            if evidence.get("future_tape_visible") is not False:
                raise TrainingError("BC teacher exposed future tape information")
            observations.append(np.asarray(observation, dtype=np.float32).copy())
            targets.append(np.asarray(action, dtype=np.float32).copy())
            observation, _, terminated, _, _ = environment.step(action)

    observation_array = np.asarray(observations, dtype=np.float32)
    target_array = np.asarray(targets, dtype=np.float32)
    _validate_imitation_dataset(
        observation_array,
        target_array,
        label="BC",
    )
    return observation_array, target_array


def array_digest(*arrays: np.ndarray) -> str:
    """Hash numerical state with dtype and shape included."""

    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def state_digest(state: dict[str, torch.Tensor]) -> str:
    """Hash a named Torch state deterministically."""

    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def actor_state(model: PPO) -> dict[str, torch.Tensor]:
    """Return exactly the parameters that determine actions."""

    state = {
        f"policy_net.{name}": parameter.detach().cpu().clone()
        for name, parameter in (
            model.policy.mlp_extractor.policy_net.named_parameters()
        )
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
    """Freeze all action parameters and expose only critic parameters."""

    for parameter in model.policy.parameters():
        parameter.requires_grad_(False)
    critic_parameters = [
        *model.policy.mlp_extractor.value_net.parameters(),
        *model.policy.value_net.parameters(),
    ]
    for parameter in critic_parameters:
        parameter.requires_grad_(True)
    trainable = [
        parameter for parameter in model.policy.parameters() if parameter.requires_grad
    ]
    if {id(parameter) for parameter in trainable} != {
        id(parameter) for parameter in critic_parameters
    }:
        raise TrainingError("critic warm-up trainable parameters drifted")
    return sum(parameter.numel() for parameter in trainable)


def unfreeze_policy(model: PPO) -> None:
    """Enable joint actor-critic optimization after critic warm-up."""

    for parameter in model.policy.parameters():
        parameter.requires_grad_(True)
    if not all(parameter.requires_grad for parameter in model.policy.parameters()):
        raise TrainingError("policy unfreeze contract drifted")


def rms_state(rms: RunningMeanStd) -> dict[str, Any]:
    """Copy running moments into a serializable-state representation."""

    return {
        "mean": np.asarray(rms.mean, dtype=np.float64).copy(),
        "var": np.asarray(rms.var, dtype=np.float64).copy(),
        "count": float(rms.count),
    }


def load_rms_state(rms: RunningMeanStd, state: dict[str, Any]) -> None:
    """Restore running moments without replacing the RMS object."""

    rms.mean = np.asarray(state["mean"], dtype=np.float64).copy()
    rms.var = np.asarray(state["var"], dtype=np.float64).copy()
    rms.count = float(state["count"])


def rms_digest(state: dict[str, Any]) -> str:
    """Hash copied running moments."""

    return array_digest(
        np.asarray(state["mean"], dtype=np.float64),
        np.asarray(state["var"], dtype=np.float64),
        np.asarray([state["count"]], dtype=np.float64),
    )


def normalize_observations(
    observations: np.ndarray,
    state: dict[str, Any],
) -> np.ndarray:
    """Apply the same observation transform used by VecNormalize."""

    normalized = (
        np.asarray(observations, dtype=np.float64) - state["mean"]
    ) / np.sqrt(state["var"] + 1e-8)
    return np.clip(normalized, -10.0, 10.0).astype(np.float32)


def policy_rollout_dataset(
    model: PPO,
    observation_rms: dict[str, Any],
    *,
    normalize_observation: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Label states visited by the current actor with the public teacher."""

    observations: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for scenario, tape_seed in training_scenarios():
        environment = CityRecoveryEnv(
            scenario,
            tape_seed,
            collect_evidence=False,
        )
        observation, _ = environment.reset(seed=tape_seed)
        terminated = False
        while not terminated:
            teacher_action, evidence = preparedness_teacher_action(observation)
            if evidence.get("future_tape_visible") is not False:
                raise TrainingError("DAgger teacher exposed future tape information")
            observations.append(np.asarray(observation, dtype=np.float32).copy())
            targets.append(np.asarray(teacher_action, dtype=np.float32).copy())
            policy_observation = np.asarray(observation, dtype=np.float32)
            if normalize_observation:
                policy_observation = normalize_observations(
                    policy_observation.reshape(1, -1),
                    observation_rms,
                )[0]
            rollout_action, _ = model.predict(
                policy_observation, deterministic=True
            )
            observation, _, terminated, _, _ = environment.step(rollout_action)

    observation_array = np.asarray(observations, dtype=np.float32)
    target_array = np.asarray(targets, dtype=np.float32)
    _validate_imitation_dataset(
        observation_array,
        target_array,
        label="DAgger",
    )
    return observation_array, target_array


def behavior_clone_policy(
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
    normalize_observation: bool = True,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
    """Fit the actor through BC/DAgger while leaving the critic untouched."""

    scenario, tape_seed = training_scenarios()[0]
    environment = DummyVecEnv(
        [
            lambda: CityRecoveryEnv(
                scenario,
                tape_seed,
                collect_evidence=False,
            )
        ]
    )
    try:
        model = build_model(
            environment,
            seed=seed,
            n_steps=n_steps,
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
        observation_rms = RunningMeanStd(shape=(len(OBSERVATION_ORDER),))
        observation_batches: list[np.ndarray] = []
        target_batches: list[np.ndarray] = []
        next_observations = observations
        next_targets = targets
        iteration_reports: list[dict[str, Any]] = []
        final_batch_loss = float("nan")
        full_dataset_loss = float("nan")

        for iteration, beta in enumerate(DAGGER_BETA_SCHEDULE):
            if iteration:
                next_observations, next_targets = policy_rollout_dataset(
                    model,
                    rms_state(observation_rms),
                    normalize_observation=normalize_observation,
                )
            observation_rms.update(next_observations)
            observation_batches.append(next_observations)
            target_batches.append(next_targets)
            cumulative_observations = np.concatenate(observation_batches)
            cumulative_targets = np.concatenate(target_batches)
            policy_observations = (
                normalize_observations(
                    cumulative_observations,
                    rms_state(observation_rms),
                )
                if normalize_observation
                else cumulative_observations
            )
            observation_tensor = torch.as_tensor(
                policy_observations,
                dtype=torch.float32,
            )
            target_tensor = torch.as_tensor(
                cumulative_targets,
                dtype=torch.float32,
            )
            model.policy.train()
            for _ in range(epochs):
                permutation = torch.randperm(
                    observation_tensor.shape[0],
                    generator=generator,
                )
                for start in range(0, observation_tensor.shape[0], 512):
                    indices = permutation[start : start + 512]
                    distribution = model.policy.get_distribution(
                        observation_tensor[indices]
                    )
                    predicted = distribution.distribution.mean
                    loss = torch.nn.functional.mse_loss(
                        predicted,
                        target_tensor[indices],
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor_parameters, 1.0)
                    optimizer.step()
                    final_batch_loss = float(loss.detach().cpu())

            model.policy.eval()
            with torch.no_grad():
                predicted = model.policy.get_distribution(
                    observation_tensor
                ).distribution.mean
                full_dataset_loss = float(
                    torch.nn.functional.mse_loss(
                        predicted,
                        target_tensor,
                    ).cpu()
                )
            iteration_reports.append(
                {
                    "iteration": iteration + 1,
                    "dagger_beta": beta,
                    "cumulative_observation_count": int(
                        observation_tensor.shape[0]
                    ),
                    "full_dataset_mse": full_dataset_loss,
                }
            )

        model.policy.eval()
        state = {
            name: value.detach().cpu().clone()
            for name, value in model.policy.state_dict().items()
        }
        observation_rms_state = rms_state(observation_rms)
        return state, observation_rms_state, {
            "teacher": "preparedness_teacher_action",
            "training_split_only": True,
            "dagger_beta_schedule": list(DAGGER_BETA_SCHEDULE),
            "iterations": len(DAGGER_BETA_SCHEDULE),
            "epochs_per_iteration": epochs,
            "batch_size": 512,
            "observation_count": int(cumulative_observations.shape[0]),
            "final_batch_mse": final_batch_loss,
            "full_dataset_mse": full_dataset_loss,
            "iteration_reports": iteration_reports,
            "dataset_sha256": array_digest(
                cumulative_observations,
                cumulative_targets,
            ),
            "policy_state_sha256": state_digest(state),
            "observation_normalization": normalize_observation,
            "observation_rms_sha256": rms_digest(observation_rms_state),
            "observation_rms_count": observation_rms_state["count"],
            "critic_unchanged_during_actor_only_imitation": True,
        }
    finally:
        environment.close()


def untrained_policy_state(
    *,
    seed: int,
    n_steps: int,
    batch_size: int,
    learning_rate: float,
    target_kl: float,
    ent_coef: float,
) -> dict[str, torch.Tensor]:
    """Build the byte-defined random initialization used by the no-BC ablation."""

    scenario, tape_seed = training_scenarios()[0]
    environment = DummyVecEnv(
        [
            lambda: CityRecoveryEnv(
                scenario,
                tape_seed,
                collect_evidence=False,
            )
        ]
    )
    try:
        model = build_model(
            environment,
            seed=seed,
            n_steps=n_steps,
            batch_size=min(batch_size, n_steps),
            learning_rate=learning_rate,
            target_kl=target_kl,
            ent_coef=ent_coef,
        )
        return {
            name: value.detach().cpu().clone()
            for name, value in model.policy.state_dict().items()
        }
    finally:
        environment.close()


def evaluate_development(
    model: PPO,
    normalizer: VecNormalize,
) -> dict[str, Any]:
    """Evaluate a policy deterministically on all development cases."""

    rows: list[dict[str, Any]] = []
    for family in DEVELOPMENT_FAMILIES:
        for case_seed in DEVELOPMENT_SEEDS:
            scenario = family.build(case_seed)
            tape_seed = family.tape_seed(case_seed)
            schedule = generate_disaster_tape(scenario, tape_seed)
            environment = CityRecoveryEnv(
                scenario,
                tape_seed,
                schedule,
                collect_evidence=True,
            )
            observation, _ = environment.reset(seed=tape_seed)
            terminated = False
            while not terminated:
                normalized = normalizer.normalize_obs(
                    np.asarray(observation, dtype=np.float32).copy()
                )
                action, _ = model.predict(normalized, deterministic=True)
                observation, _, terminated, _, _ = environment.step(action)

            summary = summarize_trajectory(
                "trained_policy",
                environment.trajectory,
                scenario,
            )
            outcome = summary["absolute_outcome"]
            tail_minimum_services = np.asarray(
                outcome["tail_minimum_services"],
                dtype=np.float64,
            )
            recovery_targets = np.asarray(
                outcome["recovery_targets"],
                dtype=np.float64,
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
                raise TrainingError("non-finite development evaluation outcome")
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
                    "target_met_by_service": list(
                        outcome["target_met_by_service"]
                    ),
                    "tail_minimum_services": tail_minimum_services.tolist(),
                    "recovery_targets": recovery_targets.tolist(),
                }
            )

    expected_count = DEVELOPMENT_CASE_COUNT
    if expected_count != CANONICAL_DEVELOPMENT_CASE_COUNT:
        raise TrainingError("canonical development roster must contain 200 cases")
    if len(rows) != expected_count or len({row["row_id"] for row in rows}) != len(
        rows
    ):
        raise TrainingError("development evaluation roster drifted")
    reasons = Counter(
        reason
        for row in rows
        if not row["solved"]
        for reason in row["reason_codes"]
    )
    result = {
        "case_count": len(rows),
        "solved_count": sum(row["solved"] for row in rows),
        "solve_rate": sum(row["solved"] for row in rows) / len(rows),
        "mean_resilience_auc": round(
            fmean(row["resilience_auc"] for row in rows),
            10,
        ),
        "mean_minimum_tail_margin": round(
            fmean(row["minimum_tail_margin"] for row in rows),
            10,
        ),
        "hard_violation_count": sum(
            row["hard_violation_count"] for row in rows
        ),
        "maximum_conservation_residual": max(
            row["max_conservation_residual"] for row in rows
        ),
        "failure_reason_code_histogram": dict(sorted(reasons.items())),
        "rows": rows,
    }
    if (
        result["hard_violation_count"] != 0
        or result["maximum_conservation_residual"] != 0.0
    ):
        raise TrainingError("development evaluation violated city physics")
    return result


def evaluate_development_frozen(
    model: PPO,
    normalizer: VecNormalize,
) -> dict[str, Any]:
    """Evaluate without updating either normalization stream."""

    previous_training = normalizer.training
    previous_norm_reward = normalizer.norm_reward
    normalizer.training = False
    normalizer.norm_reward = False
    try:
        return evaluate_development(model, normalizer)
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


def target_kl_diagnostics(
    iterations: Sequence[dict[str, Any]],
    target_kl: float,
) -> dict[str, Any]:
    """Summarize SB3's early-stop guard without imposing a second KL ceiling."""

    values = [float(iteration["approx_kl"]) for iteration in iterations]
    early_stops = [
        iteration
        for iteration in iterations
        if iteration["early_stop_detected_before_final_epoch"]
    ]
    return {
        "target_kl": target_kl,
        "reported_approx_kl_max": max(values) if values else None,
        "reported_approx_kl_rows_above_target": sum(
            value > target_kl for value in values
        ),
        "target_kl_guard_enabled_on_every_iteration": bool(iterations)
        and all(iteration["target_kl_guard_enabled"] for iteration in iterations),
        "early_stop_detected_before_final_epoch_count": len(early_stops),
        "early_stop_detection_count_is_lower_bound": True,
        "additional_reported_kl_ceiling_applied": False,
        "semantics": (
            "Stable-Baselines3 checks approximate KL after a minibatch update and "
            "then stops the remaining epoch loop. A reported maximum above the "
            "target is expected diagnostic evidence that the guard engaged."
        ),
    }


def early_stop_row_summary(
    iterations: Sequence[dict[str, Any]],
) -> dict[str, int]:
    """Count update rows that stopped early or completed every PPO epoch."""

    early_stop_rows = sum(
        bool(iteration["early_stop_detected_before_final_epoch"])
        for iteration in iterations
    )
    return {
        "iteration_row_count": len(iterations),
        "early_stop_row_count": early_stop_rows,
        "full_epoch_row_count": len(iterations) - early_stop_rows,
    }


def diagnostic_rows_valid(
    rows: Sequence[dict[str, Any]],
    expected_count: int,
) -> bool:
    """Check the per-iteration telemetry schema and action dimensionality."""

    required = {
        *DIAGNOSTIC_NAMES,
        "action_std_mean",
        "action_std_min",
        "action_std_max",
        "action_std_by_dimension",
        "target_kl_guard_enabled",
    }
    return len(rows) == expected_count and all(
        required <= set(row)
        and len(row["action_std_by_dimension"]) == len(ACTION_ORDER)
        for row in rows
    )


def return_rms_continuity_valid(
    result: dict[str, Any],
    initial_return_rms_sha256: str,
    *,
    warmup_transitions: int,
    active_transitions: int,
) -> bool:
    """Verify that reward moments continued without reset across both phases."""

    warmup = result["critic_warmup"]
    final = result["normalization"]
    before_count = float(warmup["return_rms_before_count"])
    after_count = float(warmup["return_rms_after_count"])
    final_count = float(final["return_rms_count"])
    return bool(
        warmup["return_rms_before_sha256"] == initial_return_rms_sha256
        and before_count < after_count <= final_count
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


def write_receipt(path: Path, payload: dict[str, Any]) -> None:
    """Atomically create a receipt while refusing to replace existing evidence."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise TrainingError(f"refusing to overwrite training receipt: {path}")
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    rendered = json.dumps(
        payload,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.rename(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the single canonical training configuration."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transitions", type=int, default=DEFAULT_TRANSITIONS)
    parser.add_argument("--lanes", type=int, default=DEFAULT_LANES)
    parser.add_argument("--n-steps", type=int, default=DEFAULT_N_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--policy-seed", type=int, default=DEFAULT_POLICY_SEED)
    parser.add_argument("--bc-epochs", type=int, default=DEFAULT_BC_EPOCHS)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
    )
    parser.add_argument("--target-kl", type=float, default=DEFAULT_TARGET_KL)
    parser.add_argument("--ent-coef", type=float, default=DEFAULT_ENT_COEF)
    parser.add_argument(
        "--reward-profile",
        choices=("v3_equivalent", "risk_averse"),
        default=DEFAULT_REWARD_PROFILE,
        help="training reward treatment; runtime physics and outcome stay fixed",
    )
    parser.add_argument(
        "--preparedness-alignment-coefficient",
        type=float,
        default=None,
        help=(
            "override preparedness alignment reward weight; defaults to 10.0 "
            "for v3_equivalent and 2.0 for risk_averse"
        ),
    )
    parser.add_argument(
        "--bc-warm-start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="initialize the actor from BC/DAgger rather than random weights",
    )
    parser.add_argument(
        "--vec-normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="apply VecNormalize observation and reward transforms",
    )
    parser.add_argument(
        "--critic-warmup-min-transitions",
        type=int,
        default=DEFAULT_CRITIC_WARMUP_MIN_TRANSITIONS,
    )
    parser.add_argument(
        "--critic-warmup-max-transitions",
        type=int,
        default=DEFAULT_CRITIC_WARMUP_MAX_TRANSITIONS,
    )
    parser.add_argument(
        "--critic-ev-threshold",
        type=float,
        default=CRITIC_EXPLAINED_VARIANCE_GATE,
    )
    parser.add_argument(
        "--freeze-observation-rms",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="keep BC observation moments fixed while reward moments update",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        required=True,
        help="new receipt path; existing files are never overwritten",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help=(
            "new directory for atomic milestone bundles; defaults beside the "
            "receipt"
        ),
    )
    return parser.parse_args(argv)


def reset_policy_seed(seed: int) -> None:
    """Reset every policy-side random generator."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_random_seed(seed, using_cuda=False)


def learning_milestones(
    total_transitions: int,
    rollout_size: int,
) -> list[int]:
    """Return visible PPO checkpoints ending exactly at the requested budget."""

    if total_transitions <= 0 or rollout_size <= 0 or total_transitions % rollout_size:
        raise TrainingError("PPO transitions must contain complete rollouts")
    milestones = [
        milestone
        for milestone in EVALUATION_MILESTONES
        if milestone <= total_transitions
    ]
    if total_transitions not in milestones:
        milestones.append(total_transitions)
    if any(milestone % rollout_size for milestone in milestones):
        raise TrainingError("evaluation milestones must contain complete rollouts")
    return milestones


def validate_runtime_config(args: argparse.Namespace) -> int:
    """Validate budgets and return the vectorized rollout size."""

    rollout_size = args.lanes * args.n_steps
    learning_milestones(args.transitions, rollout_size)
    if (
        args.lanes <= 0
        or args.n_steps <= 0
        or args.batch_size <= 0
        or rollout_size % args.batch_size
        or args.critic_warmup_min_transitions < rollout_size
        or args.critic_warmup_min_transitions % rollout_size
        or args.critic_warmup_max_transitions < args.critic_warmup_min_transitions
        or args.critic_warmup_max_transitions % rollout_size
    ):
        raise TrainingError(
            "training and critic warm-up must divide into complete rollouts, "
            "and batch size must divide the rollout"
        )
    if (
        args.bc_epochs <= 0
        or not np.isfinite(args.critic_ev_threshold)
        or not np.isfinite(args.learning_rate)
        or args.learning_rate <= 0.0
        or not np.isfinite(args.target_kl)
        or args.target_kl <= 0.0
        or not np.isfinite(args.ent_coef)
        or args.ent_coef < 0.0
        or (
            args.preparedness_alignment_coefficient is not None
            and (
                not np.isfinite(args.preparedness_alignment_coefficient)
                or args.preparedness_alignment_coefficient < 0.0
            )
        )
    ):
        raise TrainingError("optimizer or critic warm-up arguments are invalid")
    output = _resolve_output_path(args.json_output)
    if output.exists():
        raise TrainingError(f"refusing to overwrite training receipt: {output}")
    checkpoint_directory = resolve_checkpoint_directory(args)
    if checkpoint_directory.exists():
        raise TrainingError(
            "refusing to reuse checkpoint directory: "
            f"{checkpoint_directory}"
        )
    return rollout_size


def _resolve_output_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def resolve_checkpoint_directory(args: argparse.Namespace) -> Path:
    """Resolve the create-new checkpoint root associated with a receipt."""

    if args.checkpoint_dir is not None:
        return _resolve_output_path(args.checkpoint_dir)
    receipt = _resolve_output_path(args.json_output)
    return receipt.with_name(f"{receipt.stem}-checkpoints")


def resolved_training_config(
    args: argparse.Namespace,
    *,
    rollout_size: int,
    preparedness_alignment_coefficient: float,
) -> dict[str, Any]:
    """Build the single config object shared by bundles and the receipt."""

    return {
        "active_actor_critic_transitions": args.transitions,
        "critic_warmup_min_transitions": args.critic_warmup_min_transitions,
        "critic_warmup_max_transitions": args.critic_warmup_max_transitions,
        "critic_explained_variance_threshold": args.critic_ev_threshold,
        "lanes": args.lanes,
        "n_steps_per_lane": args.n_steps,
        "rollout_size": rollout_size,
        "batch_size": args.batch_size,
        "n_epochs": 5,
        "learning_rate": args.learning_rate,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.15,
        "ent_coef": args.ent_coef,
        "reward_profile": args.reward_profile,
        "preparedness_alignment_coefficient": (
            preparedness_alignment_coefficient
        ),
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "log_std_init": -1.5,
        "target_kl": args.target_kl,
        "target_kl_semantics": (
            "Stable-Baselines3 early-stops the remaining epoch loop after "
            "observing an update above the target"
        ),
        "use_sde": False,
        "vec_normalize": args.vec_normalize,
        "freeze_observation_rms": (
            args.freeze_observation_rms or not args.vec_normalize
        ),
        "policy_seed": args.policy_seed,
        "bc_epochs": args.bc_epochs,
        "bc_warm_start": args.bc_warm_start,
        "development_case_count": DEVELOPMENT_CASE_COUNT,
        "evaluation_milestones": learning_milestones(
            args.transitions,
            rollout_size,
        ),
    }


def _stdout_summary(payload: dict[str, Any]) -> dict[str, Any]:
    printable = json.loads(json.dumps(payload, allow_nan=False))
    for evaluation in printable["development_curve"].values():
        evaluation.pop("rows", None)
    printable["development"] = {
        key: value
        for key, value in printable["development"].items()
        if key != "rows"
    }
    for section_name in ("critic_warmup", "ppo"):
        section = printable[section_name]
        iterations = section.pop("iterations", [])
        section["iterations_omitted_from_stdout"] = len(iterations)
    return printable


def main(argv: Sequence[str] | None = None) -> int:
    """Run BC/DAgger -> critic warm-up -> PPO -> evaluation -> receipt."""

    args = parse_args(argv)
    rollout_size = validate_runtime_config(args)
    receipt_path = _resolve_output_path(args.json_output)
    checkpoint_directory = resolve_checkpoint_directory(args)

    torch.set_num_threads(min(12, os.cpu_count() or 1))
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    reset_policy_seed(args.policy_seed)

    started = time.perf_counter()
    preparedness_alignment_coefficient = (
        10.0
        if args.reward_profile == "v3_equivalent"
        else 2.0
    )
    if args.preparedness_alignment_coefficient is not None:
        preparedness_alignment_coefficient = float(
            args.preparedness_alignment_coefficient
        )
    training_config = resolved_training_config(
        args,
        rollout_size=rollout_size,
        preparedness_alignment_coefficient=(
            preparedness_alignment_coefficient
        ),
    )
    observations, targets = behavior_cloning_dataset()
    initial_state, initial_observation_rms, bc_receipt = behavior_clone_policy(
        observations,
        targets,
        seed=args.policy_seed,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        epochs=args.bc_epochs,
        learning_rate=args.learning_rate,
        target_kl=args.target_kl,
        ent_coef=args.ent_coef,
        normalize_observation=args.vec_normalize,
    )
    if not args.bc_warm_start:
        initial_state = untrained_policy_state(
            seed=args.policy_seed,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            target_kl=args.target_kl,
            ent_coef=args.ent_coef,
        )
        bc_receipt = {
            **bc_receipt,
            "actor_warm_start_applied": False,
            "counterfactual_untrained_policy_state_sha256": state_digest(
                initial_state
            ),
        }
    else:
        bc_receipt = {**bc_receipt, "actor_warm_start_applied": True}
    training_contract = training_roster_and_tapes_contract()

    raw_environment: SubprocVecEnv | None = None
    environment: VecNormalize | None = None
    try:
        reset_policy_seed(args.policy_seed)
        raw_environment = spawn_environment(
            args.lanes,
            args.policy_seed,
            reward_profile=args.reward_profile,
            preparedness_alignment_coefficient=(
                preparedness_alignment_coefficient
            ),
        )
        environment = VecNormalize(
            raw_environment,
            training=True,
            norm_obs=args.vec_normalize,
            norm_reward=args.vec_normalize,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=0.99,
            epsilon=1e-8,
        )
        if args.freeze_observation_rms or not args.vec_normalize:
            fixed_observation_rms = FreezableRunningMeanStd(
                shape=(len(OBSERVATION_ORDER),)
            )
            load_rms_state(fixed_observation_rms, initial_observation_rms)
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
        initial_hashes = {
            "policy_sha256": state_digest(model.policy.state_dict()),
            "actor_sha256": state_digest(actor_state(model)),
            "observation_rms_sha256": rms_digest(rms_state(environment.obs_rms)),
            "return_rms_sha256": rms_digest(rms_state(environment.ret_rms)),
        }
        if (
            args.bc_warm_start
            and initial_hashes["policy_sha256"]
            != bc_receipt["policy_state_sha256"]
        ):
            raise TrainingError("BC initialization changed during PPO construction")
        if (
            initial_hashes["observation_rms_sha256"]
            != bc_receipt["observation_rms_sha256"]
        ):
            raise TrainingError("observation RMS changed during PPO construction")
        if model.policy.optimizer.state:
            raise TrainingError("PPO optimizer must start with empty state")

        initial_evaluation = development_curve_evaluation(
            evaluate_development_frozen(model, environment),
            active_actor_critic_transitions=0,
            total_environment_transitions=0,
        )
        development_curve: dict[str, dict[str, Any]] = {
            "bc_initialization": initial_evaluation
        }

        actor_hash_before_warmup = state_digest(actor_state(model))
        observation_rms_before_warmup = rms_state(environment.obs_rms)
        return_rms_before_warmup = rms_state(environment.ret_rms)
        critic_parameter_count = freeze_actor_for_critic_warmup(model)
        model.set_diagnostic_phase("critic_warmup")
        warmup_metric_start = len(model.training_iterations)
        warmup_completed = 0
        explained_variance_before: float | None = None
        explained_variance_after: float | None = None

        while warmup_completed < args.critic_warmup_max_transitions:
            model.learn(
                total_timesteps=rollout_size,
                reset_num_timesteps=warmup_completed == 0,
                progress_bar=False,
            )
            warmup_completed += rollout_size
            if int(model.num_timesteps) != warmup_completed:
                raise TrainingError("critic warm-up missed a rollout boundary")
            latest_ev = float(
                model.training_iterations[-1]["explained_variance"]
            )
            if explained_variance_before is None:
                explained_variance_before = latest_ev
            explained_variance_after = latest_ev
            if (
                warmup_completed >= args.critic_warmup_min_transitions
                and latest_ev > args.critic_ev_threshold
            ):
                break

        warmup_metrics = model.training_iterations[warmup_metric_start:]
        critic_gate_passed = bool(
            explained_variance_after is not None
            and explained_variance_after > args.critic_ev_threshold
        )
        actor_hash_after_warmup = state_digest(actor_state(model))
        observation_rms_after_warmup = rms_state(environment.obs_rms)
        return_rms_after_warmup = rms_state(environment.ret_rms)
        if actor_hash_after_warmup != actor_hash_before_warmup:
            raise TrainingError("actor changed during critic-only warm-up")

        development_curve["post_critic_warmup"] = development_curve_evaluation(
            evaluate_development_frozen(model, environment),
            active_actor_critic_transitions=0,
            total_environment_transitions=warmup_completed,
        )

        active_completed = 0
        active_metrics: list[dict[str, Any]] = []
        milestone_states: dict[str, dict[str, Any]] = {}
        checkpoint_bundles: dict[str, dict[str, Any]] = {}
        if critic_gate_passed:
            unfreeze_policy(model)
            model.set_diagnostic_phase("actor_critic_training")
            active_metric_start = len(model.training_iterations)
            for milestone in learning_milestones(args.transitions, rollout_size):
                model.learn(
                    total_timesteps=milestone - active_completed,
                    reset_num_timesteps=False,
                    progress_bar=False,
                )
                active_completed = milestone
                expected_total = warmup_completed + active_completed
                if int(model.num_timesteps) != expected_total:
                    raise TrainingError("PPO missed a transition milestone")
                development_curve[f"ppo_{milestone}_transitions"] = (
                    development_curve_evaluation(
                        evaluate_development_frozen(model, environment),
                        active_actor_critic_transitions=milestone,
                        total_environment_transitions=expected_total,
                    )
                )
                milestone_states[str(milestone)] = {
                    "policy_sha256": state_digest(model.policy.state_dict()),
                    "actor_sha256": state_digest(actor_state(model)),
                    "observation_rms_sha256": rms_digest(
                        rms_state(environment.obs_rms)
                    ),
                    "return_rms_sha256": rms_digest(
                        rms_state(environment.ret_rms)
                    ),
                    "return_rms_count": rms_state(environment.ret_rms)["count"],
                }
                checkpoint_id = f"seed-{args.policy_seed}-ppo-{milestone}"
                try:
                    checkpoint_bundles[str(milestone)] = (
                        persist_checkpoint_bundle(
                            checkpoint_directory / f"ppo-{milestone}",
                            model=model,
                            normalizer=environment,
                            training_config=training_config,
                            seed=args.policy_seed,
                            milestone=milestone,
                            checkpoint_id=checkpoint_id,
                            active_actor_critic_transitions=milestone,
                        )
                    )
                except TrainingArtifactError as exc:
                    raise TrainingError(
                        f"durable checkpoint publication failed: {checkpoint_id}"
                    ) from exc
            active_metrics = model.training_iterations[active_metric_start:]

        final_key = (
            f"ppo_{args.transitions}_transitions"
            if active_completed == args.transitions
            else "post_critic_warmup"
        )
        final_evaluation = development_curve[final_key]
        observation_rms_final = rms_state(environment.obs_rms)
        return_rms_final = rms_state(environment.ret_rms)
        normalization = {
            "norm_obs": args.vec_normalize,
            "norm_reward_during_training": args.vec_normalize,
            "norm_reward_during_evaluation": False,
            "training_during_evaluation": False,
            "observation_rms_frozen": (
                args.freeze_observation_rms or not args.vec_normalize
            ),
            "observation_rms_sha256": rms_digest(observation_rms_final),
            "observation_rms_count": observation_rms_final["count"],
            "return_rms_sha256": rms_digest(return_rms_final),
            "return_rms_count": return_rms_final["count"],
        }
        critic_warmup = {
            "actor_frozen": True,
            "actor_parameters_byte_identical": True,
            "actor_sha256_before": actor_hash_before_warmup,
            "actor_sha256_after": actor_hash_after_warmup,
            "critic_trainable_parameter_count": critic_parameter_count,
            "minimum_transitions": args.critic_warmup_min_transitions,
            "maximum_transitions": args.critic_warmup_max_transitions,
            "completed_transitions": warmup_completed,
            "explained_variance_threshold": args.critic_ev_threshold,
            "first_rollout_explained_variance": explained_variance_before,
            "last_warmup_rollout_explained_variance": explained_variance_after,
            "observation_rms_before_sha256": rms_digest(
                observation_rms_before_warmup
            ),
            "observation_rms_before_count": observation_rms_before_warmup["count"],
            "observation_rms_after_sha256": rms_digest(
                observation_rms_after_warmup
            ),
            "observation_rms_after_count": observation_rms_after_warmup["count"],
            "return_rms_before_sha256": rms_digest(return_rms_before_warmup),
            "return_rms_before_count": return_rms_before_warmup["count"],
            "return_rms_after_sha256": rms_digest(return_rms_after_warmup),
            "return_rms_after_count": return_rms_after_warmup["count"],
            "diagnostic_reward_units": "vecnormalize_normalized",
            "gate_passed": critic_gate_passed,
            "early_stop_rows": early_stop_row_summary(warmup_metrics),
            "iterations": warmup_metrics,
        }
        ppo = {
            "actor_unfrozen": critic_gate_passed,
            "active_actor_critic_transitions": active_completed,
            "total_environment_transitions": int(model.num_timesteps),
            **target_kl_diagnostics(active_metrics, args.target_kl),
            "final_rollout_explained_variance": (
                float(active_metrics[-1]["explained_variance"])
                if active_metrics
                else None
            ),
            "diagnostic_reward_units": "vecnormalize_normalized",
            "early_stop_rows": early_stop_row_summary(active_metrics),
            "iterations": active_metrics,
        }
        continuity_result = {
            "critic_warmup": critic_warmup,
            "normalization": normalization,
        }
        return_rms_continuous = return_rms_continuity_valid(
            continuity_result,
            initial_hashes["return_rms_sha256"],
            warmup_transitions=warmup_completed,
            active_transitions=active_completed,
        )
        observation_rms_continuous = (
            not (args.freeze_observation_rms or not args.vec_normalize)
            or critic_warmup["observation_rms_before_sha256"]
            == critic_warmup["observation_rms_after_sha256"]
            == normalization["observation_rms_sha256"]
        )
        telemetry_complete = diagnostic_rows_valid(
            warmup_metrics,
            warmup_completed // rollout_size,
        ) and diagnostic_rows_valid(
            active_metrics,
            active_completed // rollout_size,
        )
        training_complete = (
            critic_gate_passed
            and active_completed == args.transitions
            and return_rms_continuous
            and observation_rms_continuous
            and telemetry_complete
            and len(checkpoint_bundles)
            == len(training_config["evaluation_milestones"])
        )
        elapsed = time.perf_counter() - started
        payload: dict[str, Any] = {
            "schema_version": 1,
            "tool": TOOL_ID,
            "status": (
                "complete" if training_complete else "critic_warmup_incomplete"
            ),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "training_split": "train",
            "evaluation_split": "dev",
            "development_case_count": DEVELOPMENT_CASE_COUNT,
            "final_split_used": False,
            "flow": [
                "behavior_cloning_and_dagger",
                "actor_frozen_critic_warmup",
                "ppo_actor_critic_training",
                "deterministic_development_evaluation",
                "create_new_receipt",
            ],
            "config": training_config,
            "transition_counts": {
                "critic_warmup": warmup_completed,
                "active_actor_critic": active_completed,
                "total_environment": int(model.num_timesteps),
            },
            "behavior_cloning": bc_receipt,
            "initialization": initial_hashes,
            "critic_warmup": critic_warmup,
            "ppo": ppo,
            "normalization": normalization,
            "development_curve": development_curve,
            "development": final_evaluation,
            "milestone_states": milestone_states,
            "checkpoint_bundles": checkpoint_bundles,
            "training_roster_and_tapes": training_contract,
            "checks": {
                "actor_unchanged_during_critic_warmup": (
                    actor_hash_before_warmup == actor_hash_after_warmup
                ),
                "critic_explained_variance_gate_passed": critic_gate_passed,
                "return_rms_continuous_without_reset": return_rms_continuous,
                "observation_rms_contract_preserved": (
                    observation_rms_continuous
                ),
                "per_iteration_telemetry_complete": telemetry_complete,
                "development_hard_violations_zero": all(
                    evaluation["hard_violation_count"] == 0
                    for evaluation in development_curve.values()
                ),
                "development_conservation_residuals_zero": all(
                    evaluation["maximum_conservation_residual"] == 0.0
                    for evaluation in development_curve.values()
                ),
                "development_only_no_final_split_used": True,
                "training_complete": training_complete,
                "all_registered_checkpoints_persisted": (
                    len(checkpoint_bundles)
                    == len(training_config["evaluation_milestones"])
                ),
            },
            "runtime_versions": runtime_versions(),
            "elapsed_seconds": round(elapsed, 3),
            "training_fps": round(
                (warmup_completed + active_completed) / elapsed,
                3,
            ),
        }
        write_receipt(receipt_path, payload)
        print(
            json.dumps(
                _stdout_summary(payload),
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if training_complete else 3
    finally:
        if environment is not None:
            environment.close()
        elif raw_environment is not None:
            raw_environment.close()


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    raise SystemExit(main())
