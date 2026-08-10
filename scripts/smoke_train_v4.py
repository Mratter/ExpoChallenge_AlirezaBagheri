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
from dataclasses import dataclass
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
REPORTED_APPROX_KL_STABILITY_MULTIPLIER = 1.5
REPORTED_APPROX_KL_STABILITY_DEFINITION = (
    "maximum recorded per-iteration SB3 approximate KL must be <= "
    "1.5 * configured target_kl"
)


class SmokeError(RuntimeError):
    """Raised when the matched smoke contract cannot be honored."""


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
            rows.append(
                {
                    "row_id": f"{family.id}:{case_seed}",
                    "solved": bool(outcome["solved"]),
                    "reason_codes": list(outcome["reason_codes"]),
                    "resilience_auc": float(summary["rauc"]),
                    "hard_violation_count": int(summary["hard_violation_count"]),
                    "max_conservation_residual": float(
                        summary["max_logistics_conservation_residual"]
                    ),
                    "target_met_by_service": list(outcome["target_met_by_service"]),
                    "tail_minimum_services": list(outcome["tail_minimum_services"]),
                    "recovery_targets": list(outcome["recovery_targets"]),
                }
            )
    reasons = Counter(
        reason for row in rows if not row["solved"] for reason in row["reason_codes"]
    )
    return {
        "case_count": len(rows),
        "solved_count": sum(row["solved"] for row in rows),
        "solve_rate": sum(row["solved"] for row in rows) / len(rows),
        "mean_resilience_auc": round(fmean(row["resilience_auc"] for row in rows), 10),
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


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise SmokeError(f"refusing to overwrite smoke result: {path}")
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
    os.rename(temporary, path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gate-mode",
        choices=("learning", "reward"),
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


def learning_milestones(total_transitions: int, rollout_size: int) -> list[int]:
    if total_transitions == ACTIVE_MILESTONES[-1] and all(
        milestone % rollout_size == 0 for milestone in ACTIVE_MILESTONES
    ):
        return list(ACTIVE_MILESTONES)
    return [total_transitions]


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
                "the reward gate requires the registered matched profile order"
            )
    if len(set(profiles)) != len(profiles):
        raise SmokeError("reward profiles must be unique")
    rollout_size = args.lanes * args.n_steps
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
            if critic_gate_passed:
                unfreeze_policy(model)
                model.set_diagnostic_phase("actor_critic_training")
                active_metric_start = len(model.training_iterations)
                for milestone in learning_milestones(
                    args.transitions, rollout_size
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
                active_metrics = model.training_iterations[active_metric_start:]

            profile_elapsed = time.perf_counter() - profile_started
            final_evaluation = development_curve[
                f"active_actor_critic_{args.transitions}_transitions"
                if active_completed == args.transitions
                else "post_critic_warmup"
            ]
            active_kl_summary = reported_approx_kl_summary(
                active_metrics, args.target_kl
            )
            active_final_ev = (
                float(active_metrics[-1]["explained_variance"])
                if active_metrics
                else float("-inf")
            )
            approx_kl_stable = active_kl_summary[
                "reported_approx_kl_stable"
            ]
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
                },
                "development_curve": development_curve,
                "development": final_evaluation,
                "learning_gate": {
                    "initial_solved_count": initial_evaluation["solved_count"],
                    "final_solved_count": final_evaluation["solved_count"],
                    "solve_gain": solve_gain,
                    "minimum_solve_gain": args.minimum_learning_solve_gain,
                    "gate_passed": learning_gate_passed,
                },
            }
        finally:
            if environment is not None:
                environment.close()
            elif raw_environment is not None:
                raw_environment.close()

    payload: dict[str, Any] = {
        "schema_version": 1,
        "tool": "smoke_train_v4.py",
        "status": "developmental_ppo_gate_nonauthorizing",
        "authorizing": False,
        "split": "dev",
        "final_split_used": False,
        "same_tapes": True,
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
            "reported_approx_kl_stability_limit": (
                REPORTED_APPROX_KL_STABILITY_MULTIPLIER * args.target_kl
            ),
            "reported_approx_kl_stability_multiplier": (
                REPORTED_APPROX_KL_STABILITY_MULTIPLIER
            ),
            "reported_approx_kl_stability_definition": (
                REPORTED_APPROX_KL_STABILITY_DEFINITION
            ),
            "use_sde": False,
            "vec_normalize": True,
            "observation_rms_frozen_during_ppo": (
                args.freeze_observation_rms
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
    else:
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
    if args.json_output is not None:
        write_json_atomic(args.json_output.resolve(), payload)
    printable = json.loads(json.dumps(payload))
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
    print(json.dumps(printable, indent=2, sort_keys=True))
    return 0 if gate_passed else 3


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    raise SystemExit(main())
