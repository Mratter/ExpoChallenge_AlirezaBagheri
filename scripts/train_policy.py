from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import onnxruntime as ort
import stable_baselines3
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.utils import set_random_seed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.artifact import (  # noqa: E402
    ARTIFACT_LICENSE,
    LEGACY_POLICY_SHA256,
    POLICY_ARTIFACT_TYPE,
    POLICY_ID,
    POLICY_SCHEMA_VERSION,
    POLICY_VERSION,
)
from backend.app.scenarios import TRAINING_FAMILIES, TRAINING_SEEDS  # noqa: E402
from backend.app.simulator import (  # noqa: E402
    ACTION_ORDER,
    OBSERVATION_ORDER,
    CityRecoveryEnv,
    CyclingScenarioEnv,
    action_to_proposal,
    project_capped_simplex,
)

TRAIN_SEED = 17017
DEFAULT_TIMESTEPS = 30_000
SB3_PATH = ROOT / "artifacts" / "city_recovery_ppo.v1.zip"
ONNX_PATH = ROOT / "artifacts" / "city_recovery_ppo.v1.onnx"
METADATA_PATH = ROOT / "artifacts" / "city_recovery_ppo.v1.metadata.json"
LEGACY_PATH = ROOT / "artifacts" / "frozen_policy.v1.json"
PARITY_PATH = ROOT / "evaluation" / "policy_parity.v1.json"
MANIFEST_PATH = ROOT / "artifacts" / "manifest.lock.json"


class OnnxablePolicy(torch.nn.Module):
    def __init__(self, policy: torch.nn.Module):
        super().__init__()
        self.policy = policy

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        action = self.policy._predict(observation, deterministic=True)  # type: ignore[attr-defined]
        return torch.clamp(action, -1.0, 1.0)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def training_scenarios() -> list[tuple[Any, int]]:
    return [
        (family.build(seed), seed)
        for family in TRAINING_FAMILIES
        for seed in TRAINING_SEEDS
    ]


def train(timesteps: int) -> PPO:
    random.seed(TRAIN_SEED)
    np.random.seed(TRAIN_SEED)
    torch.manual_seed(TRAIN_SEED)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    set_random_seed(TRAIN_SEED, using_cuda=False)
    scenarios = training_scenarios()
    check_env(CityRecoveryEnv(*scenarios[0]), warn=True, skip_render_check=True)
    environment = CyclingScenarioEnv(scenarios)
    model = PPO(
        "MlpPolicy",
        environment,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=4,
        gamma=0.98,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.003,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs={
            "activation_fn": torch.nn.Tanh,
            "net_arch": {"pi": [32, 32], "vf": [32, 32]},
            "ortho_init": True,
        },
        seed=TRAIN_SEED,
        device="cpu",
        verbose=0,
    )
    model.learn(total_timesteps=timesteps, progress_bar=False)
    model.save(SB3_PATH)
    return model


def export_onnx(model: PPO) -> None:
    wrapper = OnnxablePolicy(model.policy).eval()
    dummy = torch.zeros((1, len(OBSERVATION_ORDER)), dtype=torch.float32)
    torch.onnx.export(
        wrapper,
        dummy,
        ONNX_PATH,
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )


def parity_cases(model: PPO) -> list[tuple[np.ndarray, float, np.ndarray, np.ndarray]]:
    cases: list[tuple[np.ndarray, float, np.ndarray, np.ndarray]] = []
    for family_index, family in enumerate(TRAINING_FAMILIES):
        seed = 180100 + family_index
        scenario = family.build(seed)
        env = CityRecoveryEnv(scenario, seed)
        observation, _ = env.reset(seed=seed)
        terminated = False
        while not terminated and len(cases) < (family_index + 1) * 8:
            context = env.current_context()
            cases.append(
                (
                    observation.copy(),
                    context.available_budget,
                    context.lower.copy(),
                    context.upper.copy(),
                )
            )
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, _, _ = env.step(action)
    if len(cases) < 20:
        raise RuntimeError("not enough parity observations were generated")
    return cases


def build_parity_report(model: PPO) -> dict[str, Any]:
    options = ort.SessionOptions()
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        ONNX_PATH.read_bytes(),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    max_action_error = 0.0
    max_proposal_error = 0.0
    max_projected_error = 0.0
    cases = parity_cases(model)
    for observation, budget, lower, upper in cases:
        pytorch_action, _ = model.predict(observation, deterministic=True)
        onnx_action = session.run(
            ["action"], {"observation": observation.reshape(1, -1).astype(np.float32)}
        )[0][0]
        pytorch_action = np.asarray(pytorch_action, dtype=np.float64)
        onnx_action = np.asarray(onnx_action, dtype=np.float64)
        max_action_error = max(
            max_action_error, float(np.max(np.abs(pytorch_action - onnx_action)))
        )
        pytorch_proposal = action_to_proposal(pytorch_action, budget)
        onnx_proposal = action_to_proposal(onnx_action, budget)
        max_proposal_error = max(
            max_proposal_error, float(np.max(np.abs(pytorch_proposal - onnx_proposal)))
        )
        pytorch_allocation, _ = project_capped_simplex(
            pytorch_proposal, budget, lower, upper
        )
        onnx_allocation, _ = project_capped_simplex(onnx_proposal, budget, lower, upper)
        max_projected_error = max(
            max_projected_error,
            float(np.max(np.abs(pytorch_allocation - onnx_allocation))),
        )
    action_tolerance = 1e-5
    projected_tolerance = 1e-4
    return {
        "action_tolerance": action_tolerance,
        "cases": len(cases),
        "max_action_abs_error": max_action_error,
        "max_pre_projector_proposal_abs_error": max_proposal_error,
        "max_projected_allocation_abs_error": max_projected_error,
        "onnx_sha256": sha256(ONNX_PATH),
        "onnxruntime_version": ort.__version__,
        "passed": (
            max_action_error <= action_tolerance
            and max_projected_error <= projected_tolerance
        ),
        "projected_allocation_tolerance": projected_tolerance,
        "providers": session.get_providers(),
        "pytorch_version": torch.__version__,
        "sb3_checkpoint_sha256": sha256(SB3_PATH),
        "schema_version": "1.0.0",
    }


def build_metadata(timesteps: int, parity_sha256: str) -> dict[str, Any]:
    return {
        "action_order": list(ACTION_ORDER),
        "artifact_type": POLICY_ARTIFACT_TYPE,
        "disclosure": (
            "Stable-Baselines3 PPO trained only on authored synthetic, non-empirical city "
            "scenarios; local simulation evidence only, not operational guidance."
        ),
        "export": {
            "deterministic": True,
            "format": "ONNX",
            "input_name": "observation",
            "onnx_sha256": sha256(ONNX_PATH),
            "opset": 17,
            "output_name": "action",
            "runtime_provider": "CPUExecutionProvider",
        },
        "id": POLICY_ID,
        "legacy_candidate": {
            "artifact_type": "deterministic_linear_policy_candidate",
            "disclosure": (
                "Accepted deterministic grid-selected synthetic linear candidate; "
                "not PPO and not used for Feature Complete inference."
            ),
            "id": "frozen-policy-candidate-v1",
            "is_ppo": False,
            "sha256": LEGACY_POLICY_SHA256,
        },
        "observation_order": list(OBSERVATION_ORDER),
        "parity": {
            "report_path": "evaluation/policy_parity.v1.json",
            "report_sha256": parity_sha256,
        },
        "sb3_checkpoint_sha256": sha256(SB3_PATH),
        "schema_version": POLICY_SCHEMA_VERSION,
        "training": {
            "algorithm": "PPO",
            "device": "cpu",
            "environment": "CityRecoveryEnv-v1",
            "family_ids": [family.id for family in TRAINING_FAMILIES],
            "library": "stable-baselines3",
            "library_version": stable_baselines3.__version__,
            "scenario_seed_count": len(TRAINING_SEEDS),
            "scenario_unit_count": len(TRAINING_FAMILIES) * len(TRAINING_SEEDS),
            "seed": TRAIN_SEED,
            "synthetic_only": True,
            "timesteps": timesteps,
            "torch_version": torch.__version__,
        },
        "version": POLICY_VERSION,
    }


def manifest_record(
    artifact_id: str, path: Path, relative_path: str, role: str, source: str
) -> dict[str, Any]:
    return {
        "bytes": path.stat().st_size,
        "id": artifact_id,
        "license": ARTIFACT_LICENSE,
        "path": relative_path,
        "role": role,
        "sha256": sha256(path),
        "source": source,
    }


def build_manifest() -> dict[str, Any]:
    return {
        "artifacts": [
            manifest_record(
                "accepted-linear-candidate-v1",
                LEGACY_PATH,
                "artifacts/frozen_policy.v1.json",
                "accepted_legacy_linear_candidate",
                "scripts/build_policy_artifact.py",
            ),
            manifest_record(
                "city-recovery-ppo-v1-checkpoint",
                SB3_PATH,
                "artifacts/city_recovery_ppo.v1.zip",
                "training_checkpoint",
                "scripts/train_policy.py",
            ),
            manifest_record(
                "city-recovery-ppo-v1-onnx",
                ONNX_PATH,
                "artifacts/city_recovery_ppo.v1.onnx",
                "runtime_policy",
                "scripts/train_policy.py",
            ),
            manifest_record(
                "city-recovery-ppo-v1-metadata",
                METADATA_PATH,
                "artifacts/city_recovery_ppo.v1.metadata.json",
                "policy_metadata",
                "scripts/train_policy.py",
            ),
            manifest_record(
                "city-recovery-ppo-v1-parity",
                PARITY_PATH,
                "evaluation/policy_parity.v1.json",
                "pytorch_onnx_parity_evidence",
                "scripts/train_policy.py",
            ),
        ],
        "project": "AI17",
        "version": 2,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    args = parser.parse_args()
    if args.timesteps < 1:
        raise ValueError("timesteps must be positive")
    if sha256(LEGACY_PATH) != LEGACY_POLICY_SHA256:
        raise RuntimeError("accepted legacy candidate changed before PPO training")
    model = train(args.timesteps)
    export_onnx(model)
    report = build_parity_report(model)
    if not report["passed"]:
        raise RuntimeError(f"PyTorch/ONNX parity failed: {report}")
    write_json(PARITY_PATH, report)
    metadata = build_metadata(args.timesteps, sha256(PARITY_PATH))
    write_json(METADATA_PATH, metadata)
    write_json(MANIFEST_PATH, build_manifest())
    print(
        json.dumps(
            {
                "max_action_abs_error": report["max_action_abs_error"],
                "max_projected_allocation_abs_error": report[
                    "max_projected_allocation_abs_error"
                ],
                "onnx_sha256": sha256(ONNX_PATH),
                "parity_cases": report["cases"],
                "sb3_checkpoint_sha256": sha256(SB3_PATH),
                "timesteps": args.timesteps,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
