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

from backend.app.scenarios import TRAINING_FAMILIES, TRAINING_SEEDS  # noqa: E402
from backend.app.simulator import (  # noqa: E402
    ACTION_ORDER,
    OBSERVATION_ORDER,
    action_to_proposal,
    project_capped_simplex,
)
from backend.app.simulator_v2 import (  # noqa: E402
    AFTERSHOCK_DAY_ONE_SCALE,
    AFTERSHOCK_DAY_TWO_SCALE,
    DELAYED_DELIVERY_FRACTION,
    DEPOT_CAPACITY,
    DEPOT_DAMAGE_PENALTY_CAP,
    DEPOT_DAMAGE_SCALE,
    DEPOT_THROUGHPUT_FLOOR,
    ENGINE_V2_SPEC_SHA256,
    FOOD_SPOILAGE_RATE,
    IMMEDIATE_DELIVERY_FRACTION,
    OBSERVATION_ORDER_V2,
    RESERVE_DRAW_FRACTION,
    ROAD_CAPACITY_FLOOR,
    TRANSFER_DAILY_CAP_FRACTION,
    TRANSFER_DONOR_RESERVE_FRACTION,
    TRANSFER_MIN_THROUGHPUT,
    TRANSFER_RECEIVER_TARGET_FRACTION,
    TRANSFER_STARVED_FRACTION,
    TRANSFER_SURPLUS_FRACTION,
    CityRecoveryEnvV2,
    CyclingScenarioEnvV2,
)

TRAIN_SEED = 17017
DEFAULT_TIMESTEPS = 30_000
PARITY_SEEDS = (280100, 280101, 280102, 280103)
POLICY_ID = "city-recovery-sb3-ppo-v2"
POLICY_VERSION = "2.0.0"
POLICY_SCHEMA_VERSION = "3.0.0"
POLICY_ARTIFACT_TYPE = "stable_baselines3_ppo"
ARTIFACT_LICENSE = "CC0-1.0"
DISCLOSURE = (
    "Stable-Baselines3 PPO trained only on structurally realistic, authored-synthetic, "
    "not empirically calibrated to real disasters; local simulation evidence only, not "
    "operational guidance."
)
LEGACY_CANDIDATE = {
    "artifact_type": "deterministic_linear_policy_candidate",
    "disclosure": (
        "Accepted deterministic grid-selected synthetic linear candidate; not PPO and not "
        "used for Feature Complete inference."
    ),
    "id": "frozen-policy-candidate-v1",
    "is_ppo": False,
    "sha256": "23762a44d67e83dd487558d595d3d9ed5f5e406915f488a076ac21190ab9a6e3",
}

SB3_PATH = ROOT / "artifacts" / "city_recovery_ppo.v2.zip"
ONNX_PATH = ROOT / "artifacts" / "city_recovery_ppo.v2.onnx"
METADATA_PATH = ROOT / "artifacts" / "city_recovery_ppo.v2.metadata.json"
PARITY_PATH = ROOT / "evaluation" / "policy_parity.v2.json"
PROTOCOL_PATH = ROOT / "evaluation" / "protocol.v2.json"

# These released v1 files are inputs, never outputs. A v2 build stops if any differs before or
# after training. This guard deliberately does not include the additive manifest-v3 work.
V1_IMMUTABLE_SHA256 = {
    "artifacts/city_recovery_ppo.v1.metadata.json": (
        "becc2eed1e552e9a503c3210d2ebae18eeccc593c9a7d716fae11e1e69b1c62e"
    ),
    "artifacts/city_recovery_ppo.v1.onnx": (
        "983b7090e9cfc761b7b2118a24cff907abfc9caa74036cfb16bd9218346b11d8"
    ),
    "artifacts/city_recovery_ppo.v1.zip": (
        "f270bc720e7d2866d293feab27692d3ac9542d064d275b13c33f4d960dad4e33"
    ),
    "artifacts/frozen_policy.v1.json": (
        "23762a44d67e83dd487558d595d3d9ed5f5e406915f488a076ac21190ab9a6e3"
    ),
    "evaluation/feature_complete_report.v1.json": (
        "fea00d1bf578c7d52cad816eed732a58ffb3f9b809c2788ba35c601e976f9351"
    ),
    "evaluation/gate2-evidence.json": (
        "82aa655ecff8c91db99d8db72ec561955ca23badf86689285424a6e90a5c74df"
    ),
    "evaluation/policy_parity.v1.json": (
        "20d87aafc638f3c6e7942a1578eea0710e0cd083c5a2054063f1813a76916a82"
    ),
    "evaluation/protocol.v1.json": (
        "b36bba8dba6948b6b2a29170f6e5a9f7ebf012f95ce859edcece87bb5c9c5655"
    ),
}
EXPECTED_HYPERPARAMETERS = {
    "batch_size": 64,
    "clip_range": 0.2,
    "ent_coef": 0.003,
    "gae_lambda": 0.95,
    "gamma": 0.98,
    "learning_rate": 0.0003,
    "max_grad_norm": 0.5,
    "n_epochs": 4,
    "n_steps": 256,
    "orthogonal_initialization": True,
    "policy_activation": "Tanh",
    "policy_network": [32, 32],
    "value_function_network": [32, 32],
    "vf_coef": 0.5,
}


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


def assert_v1_unchanged() -> None:
    for relative_path, expected in V1_IMMUTABLE_SHA256.items():
        path = ROOT / relative_path
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"released v1 evidence changed or is missing: {relative_path}")


def read_protocol() -> tuple[dict[str, Any], str]:
    payload = PROTOCOL_PATH.read_bytes()
    protocol = json.loads(payload.decode("utf-8"))
    if protocol.get("candidate", {}).get("id") != POLICY_ID:
        raise RuntimeError("v2 protocol candidate id drifted")
    if protocol.get("candidate", {}).get("observation_count") != 33:
        raise RuntimeError("v2 protocol must preregister 33 observations")
    if protocol.get("candidate", {}).get("action_space_unchanged") is not True:
        raise RuntimeError("v2 protocol must keep the five-way action space unchanged")
    engine = protocol.get("environment", {})
    if engine.get("id") != "city-recovery-env-v2":
        raise RuntimeError("v2 protocol environment id drifted")
    if engine.get("spec_sha256") != ENGINE_V2_SPEC_SHA256:
        raise RuntimeError("v2 protocol engine specification checksum drifted")
    regimen = protocol.get("training_regimen", {})
    if regimen.get("fixed_seed") != TRAIN_SEED or regimen.get("timesteps") != DEFAULT_TIMESTEPS:
        raise RuntimeError("v2 protocol training regimen drifted from this build script")
    if regimen.get("hyperparameters") != EXPECTED_HYPERPARAMETERS:
        raise RuntimeError("v2 protocol PPO hyperparameters drifted from this build script")
    if regimen.get("training_family_ids") != [family.id for family in TRAINING_FAMILIES]:
        raise RuntimeError("v2 protocol training families drifted from authored code")
    if regimen.get("training_seeds") != list(TRAINING_SEEDS):
        raise RuntimeError("v2 protocol training seeds drifted from authored code")
    if len(OBSERVATION_ORDER_V2) != 33:
        raise RuntimeError("CityRecoveryEnvV2 observation order must contain 33 features")
    if tuple(OBSERVATION_ORDER_V2[: len(OBSERVATION_ORDER)]) != tuple(OBSERVATION_ORDER):
        raise RuntimeError("CityRecoveryEnvV2 must append features after the complete v1 order")
    parity = protocol.get("parity_regimen", {})
    expected_parity_mapping = {
        family.id: PARITY_SEEDS[index]
        for index, family in enumerate(TRAINING_FAMILIES)
    }
    if parity.get("family_seed_mapping") != expected_parity_mapping:
        raise RuntimeError("v2 protocol parity seeds drifted from this build script")
    if parity.get("observation_case_count") != 32:
        raise RuntimeError("v2 protocol must preregister 32 parity observations")
    if parity.get("action_absolute_tolerance") != 1e-5:
        raise RuntimeError("v2 protocol action parity tolerance drifted")
    if parity.get("projected_allocation_absolute_tolerance") != 1e-4:
        raise RuntimeError("v2 protocol allocation parity tolerance drifted")
    if parity.get("onnx_opset") != 17:
        raise RuntimeError("v2 protocol ONNX opset drifted")
    authored = protocol.get("authored_coefficients", {})
    delivery = authored.get("allocation_delivery", {})
    depot = authored.get("depot_stock", {})
    damage = authored.get("depot_damage", {})
    spoilage = authored.get("food_spoilage", {})
    transfer = authored.get("inter_depot_transfer", {})
    road = authored.get("road_capacity", {})
    cluster = authored.get("aftershock_hazard_clustering", {})
    exact_checks = (
        (delivery.get("same_day_fraction"), IMMEDIATE_DELIVERY_FRACTION),
        (delivery.get("next_day_fraction"), DELAYED_DELIVERY_FRACTION),
        (depot.get("capacity_units_by_service"), DEPOT_CAPACITY.tolist()),
        (depot.get("reserve_draw_capacity_fraction"), RESERVE_DRAW_FRACTION),
        (damage.get("factor_floor"), DEPOT_THROUGHPUT_FLOOR),
        (damage.get("penalty_cap"), DEPOT_DAMAGE_PENALTY_CAP),
        (damage.get("penalty_scale"), DEPOT_DAMAGE_SCALE),
        (spoilage.get("daily_fraction_of_remaining_food_stock"), FOOD_SPOILAGE_RATE),
        (road.get("floor"), ROAD_CAPACITY_FLOOR),
        (transfer.get("receiver_stock_fraction_below"), TRANSFER_STARVED_FRACTION),
        (transfer.get("donor_stock_fraction_above"), TRANSFER_SURPLUS_FRACTION),
        (
            transfer.get("donor_post_transfer_reserve_fraction"),
            TRANSFER_DONOR_RESERVE_FRACTION,
        ),
        (transfer.get("receiver_target_fraction"), TRANSFER_RECEIVER_TARGET_FRACTION),
        (transfer.get("daily_transfer_capacity_fraction"), TRANSFER_DAILY_CAP_FRACTION),
        (transfer.get("receiver_throughput_at_least"), TRANSFER_MIN_THROUGHPUT),
        (transfer.get("donor_throughput_at_least"), TRANSFER_MIN_THROUGHPUT),
        (
            cluster.get("day_1_previous_quake_severity_multiplier"),
            AFTERSHOCK_DAY_ONE_SCALE,
        ),
        (
            cluster.get("day_2_previous_quake_severity_multiplier"),
            AFTERSHOCK_DAY_TWO_SCALE,
        ),
    )
    if any(actual != expected for actual, expected in exact_checks):
        raise RuntimeError("v2 protocol coefficients drifted from CityRecoveryEnvV2")
    return protocol, hashlib.sha256(payload).hexdigest()


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
    check_env(CityRecoveryEnvV2(*scenarios[0]), warn=True, skip_render_check=True)
    environment = CyclingScenarioEnvV2(scenarios)
    model = PPO(
        "MlpPolicy",
        environment,
        learning_rate=EXPECTED_HYPERPARAMETERS["learning_rate"],
        n_steps=EXPECTED_HYPERPARAMETERS["n_steps"],
        batch_size=EXPECTED_HYPERPARAMETERS["batch_size"],
        n_epochs=EXPECTED_HYPERPARAMETERS["n_epochs"],
        gamma=EXPECTED_HYPERPARAMETERS["gamma"],
        gae_lambda=EXPECTED_HYPERPARAMETERS["gae_lambda"],
        clip_range=EXPECTED_HYPERPARAMETERS["clip_range"],
        ent_coef=EXPECTED_HYPERPARAMETERS["ent_coef"],
        vf_coef=EXPECTED_HYPERPARAMETERS["vf_coef"],
        max_grad_norm=EXPECTED_HYPERPARAMETERS["max_grad_norm"],
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
    dummy = torch.zeros((1, len(OBSERVATION_ORDER_V2)), dtype=torch.float32)
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
        seed = PARITY_SEEDS[family_index]
        env = CityRecoveryEnvV2(family.build(seed), seed)
        observation, _ = env.reset(seed=seed)
        terminated = False
        family_cases = 0
        while not terminated and family_cases < 8:
            context = env.current_context()
            cases.append(
                (
                    observation.copy(),
                    context.available_budget,
                    context.lower.copy(),
                    context.upper.copy(),
                )
            )
            family_cases += 1
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, _, _ = env.step(action)
    if len(cases) != len(TRAINING_FAMILIES) * 8:
        raise RuntimeError("v2 parity set must contain eight observations per training family")
    return cases


def build_parity_report(model: PPO, protocol_sha256: str) -> dict[str, Any]:
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
        "environment": "CityRecoveryEnv-v2",
        "engine_spec_sha256": ENGINE_V2_SPEC_SHA256,
        "max_action_abs_error": max_action_error,
        "max_pre_projector_proposal_abs_error": max_proposal_error,
        "max_projected_allocation_abs_error": max_projected_error,
        "observation_count": len(OBSERVATION_ORDER_V2),
        "onnx_sha256": sha256(ONNX_PATH),
        "onnxruntime_version": ort.__version__,
        "passed": (
            max_action_error <= action_tolerance
            and max_projected_error <= projected_tolerance
        ),
        "projected_allocation_tolerance": projected_tolerance,
        "protocol_sha256": protocol_sha256,
        "providers": session.get_providers(),
        "pytorch_version": torch.__version__,
        "sb3_checkpoint_sha256": sha256(SB3_PATH),
        "schema_version": "2.0.0",
    }


def build_metadata(timesteps: int, parity_sha256: str, protocol_sha256: str) -> dict[str, Any]:
    return {
        "action_order": list(ACTION_ORDER),
        "artifact_type": POLICY_ARTIFACT_TYPE,
        "disclosure": DISCLOSURE,
        "environment": {
            "change_log": "ENGINE_V2_CHANGELOG.md",
            "engine_spec_sha256": ENGINE_V2_SPEC_SHA256,
            "id": "CityRecoveryEnv-v2",
            "observation_count": len(OBSERVATION_ORDER_V2),
            "protocol_path": "evaluation/protocol.v2.json",
            "protocol_sha256": protocol_sha256,
        },
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
        "legacy_candidate": LEGACY_CANDIDATE,
        "license": ARTIFACT_LICENSE,
        "observation_order": list(OBSERVATION_ORDER_V2),
        "parity": {
            "action_tolerance": 1e-5,
            "projected_allocation_tolerance": 1e-4,
            "report_path": "evaluation/policy_parity.v2.json",
            "report_sha256": parity_sha256,
        },
        "sb3_checkpoint_sha256": sha256(SB3_PATH),
        "predecessor_policy": {
            "id": "city-recovery-sb3-ppo-v1",
            "onnx_sha256": (
                "983b7090e9cfc761b7b2118a24cff907abfc9caa74036cfb16bd9218346b11d8"
            ),
            "preserved": True,
            "version": "1.0.0",
        },
        "schema_version": POLICY_SCHEMA_VERSION,
        "training": {
            "algorithm": "PPO",
            "device": "cpu",
            "engine_spec_sha256": ENGINE_V2_SPEC_SHA256,
            "environment": "CityRecoveryEnv-v2",
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
        "v1_bundle_preserved": dict(sorted(V1_IMMUTABLE_SHA256.items())),
        "version": POLICY_VERSION,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and export the additive CityRecoveryEnv-v2 PPO bundle."
    )
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    args = parser.parse_args()
    if args.timesteps != DEFAULT_TIMESTEPS:
        raise ValueError(
            f"protocol.v2 preregisters exactly {DEFAULT_TIMESTEPS} timesteps; "
            "create a new protocol version to change it"
        )
    assert_v1_unchanged()
    _, protocol_sha256 = read_protocol()
    model = train(args.timesteps)
    export_onnx(model)
    report = build_parity_report(model, protocol_sha256)
    if not report["passed"]:
        raise RuntimeError(f"PyTorch/ONNX v2 parity failed: {report}")
    write_json(PARITY_PATH, report)
    metadata = build_metadata(args.timesteps, sha256(PARITY_PATH), protocol_sha256)
    write_json(METADATA_PATH, metadata)
    assert_v1_unchanged()
    print(
        json.dumps(
            {
                "environment": "CityRecoveryEnv-v2",
                "max_action_abs_error": report["max_action_abs_error"],
                "max_projected_allocation_abs_error": report[
                    "max_projected_allocation_abs_error"
                ],
                "onnx_sha256": sha256(ONNX_PATH),
                "parity_cases": report["cases"],
                "protocol_sha256": protocol_sha256,
                "sb3_checkpoint_sha256": sha256(SB3_PATH),
                "timesteps": args.timesteps,
                "v1_preserved": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
