from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.scenarios import (  # noqa: E402
    HELD_OUT_FAMILIES,
    HELD_OUT_SEEDS,
    TRAINING_FAMILIES,
    TRAINING_SEEDS,
)
from backend.app.simulator import ACTION_ORDER, canonical_hash, canonical_json_bytes  # noqa: E402
from backend.app.simulator_v2 import (  # noqa: E402
    ENGINE_V2_SPEC_SHA256,
    OBSERVATION_ORDER_V2,
    compare_v2,
)

PROTOCOL_PATH = ROOT / "evaluation" / "protocol.v2.json"
REPORT_PATH = ROOT / "evaluation" / "feature_complete_report.v2.json"
METADATA_PATH = ROOT / "artifacts" / "city_recovery_ppo.v2.metadata.json"
SB3_PATH = ROOT / "artifacts" / "city_recovery_ppo.v2.zip"
ONNX_PATH = ROOT / "artifacts" / "city_recovery_ppo.v2.onnx"
PARITY_PATH = ROOT / "evaluation" / "policy_parity.v2.json"
BOOTSTRAP_SEED = 2717
BOOTSTRAP_SAMPLES = 5_000
DISCLOSURE = (
    "structurally realistic, authored-synthetic, not empirically calibrated to real disasters"
)
CONSTRAINT_TOLERANCE = 1e-7
LOGISTICS_TOLERANCE = 1e-6


@dataclass(frozen=True)
class EvaluationPolicyBundle:
    metadata: dict[str, Any]
    session: ort.InferenceSession
    onnx_sha256: str
    sb3_sha256: str
    metadata_sha256: str
    parity_sha256: str


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_protocol() -> tuple[dict[str, Any], str]:
    payload = PROTOCOL_PATH.read_bytes()
    protocol = json.loads(payload.decode("utf-8"))
    expected_families = [family.id for family in HELD_OUT_FAMILIES]
    if protocol.get("held_out_scenario_families") != expected_families:
        raise RuntimeError("v2 protocol held-out families drifted from authored code")
    if len(expected_families) != 5:
        raise RuntimeError("v2 protocol requires exactly five held-out families")
    if protocol.get("evaluation_seeds") != list(HELD_OUT_SEEDS) or len(HELD_OUT_SEEDS) != 8:
        raise RuntimeError("v2 protocol requires the eight fixed held-out seeds")
    if protocol.get("determinism", {}).get("repeats") != 5:
        raise RuntimeError("v2 protocol requires five exact repeats per case")
    if protocol.get("synthetic_disclosure") != DISCLOSURE:
        raise RuntimeError("v2 protocol synthetic disclosure drifted")
    freeze = protocol.get("coefficient_freeze", {})
    if freeze.get("no_post_hoc_tuning") is not True:
        raise RuntimeError("v2 protocol must prohibit post-hoc tuning")
    if freeze.get("finalized_before_training_and_evaluation") is not True:
        raise RuntimeError("v2 coefficients were not preregistered as final")
    training_ids = {family.id for family in TRAINING_FAMILIES}
    if training_ids.intersection(expected_families):
        raise RuntimeError("v2 training and held-out family ids overlap")
    if set(TRAINING_SEEDS).intersection(HELD_OUT_SEEDS):
        raise RuntimeError("v2 training and held-out seeds overlap")
    exclusions = protocol.get("training_exclusions", {})
    if exclusions.get("seeds") != list(HELD_OUT_SEEDS):
        raise RuntimeError("v2 protocol training exclusions are incomplete")
    if protocol.get("candidate", {}).get("observation_count") != len(OBSERVATION_ORDER_V2):
        raise RuntimeError("v2 protocol observation count drifted from the environment")
    if len(OBSERVATION_ORDER_V2) != 33 or len(ACTION_ORDER) != 5:
        raise RuntimeError("v2 must have 33 observations and the unchanged five actions")
    engine = protocol.get("environment", {})
    if engine.get("id") != "city-recovery-env-v2":
        raise RuntimeError("v2 protocol environment id drifted")
    if engine.get("spec_sha256") != ENGINE_V2_SPEC_SHA256:
        raise RuntimeError("v2 protocol engine specification checksum drifted")
    return protocol, hashlib.sha256(payload).hexdigest()


def load_evaluation_bundle(protocol_sha256: str) -> EvaluationPolicyBundle:
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    parity = json.loads(PARITY_PATH.read_text(encoding="utf-8"))
    onnx_sha = sha256(ONNX_PATH)
    sb3_sha = sha256(SB3_PATH)
    parity_sha = sha256(PARITY_PATH)
    if metadata.get("id") != "city-recovery-sb3-ppo-v2":
        raise RuntimeError("v2 policy metadata id is invalid")
    if metadata.get("observation_order") != list(OBSERVATION_ORDER_V2):
        raise RuntimeError("v2 policy metadata observation order drifted")
    if metadata.get("action_order") != list(ACTION_ORDER):
        raise RuntimeError("v2 policy metadata action order drifted")
    environment = metadata.get("environment", {})
    if environment.get("id") != "CityRecoveryEnv-v2":
        raise RuntimeError("v2 policy metadata environment is invalid")
    if environment.get("engine_spec_sha256") != ENGINE_V2_SPEC_SHA256:
        raise RuntimeError("v2 policy metadata engine specification drifted")
    if environment.get("protocol_sha256") != protocol_sha256:
        raise RuntimeError("v2 policy metadata protocol checksum is inconsistent")
    if metadata.get("export", {}).get("onnx_sha256") != onnx_sha:
        raise RuntimeError("v2 policy metadata ONNX checksum is inconsistent")
    if metadata.get("sb3_checkpoint_sha256") != sb3_sha:
        raise RuntimeError("v2 policy metadata checkpoint checksum is inconsistent")
    metadata_parity = metadata.get("parity", {})
    if metadata_parity.get("report_sha256") != parity_sha:
        raise RuntimeError("v2 policy metadata parity checksum is inconsistent")
    if metadata_parity.get("action_tolerance") != 1e-5:
        raise RuntimeError("v2 policy metadata action tolerance is invalid")
    if metadata_parity.get("projected_allocation_tolerance") != 1e-4:
        raise RuntimeError("v2 policy metadata projected allocation tolerance is invalid")
    if parity.get("passed") is not True or parity.get("cases", 0) < 32:
        raise RuntimeError("v2 PyTorch/ONNX parity did not pass enough cases")
    if parity.get("onnx_sha256") != onnx_sha or parity.get("sb3_checkpoint_sha256") != sb3_sha:
        raise RuntimeError("v2 parity artifact hashes are inconsistent")
    if parity.get("protocol_sha256") != protocol_sha256:
        raise RuntimeError("v2 parity protocol checksum is inconsistent")
    if parity.get("engine_spec_sha256") != ENGINE_V2_SPEC_SHA256:
        raise RuntimeError("v2 parity engine specification checksum is inconsistent")
    if float(parity["max_action_abs_error"]) > float(parity["action_tolerance"]):
        raise RuntimeError("v2 action parity tolerance was exceeded")
    if float(parity["max_projected_allocation_abs_error"]) > float(
        parity["projected_allocation_tolerance"]
    ):
        raise RuntimeError("v2 projected allocation parity tolerance was exceeded")
    payload = ONNX_PATH.read_bytes()
    model = onnx.load_model_from_string(payload)
    onnx.checker.check_model(model)
    options = ort.SessionOptions()
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        payload, sess_options=options, providers=["CPUExecutionProvider"]
    )
    output = session.run(
        ["action"],
        {"observation": np.zeros((1, len(OBSERVATION_ORDER_V2)), dtype=np.float32)},
    )[0]
    if np.asarray(output).shape != (1, 5) or not np.all(np.isfinite(output)):
        raise RuntimeError("v2 ONNX smoke inference returned an invalid action")
    return EvaluationPolicyBundle(
        metadata=metadata,
        session=session,
        onnx_sha256=onnx_sha,
        sb3_sha256=sb3_sha,
        metadata_sha256=sha256(METADATA_PATH),
        parity_sha256=parity_sha,
    )


def _vector(logistics: dict[str, Any], name: str) -> np.ndarray:
    values = np.asarray(logistics.get(name), dtype=np.float64)
    if values.shape != (5,) or not np.all(np.isfinite(values)):
        raise RuntimeError(f"v2 logistics field {name} must contain five finite values")
    return values


def assert_invariants(result: dict[str, Any]) -> None:
    schedule = result["shock_schedule"]
    if result.get("schema_version") != "3.0.0":
        raise RuntimeError("v2 comparison schema must be 3.0.0")
    if result.get("engine_version") != "city-recovery-env-v2":
        raise RuntimeError("v2 comparison engine version is missing")
    environment = result.get("environment")
    if environment != {
        "action_count": 5,
        "id": "CityRecoveryEnv-v2",
        "observation_count": 33,
        "spec_sha256": ENGINE_V2_SPEC_SHA256,
        "version": "2.0.0",
    }:
        raise RuntimeError("v2 comparison environment provenance is invalid")
    if result.get("engine_spec_sha256") != ENGINE_V2_SPEC_SHA256:
        raise RuntimeError("v2 comparison engine specification drifted")
    if canonical_hash(result.get("engine_spec")) != ENGINE_V2_SPEC_SHA256:
        raise RuntimeError("v2 comparison engine specification hash is invalid")
    if result.get("observation_order") != list(OBSERVATION_ORDER_V2):
        raise RuntimeError("v2 comparison observation order drifted")
    if result.get("action_order") != list(ACTION_ORDER):
        raise RuntimeError("v2 comparison action order drifted")
    predecessor = result.get("policy", {}).get("predecessor_policy")
    if predecessor != {
        "id": "city-recovery-sb3-ppo-v1",
        "onnx_sha256": (
            "983b7090e9cfc761b7b2118a24cff907abfc9caa74036cfb16bd9218346b11d8"
        ),
        "preserved": True,
        "version": "1.0.0",
    }:
        raise RuntimeError("v2 comparison predecessor provenance is invalid")
    for planner_name in ("candidate", "baseline"):
        planner = result[planner_name]
        if planner["constraint_violations"] != 0:
            raise RuntimeError(f"{planner_name} reported a hard allocation violation")
        previous_pending = np.zeros(5, dtype=np.float64)
        previous_stock_end: np.ndarray | None = None
        for day_index, (day, shock) in enumerate(
            zip(planner["trajectory"], schedule, strict=True)
        ):
            if day["shock"] != shock:
                raise RuntimeError(f"{planner_name} received a different v2 shock tape")
            allocation = np.asarray(day["allocation"], dtype=np.float64)
            lower = np.asarray(day["lower_bounds"], dtype=np.float64)
            upper = np.asarray(day["upper_bounds"], dtype=np.float64)
            budget = float(day["available_budget"])
            if abs(float(allocation.sum()) - budget) > CONSTRAINT_TOLERANCE:
                raise RuntimeError(f"{planner_name} allocation sum invariant failed")
            if np.any(allocation < lower - CONSTRAINT_TOLERANCE):
                raise RuntimeError(f"{planner_name} allocation lower invariant failed")
            if np.any(allocation > upper + CONSTRAINT_TOLERANCE):
                raise RuntimeError(f"{planner_name} allocation upper invariant failed")
            if any(value != 0 for value in day["projection"]["violation_breakdown"].values()):
                raise RuntimeError(f"{planner_name} serialized allocation evidence is nonzero")

            logistics = day.get("logistics")
            if not isinstance(logistics, dict):
                raise RuntimeError(f"{planner_name} day {day_index + 1} lacks a logistics ledger")
            capacity = _vector(logistics, "depot_capacity")
            stock_before = _vector(logistics, "depot_stock_before")
            pending_arrivals = _vector(logistics, "pending_arrivals")
            pending_landed = _vector(logistics, "pending_arrivals_landed")
            pending_held = _vector(logistics, "pending_arrivals_held")
            stock_after_pending = _vector(logistics, "depot_stock_after_pending")
            damage_penalty = _vector(logistics, "depot_damage_penalty")
            damage_days = _vector(logistics, "depot_damage_days_remaining")
            damage = _vector(logistics, "depot_damage_factor")
            road_capacity = float(logistics.get("road_capacity"))
            throughput = _vector(logistics, "throughput_factor")
            transfers = logistics.get("mutual_aid_transfers")
            transfer_net = _vector(logistics, "mutual_aid_net")
            stock_ready = _vector(logistics, "depot_stock_ready")
            same_day_scheduled = _vector(logistics, "same_day_delivery_scheduled")
            same_day_landed = _vector(logistics, "same_day_delivery_landed")
            same_day_held = _vector(logistics, "same_day_delivery_held")
            delayed_scheduled = _vector(logistics, "delayed_delivery_scheduled")
            reserve = _vector(logistics, "repair_reserve")
            request = _vector(logistics, "repair_request")
            dispatch = _vector(logistics, "repair_dispatch")
            repair_supply = _vector(logistics, "repair_supply")
            spoilage = _vector(logistics, "spoilage")
            stock_end = _vector(logistics, "depot_stock_end")
            pending_next = _vector(logistics, "pending_next_day")
            overflow = _vector(logistics, "capacity_overflow")
            residual = _vector(logistics, "conservation_residual")

            if np.max(np.abs(capacity - 400.0)) > LOGISTICS_TOLERANCE:
                raise RuntimeError(f"{planner_name} depot capacity drifted")
            if np.max(np.abs(pending_arrivals - previous_pending)) > LOGISTICS_TOLERANCE:
                raise RuntimeError(f"{planner_name} pending delivery carryover failed")
            if previous_stock_end is not None and np.max(
                np.abs(stock_before - previous_stock_end)
            ) > LOGISTICS_TOLERANCE:
                raise RuntimeError(f"{planner_name} depot stock carryover failed")
            if np.max(
                np.abs(pending_arrivals - pending_landed - pending_held)
            ) > LOGISTICS_TOLERANCE:
                raise RuntimeError(f"{planner_name} pending arrival split failed")
            if np.max(
                np.abs(stock_after_pending - stock_before - pending_landed)
            ) > LOGISTICS_TOLERANCE:
                raise RuntimeError(f"{planner_name} pending receipt stock ledger failed")
            if np.any(stock_before < -LOGISTICS_TOLERANCE) or np.any(
                stock_before > capacity + LOGISTICS_TOLERANCE
            ):
                raise RuntimeError(f"{planner_name} depot stock-before bounds failed")
            if np.any(stock_end < -LOGISTICS_TOLERANCE) or np.any(
                stock_end > capacity + LOGISTICS_TOLERANCE
            ):
                raise RuntimeError(f"{planner_name} depot stock-end bounds failed")
            if np.any(pending_next < -LOGISTICS_TOLERANCE) or np.any(
                pending_held < -LOGISTICS_TOLERANCE
            ) or np.any(same_day_held < -LOGISTICS_TOLERANCE):
                raise RuntimeError(f"{planner_name} delivery queue became negative")
            if not np.isfinite(road_capacity) or not (
                0.40 - LOGISTICS_TOLERANCE
                <= road_capacity
                <= 1.0 + LOGISTICS_TOLERANCE
            ):
                raise RuntimeError(f"{planner_name} road capacity bounds failed")
            expected_road = 0.40 + 0.60 * float(day["services_after_shock"][0])
            if abs(road_capacity - expected_road) > LOGISTICS_TOLERANCE:
                raise RuntimeError(f"{planner_name} road capacity formula drifted")
            if np.any(throughput < -LOGISTICS_TOLERANCE) or np.any(
                throughput > 1 + LOGISTICS_TOLERANCE
            ):
                raise RuntimeError(f"{planner_name} throughput bounds failed")
            if np.any(damage < 0.30 - LOGISTICS_TOLERANCE) or np.any(
                damage > 1 + LOGISTICS_TOLERANCE
            ):
                raise RuntimeError(f"{planner_name} depot damage-factor bounds failed")
            if np.any(damage_penalty < -LOGISTICS_TOLERANCE) or np.any(
                damage_penalty > 0.72 + LOGISTICS_TOLERANCE
            ) or np.any(damage_days < -LOGISTICS_TOLERANCE):
                raise RuntimeError(f"{planner_name} depot damage ledger bounds failed")
            expected_throughput = damage.copy()
            expected_throughput[1:] *= road_capacity
            if np.max(np.abs(throughput - expected_throughput)) > LOGISTICS_TOLERANCE:
                raise RuntimeError(f"{planner_name} throughput formula drifted")
            if not isinstance(transfers, list) or len(transfers) > 1:
                raise RuntimeError(f"{planner_name} mutual-aid transfer count failed")
            if abs(float(transfer_net.sum())) > LOGISTICS_TOLERANCE:
                raise RuntimeError(f"{planner_name} mutual-aid transfer created stock")
            if float(np.maximum(0.0, transfer_net).sum()) > 24.0 + LOGISTICS_TOLERANCE:
                raise RuntimeError(f"{planner_name} mutual-aid daily cap failed")
            if transfers:
                receiver = int(np.argmax(transfer_net))
                donor = int(np.argmin(transfer_net))
                if throughput[receiver] < 0.55 - LOGISTICS_TOLERANCE or throughput[
                    donor
                ] < 0.55 - LOGISTICS_TOLERANCE:
                    raise RuntimeError(f"{planner_name} mutual-aid route gate failed")
                before_fraction = stock_after_pending / capacity
                if before_fraction[receiver] >= 0.15 + LOGISTICS_TOLERANCE:
                    raise RuntimeError(f"{planner_name} mutual-aid receiver gate failed")
                if before_fraction[donor] <= 0.42 - LOGISTICS_TOLERANCE:
                    raise RuntimeError(f"{planner_name} mutual-aid donor gate failed")
                if stock_ready[donor] < 0.35 * capacity[donor] - LOGISTICS_TOLERANCE:
                    raise RuntimeError(f"{planner_name} mutual-aid donor reserve failed")
            if np.max(
                np.abs(stock_ready - stock_after_pending - transfer_net)
            ) > LOGISTICS_TOLERANCE:
                raise RuntimeError(f"{planner_name} mutual-aid stock ledger failed")
            if np.max(
                np.abs(same_day_scheduled - 0.65 * allocation)
            ) > LOGISTICS_TOLERANCE:
                raise RuntimeError(f"{planner_name} same-day delivery fraction drifted")
            if np.max(
                np.abs(same_day_scheduled - same_day_landed - same_day_held)
            ) > LOGISTICS_TOLERANCE:
                raise RuntimeError(f"{planner_name} same-day landing split failed")
            if np.max(
                np.abs(delayed_scheduled - 0.35 * allocation)
            ) > LOGISTICS_TOLERANCE:
                raise RuntimeError(f"{planner_name} delayed delivery fraction drifted")
            expected_pending = pending_held + same_day_held + delayed_scheduled
            if np.max(np.abs(pending_next - expected_pending)) > LOGISTICS_TOLERANCE:
                raise RuntimeError(f"{planner_name} next-day queue ledger failed")
            if np.max(
                np.abs(overflow - pending_held - same_day_held)
            ) > LOGISTICS_TOLERANCE:
                raise RuntimeError(f"{planner_name} capacity-overflow evidence failed")
            shocked = np.asarray(day["services_after_shock"], dtype=np.float64)
            expected_reserve = 0.04 * capacity * (1.0 - shocked)
            if np.max(np.abs(reserve - expected_reserve)) > LOGISTICS_TOLERANCE:
                raise RuntimeError(f"{planner_name} repair reserve formula drifted")
            if np.max(np.abs(request - allocation - reserve)) > LOGISTICS_TOLERANCE:
                raise RuntimeError(f"{planner_name} repair request formula drifted")
            available_stock = stock_ready + same_day_landed
            if np.max(
                np.abs(dispatch - np.minimum(available_stock, request))
            ) > LOGISTICS_TOLERANCE:
                raise RuntimeError(f"{planner_name} repair dispatch formula drifted")
            if np.max(np.abs(repair_supply - dispatch * throughput)) > LOGISTICS_TOLERANCE:
                raise RuntimeError(f"{planner_name} repair throughput formula drifted")
            before_spoilage = available_stock - repair_supply
            expected_spoilage = np.zeros(5, dtype=np.float64)
            expected_spoilage[2] = 0.006 * before_spoilage[2]
            if np.max(np.abs(spoilage - expected_spoilage)) > LOGISTICS_TOLERANCE:
                raise RuntimeError(f"{planner_name} food spoilage formula drifted")
            if np.max(np.abs(residual)) > LOGISTICS_TOLERANCE:
                raise RuntimeError(f"{planner_name} logistics conservation residual failed")
            recomputed = (
                stock_before
                + pending_arrivals
                + allocation
                + transfer_net
                - repair_supply
                - spoilage
                - stock_end
                - pending_next
            )
            if np.max(np.abs(recomputed)) > LOGISTICS_TOLERANCE:
                raise RuntimeError(f"{planner_name} depot conservation equation failed")
            direct_stock = (
                stock_before
                + pending_landed
                + transfer_net
                + same_day_landed
                - repair_supply
                - spoilage
            )
            if np.max(np.abs(stock_end - direct_stock)) > LOGISTICS_TOLERANCE:
                raise RuntimeError(f"{planner_name} direct stock conservation failed")
            if planner_name == "baseline":
                evidence = day.get("planner_evidence")
                if not isinstance(evidence, dict):
                    raise RuntimeError("baseline v2 GLOP evidence is missing")
                evidence_x = np.asarray(
                    evidence.get("allocation_solution"), dtype=np.float64
                )
                evidence_y = np.asarray(
                    evidence.get("dispatch_solution"), dtype=np.float64
                )
                if evidence_x.shape != (5,) or np.max(
                    np.abs(evidence_x - allocation)
                ) > LOGISTICS_TOLERANCE:
                    raise RuntimeError("baseline v2 allocation solution drifted")
                if evidence_y.shape != (5,) or np.max(
                    np.abs(evidence_y - dispatch)
                ) > LOGISTICS_TOLERANCE:
                    raise RuntimeError("baseline v2 dispatch solution drifted")
                if not isinstance(evidence.get("objective_coefficients"), list):
                    raise RuntimeError("baseline v2 objective coefficients are missing")
            previous_pending = pending_next
            previous_stock_end = stock_end


def mean(values: list[float]) -> float:
    return round(float(np.mean(np.asarray(values, dtype=np.float64))), 8)


def paired_interval(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    indices = rng.integers(0, len(array), size=(BOOTSTRAP_SAMPLES, len(array)))
    means = array[indices].mean(axis=1)
    return {
        "bootstrap_seed": BOOTSTRAP_SEED,
        "method": "paired nonparametric bootstrap over complete scenario-seed units",
        "samples": BOOTSTRAP_SAMPLES,
        "mean": round(float(array.mean()), 8),
        "lower_95": round(float(np.quantile(means, 0.025)), 8),
        "upper_95": round(float(np.quantile(means, 0.975)), 8),
    }


def logistics_metrics(planner: dict[str, Any]) -> dict[str, float]:
    stock_fractions: list[float] = []
    queued_delivery_unit_days = 0.0
    throughput_constrained_unit_days = 0.0
    transfer_units = 0.0
    food_spoilage = 0.0
    for day in planner["trajectory"]:
        logistics = day["logistics"]
        capacity = _vector(logistics, "depot_capacity")
        stock_end = _vector(logistics, "depot_stock_end")
        pending_next = _vector(logistics, "pending_next_day")
        transfer_net = _vector(logistics, "mutual_aid_net")
        dispatch = _vector(logistics, "repair_dispatch")
        repair_supply = _vector(logistics, "repair_supply")
        spoilage = _vector(logistics, "spoilage")
        stock_fractions.extend((stock_end / capacity).tolist())
        queued_delivery_unit_days += float(pending_next.sum())
        throughput_constrained_unit_days += float(
            np.maximum(0.0, dispatch - repair_supply).sum()
        )
        transfer_units += float(np.maximum(0.0, transfer_net).sum())
        food_spoilage += float(spoilage[2])
    return {
        "food_spoilage_units": round(food_spoilage, 8),
        "inter_depot_transfer_units": round(transfer_units, 8),
        "mean_depot_stock_fraction": round(float(np.mean(stock_fractions)), 8),
        "queued_delivery_unit_days": round(queued_delivery_unit_days, 8),
        "throughput_constrained_unit_days": round(throughput_constrained_unit_days, 8),
    }


def case_metrics(planner: dict[str, Any]) -> dict[str, Any]:
    return {
        "critical_service_days": planner["critical_service_days"],
        "days_to_pre_shock_recovery_after_largest_loss": planner[
            "days_to_pre_shock_recovery_after_largest_loss"
        ],
        **logistics_metrics(planner),
        "post_shock_recovery_shortfall_auc": planner[
            "post_shock_recovery_shortfall_auc"
        ],
        "rauc": planner["rauc"],
        "trajectory_sha256": planner["trajectory_sha256"],
    }


def aggregate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = (
        "rauc",
        "post_shock_recovery_shortfall_auc",
        "days_to_pre_shock_recovery_after_largest_loss",
        "critical_service_days",
        "mean_depot_stock_fraction",
        "queued_delivery_unit_days",
        "throughput_constrained_unit_days",
        "inter_depot_transfer_units",
        "food_spoilage_units",
    )
    result: dict[str, Any] = {}
    for metric in metrics:
        candidate = [float(case["candidate"][metric]) for case in cases]
        baseline = [float(case["baseline"][metric]) for case in cases]
        deltas = [left - right for left, right in zip(candidate, baseline, strict=True)]
        result[metric] = {
            "baseline_mean": mean(baseline),
            "candidate_mean": mean(candidate),
            "candidate_minus_baseline": mean(deltas),
        }
        if metric == "rauc":
            result[metric]["paired_95_interval"] = paired_interval(deltas)
    return result


def main() -> None:
    protocol, protocol_sha256 = read_protocol()
    bundle = load_evaluation_bundle(protocol_sha256)
    repeats = int(protocol["determinism"]["repeats"])
    cases: list[dict[str, Any]] = []
    determinism_mismatches = 0
    violation_totals = {
        "candidate": defaultdict(int),
        "baseline": defaultdict(int),
    }
    for family in HELD_OUT_FAMILIES:
        for seed in HELD_OUT_SEEDS:
            scenario = family.build(seed)
            results = [compare_v2(scenario, seed, bundle) for _ in range(repeats)]
            result_bytes = [canonical_json_bytes(result) for result in results]
            hashes = [hashlib.sha256(payload).hexdigest() for payload in result_bytes]
            if len(set(result_bytes)) != 1:
                determinism_mismatches += 1
            first = results[0]
            assert_invariants(first)
            for planner_name in ("candidate", "baseline"):
                for key, value in first[planner_name]["violation_breakdown"].items():
                    violation_totals[planner_name][key] += value
            cases.append(
                {
                    "baseline": case_metrics(first["baseline"]),
                    "candidate": case_metrics(first["candidate"]),
                    "family_id": family.id,
                    "repeat_result_sha256": hashes,
                    "result_sha256": hashes[0],
                    "scenario_sha256": canonical_hash(scenario.model_dump(mode="json")),
                    "seed": seed,
                    "shock_schedule_sha256": first["shock_schedule_sha256"],
                }
            )
    if determinism_mismatches:
        raise RuntimeError(f"v2 determinism failed for {determinism_mismatches} cases")
    aggregate_metrics = aggregate(cases)
    by_family = {
        family.id: aggregate([case for case in cases if case["family_id"] == family.id])
        for family in HELD_OUT_FAMILIES
    }
    rauc_delta = aggregate_metrics["rauc"]["candidate_minus_baseline"]
    if rauc_delta > 0:
        outcome = "measured_resilience_improvement"
        statement = (
            "The fixed v2 PPO/ONNX candidate has higher mean resilience AUC on this "
            "authored-synthetic held-out protocol. Every registered recovery and logistics "
            "delta is reported separately."
        )
    else:
        outcome = "measured_tradeoff_or_baseline_advantage"
        statement = (
            "The fixed v2 PPO/ONNX candidate does not improve mean resilience AUC on this "
            "authored-synthetic held-out protocol; the measured resilience, recovery, and "
            "logistics trade-off is reported without changing the protocol."
        )
    wins = sum(case["candidate"]["rauc"] > case["baseline"]["rauc"] for case in cases)
    losses = sum(case["candidate"]["rauc"] < case["baseline"]["rauc"] for case in cases)
    report = {
        "aggregate": aggregate_metrics,
        "baseline_id": "ortools-glop-visible-v2",
        "by_family": by_family,
        "candidate_id": bundle.metadata["id"],
        "cases": cases,
        "coefficient_freeze_honored": True,
        "determinism": {
            "canonical_result_bytes": "identical",
            "mismatches": determinism_mismatches,
            "repeats_per_case": repeats,
            "total_executions": len(cases) * repeats,
        },
        "evaluation_case_count": len(cases),
        "engine_spec_sha256": ENGINE_V2_SPEC_SHA256,
        "held_out_family_count": len(HELD_OUT_FAMILIES),
        "limitations": [
            DISCLOSURE,
            (
                "The paired interval describes this authored finite protocol; it is not "
                "population or causal uncertainty."
            ),
            "No municipal effectiveness, equity, safety, or deployment claim is supported.",
        ],
        "onnx_sha256": bundle.onnx_sha256,
        "outcome": outcome,
        "outcome_statement": statement,
        "parity_report_sha256": bundle.parity_sha256,
        "protocol_sha256": protocol_sha256,
        "resilience_case_counts": {
            "baseline_higher": losses,
            "candidate_higher": wins,
            "ties": len(cases) - wins - losses,
        },
        "sb3_checkpoint_sha256": bundle.sb3_sha256,
        "schema_version": "2.0.0",
        "split": {
            "evaluation_seeds": list(HELD_OUT_SEEDS),
            "held_out_family_ids": [family.id for family in HELD_OUT_FAMILIES],
            "training_family_ids": [family.id for family in TRAINING_FAMILIES],
            "training_seeds": list(TRAINING_SEEDS),
            "unit": protocol["split_unit"],
        },
        "synthetic_disclosure": DISCLOSURE,
        "synthetic_only": True,
        "violation_totals": {
            planner: dict(values) for planner, values in violation_totals.items()
        },
    }
    REPORT_PATH.write_bytes(canonical_json_bytes(report) + b"\n")
    print(
        json.dumps(
            {
                "baseline_rauc": aggregate_metrics["rauc"]["baseline_mean"],
                "candidate_rauc": aggregate_metrics["rauc"]["candidate_mean"],
                "cases": len(cases),
                "determinism_mismatches": determinism_mismatches,
                "executions": len(cases) * repeats,
                "outcome": outcome,
                "rauc_delta": rauc_delta,
                "report_sha256": sha256(REPORT_PATH),
                "violations": sum(
                    sum(values.values()) for values in violation_totals.values()
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
