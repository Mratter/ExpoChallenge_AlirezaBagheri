from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def fetch(url: str, payload: dict[str, Any] | None = None) -> tuple[int, bytes]:
    body = None if payload is None else json.dumps(payload, separators=(",", ":")).encode()
    request = urllib.request.Request(
        url,
        data=body,
        method="GET" if body is None else "POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def wait_ready(base_url: str) -> None:
    for _ in range(80):
        try:
            status, body = fetch(f"{base_url}/health/ready")
            if status == 200 and json.loads(body)["status"] == "ready":
                return
        except (OSError, ValueError, KeyError):
            pass
        time.sleep(0.25)
    raise RuntimeError("service did not become ready within 20 seconds")


def assert_constraints(result: dict[str, Any]) -> int:
    if result["services"] != [
        "transport",
        "housing",
        "food",
        "healthcare",
        "public_services",
    ]:
        raise AssertionError("service ordering changed")
    schedule = result["shock_schedule"]
    checks = 0
    for planner_name in ("baseline", "candidate"):
        trajectory = result[planner_name]["trajectory"]
        if len(trajectory) != result["scenario"]["horizon_days"]:
            raise AssertionError(f"{planner_name} trajectory is incomplete")
        for day, shock in zip(trajectory, schedule, strict=True):
            if day["shock"] != shock:
                raise AssertionError(f"{planner_name} did not receive shared shock tape")
            budget = day["available_budget"]
            allocation_sum = sum(day["allocation"])
            if abs(allocation_sum - budget) > 1e-7:
                raise AssertionError(f"{planner_name} allocation does not sum to budget")
            if allocation_sum > budget + 1e-7:
                raise AssertionError(f"{planner_name} allocation exceeds budget")
            for lower, allocation, upper in zip(
                day["lower_bounds"], day["allocation"], day["upper_bounds"], strict=True
            ):
                if allocation < lower - 1e-7 or allocation > upper + 1e-7:
                    raise AssertionError(f"{planner_name} allocation violates a bound")
                checks += 1
            if any(day["projection"]["violation_breakdown"].values()):
                raise AssertionError(f"{planner_name} serialized violations are nonzero")
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:4117")
    parser.add_argument("--capture-result", type=Path)
    parser.add_argument("--restore-result", type=Path)
    args = parser.parse_args()
    wait_ready(args.base_url)

    if args.restore_result is not None:
        expected = args.restore_result.read_bytes()
        captured = json.loads(expected)
        result_id = captured["result_id"]
        status, restored = fetch(f"{args.base_url}/api/v1/simulations/{result_id}")
        if status != 200 or restored != expected:
            raise AssertionError("persisted result changed after process restart")
        print(
            json.dumps(
                {
                    "response_sha256": hashlib.sha256(restored).hexdigest(),
                    "restored_result_id": result_id,
                    "status": "restart-persistence-passed",
                },
                sort_keys=True,
            )
        )
        return

    status, homepage = fetch(f"{args.base_url}/")
    if status != 200 or b"Civic Relay" not in homepage:
        raise AssertionError("compiled frontend was not served")
    status, meta_bytes = fetch(f"{args.base_url}/api/v1/meta")
    meta = json.loads(meta_bytes)
    if status != 200 or meta["default_seed"] != 20260714:
        raise AssertionError("runtime metadata is incomplete")
    if meta["schema_version"] != "2.0.0":
        raise AssertionError("runtime API schema metadata is incomplete")
    if (
        meta["model"]["artifact_type"] != "stable_baselines3_ppo"
        or meta["model"]["algorithm"] != "PPO"
        or not meta["model"]["onnx_sha256"]
    ):
        raise AssertionError("runtime policy metadata is incomplete")
    if meta["model"]["legacy_candidate"]["is_ppo"] is not False:
        raise AssertionError("accepted linear candidate was relabeled PPO")
    if meta["dataset"]["version"] != "2.0.0" or meta["dataset"]["empirical"] is not False:
        raise AssertionError("runtime dataset metadata is incomplete")

    unseen = {
        "seed": 118773,
        "scenario": {
            "name": "Unseen west-corridor recovery",
            "horizon_days": 11,
            "daily_budget": 147,
            "initial_services": [0.19, 0.57, 0.33, 0.46, 0.24],
            "priorities": [1.7, 0.8, 1.3, 1.6, 0.9],
            "shock_probability": 0.23,
            "severity_min": 0.09,
            "severity_max": 0.31,
            "forced_shock": {"day": 7, "type": "weather", "severity": 0.24},
        },
    }
    responses = [fetch(f"{args.base_url}/api/v1/simulations/compare", unseen) for _ in range(5)]
    if any(status != 200 for status, _ in responses):
        raise AssertionError("unseen comparison request failed")
    if len({body for _, body in responses}) != 1:
        raise AssertionError("canonical response changed across five identical runs")
    result = json.loads(responses[0][1])
    checks = assert_constraints(result)
    result_id = result["result_id"]
    restore_status, restore_bytes = fetch(f"{args.base_url}/api/v1/simulations/{result_id}")
    if restore_status != 200 or restore_bytes != responses[0][1]:
        raise AssertionError("persisted comparison did not restore byte-identically")
    if args.capture_result is not None:
        args.capture_result.write_bytes(responses[0][1])
    index_status, index_bytes = fetch(f"{args.base_url}/api/v1/simulations")
    index = json.loads(index_bytes)
    if index_status != 200 or result_id not in {item["result_id"] for item in index["results"]}:
        raise AssertionError("persisted comparison is absent from the deterministic index")

    invalid_status, invalid_bytes = fetch(
        f"{args.base_url}/api/v1/simulations/compare",
        {"seed": 2, "scenario": {"horizon_days": 31}},
    )
    invalid = json.loads(invalid_bytes)
    if invalid_status != 422 or invalid["error"]["code"] != "INVALID_SCENARIO":
        raise AssertionError("invalid scenario did not fail with a structured error")

    print(
        json.dumps(
            {
                "baseline_rauc": result["baseline"]["rauc"],
                "candidate_rauc": result["candidate"]["rauc"],
                "constraint_checks": checks,
                "repeats": 5,
                "response_sha256": hashlib.sha256(responses[0][1]).hexdigest(),
                "restored_result_id": result_id,
                "schedule_sha256": result["shock_schedule_sha256"],
                "status": "runtime-verification-passed",
                "unseen_seed": unseen["seed"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
