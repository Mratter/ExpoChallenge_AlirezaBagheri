#!/usr/bin/env python3
"""Run the fixed privileged CEM diagnostic on canonical 200-case splits.

The runner reuses the exact tuned-rule, MPC, and full-sequence CEM
implementations from :mod:`scripts.headroom`.  It never loads or evaluates a
learned policy on the final split.  Results are written as immutable per-case
shards under an out-of-repository output root so a Windows worker-pool failure
does not discard completed compute.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

# Spawned workers must inherit these limits before NumPy or OR-Tools imports.
for _thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.shared_evidence import (  # noqa: E402
    canonical_hash,
    file_sha256,
    fsync_parent,
    load_json_object,
    split_contract,
    wilson_interval,
)
from backend.app.city.scenarios import (  # noqa: E402
    DEVELOPMENT_FAMILIES,
    DEVELOPMENT_SEEDS,
    FINAL_FAMILIES,
    FINAL_SEEDS,
    generate_disaster_tape,
)
from scripts.headroom import (  # noqa: E402
    MPC_HORIZONS,
    HeadroomCase,
    HeadroomError,
    MPCConfig,
    OracleConfig,
    PlannerResult,
    _assert_result_equivalent,
    _result_from_receipt,
    aggregate_results,
    rollout_actions,
    run_mpc_case,
    run_oracle_case,
    select_best_mpc_k,
    tuned_rollout,
)

TOOL_ID = "run_oracle_study.py"
SCHEMA_VERSION = 1
CANONICAL_CASE_COUNT = 200
HISTORICAL_RECEIPT = (
    ROOT / "internal" / "developmental_runs" / "v4" / "headroom-probe-v4-dev.json"
)
HISTORICAL_RECEIPT_SHA256 = (
    "f037c98d8fec483dfa6b5c9c1691842597a4163c7d1ee6f3e72618f987d671b9"
)
DEFAULT_REFERENCE_EVIDENCE = (
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
SHIPPED_ONNX_SHA256 = (
    "a9f5e9b41be57d7cd34623725a5ab4067aa75fbab16dc666cecc3c0a06c26483"
)

HISTORICAL_MPC_CONFIG = MPCConfig(
    population=48,
    elite_count=6,
    iterations=5,
    fantasies=4,
    initial_std=0.35,
    std_floor=0.05,
    smoothing=0.80,
)
HISTORICAL_ORACLE_CONFIG = OracleConfig(
    population=512,
    elite_fraction=0.10,
    min_iterations=20,
    max_iterations=40,
    patience=6,
    initial_std=0.25,
    std_floor=0.03,
    smoothing=0.75,
)


class OracleStudyError(RuntimeError):
    """Raised when the fixed oracle study contract cannot be established."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _source_identity() -> dict[str, str]:
    paths = (
        "scripts/run_oracle_study.py",
        "scripts/headroom.py",
        "backend/app/shared_evidence.py",
        "backend/app/city/environment.py",
        "backend/app/city/scenarios.py",
        "backend/app/city/outcome.py",
        "backend/app/city/physics.py",
        "backend/app/city/planners.py",
        "backend/app/city/optimizer.py",
    )
    return {path: file_sha256(ROOT / path) for path in paths}


def build_cases(split: str) -> list[HeadroomCase]:
    """Build one canonical split without involving a learned policy."""

    if split == "dev":
        families, seeds = DEVELOPMENT_FAMILIES, DEVELOPMENT_SEEDS
    elif split == "final":
        families, seeds = FINAL_FAMILIES, FINAL_SEEDS
    else:
        raise OracleStudyError(f"unsupported split: {split}")
    cases: list[HeadroomCase] = []
    for family in families:
        for case_seed in seeds:
            scenario = family.build(case_seed)
            tape_seed = family.tape_seed(case_seed)
            cases.append(
                HeadroomCase(
                    row_id=f"{family.id}:{case_seed}",
                    family_id=family.id,
                    case_seed=case_seed,
                    tape_seed=tape_seed,
                    scenario=scenario,
                    schedule=tuple(generate_disaster_tape(scenario, tape_seed)),
                )
            )
    if len(cases) != CANONICAL_CASE_COUNT or len({case.row_id for case in cases}) != CANONICAL_CASE_COUNT:
        raise OracleStudyError(f"{split} must contain 200 unique cases")
    return cases


def _case_contract(case: HeadroomCase) -> dict[str, Any]:
    return {
        "row_id": case.row_id,
        "family_id": case.family_id,
        "case_seed": case.case_seed,
        "tape_seed": case.tape_seed,
        "tape_sha256": canonical_hash([asdict(shock) for shock in case.schedule]),
    }


def _atomic_create_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise OracleStudyError(f"refusing to overwrite existing evidence: {path}")
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        fsync_parent(path)
    except FileExistsError as exc:
        raise OracleStudyError(f"refusing to overwrite existing evidence: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    return load_json_object(path, label, error_type=OracleStudyError)


def _study_contract(splits: Sequence[str]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL_ID,
        "git_commit": _git_commit(),
        "splits": list(splits),
        "split_contracts": {
            "dev": split_contract("dev", DEVELOPMENT_FAMILIES, DEVELOPMENT_SEEDS),
            "final": split_contract("final", FINAL_FAMILIES, FINAL_SEEDS),
        },
        "mpc_config": asdict(HISTORICAL_MPC_CONFIG),
        "oracle_config": asdict(HISTORICAL_ORACLE_CONFIG),
        "objective": ["solved", "minimum_tail_margin", "resilience_auc"],
        "rng_namespace": "headroom-oracle-v1:{row_id}:{tape_seed}",
        "warm_starts": [
            "tuned_rule_preparedness_multiplier_10_cap_0.50",
            "globally_selected_mpc_horizon_from_k_1_3_5",
        ],
        "historical_reference": {
            "path": str(HISTORICAL_RECEIPT.relative_to(ROOT)).replace("\\", "/"),
            "sha256": HISTORICAL_RECEIPT_SHA256,
            "solved_count": 37,
            "case_count": 40,
            "scope": "original_40_case_development_subset",
        },
        "source_identity": _source_identity(),
        "scientific_disclosure": {
            "privileged_clairvoyant": True,
            "full_future_shock_tape_visible": True,
            "anytime_achieved_lower_bound": True,
            "mathematical_optimum_claimed": False,
            "infeasibility_certificate_claimed": False,
            "submission_baseline": False,
            "model_selection_used": False,
        },
    }


def _record_path(root: Path, split: str, phase: str, index: int) -> Path:
    return root / split / phase / f"{index:03d}.json"


def _wrap_record(
    *,
    contract_sha256: str,
    split: str,
    phase: str,
    index: int,
    case: HeadroomCase,
    payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "contract_sha256": contract_sha256,
        "split": split,
        "phase": phase,
        "index": index,
        "case": _case_contract(case),
        "payload": payload,
    }


def _load_record(
    path: Path,
    *,
    contract_sha256: str,
    split: str,
    phase: str,
    index: int,
    case: HeadroomCase,
) -> dict[str, Any]:
    record = _load_json(path, f"{split} {phase} case record")
    expected = {
        "contract_sha256": contract_sha256,
        "split": split,
        "phase": phase,
        "index": index,
        "case": _case_contract(case),
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise OracleStudyError(f"stale or mismatched record {path}: {key}")
    if not isinstance(record.get("payload"), dict):
        raise OracleStudyError(f"record payload is invalid: {path}")
    return record["payload"]


def run_preparation_case(job: tuple[HeadroomCase, MPCConfig]) -> dict[str, Any]:
    """Produce the exact tuned and k=1/3/5 MPC warm starts for one case."""

    case, config = job
    compact, tuned_actions = tuned_rollout(case)
    tuned_replay = rollout_actions(case, tuned_actions, collect_evidence=True)
    _assert_result_equivalent(compact, tuned_replay, f"{case.row_id} tuned")
    mpc = run_mpc_case((case, config))
    horizons = {
        horizon: {
            key: value
            for key, value in mpc["horizons"][horizon].items()
            if key != "days"
        }
        for horizon in ("1", "3", "5")
    }
    return {
        "row_id": case.row_id,
        "tuned": {
            "result": tuned_replay.as_receipt(),
            "actions": tuned_actions.tolist(),
        },
        "mpc": {
            "horizons": horizons,
            "worker_runtime": mpc["worker_runtime"],
        },
    }


def _run_missing_parallel(
    *,
    root: Path,
    contract_sha256: str,
    split: str,
    phase: str,
    cases: Sequence[HeadroomCase],
    jobs: Sequence[Any],
    worker: Callable[[Any], dict[str, Any]],
    workers: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any] | None] = [None] * len(cases)
    missing: list[int] = []
    for index, case in enumerate(cases):
        path = _record_path(root, split, phase, index)
        if path.exists():
            results[index] = _load_record(
                path,
                contract_sha256=contract_sha256,
                split=split,
                phase=phase,
                index=index,
                case=case,
            )
        else:
            missing.append(index)
    if not missing:
        return [value for value in results if value is not None]

    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        futures = {executor.submit(worker, jobs[index]): index for index in missing}
        completed_count = len(cases) - len(missing)
        for future in as_completed(futures):
            index = futures[future]
            payload = future.result()
            case = cases[index]
            if payload.get("row_id") != case.row_id:
                raise OracleStudyError(f"worker returned wrong row for {case.row_id}")
            record = _wrap_record(
                contract_sha256=contract_sha256,
                split=split,
                phase=phase,
                index=index,
                case=case,
                payload=payload,
            )
            _atomic_create_json(_record_path(root, split, phase, index), record)
            results[index] = payload
            completed_count += 1
            print(
                f"{split} {phase}: {completed_count}/{len(cases)} {case.row_id}",
                flush=True,
            )
    if any(value is None for value in results):
        raise OracleStudyError(f"{split} {phase} did not produce every case")
    return [value for value in results if value is not None]


def _parallel_with_fallback(
    *,
    root: Path,
    contract_sha256: str,
    split: str,
    phase: str,
    cases: Sequence[HeadroomCase],
    jobs: Sequence[Any],
    worker: Callable[[Any], dict[str, Any]],
    workers: int,
    runtime_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    started = time.perf_counter()
    try:
        values = _run_missing_parallel(
            root=root,
            contract_sha256=contract_sha256,
            split=split,
            phase=phase,
            cases=cases,
            jobs=jobs,
            worker=worker,
            workers=workers,
        )
        runtime_events.append(
            {
                "split": split,
                "phase": phase,
                "workers": workers,
                "fallback": False,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        )
        return values
    except BrokenProcessPool as exc:
        if workers < 8:
            raise
        print(
            f"{split} {phase}: worker pool failed at {workers}; resuming missing shards at 4",
            file=sys.stderr,
            flush=True,
        )
        runtime_events.append(
            {
                "split": split,
                "phase": phase,
                "workers": workers,
                "fallback": True,
                "fallback_workers": 4,
                "reason": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds_before_fallback": round(time.perf_counter() - started, 3),
            }
        )
        return _run_missing_parallel(
            root=root,
            contract_sha256=contract_sha256,
            split=split,
            phase=phase,
            cases=cases,
            jobs=jobs,
            worker=worker,
            workers=4,
        )


def _select_mpc(
    preparations: Sequence[dict[str, Any]], cases: Sequence[HeadroomCase]
) -> tuple[int, dict[str, Any]]:
    rows_by_horizon: dict[int, dict[str, PlannerResult]] = {
        horizon: {
            case.row_id: _result_from_receipt(
                preparations[index]["mpc"]["horizons"][str(horizon)]["result"]
            )
            for index, case in enumerate(cases)
        }
        for horizon in MPC_HORIZONS
    }
    return select_best_mpc_k(rows_by_horizon)


def _validate_invariants(label: str, rows: Sequence[PlannerResult]) -> None:
    if len(rows) != CANONICAL_CASE_COUNT:
        raise OracleStudyError(f"{label} must contain 200 results")
    if any(row.hard_violation_count != 0 for row in rows):
        raise OracleStudyError(f"{label} produced a hard violation")
    if any(row.maximum_conservation_residual != 0.0 for row in rows):
        raise OracleStudyError(f"{label} produced a conservation residual")


def build_reference_comparison(
    *,
    cases: Sequence[HeadroomCase],
    oracle_rows: Sequence[PlannerResult],
    reference_path: Path,
) -> dict[str, Any]:
    """Join dev oracle rows to already accepted shipped-ONNX evidence."""

    if file_sha256(reference_path) != SHIPPED_PARITY_SHA256:
        raise OracleStudyError("shipped parity receipt hash mismatch")
    receipt = _load_json(reference_path, "shipped parity receipt")
    parity = receipt.get("parity")
    if (
        receipt.get("split") != "dev"
        or receipt.get("final_split_used") is not False
        or not isinstance(parity, dict)
        or parity.get("passed") is not True
        or parity.get("rows_sha256") != SHIPPED_PARITY_ROWS_SHA256
        or receipt.get("onnx_artifact", {}).get("sha256") != SHIPPED_ONNX_SHA256
    ):
        raise OracleStudyError("shipped parity receipt contract mismatch")
    rows = parity.get("rows")
    if not isinstance(rows, list) or len(rows) != CANONICAL_CASE_COUNT:
        raise OracleStudyError("shipped parity rows must contain 200 cases")

    both = policy_only = oracle_only = neither = 0
    oracle_only_ids: list[str] = []
    policy_only_ids: list[str] = []
    joined_rows: list[dict[str, Any]] = []
    for case, oracle, policy_row in zip(cases, oracle_rows, rows, strict=True):
        contract = _case_contract(case)
        onnx = policy_row.get("onnx")
        if (
            policy_row.get("row_id") != case.row_id
            or policy_row.get("tape_seed") != case.tape_seed
            or policy_row.get("tape_sha256") != contract["tape_sha256"]
            or not isinstance(onnx, dict)
            or onnx.get("row_id") != case.row_id
            or int(onnx.get("hard_violation_count", -1)) != 0
            or float(onnx.get("maximum_conservation_residual", -1.0)) != 0.0
        ):
            raise OracleStudyError(f"shipped parity row mismatch: {case.row_id}")
        policy_solved = bool(onnx["solved"])
        if policy_solved and oracle.solved:
            both += 1
            classification = "both"
        elif policy_solved:
            policy_only += 1
            policy_only_ids.append(case.row_id)
            classification = "policy_only"
        elif oracle.solved:
            oracle_only += 1
            oracle_only_ids.append(case.row_id)
            classification = "oracle_only_contested"
        else:
            neither += 1
            classification = "neither"
        joined_rows.append(
            {
                "row_id": case.row_id,
                "tape_seed": case.tape_seed,
                "tape_sha256": contract["tape_sha256"],
                "shipped_policy_solved": policy_solved,
                "oracle_solved": oracle.solved,
                "classification": classification,
            }
        )
    policy_count = both + policy_only
    oracle_count = both + oracle_only
    if policy_count != 178:
        raise OracleStudyError("shipped policy dev solved count drifted from 178")
    return {
        "reference": {
            "policy": "shipped city_recovery_ppo.v4.onnx",
            "policy_evaluated_in_this_run": False,
            "parity_receipt_sha256": SHIPPED_PARITY_SHA256,
            "parity_rows_sha256": SHIPPED_PARITY_ROWS_SHA256,
            "onnx_sha256": SHIPPED_ONNX_SHA256,
        },
        "pairing": {
            "both": both,
            "policy_only": policy_only,
            "oracle_only": oracle_only,
            "neither": neither,
        },
        "shipped_policy_solved_count": policy_count,
        "oracle_solved_count": oracle_count,
        "oracle_only_contested_count": oracle_only,
        "oracle_only_contested_row_ids": oracle_only_ids,
        "policy_only_count": policy_only,
        "policy_only_row_ids": policy_only_ids,
        "known_feasible_union_count": both + policy_only + oracle_only,
        "remaining_provable_headroom_cases": oracle_only,
        "rows": joined_rows,
    }


def _runtime_records(values: Sequence[dict[str, Any]], path: Sequence[str]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for value in values:
        current: Any = value
        for key in path:
            current = current[key]
        unique[canonical_hash(current)] = current
    return sorted(unique.values(), key=lambda item: int(item["pid"]))


def _build_split_receipt(
    *,
    split: str,
    cases: Sequence[HeadroomCase],
    preparations: Sequence[dict[str, Any]],
    oracle_values: Sequence[dict[str, Any]],
    selected_horizon: int,
    mpc_selection: dict[str, Any],
    contract: dict[str, Any],
    contract_sha256: str,
    output_root: Path,
    workers: int,
    runtime_events: Sequence[dict[str, Any]],
    elapsed_seconds: float,
    reference_path: Path,
) -> dict[str, Any]:
    tuned_rows: list[PlannerResult] = []
    selected_mpc_rows: list[PlannerResult] = []
    oracle_rows: list[PlannerResult] = []
    rows: list[dict[str, Any]] = []
    total_candidate_evaluations = 0
    total_simulated_transitions = 0
    for index, case in enumerate(cases):
        preparation = preparations[index]
        oracle_value = oracle_values[index]
        tuned = _result_from_receipt(preparation["tuned"]["result"])
        mpc_payload = preparation["mpc"]["horizons"][str(selected_horizon)]
        mpc_compact = _result_from_receipt(mpc_payload["result"])
        mpc_actions = np.asarray(mpc_payload["actions"], dtype=np.float64)
        mpc_replay = rollout_actions(case, mpc_actions, collect_evidence=True)
        _assert_result_equivalent(mpc_compact, mpc_replay, f"{case.row_id} selected MPC")
        oracle = _result_from_receipt(oracle_value["result"])
        search_invariants = oracle_value["search_wide_invariants"]
        if (
            int(search_invariants["maximum_hard_violation_count"]) != 0
            or float(search_invariants["maximum_conservation_residual"]) != 0.0
        ):
            raise OracleStudyError(f"oracle search invariant failed: {case.row_id}")
        tuned_rows.append(tuned)
        selected_mpc_rows.append(mpc_replay)
        oracle_rows.append(oracle)
        budget = oracle_value["budget"]
        total_candidate_evaluations += int(budget["candidate_evaluations"])
        total_simulated_transitions += int(budget["simulated_transitions"])
        rows.append(
            {
                **_case_contract(case),
                "tuned_rule": tuned.as_receipt(),
                f"selected_mpc_k{selected_horizon}": mpc_replay.as_receipt(),
                "clairvoyant_oracle_cem": oracle.as_receipt(),
                "oracle_budget": budget,
                "oracle_warm_start": oracle_value["warm_start"],
                "oracle_search_wide_invariants": search_invariants,
                "oracle_worker_runtime": oracle_value["worker_runtime"],
            }
        )
    _validate_invariants("tuned rule", tuned_rows)
    _validate_invariants("selected MPC", selected_mpc_rows)
    _validate_invariants("clairvoyant oracle", oracle_rows)
    oracle_aggregate = aggregate_results(oracle_rows)
    solved_count = int(oracle_aggregate["solved_count"])
    oracle_aggregate["wilson_95"] = wilson_interval(
        solved_count, CANONICAL_CASE_COUNT, digits=10
    )
    reference = (
        build_reference_comparison(
            cases=cases,
            oracle_rows=oracle_rows,
            reference_path=reference_path,
        )
        if split == "dev"
        else None
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL_ID,
        "status": "complete_privileged_clairvoyant_oracle_diagnostic",
        "created_at_utc": _utc_now(),
        "authorizing": False,
        "model_selection_used": False,
        "submission_baseline": False,
        "privileged_diagnostic": True,
        "split": split,
        "split_contract": contract["split_contracts"][split],
        "case_count": CANONICAL_CASE_COUNT,
        "final_split_used": split == "final",
        "learned_v4_policy_evaluated": False,
        "learned_v4_final_evaluated": False,
        "full_future_shock_tape_visible": True,
        "interpretation": {
            "anytime_achieved_lower_bound": True,
            "mathematical_optimum_claimed": False,
            "infeasibility_certificate_claimed": False,
            "wording": (
                "Privileged CEM achieved lower-bound reference on identical cases; "
                "not a proven mathematical ceiling and not a submission baseline."
            ),
        },
        "study_contract_sha256": contract_sha256,
        "source_identity": contract["source_identity"],
        "historical_reference": contract["historical_reference"],
        "mpc_warm_start": {
            "config": asdict(HISTORICAL_MPC_CONFIG),
            "selection": mpc_selection,
        },
        "oracle": {
            "config": asdict(HISTORICAL_ORACLE_CONFIG),
            "objective": contract["objective"],
            "rng_namespace": contract["rng_namespace"],
            "aggregate": oracle_aggregate,
            "total_candidate_evaluations": total_candidate_evaluations,
            "total_simulated_transitions": total_simulated_transitions,
        },
        "planner_aggregates": {
            "tuned_rule": aggregate_results(tuned_rows),
            f"selected_mpc_k{selected_horizon}": aggregate_results(selected_mpc_rows),
            "clairvoyant_oracle_cem": oracle_aggregate,
        },
        "development_shipped_policy_comparison": reference,
        "rows": rows,
        "rows_sha256": canonical_hash(rows),
        "invariants": {
            "case_count_exactly_200": len(rows) == CANONICAL_CASE_COUNT,
            "row_ids_unique": len({row["row_id"] for row in rows}) == CANONICAL_CASE_COUNT,
            "all_planner_hard_violation_counts_zero": True,
            "all_planner_conservation_residuals_exactly_zero": True,
            "all_oracle_search_hard_violation_counts_zero": True,
            "all_oracle_search_conservation_residuals_exactly_zero": True,
            "production_decoder_and_exact_projection_used": True,
            "learned_policy_loaded_or_run": False,
        },
        "runtime": {
            "output_root": str(output_root),
            "requested_workers": workers,
            "multiprocessing_start_method": "spawn",
            "events": list(runtime_events),
            "preparation_worker_runtime_records": _runtime_records(
                preparations, ("mpc", "worker_runtime")
            ),
            "oracle_worker_runtime_records": _runtime_records(
                oracle_values, ("worker_runtime",)
            ),
            "native_thread_limits": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
            "elapsed_seconds": round(elapsed_seconds, 3),
        },
    }


def _run_split(
    *,
    split: str,
    root: Path,
    contract: dict[str, Any],
    contract_sha256: str,
    workers: int,
    reference_path: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    cases = build_cases(split)
    runtime_events: list[dict[str, Any]] = []
    preparations = _parallel_with_fallback(
        root=root,
        contract_sha256=contract_sha256,
        split=split,
        phase="preparation",
        cases=cases,
        jobs=[(case, HISTORICAL_MPC_CONFIG) for case in cases],
        worker=run_preparation_case,
        workers=workers,
        runtime_events=runtime_events,
    )
    selected_horizon, mpc_selection = _select_mpc(preparations, cases)
    print(f"{split}: globally selected MPC k={selected_horizon}", flush=True)
    oracle_jobs = [
        (
            case,
            np.asarray(preparations[index]["tuned"]["actions"], dtype=np.float64),
            np.asarray(
                preparations[index]["mpc"]["horizons"][str(selected_horizon)][
                    "actions"
                ],
                dtype=np.float64,
            ),
            HISTORICAL_ORACLE_CONFIG,
        )
        for index, case in enumerate(cases)
    ]
    oracle_values = _parallel_with_fallback(
        root=root,
        contract_sha256=contract_sha256,
        split=split,
        phase="oracle",
        cases=cases,
        jobs=oracle_jobs,
        worker=run_oracle_case,
        workers=workers,
        runtime_events=runtime_events,
    )
    receipt = _build_split_receipt(
        split=split,
        cases=cases,
        preparations=preparations,
        oracle_values=oracle_values,
        selected_horizon=selected_horizon,
        mpc_selection=mpc_selection,
        contract=contract,
        contract_sha256=contract_sha256,
        output_root=root,
        workers=workers,
        runtime_events=runtime_events,
        elapsed_seconds=time.perf_counter() - started,
        reference_path=reference_path,
    )
    receipt_path = root / split / "receipt.json"
    if receipt_path.exists():
        existing = _load_json(receipt_path, f"{split} receipt")
        if existing.get("rows_sha256") != receipt["rows_sha256"]:
            raise OracleStudyError(f"existing {split} receipt does not match shards")
        return existing
    _atomic_create_json(receipt_path, receipt)
    return receipt


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--splits", nargs="+", choices=("dev", "final"), default=("dev", "final")
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--reference-evidence", type=Path, default=DEFAULT_REFERENCE_EVIDENCE
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> tuple[Path, tuple[str, ...], Path]:
    root = args.output_root.resolve()
    if not args.output_root.is_absolute():
        raise OracleStudyError("--output-root must be absolute and out of repository")
    try:
        root.relative_to(ROOT)
    except ValueError:
        pass
    else:
        raise OracleStudyError("--output-root must be outside the repository")
    splits = tuple(dict.fromkeys(args.splits))
    if not splits or any(split not in {"dev", "final"} for split in splits):
        raise OracleStudyError("--splits must contain dev and/or final")
    if not 1 <= args.workers <= 16:
        raise OracleStudyError("--workers must be in [1, 16]")
    if root.exists() and not args.resume:
        raise OracleStudyError("output root exists; use --resume only for this exact study")
    if not root.exists() and args.resume:
        raise OracleStudyError("cannot resume a missing output root")
    reference = args.reference_evidence
    if not reference.is_absolute():
        reference = (ROOT / reference).resolve()
    return root, splits, reference


def main() -> int:
    args = _parse_args()
    root, splits, reference_path = _validate_args(args)
    if "dev" in splits and not reference_path.is_file():
        raise OracleStudyError("development run requires shipped parity evidence")
    if file_sha256(HISTORICAL_RECEIPT) != HISTORICAL_RECEIPT_SHA256:
        raise OracleStudyError("historical 37/40 receipt changed")
    contract = _study_contract(splits)
    contract_sha256 = canonical_hash(contract)
    protocol_path = root / "protocol.json"
    if root.exists():
        protocol = _load_json(protocol_path, "oracle study protocol")
        if protocol.get("contract_sha256") != contract_sha256 or protocol.get(
            "contract"
        ) != contract:
            raise OracleStudyError("resume contract differs from original study")
    else:
        root.mkdir(parents=True, exist_ok=False)
        _atomic_create_json(
            protocol_path,
            {
                "created_at_utc": _utc_now(),
                "contract_sha256": contract_sha256,
                "contract": contract,
            },
        )
    receipts: dict[str, dict[str, Any]] = {}
    for split in splits:
        receipts[split] = _run_split(
            split=split,
            root=root,
            contract=contract,
            contract_sha256=contract_sha256,
            workers=args.workers,
            reference_path=reference_path,
        )
        aggregate = receipts[split]["oracle"]["aggregate"]
        print(
            f"{split} complete: {aggregate['solved_count']}/200 "
            f"Wilson95={aggregate['wilson_95']}",
            flush=True,
        )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL_ID,
        "status": "complete",
        "created_at_utc": _utc_now(),
        "contract_sha256": contract_sha256,
        "splits": {
            split: {
                "receipt": f"{split}/receipt.json",
                "receipt_sha256": file_sha256(root / split / "receipt.json"),
                "solved_count": receipts[split]["oracle"]["aggregate"]["solved_count"],
                "solve_rate": receipts[split]["oracle"]["aggregate"]["solve_rate"],
                "wilson_95": receipts[split]["oracle"]["aggregate"]["wilson_95"],
                "learned_v4_policy_evaluated": False,
            }
            for split in splits
        },
        "learned_v4_final_evaluated": False,
        "historical_37_of_40_preserved": True,
    }
    summary_path = root / "summary.json"
    if not summary_path.exists():
        _atomic_create_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        raise SystemExit(main())
    except (HeadroomError, OracleStudyError) as error:
        print(f"oracle study failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
