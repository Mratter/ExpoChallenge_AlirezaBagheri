#!/usr/bin/env python3
"""Build privileged CEM demonstrations on the authored training split only.

This preparation tool never trains or evaluates a learned policy.  It runs the
registered tuned-rule/MPC warm starts and full-sequence CEM budget on the 192
authored training cases, then replays each winning action sequence to capture
the 73 public observations available to a future causal student.  Immutable
per-case shards live under an absolute out-of-repository root so interrupted
Windows worker pools can resume without repeating completed cases.
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

from backend.app.city.environment import (  # noqa: E402
    ACTION_ORDER,
    ACTION_SIZE,
    OBSERVATION_ORDER,
    OBSERVATION_SIZE,
    CityRecoveryEnv,
)
from backend.app.city.scenarios import (  # noqa: E402
    TRAINING_FAMILIES,
    TRAINING_SEEDS,
    generate_disaster_tape,
)
from backend.app.shared_evidence import (  # noqa: E402
    canonical_hash,
    file_sha256,
    fsync_parent,
    function_source_sha256,
    load_json_object,
    split_contract,
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

TOOL_ID = "run_training_oracle_trajectories.py"
SCHEMA_VERSION = 1
TRAINING_CASE_COUNT = 192
HORIZON_DAYS = 30
TRAINING_OBSERVATION_COUNT = TRAINING_CASE_COUNT * HORIZON_DAYS
DEFAULT_WORKERS = 8
FALLBACK_WORKERS = 4

# These are the registered settings used for the historical 37/40 result and
# the matched 200-case oracle study.  They are repeated here because importing
# the study publisher would also import non-training split rosters.
REGISTERED_MPC_CONFIG = MPCConfig(
    population=48,
    elite_count=6,
    iterations=5,
    fantasies=4,
    initial_std=0.35,
    std_floor=0.05,
    smoothing=0.80,
)
REGISTERED_ORACLE_CONFIG = OracleConfig(
    population=512,
    elite_fraction=0.10,
    min_iterations=20,
    max_iterations=40,
    patience=6,
    initial_std=0.25,
    std_floor=0.03,
    smoothing=0.75,
)


class TrainingOracleError(RuntimeError):
    """Raised when the training-only trajectory contract cannot be honored."""


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


def _worktree_is_clean() -> bool:
    completed = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return not completed.stdout.strip()


def _source_identity() -> dict[str, str]:
    paths = (
        "scripts/run_training_oracle_trajectories.py",
        "scripts/headroom.py",
        "scripts/train_policy.py",
        "scripts/training_artifacts.py",
        "backend/app/shared_evidence.py",
        "backend/app/city/environment.py",
        "backend/app/city/scenarios.py",
        "backend/app/city/outcome.py",
        "backend/app/city/physics.py",
        "backend/app/city/planners.py",
        "backend/app/city/optimizer.py",
    )
    return {path: file_sha256(ROOT / path) for path in paths}


def build_training_cases() -> list[HeadroomCase]:
    """Build exactly the authored 6-family by 32-seed training roster."""

    cases: list[HeadroomCase] = []
    for family in TRAINING_FAMILIES:
        for case_seed in TRAINING_SEEDS:
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
    if (
        len(cases) != TRAINING_CASE_COUNT
        or len({case.row_id for case in cases}) != TRAINING_CASE_COUNT
        or len(TRAINING_FAMILIES) != 6
        or TRAINING_SEEDS != tuple(range(810000, 810032))
        or any(not case.family_id.startswith("v3_train_") for case in cases)
    ):
        raise TrainingOracleError("training roster must remain 6 x 32 = 192 cases")
    if any(case.scenario.horizon_days != HORIZON_DAYS for case in cases):
        raise TrainingOracleError("training horizon must remain 30 days")
    return cases


def _case_contract(case: HeadroomCase) -> dict[str, Any]:
    return {
        "row_id": case.row_id,
        "family_id": case.family_id,
        "case_seed": case.case_seed,
        "tape_seed": case.tape_seed,
        "scenario_sha256": canonical_hash(case.scenario.model_dump(mode="json")),
        "tape_sha256": canonical_hash(
            [asdict(shock) for shock in case.schedule]
        ),
    }


def study_contract() -> dict[str, Any]:
    """Return the immutable, training-only compute and dataset contract."""

    if OBSERVATION_SIZE != 73 or len(OBSERVATION_ORDER) != 73:
        raise TrainingOracleError("public observation contract must contain 73 values")
    if ACTION_SIZE != 22 or len(ACTION_ORDER) != 22:
        raise TrainingOracleError("policy action contract must contain 22 values")
    cases = build_training_cases()
    ordered_cases = [_case_contract(case) for case in cases]
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL_ID,
        "git_commit": _git_commit(),
        "split": split_contract("train", TRAINING_FAMILIES, TRAINING_SEEDS),
        "ordered_case_contract_sha256": canonical_hash(ordered_cases),
        "access_contract": {
            "training_split_used": True,
            "development_split_used": False,
            "final_split_used": False,
            "learned_policy_loaded_or_run": False,
        },
        "mpc_config": asdict(REGISTERED_MPC_CONFIG),
        "oracle_config": asdict(REGISTERED_ORACLE_CONFIG),
        "objective": ["solved", "minimum_tail_margin", "resilience_auc"],
        "rng_namespace": "headroom-oracle-v1:{row_id}:{tape_seed}",
        "warm_starts": [
            "tuned_rule_preparedness_multiplier_10_cap_0.50",
            "globally_selected_training_mpc_horizon_from_k_1_3_5",
        ],
        "demonstration_contract": {
            "case_count": TRAINING_CASE_COUNT,
            "horizon_days": HORIZON_DAYS,
            "row_count": TRAINING_OBSERVATION_COUNT,
            "observation_count": OBSERVATION_SIZE,
            "action_count": ACTION_SIZE,
            "observation_dtype": "float32",
            "action_target_dtype": "float32",
            "observation_order": list(OBSERVATION_ORDER),
            "action_order": list(ACTION_ORDER),
            "student_input_future_tape_visible": False,
            "teacher_target_uses_full_future_tape": True,
        },
        "future_student_contract": {
            "training_method": "behavior_cloning_only",
            "dagger_used": False,
            "ppo_used": False,
            "actor_architecture": [384, 256, 128],
            "activation": "SiLU",
            "orthogonal_initialization": True,
            "log_standard_deviation_initialization": -1.5,
            "normalization": {
                "fit_source": "all 5760 raw training observations only",
                "implementation": "stable_baselines3.common.running_mean_std.RunningMeanStd",
                "epsilon": 1e-8,
                "clip_observation": 10.0,
                "frozen_during_student_evaluation": True,
            },
            "implementation_references": {
                function_name: function_source_sha256(
                    ROOT,
                    "scripts/train_policy.py",
                    function_name,
                    error_type=TrainingOracleError,
                )
                for function_name in (
                    "build_model",
                    "normalize_observations",
                    "behavior_clone_policy",
                )
            },
        },
        "scientific_disclosure": {
            "privileged_teacher": True,
            "full_future_shock_tape_visible_to_teacher": True,
            "full_future_shock_tape_visible_to_student": False,
            "anytime_achieved_lower_bound": True,
            "mathematical_optimum_claimed": False,
            "student_performance_claimed": False,
        },
        "source_identity": _source_identity(),
    }


def _atomic_create_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise TrainingOracleError(f"refusing to overwrite existing evidence: {path}")
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
        raise TrainingOracleError(
            f"refusing to overwrite existing evidence: {path}"
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    return load_json_object(path, label, error_type=TrainingOracleError)


def _record_path(root: Path, phase: str, index: int) -> Path:
    return root / "training" / phase / f"{index:03d}.json"


def _wrap_record(
    *,
    contract_sha256: str,
    phase: str,
    index: int,
    case: HeadroomCase,
    payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "contract_sha256": contract_sha256,
        "split": "train",
        "phase": phase,
        "index": index,
        "case": _case_contract(case),
        "payload": payload,
    }


def _load_record(
    path: Path,
    *,
    contract_sha256: str,
    phase: str,
    index: int,
    case: HeadroomCase,
) -> dict[str, Any]:
    record = _load_json(path, f"training {phase} case record")
    expected = {
        "contract_sha256": contract_sha256,
        "split": "train",
        "phase": phase,
        "index": index,
        "case": _case_contract(case),
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise TrainingOracleError(f"stale or mismatched record {path}: {key}")
    payload = record.get("payload")
    if not isinstance(payload, dict) or payload.get("row_id") != case.row_id:
        raise TrainingOracleError(f"record payload is invalid: {path}")
    return payload


def run_preparation_case(job: tuple[HeadroomCase, MPCConfig]) -> dict[str, Any]:
    """Produce the tuned and k=1/3/5 MPC warm starts for one training case."""

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


def collect_public_demonstration(
    case: HeadroomCase,
    actions: np.ndarray,
) -> dict[str, Any]:
    """Replay oracle targets and capture only observations exposed by ``reset/step``."""

    sequence = np.asarray(actions, dtype=np.float64)
    if sequence.shape != (HORIZON_DAYS, ACTION_SIZE):
        raise TrainingOracleError(
            f"{case.row_id} action sequence must have shape "
            f"({HORIZON_DAYS}, {ACTION_SIZE})"
        )
    if not np.all(np.isfinite(sequence)) or np.any(np.abs(sequence) > 1.0):
        raise TrainingOracleError(f"{case.row_id} action sequence is invalid")

    environment = CityRecoveryEnv(
        case.scenario,
        case.tape_seed,
        list(case.schedule),
        collect_evidence=True,
    )
    observation, reset_evidence = environment.reset(seed=case.tape_seed)
    if reset_evidence.get("shock_schedule_sha256") != _case_contract(case)[
        "tape_sha256"
    ]:
        raise TrainingOracleError(f"{case.row_id} replay tape identity drifted")
    observations: list[np.ndarray] = []
    terminated = False
    for day_index, action in enumerate(sequence):
        current = np.asarray(observation, dtype=np.float32)
        if current.shape != (OBSERVATION_SIZE,) or not np.all(np.isfinite(current)):
            raise TrainingOracleError(
                f"{case.row_id} day {day_index + 1} observation is invalid"
            )
        observations.append(current.copy())
        observation, _, terminated, truncated, _ = environment.step(action)
        if truncated:
            raise TrainingOracleError(f"{case.row_id} replay unexpectedly truncated")
        if terminated != (day_index + 1 == HORIZON_DAYS):
            raise TrainingOracleError(f"{case.row_id} replay horizon drifted")
    if not terminated or len(observations) != HORIZON_DAYS:
        raise TrainingOracleError(f"{case.row_id} replay did not reach day 30")

    observation_array = np.asarray(observations, dtype=np.float32)
    target_array = sequence.astype(np.float32)
    observation_rows = observation_array.tolist()
    target_rows = target_array.tolist()
    return {
        "row_id": case.row_id,
        "input_contract": "73_public_causal_observations",
        "student_input_future_tape_visible": False,
        "teacher_target_uses_full_future_tape": True,
        "observation_dtype": "float32",
        "target_dtype": "float32",
        "observation_shape": [HORIZON_DAYS, OBSERVATION_SIZE],
        "target_shape": [HORIZON_DAYS, ACTION_SIZE],
        "observations": observation_rows,
        "targets": target_rows,
        "observations_sha256": canonical_hash(observation_rows),
        "targets_sha256": canonical_hash(target_rows),
        "dataset_sha256": canonical_hash(
            {"observations": observation_rows, "targets": target_rows}
        ),
    }


def run_oracle_trajectory_case(
    job: tuple[HeadroomCase, np.ndarray, np.ndarray, OracleConfig],
) -> dict[str, Any]:
    """Run the registered oracle and attach its public-state imitation pairs."""

    oracle = run_oracle_case(job)
    case = job[0]
    actions = np.asarray(oracle["actions"], dtype=np.float64)
    replay = rollout_actions(case, actions, collect_evidence=True)
    compact = _result_from_receipt(oracle["result"])
    _assert_result_equivalent(compact, replay, f"{case.row_id} oracle dataset")
    return {
        **oracle,
        "demonstration": collect_public_demonstration(case, actions),
    }


def _run_missing_parallel(
    *,
    root: Path,
    contract_sha256: str,
    phase: str,
    cases: Sequence[HeadroomCase],
    jobs: Sequence[Any],
    worker: Callable[[Any], dict[str, Any]],
    workers: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any] | None] = [None] * len(cases)
    missing: list[int] = []
    for index, case in enumerate(cases):
        path = _record_path(root, phase, index)
        if path.exists():
            results[index] = _load_record(
                path,
                contract_sha256=contract_sha256,
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
                raise TrainingOracleError(
                    f"worker returned wrong row for {case.row_id}"
                )
            _atomic_create_json(
                _record_path(root, phase, index),
                _wrap_record(
                    contract_sha256=contract_sha256,
                    phase=phase,
                    index=index,
                    case=case,
                    payload=payload,
                ),
            )
            results[index] = payload
            completed_count += 1
            print(
                f"train {phase}: {completed_count}/{len(cases)} {case.row_id}",
                flush=True,
            )
    if any(value is None for value in results):
        raise TrainingOracleError(f"training {phase} did not produce every case")
    return [value for value in results if value is not None]


def _parallel_with_fallback(
    *,
    root: Path,
    contract_sha256: str,
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
            phase=phase,
            cases=cases,
            jobs=jobs,
            worker=worker,
            workers=workers,
        )
        runtime_events.append(
            {
                "phase": phase,
                "workers": workers,
                "fallback": False,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        )
        return values
    except BrokenProcessPool as exc:
        if workers < DEFAULT_WORKERS:
            raise
        print(
            f"train {phase}: pool failed at {workers}; resuming at "
            f"{FALLBACK_WORKERS} workers",
            file=sys.stderr,
            flush=True,
        )
        runtime_events.append(
            {
                "phase": phase,
                "workers": workers,
                "fallback": True,
                "fallback_workers": FALLBACK_WORKERS,
                "reason": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds_before_fallback": round(
                    time.perf_counter() - started, 3
                ),
            }
        )
        return _run_missing_parallel(
            root=root,
            contract_sha256=contract_sha256,
            phase=phase,
            cases=cases,
            jobs=jobs,
            worker=worker,
            workers=FALLBACK_WORKERS,
        )


def _select_mpc(
    preparations: Sequence[dict[str, Any]],
    cases: Sequence[HeadroomCase],
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


def _build_receipt(
    *,
    root: Path,
    contract: dict[str, Any],
    contract_sha256: str,
    cases: Sequence[HeadroomCase],
    oracle_values: Sequence[dict[str, Any]],
    selected_horizon: int,
    mpc_selection: dict[str, Any],
    workers: int,
    runtime_events: Sequence[dict[str, Any]],
    elapsed_seconds: float,
) -> dict[str, Any]:
    results: list[PlannerResult] = []
    rows: list[dict[str, Any]] = []
    total_candidate_evaluations = 0
    total_simulated_transitions = 0
    for index, (case, oracle) in enumerate(
        zip(cases, oracle_values, strict=True)
    ):
        result = _result_from_receipt(oracle["result"])
        demonstration = oracle["demonstration"]
        observations = np.asarray(demonstration.get("observations"), dtype=np.float32)
        targets = np.asarray(demonstration.get("targets"), dtype=np.float32)
        if (
            demonstration.get("observation_shape") != [HORIZON_DAYS, OBSERVATION_SIZE]
            or demonstration.get("target_shape") != [HORIZON_DAYS, ACTION_SIZE]
            or observations.shape != (HORIZON_DAYS, OBSERVATION_SIZE)
            or targets.shape != (HORIZON_DAYS, ACTION_SIZE)
            or not np.all(np.isfinite(observations))
            or not np.all(np.isfinite(targets))
            or np.any(np.abs(targets) > 1.0)
            or demonstration.get("observations_sha256")
            != canonical_hash(observations.tolist())
            or demonstration.get("targets_sha256")
            != canonical_hash(targets.tolist())
            or demonstration.get("dataset_sha256")
            != canonical_hash(
                {
                    "observations": observations.tolist(),
                    "targets": targets.tolist(),
                }
            )
            or not np.array_equal(
                targets,
                np.asarray(oracle.get("actions"), dtype=np.float64).astype(
                    np.float32
                ),
            )
        ):
            raise TrainingOracleError(
                f"demonstration contract drifted: {case.row_id}"
            )
        search = oracle["search_wide_invariants"]
        if (
            int(search["maximum_hard_violation_count"]) != 0
            or float(search["maximum_conservation_residual"]) != 0.0
            or result.hard_violation_count != 0
            or result.maximum_conservation_residual != 0.0
        ):
            raise TrainingOracleError(f"oracle invariant failed: {case.row_id}")
        results.append(result)
        budget = oracle["budget"]
        total_candidate_evaluations += int(budget["candidate_evaluations"])
        total_simulated_transitions += int(budget["simulated_transitions"])
        shard = _record_path(root, "oracle", index)
        rows.append(
            {
                **_case_contract(case),
                "shard": str(shard.relative_to(root)).replace("\\", "/"),
                "shard_sha256": file_sha256(shard),
                "observations_sha256": demonstration["observations_sha256"],
                "targets_sha256": demonstration["targets_sha256"],
                "dataset_sha256": demonstration["dataset_sha256"],
                "oracle_result": result.as_receipt(),
                "oracle_budget": budget,
            }
        )
    aggregate = aggregate_results(results)
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL_ID,
        "status": "complete_training_oracle_demonstrations",
        "created_at_utc": _utc_now(),
        "contract_sha256": contract_sha256,
        "split": contract["split"],
        "access_contract": contract["access_contract"],
        "case_count": len(rows),
        "demonstration_row_count": len(rows) * HORIZON_DAYS,
        "observation_count": OBSERVATION_SIZE,
        "action_count": ACTION_SIZE,
        "privileged_teacher": True,
        "causal_student_inputs_only": True,
        "student_trained": False,
        "mpc_warm_start": {
            "config": asdict(REGISTERED_MPC_CONFIG),
            "selection": mpc_selection,
        },
        "oracle": {
            "config": asdict(REGISTERED_ORACLE_CONFIG),
            "aggregate": aggregate,
            "selected_mpc_horizon": selected_horizon,
            "total_candidate_evaluations": total_candidate_evaluations,
            "total_simulated_transitions": total_simulated_transitions,
        },
        "future_student_contract": contract["future_student_contract"],
        "rows": rows,
        "rows_sha256": canonical_hash(rows),
        "dataset_index_sha256": canonical_hash(
            [
                {
                    "row_id": row["row_id"],
                    "dataset_sha256": row["dataset_sha256"],
                }
                for row in rows
            ]
        ),
        "invariants": {
            "case_count_exactly_192": len(rows) == TRAINING_CASE_COUNT,
            "demonstration_rows_exactly_5760": (
                len(rows) * HORIZON_DAYS == TRAINING_OBSERVATION_COUNT
            ),
            "row_ids_unique": len({row["row_id"] for row in rows})
            == TRAINING_CASE_COUNT,
            "observation_dimension_exactly_73": OBSERVATION_SIZE == 73,
            "action_dimension_exactly_22": ACTION_SIZE == 22,
            "all_hard_violation_counts_zero": True,
            "all_conservation_residuals_exactly_zero": True,
            "development_split_used": False,
            "final_split_used": False,
            "learned_policy_loaded_or_run": False,
        },
        "runtime": {
            "output_root": str(root),
            "requested_workers": workers,
            "multiprocessing_start_method": "spawn",
            "fallback_workers_after_broken_pool": FALLBACK_WORKERS,
            "events": list(runtime_events),
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


def run_study(
    *,
    root: Path,
    contract: dict[str, Any],
    contract_sha256: str,
    workers: int,
) -> dict[str, Any]:
    """Run or resume both phases and create the external dataset receipt."""

    started = time.perf_counter()
    cases = build_training_cases()
    runtime_events: list[dict[str, Any]] = []
    preparations = _parallel_with_fallback(
        root=root,
        contract_sha256=contract_sha256,
        phase="preparation",
        cases=cases,
        jobs=[(case, REGISTERED_MPC_CONFIG) for case in cases],
        worker=run_preparation_case,
        workers=workers,
        runtime_events=runtime_events,
    )
    selected_horizon, mpc_selection = _select_mpc(preparations, cases)
    print(f"train: globally selected MPC k={selected_horizon}", flush=True)
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
            REGISTERED_ORACLE_CONFIG,
        )
        for index, case in enumerate(cases)
    ]
    oracle_values = _parallel_with_fallback(
        root=root,
        contract_sha256=contract_sha256,
        phase="oracle",
        cases=cases,
        jobs=oracle_jobs,
        worker=run_oracle_trajectory_case,
        workers=workers,
        runtime_events=runtime_events,
    )
    receipt = _build_receipt(
        root=root,
        contract=contract,
        contract_sha256=contract_sha256,
        cases=cases,
        oracle_values=oracle_values,
        selected_horizon=selected_horizon,
        mpc_selection=mpc_selection,
        workers=workers,
        runtime_events=runtime_events,
        elapsed_seconds=time.perf_counter() - started,
    )
    receipt_path = root / "training" / "receipt.json"
    if receipt_path.exists():
        existing = _load_json(receipt_path, "training oracle receipt")
        if existing.get("rows_sha256") != receipt["rows_sha256"]:
            raise TrainingOracleError("existing receipt does not match shards")
        return existing
    _atomic_create_json(receipt_path, receipt)
    return receipt


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--resume", action="store_true")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> Path:
    if not args.output_root.is_absolute():
        raise TrainingOracleError("--output-root must be absolute")
    root = args.output_root.resolve()
    try:
        root.relative_to(ROOT)
    except ValueError:
        pass
    else:
        raise TrainingOracleError("--output-root must be outside the repository")
    if root == Path(root.anchor):
        raise TrainingOracleError("--output-root cannot be a filesystem root")
    if not 1 <= args.workers <= 16:
        raise TrainingOracleError("--workers must be in [1, 16]")
    if args.resume and not args.execute:
        raise TrainingOracleError("--resume is valid only with --execute")
    if args.execute:
        if root.exists() and not args.resume:
            raise TrainingOracleError(
                "output root exists; use --resume only for this exact study"
            )
        if not root.exists() and args.resume:
            raise TrainingOracleError("cannot resume a missing output root")
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    root = _validate_args(args)
    contract = study_contract()
    contract_sha256 = canonical_hash(contract)
    preflight = {
        "tool": TOOL_ID,
        "status": "ready" if _worktree_is_clean() else "blocked_dirty_worktree",
        "compute_started": False,
        "filesystem_written": False,
        "output_root": str(root),
        "workers": args.workers,
        "case_count": TRAINING_CASE_COUNT,
        "demonstration_row_count": TRAINING_OBSERVATION_COUNT,
        "contract_sha256": contract_sha256,
        "training_split_only": True,
        "development_split_used": False,
        "final_split_used": False,
    }
    if args.preflight:
        print(json.dumps(preflight, indent=2, sort_keys=True), flush=True)
        return 0 if preflight["status"] == "ready" else 3
    if not _worktree_is_clean():
        raise TrainingOracleError("refusing expensive compute from a dirty worktree")

    protocol_path = root / "protocol.json"
    if root.exists():
        protocol = _load_json(protocol_path, "training oracle protocol")
        if (
            protocol.get("contract_sha256") != contract_sha256
            or protocol.get("contract") != contract
        ):
            raise TrainingOracleError("resume contract differs from original study")
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
    receipt = run_study(
        root=root,
        contract=contract,
        contract_sha256=contract_sha256,
        workers=args.workers,
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "receipt": str(root / "training" / "receipt.json"),
                "case_count": receipt["case_count"],
                "demonstration_row_count": receipt["demonstration_row_count"],
                "oracle_solved_count": receipt["oracle"]["aggregate"][
                    "solved_count"
                ],
                "dataset_index_sha256": receipt["dataset_index_sha256"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        raise SystemExit(main())
    except (HeadroomError, TrainingOracleError) as error:
        print(f"training oracle trajectory run failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
