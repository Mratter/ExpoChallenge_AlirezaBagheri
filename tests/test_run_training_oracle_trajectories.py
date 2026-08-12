from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from backend.app.shared_evidence import canonical_hash, file_sha256
from scripts.headroom import tuned_rollout
from scripts.run_training_oracle_trajectories import (
    ACTION_SIZE,
    HORIZON_DAYS,
    OBSERVATION_SIZE,
    REGISTERED_MPC_CONFIG,
    REGISTERED_ORACLE_CONFIG,
    TRAINING_CASE_COUNT,
    TRAINING_OBSERVATION_COUNT,
    TrainingOracleError,
    _atomic_create_json,
    _load_record,
    _run_missing_parallel,
    _validate_args,
    _wrap_record,
    build_training_cases,
    collect_public_demonstration,
    main,
    study_contract,
)


def test_training_roster_is_exactly_six_by_thirty_two() -> None:
    cases = build_training_cases()

    assert len(cases) == TRAINING_CASE_COUNT == 192
    assert len({case.row_id for case in cases}) == 192
    assert len({case.family_id for case in cases}) == 6
    assert [case.case_seed for case in cases[:32]] == list(
        range(810000, 810032)
    )
    assert all(case.row_id.startswith("v3_train_") for case in cases)
    assert all(case.scenario.horizon_days == HORIZON_DAYS for case in cases)


def test_contract_pins_registered_budget_and_causal_student_boundary() -> None:
    contract = study_contract()

    assert contract["split"]["id"] == "train"
    assert contract["split"]["cartesian_case_count"] == 192
    assert len(contract["ordered_case_contract_sha256"]) == 64
    assert contract["access_contract"] == {
        "training_split_used": True,
        "development_split_used": False,
        "final_split_used": False,
        "learned_policy_loaded_or_run": False,
    }
    assert contract["mpc_config"] == REGISTERED_MPC_CONFIG.__dict__
    assert contract["oracle_config"] == REGISTERED_ORACLE_CONFIG.__dict__
    assert contract["oracle_config"] == {
        "population": 512,
        "elite_fraction": 0.10,
        "min_iterations": 20,
        "max_iterations": 40,
        "patience": 6,
        "initial_std": 0.25,
        "std_floor": 0.03,
        "smoothing": 0.75,
    }
    demonstration = contract["demonstration_contract"]
    assert demonstration["row_count"] == TRAINING_OBSERVATION_COUNT == 5760
    assert demonstration["observation_count"] == OBSERVATION_SIZE == 73
    assert demonstration["action_count"] == ACTION_SIZE == 22
    assert demonstration["student_input_future_tape_visible"] is False
    assert demonstration["teacher_target_uses_full_future_tape"] is True
    assert contract["future_student_contract"]["training_method"] == (
        "behavior_cloning_only"
    )
    assert contract["future_student_contract"]["actor_architecture"] == [
        384,
        256,
        128,
    ]
    assert set(
        contract["future_student_contract"]["implementation_references"]
    ) == {"build_model", "normalize_observations", "behavior_clone_policy"}


def test_registered_budget_matches_preserved_historical_receipt() -> None:
    root = Path(__file__).parents[1]
    receipt_path = (
        root
        / "internal"
        / "developmental_runs"
        / "v4"
        / "headroom-probe-v4-dev.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

    assert file_sha256(receipt_path) == (
        "f037c98d8fec483dfa6b5c9c1691842597a4163c7d1ee6f3e72618f987d671b9"
    )
    assert receipt["mpc"]["config"] == REGISTERED_MPC_CONFIG.__dict__
    assert receipt["oracle"]["config"] == REGISTERED_ORACLE_CONFIG.__dict__
    assert receipt["oracle"]["solved_count"] == 37


def test_source_has_no_nontraining_roster_or_learned_policy_access() -> None:
    source = (
        Path(__file__).parents[1]
        / "scripts"
        / "run_training_oracle_trajectories.py"
    ).read_text(encoding="utf-8")

    forbidden = (
        "FINAL_FAMILIES",
        "FINAL_SEEDS",
        "DEVELOPMENT_FAMILIES",
        "DEVELOPMENT_SEEDS",
        "scripts.evaluate",
        "model.policy",
    )
    assert not any(token in source for token in forbidden)
    assert 'split_contract("train", TRAINING_FAMILIES, TRAINING_SEEDS)' in source
    assert '"final_split_used": False' in source


def test_public_demonstration_has_exact_shapes_and_stable_hashes() -> None:
    case = build_training_cases()[0]
    _, actions = tuned_rollout(case)

    demonstration = collect_public_demonstration(case, actions)

    assert demonstration["observation_shape"] == [30, 73]
    assert demonstration["target_shape"] == [30, 22]
    assert demonstration["student_input_future_tape_visible"] is False
    assert demonstration["teacher_target_uses_full_future_tape"] is True
    assert demonstration["observations_sha256"] == canonical_hash(
        demonstration["observations"]
    )
    assert demonstration["targets_sha256"] == canonical_hash(
        demonstration["targets"]
    )
    assert demonstration["dataset_sha256"] == canonical_hash(
        {
            "observations": demonstration["observations"],
            "targets": demonstration["targets"],
        }
    )
    with pytest.raises(TrainingOracleError, match="action sequence must have shape"):
        collect_public_demonstration(case, np.zeros((1, ACTION_SIZE)))


def test_resume_loads_contract_bound_shards_without_starting_pool(
    tmp_path: Path,
) -> None:
    cases = build_training_cases()[:2]
    contract_sha256 = "c" * 64
    expected: list[dict[str, str]] = []
    for index, case in enumerate(cases):
        payload = {"row_id": case.row_id, "value": str(index)}
        expected.append(payload)
        path = tmp_path / "training" / "oracle" / f"{index:03d}.json"
        _atomic_create_json(
            path,
            _wrap_record(
                contract_sha256=contract_sha256,
                phase="oracle",
                index=index,
                case=case,
                payload=payload,
            ),
        )
        assert _load_record(
            path,
            contract_sha256=contract_sha256,
            phase="oracle",
            index=index,
            case=case,
        ) == payload

    def forbidden_worker(_: object) -> dict[str, str]:
        raise AssertionError("resume should not submit completed shards")

    resumed = _run_missing_parallel(
        root=tmp_path,
        contract_sha256=contract_sha256,
        phase="oracle",
        cases=cases,
        jobs=[None, None],
        worker=forbidden_worker,
        workers=8,
    )

    assert resumed == expected


def test_execution_requires_absolute_external_root_and_explicit_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relative = argparse.Namespace(
        output_root=Path("relative"),
        workers=8,
        resume=False,
        preflight=True,
        execute=False,
    )
    with pytest.raises(TrainingOracleError, match="must be absolute"):
        _validate_args(relative)

    in_repo = argparse.Namespace(
        output_root=Path(__file__).parents[1] / "forbidden-output",
        workers=8,
        resume=False,
        preflight=True,
        execute=False,
    )
    with pytest.raises(TrainingOracleError, match="outside the repository"):
        _validate_args(in_repo)

    output = tmp_path / "preflight-does-not-write"
    monkeypatch.setattr(
        "scripts.run_training_oracle_trajectories._worktree_is_clean",
        lambda: True,
    )
    assert main(["--output-root", str(output), "--preflight"]) == 0
    assert not output.exists()
