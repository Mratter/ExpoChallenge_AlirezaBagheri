from __future__ import annotations

import ast
import copy
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

import scripts.headroom as headroom
from backend.app.city.environment import CityRecoveryEnv
from backend.app.city.scenarios import DEVELOPMENT_FAMILIES, DEVELOPMENT_SEEDS
from scripts.headroom import (
    HeadroomError,
    MPCConfig,
    ORIGINAL_DEVELOPMENT_SUBSET_CASE_COUNT,
    ORIGINAL_DEVELOPMENT_SUBSET_ID,
    ORIGINAL_DEVELOPMENT_SUBSET_SEEDS,
    PlannerResult,
    _antithetic_samples,
    build_development_cases,
    capture_public_snapshot,
    classify_case,
    lexicographic_key,
    load_prior_evidence,
    original_development_subset_contract,
    plan_mpc_action,
    rollout_actions,
    select_best_mpc_k,
    tuned_rollout,
    write_new_receipt,
)


def _result(
    solved: bool,
    margin: float = 0.0,
    resilience_auc: float = 0.5,
) -> PlannerResult:
    return PlannerResult(
        solved=solved,
        minimum_tail_margin=margin,
        resilience_auc=resilience_auc,
        reason_codes=() if solved else ("assessment_tail_targets_met",),
        hard_violation_count=0,
        maximum_conservation_residual=0.0,
        action_sequence_sha256=None,
    )


def test_build_development_cases_is_the_ordered_40_case_split() -> None:
    cases = build_development_cases()

    assert ORIGINAL_DEVELOPMENT_SUBSET_ID == "original_40_case_development_subset"
    assert ORIGINAL_DEVELOPMENT_SUBSET_SEEDS == tuple(range(820000, 820008))
    assert ORIGINAL_DEVELOPMENT_SUBSET_CASE_COUNT == 40
    assert DEVELOPMENT_SEEDS == tuple(range(820000, 820040))
    assert set(ORIGINAL_DEVELOPMENT_SUBSET_SEEDS) < set(DEVELOPMENT_SEEDS)
    assert len(cases) == ORIGINAL_DEVELOPMENT_SUBSET_CASE_COUNT
    assert len({case.row_id for case in cases}) == (
        ORIGINAL_DEVELOPMENT_SUBSET_CASE_COUNT
    )
    assert [case.case_seed for case in cases[:8]] == list(range(820000, 820008))
    assert all(case.row_id == f"{case.family_id}:{case.case_seed}" for case in cases)
    assert all(case.tape_seed != case.case_seed for case in cases)
    assert all(len(case.schedule) == case.scenario.horizon_days == 30 for case in cases)


def test_original_subset_contract_never_claims_the_200_case_ceiling() -> None:
    assert original_development_subset_contract() == {
        "id": "original_40_case_development_subset",
        "case_count": 40,
        "family_count": 5,
        "seed_interval": {"first": 820000, "last": 820007, "count": 8},
        "canonical_200_case_development_ceiling_claimed": False,
        "interpretation": (
            "Privileged oracle and headroom results apply only to the original "
            "40-case development subset, not the canonical 200-case development "
            "split."
        ),
    }


def test_lexicographic_key_prioritizes_solved_then_margin_then_auc() -> None:
    candidates = [
        _result(False, margin=10.0, resilience_auc=1.0),
        _result(True, margin=-0.02, resilience_auc=0.9),
        _result(True, margin=0.01, resilience_auc=0.4),
        _result(True, margin=0.01, resilience_auc=0.6),
    ]

    ordered = sorted(candidates, key=lexicographic_key)

    assert ordered == [candidates[0], candidates[1], candidates[2], candidates[3]]
    assert max(candidates, key=lexicographic_key) is candidates[3]


def test_antithetic_samples_are_bounded_and_paired() -> None:
    samples = _antithetic_samples(
        np.random.Generator(np.random.PCG64(11)),
        np.zeros((2, 3), dtype=np.float64),
        np.ones((2, 3), dtype=np.float64),
        population=4,
    )

    assert samples.shape == (4, 2, 3)
    assert np.max(np.abs(samples)) <= 1.0
    np.testing.assert_array_equal(samples[0], -samples[2])
    np.testing.assert_array_equal(samples[1], -samples[3])


def test_compact_rollout_and_full_evidence_replay_match_exactly() -> None:
    case = build_development_cases()[0]
    compact, actions = tuned_rollout(case)

    evidence = rollout_actions(case, actions, collect_evidence=True)

    assert compact.solved == evidence.solved
    assert compact.minimum_tail_margin == evidence.minimum_tail_margin
    assert compact.resilience_auc == evidence.resilience_auc
    assert compact.reason_codes == evidence.reason_codes
    assert compact.hard_violation_count == evidence.hard_violation_count == 0
    assert (
        compact.maximum_conservation_residual
        == evidence.maximum_conservation_residual
        == 0.0
    )
    assert compact.action_sequence_sha256 == evidence.action_sequence_sha256
    assert evidence.trajectory_sha256 is not None


def test_mpc_first_action_is_invariant_to_unseen_future_tape() -> None:
    case = build_development_cases()[0]
    original = CityRecoveryEnv(
        case.scenario,
        case.tape_seed,
        list(case.schedule),
        collect_evidence=False,
    )
    observation, _ = original.reset(seed=case.tape_seed)
    mutated_schedule = list(case.schedule)
    mutated_schedule[1:] = list(reversed(copy.deepcopy(mutated_schedule[1:])))
    mutated = CityRecoveryEnv(
        case.scenario,
        case.tape_seed,
        mutated_schedule,
        collect_evidence=False,
    )
    mutated_observation, _ = mutated.reset(seed=case.tape_seed)
    np.testing.assert_array_equal(observation, mutated_observation)
    config = MPCConfig(
        population=4,
        elite_count=1,
        iterations=1,
        fantasies=2,
        initial_std=0.20,
        std_floor=0.05,
        smoothing=0.80,
    )

    action, plan, _ = plan_mpc_action(
        capture_public_snapshot(original),
        observation,
        horizon=3,
        previous_plan=None,
        config=config,
    )
    mutated_action, mutated_plan, _ = plan_mpc_action(
        capture_public_snapshot(mutated),
        mutated_observation,
        horizon=3,
        previous_plan=None,
        config=config,
    )

    np.testing.assert_array_equal(action, mutated_action)
    np.testing.assert_array_equal(plan, mutated_plan)


def _prior_row(row_id: str, *, solved: bool) -> dict[str, object]:
    return {
        "row_id": row_id,
        "solved": solved,
        "recovery_targets": [0.5] * 5,
        "tail_minimum_services": [0.6 if solved else 0.4] * 5,
        "resilience_auc": 0.5,
        "reason_codes": [] if solved else ["assessment_tail_targets_met"],
        "hard_violation_count": 0,
        "max_conservation_residual": 0.0,
    }


def _write_prior_receipt(path: Path, row_ids: list[str]) -> None:
    receipt = {
        "authorizing": False,
        "split": "dev",
        "final_split_used": False,
        "selects_or_exports_policy": False,
        "config": {"policy_seed": 73},
        "profiles": {
            "published_reward": {
                "development_curve": {
                    "bc_initialization": {
                        "active_actor_critic_transitions": 0,
                        "rows": [
                            _prior_row(row_id, solved=False) for row_id in row_ids
                        ],
                    },
                    "actor_critic_100_transitions": {
                        "active_actor_critic_transitions": 100,
                        "rows": [
                            _prior_row(row_id, solved=False) for row_id in row_ids
                        ],
                    },
                    "actor_critic_200_transitions": {
                        "active_actor_critic_transitions": 200,
                        "rows": [
                            _prior_row(row_id, solved=True) for row_id in row_ids
                        ],
                    },
                }
            }
        },
    }
    path.write_text(json.dumps(receipt), encoding="utf-8")


def test_load_prior_evidence_accepts_one_receipt_and_uses_latest_stage(
    tmp_path: Path,
) -> None:
    row_ids = ["case-a", "case-b"]
    receipt_path = tmp_path / "prior.json"
    _write_prior_receipt(receipt_path, row_ids)

    evidence = load_prior_evidence(
        receipt_path, row_ids, expected_policy_seed=73
    )

    assert evidence["actor_critic_stage"] == "actor_critic_200_transitions"
    assert evidence["active_actor_critic_transitions"] == 200
    assert evidence["treatment_profile"] == "published_reward"
    assert list(evidence["bc"]) == row_ids
    assert list(evidence["ppo"]) == row_ids
    assert not any(result.solved for result in evidence["bc"].values())
    assert all(result.solved for result in evidence["ppo"].values())


def test_load_prior_evidence_binds_policy_seed(tmp_path: Path) -> None:
    receipt_path = tmp_path / "prior.json"
    _write_prior_receipt(receipt_path, ["case-a"])

    with pytest.raises(HeadroomError, match="policy seed does not match"):
        load_prior_evidence(
            receipt_path, ["case-a"], expected_policy_seed=74
        )


def test_load_prior_evidence_selects_the_original_subset_from_200_rows(
    tmp_path: Path,
) -> None:
    full_row_ids = [
        f"{family.id}:{seed}"
        for family in DEVELOPMENT_FAMILIES
        for seed in DEVELOPMENT_SEEDS
    ]
    expected_row_ids = [case.row_id for case in build_development_cases()]
    receipt_path = tmp_path / "prior-200.json"
    _write_prior_receipt(receipt_path, full_row_ids)

    evidence = load_prior_evidence(
        receipt_path,
        expected_row_ids,
        expected_policy_seed=73,
    )

    assert len(full_row_ids) == evidence["source_development_row_count"] == 200
    assert evidence["analysis_subset"] == "original_40_case_development_subset"
    assert list(evidence["bc"]) == expected_row_ids
    assert list(evidence["ppo"]) == expected_row_ids


def test_select_best_mpc_k_uses_one_global_lexicographic_winner() -> None:
    more_solves = {
        1: {"a": _result(True, -0.2, 0.4), "b": _result(True, -0.2, 0.4)},
        3: {"a": _result(True, 1.0, 1.0), "b": _result(False, 1.0, 1.0)},
        5: {"a": _result(False, 2.0, 1.0), "b": _result(True, 2.0, 1.0)},
    }
    assert select_best_mpc_k(more_solves)[0] == 1

    better_margin = {
        1: {"a": _result(True, 0.1, 0.9), "b": _result(False, 0.1, 0.9)},
        3: {"a": _result(False, 0.2, 0.1), "b": _result(True, 0.2, 0.1)},
        5: {"a": _result(True, 0.15, 1.0), "b": _result(False, 0.15, 1.0)},
    }
    assert select_best_mpc_k(better_margin)[0] == 3

    better_auc = {
        1: {"a": _result(True, 0.2, 0.5), "b": _result(False, 0.2, 0.5)},
        3: {"a": _result(False, 0.2, 0.6), "b": _result(True, 0.2, 0.6)},
        5: {"a": _result(True, 0.2, 0.7), "b": _result(False, 0.2, 0.7)},
    }
    assert select_best_mpc_k(better_auc)[0] == 5

    exact_tie = {horizon: {"a": _result(True, 0.2, 0.7)} for horizon in (1, 3, 5)}
    selected, evidence = select_best_mpc_k(exact_tie)
    assert selected == 1
    assert evidence["selected_horizon"] == 1
    assert set(evidence["aggregates"]) == {"1", "3", "5"}


def test_classify_case_has_explicit_residual_and_exhaustive_partitions() -> None:
    names = ("tuned_rule", "bc_initialization", "best_ppo", "mpc", "oracle")
    classifications = []
    for solved_values in itertools.product((False, True), repeat=len(names)):
        planners = {
            name: _result(solved)
            for name, solved in zip(names, solved_values, strict=True)
        }
        classifications.append(classify_case(planners))

    assert {item["literal_taxonomy"] for item in classifications} == {
        "saturated",
        "contested",
        "oracle_search_unsolved",
        "achieved_nonunanimous",
    }
    assert {item["decision_partition"] for item in classifications} == {
        "ppo_solved",
        "contested",
        "known_achievable_oracle_search_miss",
        "oracle_search_unsolved",
    }
    assert all(
        set(item) == {"literal_taxonomy", "decision_partition"}
        for item in classifications
    )

    residual = classify_case(
        {
            "tuned_rule": _result(False),
            "bc_initialization": _result(False),
            "best_ppo": _result(True),
            "mpc": _result(False),
            "oracle": _result(False),
        }
    )
    assert residual == {
        "literal_taxonomy": "achieved_nonunanimous",
        "decision_partition": "ppo_solved",
    }

    without_prior = classify_case(
        {
            "tuned_rule": _result(False),
            "mpc": _result(False),
            "oracle": _result(True),
        }
    )
    assert without_prior == {
        "literal_taxonomy": "contested",
        "decision_partition": "contested",
    }


def test_write_new_receipt_is_atomic_create_new_and_refuses_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(headroom, "ROOT", tmp_path)
    target = tmp_path / "internal" / "developmental_runs" / "probe.json"
    original = {"status": "first", "rows": [1, 2, 3]}

    write_new_receipt(target, original)

    first_bytes = target.read_bytes()
    assert json.loads(first_bytes) == original
    assert first_bytes.endswith(b"\n")
    assert not list(target.parent.glob("*.tmp"))

    with pytest.raises(HeadroomError, match="refusing to overwrite"):
        write_new_receipt(target, {"status": "replacement"})
    assert target.read_bytes() == first_bytes


def test_headroom_does_not_import_final_split_or_legacy_simulators() -> None:
    source_path = Path(headroom.__file__)
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }

    assert not {name for name in imported_names if name.startswith("FINAL_")}
    assert "DEVELOPMENT_SEEDS" not in source
    assert "FINAL_" not in source
    assert "simulator_v" not in source
    assert "scenarios_v" not in source
