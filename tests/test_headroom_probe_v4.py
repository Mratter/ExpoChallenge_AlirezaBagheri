from __future__ import annotations

import ast
import itertools
import json
import copy
from pathlib import Path

import numpy as np
import pytest

import scripts.headroom_probe_v4 as headroom
from scripts.headroom_probe_v4 import (
    DEFAULT_PRIOR_SUMMARY,
    HeadroomError,
    PlannerResult,
    MPCConfig,
    build_development_cases,
    capture_public_snapshot,
    classify_case,
    lexicographic_key,
    plan_mpc_action,
    rollout_actions,
    select_best_mpc_k,
    select_prior_evidence,
    tuned_rollout,
    write_new_receipt,
)
from backend.app.simulator_v4 import CityRecoveryEnvV4


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


def test_build_development_cases_is_the_frozen_ordered_40_case_split() -> None:
    cases = build_development_cases()

    assert len(cases) == 40
    assert len({case.row_id for case in cases}) == 40
    assert [case.case_seed for case in cases[:8]] == list(range(820000, 820008))
    assert all(case.family_id.startswith("v3_dev_") for case in cases)
    assert all(case.row_id == f"{case.family_id}:{case.case_seed}" for case in cases)
    assert all(case.tape_seed != case.case_seed for case in cases)
    assert all(len(case.schedule) == case.scenario.horizon_days == 30 for case in cases)


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
    original = CityRecoveryEnvV4(
        case.scenario,
        case.tape_seed,
        list(case.schedule),
        collect_evidence=False,
        reward_profile="v3_equivalent",
    )
    observation, _ = original.reset(seed=case.tape_seed)
    mutated_schedule = list(case.schedule)
    mutated_schedule[1:] = list(reversed(copy.deepcopy(mutated_schedule[1:])))
    mutated = CityRecoveryEnvV4(
        case.scenario,
        case.tape_seed,
        mutated_schedule,
        collect_evidence=False,
        reward_profile="v3_equivalent",
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


def test_select_prior_evidence_hash_validates_and_selects_attempt_09() -> None:
    row_ids = [case.row_id for case in build_development_cases()]

    evidence = select_prior_evidence(DEFAULT_PRIOR_SUMMARY, row_ids)

    assert evidence["attempt"] == 9
    assert evidence["receipt_path"].endswith(
        "ppo-learning-gate-200k-seed-37017-attempt-09.json"
    )
    assert evidence["checkpoint_available"] is False
    assert evidence["selected_or_exported_policy"] is False
    assert list(evidence["bc"]) == row_ids
    assert list(evidence["ppo"]) == row_ids
    assert sum(result.solved for result in evidence["bc"].values()) == 32
    assert sum(result.solved for result in evidence["ppo"].values()) == 33


def test_select_prior_evidence_binds_policy_seed() -> None:
    row_ids = [case.row_id for case in build_development_cases()]

    with pytest.raises(HeadroomError, match="not valid dev-only evidence"):
        select_prior_evidence(
            DEFAULT_PRIOR_SUMMARY, row_ids, expected_policy_seed=37018
        )


def test_select_prior_evidence_refuses_a_receipt_hash_mismatch(
    tmp_path: Path,
) -> None:
    receipt_path = tmp_path / "attempt-09.json"
    receipt_path.write_text("{}\n", encoding="utf-8")
    summary = {
        "authorizing": False,
        "split": "dev",
        "final_split_used": False,
        "reward_profile": "v3_equivalent",
        "policy_seed": 37017,
        "attempts": [
            {
                "attempt": 9,
                "receipt": receipt_path.name,
                "receipt_sha256": "0" * 64,
                "solved_curve": [32, 33],
                "final_mean_resilience_auc": 0.489036433,
            }
        ],
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(HeadroomError, match="prior receipt hash mismatch"):
        select_prior_evidence(summary_path, [])


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


def test_write_new_receipt_is_atomic_create_new_and_refuses_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(headroom, "ROOT", tmp_path)
    target = tmp_path / "internal" / "developmental_runs" / "v4" / "probe.json"
    original = {"status": "first", "rows": [1, 2, 3]}

    write_new_receipt(target, original)

    first_bytes = target.read_bytes()
    assert json.loads(first_bytes) == original
    assert first_bytes.endswith(b"\n")
    assert not list(target.parent.glob("*.tmp"))

    with pytest.raises(HeadroomError, match="refusing to overwrite"):
        write_new_receipt(target, {"status": "replacement"})
    assert target.read_bytes() == first_bytes


def test_headroom_probe_does_not_import_final_split_symbols() -> None:
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
    assert "FINAL_" not in source
