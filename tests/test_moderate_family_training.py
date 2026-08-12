from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import pytest

from backend.app.city.scenarios import (
    DEVELOPMENT_FAMILIES,
    DEVELOPMENT_SEEDS,
    TRAINING_FAMILIES,
)
from scripts.moderate_family_training import (
    ACTIVE_TRANSITIONS,
    CRITIC_WARMUP_PREFIX_TRANSITIONS,
    CURRENT_SELECTED_SOLVED_COUNT,
    CURVE_MILESTONES,
    INCUMBENT_SEED_MEAN_SOLVED_COUNT,
    POLICY_SEEDS,
    PROMOTION_ENDPOINT_MINIMUM,
    PROMOTION_SELECTED_MINIMUM,
    SHIPPED_ARTIFACT_SHA256,
    SOURCE_PATHS,
    TRAINING_LANES,
    WEIGHTED_FAMILY_SLOT_COUNT,
    ModerateStudyError,
    TrainingCase,
    adopted_training_arguments,
    build_difficulty_receipt,
    build_training_cases,
    development_family_aggregate,
    family_weights,
    parse_args,
    promotion_gate,
    rank_training_families,
    sampler_contract,
    select_development_checkpoint,
    summarize_seed_endpoints,
    validate_difficulty_receipt,
    warmup_prefix_balance,
    weighted_training_cases,
    write_new_json,
)


def _evaluator(counts: dict[str, int]) -> Callable[[TrainingCase], dict[str, Any]]:
    offsets = {family.id: 810000 for family in TRAINING_FAMILIES}

    def evaluate(case: TrainingCase) -> dict[str, Any]:
        solved = case.case_seed - offsets[case.family_id] < counts[case.family_id]
        return {
            **case.identity(),
            "solved": solved,
            "reason_codes": [] if solved else ["assessment_tail_targets_met"],
            "resilience_auc": 0.5,
            "minimum_tail_margin": 0.01 if solved else -0.01,
            "hard_violation_count": 0,
            "max_conservation_residual": 0.0,
            "trajectory_sha256": f"{case.case_seed:064x}",
        }

    return evaluate


def test_training_roster_is_exact_authored_6_by_32() -> None:
    cases = build_training_cases()

    assert len(cases) == 192
    assert len({case.row_id for case in cases}) == 192
    assert Counter(case.family_id for case in cases) == {
        family.id: 32 for family in TRAINING_FAMILIES
    }
    assert [case.case_seed for case in cases[:32]] == list(range(810000, 810032))


def test_family_ranking_uses_solve_count_then_neutral_family_id() -> None:
    rows = [
        {
            "family_id": family.id,
            "solved_count": value,
            "mean_minimum_tail_margin": 999.0 - index,
            "mean_resilience_auc": 999.0 - index,
        }
        for index, (family, value) in enumerate(
            zip(TRAINING_FAMILIES, (20, 20, 22, 24, 25, 26), strict=True)
        )
    ]

    ranked = rank_training_families(rows)

    tied = sorted((TRAINING_FAMILIES[0].id, TRAINING_FAMILIES[1].id))
    assert [row["family_id"] for row in ranked[:2]] == tied
    assert [row["solved_count"] for row in ranked] == [20, 20, 22, 24, 25, 26]


def test_shipped_policy_selects_weights_and_tuned_is_context_only_when_different(
    tmp_path: Path,
) -> None:
    ids = [family.id for family in TRAINING_FAMILIES]
    shipped_counts = dict(zip(ids, (25, 18, 30, 22, 28, 27), strict=True))
    tuned_counts = dict(zip(ids, (17, 29, 30, 22, 28, 27), strict=True))

    receipt = build_difficulty_receipt(
        shipped_evaluator=_evaluator(shipped_counts),
        tuned_evaluator=_evaluator(tuned_counts),
    )

    assert receipt["selection_policy"]["id"] == "shipped_v4_onnx"
    assert receipt["selection_policy"]["artifact_sha256"] == SHIPPED_ARTIFACT_SHA256
    assert "scripts/train_policy.py" in receipt["source_identity"]
    assert "scripts/training_artifacts.py" in receipt["source_identity"]
    assert set(receipt["source_identity"]) == set(SOURCE_PATHS)
    assert receipt["ranking"]["ranked_family_ids"][:2] == [ids[1], ids[3]]
    assert receipt["sampler"]["hardest_family_ids"] == [ids[1], ids[3]]
    assert receipt["sampler"]["family_weights"] == {
        family_id: (2 if family_id in (ids[1], ids[3]) else 1)
        for family_id in ids
    }
    assert receipt["contextual_tuned_rule"]["reported"] is True
    assert receipt["contextual_tuned_rule"]["ranked_family_ids"][0] == ids[0]
    path = tmp_path / "difficulty.json"
    write_new_json(path, receipt)
    assert validate_difficulty_receipt(path)["sampler"] == receipt["sampler"]

    for index, relative in enumerate(
        ("scripts/train_policy.py", "scripts/training_artifacts.py")
    ):
        tampered = json.loads(json.dumps(receipt))
        tampered["source_identity"][relative] = "0" * 64
        tampered_path = tmp_path / f"difficulty-tampered-{index}.json"
        tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
        with pytest.raises(ModerateStudyError, match="source identity drifted"):
            validate_difficulty_receipt(tampered_path)


def test_tuned_ranking_is_omitted_when_it_matches_shipped() -> None:
    ids = [family.id for family in TRAINING_FAMILIES]
    counts = dict(zip(ids, (20, 21, 22, 23, 24, 25), strict=True))

    receipt = build_difficulty_receipt(
        shipped_evaluator=_evaluator(counts),
        tuned_evaluator=_evaluator(counts),
    )

    assert receipt["contextual_tuned_rule"] == {
        "reported": False,
        "reason": "family_ordering_matches_shipped_policy",
    }


def test_weighted_cycle_is_exactly_two_to_one_and_bound_to_train_receipt(
    tmp_path: Path,
) -> None:
    hard = (TRAINING_FAMILIES[1].id, TRAINING_FAMILIES[4].id)
    weighted = weighted_training_cases(hard)
    counts = Counter(case.family_id for case in weighted)

    assert len(weighted) == 256
    assert counts == {
        family.id: (64 if family.id in hard else 32)
        for family in TRAINING_FAMILIES
    }
    expected_slots = [
        TRAINING_FAMILIES[1].id,
        TRAINING_FAMILIES[0].id,
        TRAINING_FAMILIES[4].id,
        TRAINING_FAMILIES[2].id,
        TRAINING_FAMILIES[1].id,
        TRAINING_FAMILIES[3].id,
        TRAINING_FAMILIES[4].id,
        TRAINING_FAMILIES[5].id,
    ]
    for start in range(0, len(weighted), WEIGHTED_FAMILY_SLOT_COUNT):
        block = weighted[start : start + WEIGHTED_FAMILY_SLOT_COUNT]
        assert [case.family_id for case in block] == expected_slots
        assert len({case.case_seed for case in block}) == 1
    difficulty_path = tmp_path / "difficulty.json"
    difficulty_path.write_text("{}", encoding="utf-8")
    contract = sampler_contract(
        difficulty_path,
        {"sampler": {"hardest_family_ids": list(hard)}},
    )
    assert contract["selection_policy"] == "shipped_v4_onnx"
    assert contract["selection_policy_sha256"] == SHIPPED_ARTIFACT_SHA256
    assert contract["family_weights"] == family_weights(hard)
    assert contract["family_slot_pattern"] == expected_slots
    assert contract["application_scope"] == [
        "behavior_cloning",
        "dagger_rollouts",
        "critic_warmup",
        "ppo_actor_critic_training",
    ]


def test_weighted_cycle_is_balanced_in_each_lane_at_both_warmup_prefixes() -> None:
    hard = (TRAINING_FAMILIES[1].id, TRAINING_FAMILIES[4].id)
    weights = family_weights(hard)
    balance = warmup_prefix_balance(hard)

    assert balance["lanes"] == TRAINING_LANES == 20
    assert [
        prefix["global_transitions"] for prefix in balance["prefixes"]
    ] == list(CRITIC_WARMUP_PREFIX_TRANSITIONS)
    for prefix in balance["prefixes"]:
        starts = prefix["episode_starts_per_lane"]
        assert prefix[
            "maximum_per_lane_absolute_deviation_from_weighted_target"
        ] <= 1.0
        assert len(prefix["lanes"]) == TRAINING_LANES
        for lane in prefix["lanes"]:
            counts = lane["family_episode_starts"]
            assert sum(counts.values()) == starts
            for family_id, weight in weights.items():
                target = starts * weight / WEIGHTED_FAMILY_SLOT_COUNT
                assert abs(counts[family_id] - target) <= 1.0


def test_adopted_training_invocation_is_fixed_three_seed_2m_config(
    tmp_path: Path,
) -> None:
    args = adopted_training_arguments(
        policy_seed=POLICY_SEEDS[0],
        receipt_path=tmp_path / "receipt.json",
        checkpoint_directory=tmp_path / "checkpoints",
    )
    joined = " ".join(args)

    assert ACTIVE_TRANSITIONS == 2_000_000
    assert POLICY_SEEDS == (37_017, 47_017, 57_017)
    for fragment in (
        "--transitions 2000000",
        "--lanes 20",
        "--n-steps 250",
        "--batch-size 500",
        "--learning-rate 0.000075",
        "--target-kl 0.02",
        "--ent-coef 0.003",
        "--critic-warmup-min-transitions 50000",
        "--freeze-observation-rms",
        "--reward-profile v3_equivalent",
        "--preparedness-alignment-coefficient 10.0",
    ):
        assert fragment in joined


def test_development_family_aggregate_binds_exact_200_case_roster() -> None:
    rows = []
    for family in DEVELOPMENT_FAMILIES:
        for index, case_seed in enumerate(DEVELOPMENT_SEEDS):
            rows.append(
                {
                    "row_id": f"{family.id}:{case_seed}",
                    "case_seed": case_seed,
                    "tape_seed": family.tape_seed(case_seed),
                    "solved": index < 35,
                    "hard_violation_count": 0,
                    "max_conservation_residual": 0.0,
                }
            )

    per_family = development_family_aggregate(rows)

    assert [row["solved_count"] for row in per_family] == [35] * 5
    assert sum(row["solved_count"] for row in per_family) == 175


def _endpoint(seed: int, solved: int, family_counts: list[int]) -> dict[str, Any]:
    return {
        "policy_seed": seed,
        "solved_count": solved,
        "solve_rate": solved / 200,
        "per_family": [
            {
                "family_id": family.id,
                "solved_count": count,
            }
            for family, count in zip(
                DEVELOPMENT_FAMILIES, family_counts, strict=True
            )
        ],
    }


def test_endpoint_summary_reports_incumbent_and_per_family_statistics() -> None:
    rows = [
        _endpoint(POLICY_SEEDS[0], 172, [34, 34, 35, 33, 36]),
        _endpoint(POLICY_SEEDS[1], 174, [35, 35, 35, 33, 36]),
        _endpoint(POLICY_SEEDS[2], 176, [36, 36, 35, 33, 36]),
    ]

    summary = summarize_seed_endpoints(rows)

    assert summary["mean_solved_count"] == 174
    assert summary["population_std_solved_count"] == pytest.approx(1.6329931619)
    assert summary["incumbent_comparison"]["incumbent_mean_solved_count"] == 171.4
    assert summary["incumbent_comparison"][
        "challenger_minus_incumbent_mean_solved_count"
    ] == pytest.approx(2.6)
    assert len(summary["per_family"]) == 5


def test_promotion_rule_is_conjunctive_and_uses_correct_boundaries() -> None:
    assert CURRENT_SELECTED_SOLVED_COUNT == 178
    assert PROMOTION_SELECTED_MINIMUM == 183
    assert PROMOTION_ENDPOINT_MINIMUM == 172
    assert INCUMBENT_SEED_MEAN_SOLVED_COUNT == 171.4
    assert promotion_gate(183, [172, 172, 172])["passed"] is True
    assert promotion_gate(182, [180, 180, 180])["passed"] is False
    assert promotion_gate(190, [172, 172, 170])["passed"] is False
    assert promotion_gate(190, [175, 171, 171])["passed"] is False
    assert promotion_gate(190, [173, 172, 171])["passed"] is True
    assert promotion_gate(183, [172, 172, 172])["final_evaluation_authorized"] is False


def test_selection_uses_nine_registered_curves_and_neutral_ties() -> None:
    seed_rows = []
    values = {
        POLICY_SEEDS[0]: (179, 183, 181),
        POLICY_SEEDS[1]: (180, 183, 182),
        POLICY_SEEDS[2]: (178, 181, 180),
    }
    for seed in POLICY_SEEDS:
        seed_rows.append(
            {
                "curves": [
                    {
                        "policy_seed": seed,
                        "active_actor_critic_transitions": milestone,
                        "solved_count": solved,
                    }
                    for milestone, solved in zip(
                        CURVE_MILESTONES, values[seed], strict=True
                    )
                ]
            }
        )

    winner, ranked = select_development_checkpoint(seed_rows)

    assert len(ranked) == 9
    assert winner["solved_count"] == 183
    assert winner["policy_seed"] == POLICY_SEEDS[0]
    assert winner["active_actor_critic_transitions"] == 1_000_000


def test_cli_has_no_final_mode_and_source_only_monkeypatches_trainer() -> None:
    assert parse_args(["--measure-difficulty", "--difficulty-output", "E:/x.json"])
    source = (
        Path(__file__).parents[1] / "scripts" / "moderate_family_training.py"
    ).read_text(encoding="utf-8")

    assert "FINAL_FAMILIES" not in source
    assert "FINAL_SEEDS" not in source
    assert '"--final"' not in source
    assert "trainer.training_scenarios = patched_training_scenarios" in source
    assert "trainer.spawn_environment = patched_spawn_environment" in source
    assert "SHIPPED_ARTIFACT_SHA256" in source
    assert "tuned_rule_did_not_select_family_weights" in source
