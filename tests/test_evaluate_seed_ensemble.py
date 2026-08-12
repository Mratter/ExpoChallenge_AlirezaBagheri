from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import scripts.evaluate_seed_ensemble as ensemble
from backend.app.shared_evidence import canonical_hash, file_sha256

RECEIPT = (
    ensemble.ROOT
    / "internal"
    / "developmental_runs"
    / "v4"
    / "action-mean-ensemble-5x2m-dev-200.json"
)


def _members() -> list[ensemble.EnsembleMember]:
    return [
        ensemble.EnsembleMember(
            seed=seed,
            model=SimpleNamespace(name=f"model-{seed}"),  # type: ignore[arg-type]
            normalization=SimpleNamespace(seed=seed),  # type: ignore[arg-type]
            identity={"seed": seed},
        )
        for seed in ensemble.REGISTERED_POLICY_SEEDS
    ]


def test_action_mean_uses_each_members_own_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    members = _members()
    calls: list[tuple[str, int]] = []

    def fake_actions(model: Any, normalization: Any, raw: Any) -> np.ndarray:
        calls.append((model.name, normalization.seed))
        member_index = ensemble.REGISTERED_POLICY_SEEDS.index(normalization.seed)
        return np.full((1, ensemble.ACTION_COUNT), member_index / 4, np.float32)

    monkeypatch.setattr(ensemble, "sb3_actions", fake_actions)
    actor = ensemble.ActionMeanActor(members)
    action = actor(np.zeros(73, dtype=np.float32))

    assert action.shape == (1, ensemble.ACTION_COUNT)
    assert action.dtype == np.float32
    assert np.array_equal(action, np.full_like(action, 0.5))
    assert calls == [
        (f"model-{seed}", seed) for seed in ensemble.REGISTERED_POLICY_SEEDS
    ]
    assert actor.disagreement_receipt() == {
        "raw_observation_count": 1,
        "action_component_count": ensemble.ACTION_COUNT,
        "maximum_member_action_span": 1.0,
        "mean_member_action_span": 1.0,
    }


def test_action_mean_rejects_member_subsets_and_unbounded_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ensemble.EnsembleError, match="five registered seeds"):
        ensemble.ActionMeanActor(_members()[:-1])

    monkeypatch.setattr(
        ensemble,
        "sb3_actions",
        lambda *_args: np.full((1, ensemble.ACTION_COUNT), 1.01, np.float32),
    )
    with pytest.raises(ensemble.EnsembleError, match="finite and bounded"):
        ensemble.ActionMeanActor(_members())(np.zeros(73, dtype=np.float32))


@dataclass(frozen=True)
class FakeCase:
    row_id: str
    family_id: str
    tape_sha256: str


def test_development_evaluation_aggregates_exact_roster_and_invariants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = [
        FakeCase(
            row_id=f"{family.id}:{case_index}",
            family_id=family.id,
            tape_sha256=f"{case_index + family_index * 40:064x}",
        )
        for family_index, family in enumerate(ensemble.DEVELOPMENT_FAMILIES)
        for case_index in range(40)
    ]
    solved_ids = {case.row_id for case in cases[:178]}

    def fake_rollout(case: FakeCase, actor: Any, *, label: str) -> Any:
        assert label == "five_seed_action_mean"
        actor(np.zeros(73, dtype=np.float32))
        solved = case.row_id in solved_ids
        return SimpleNamespace(
            row={
                "row_id": case.row_id,
                "family_id": case.family_id,
                "case_seed": 0,
                "tape_seed": 0,
                "tape_sha256": case.tape_sha256,
                "solved": solved,
                "reason_codes": [] if solved else ["assessment_tail_targets_met"],
                "resilience_auc": 0.5,
                "hard_violation_count": 0,
                "maximum_conservation_residual": 0.0,
                "trajectory_sha256": "0" * 64,
                "observation_trace_sha256": "1" * 64,
                "action_trace_sha256": "2" * 64,
            }
        )

    monkeypatch.setattr(ensemble, "development_cases", lambda: cases)
    monkeypatch.setattr(ensemble, "rollout_development_case", fake_rollout)
    monkeypatch.setattr(
        ensemble,
        "sb3_actions",
        lambda *_args: np.zeros((1, ensemble.ACTION_COUNT), np.float32),
    )

    result = ensemble.evaluate_ensemble(_members())

    assert result["case_count"] == 200
    assert result["solved_count"] == 178
    assert result["solve_rate"] == 0.89
    assert result["hard_violation_count"] == 0
    assert result["maximum_conservation_residual"] == 0.0
    assert sum(row["solved_count"] for row in result["per_family"]) == 178
    assert result["member_disagreement"]["raw_observation_count"] == 200
    assert len(result["rows"]) == 200


def test_tool_has_no_final_split_interface_or_final_roster_import() -> None:
    source_path = Path(ensemble.__file__)
    source = source_path.read_text(encoding="utf-8")

    assert "FINAL_FAMILIES" not in source
    assert "FINAL_SEEDS" not in source
    assert 'add_argument("--split"' not in source
    assert ensemble.DEFAULT_RECEIPT.is_relative_to(
        ensemble.ROOT / "internal" / "developmental_runs" / "v4"
    )
    cases = ensemble.development_cases()
    assert len(cases) == 200
    assert len({case.row_id for case in cases}) == 200


def test_cli_requires_only_the_external_study_root() -> None:
    parsed = ensemble.parse_args(["--study-root", "E:/study"])
    assert parsed.study_root == Path("E:/study")
    with pytest.raises(SystemExit):
        ensemble.parse_args(["--study-root", "E:/study", "--split", "final"])


def test_committed_development_receipt_is_complete_and_source_bound() -> None:
    payload = ensemble.load_json_object(
        RECEIPT,
        "development ensemble receipt",
        error_type=AssertionError,
    )

    assert payload["schema_version"] == ensemble.SCHEMA_VERSION
    assert payload["status"] == "complete_development_experiment"
    assert payload["split"] == "dev"
    assert payload["final_split_used"] is False
    assert payload["split_contract"]["id"] == "dev"
    assert payload["split_contract"]["cartesian_case_count"] == 200
    assert payload["ensemble"]["registered_policy_seeds"] == list(
        ensemble.REGISTERED_POLICY_SEEDS
    )
    members = payload["ensemble"]["members"]
    assert len(members) == 5
    assert payload["ensemble"]["members_identity_sha256"] == canonical_hash(members)
    assert [member["checkpoint_bundle"]["checkpoint_id"] for member in members] == [
        f"seed-{seed}-ppo-2000000" for seed in ensemble.REGISTERED_POLICY_SEEDS
    ]
    assert len({member["checkpoint_bundle"]["model_sha256"] for member in members}) == 5
    assert (
        len({member["checkpoint_bundle"]["obs_rms_sha256"] for member in members}) == 5
    )

    evaluation = payload["evaluation"]
    assert evaluation["case_count"] == len(evaluation["rows"]) == 200
    assert evaluation["solved_count"] == 179
    assert evaluation["solve_rate"] == 0.895
    assert evaluation["hard_violation_count"] == 0
    assert evaluation["maximum_conservation_residual"] == 0.0
    assert evaluation["rows_sha256"] == canonical_hash(evaluation["rows"])
    assert len({row["row_id"] for row in evaluation["rows"]}) == 200
    assert [row["solved_count"] for row in evaluation["per_family"]] == [
        34,
        38,
        40,
        31,
        36,
    ]
    assert evaluation["member_disagreement"]["raw_observation_count"] == 6_000
    assert evaluation["member_disagreement"]["action_component_count"] == 132_000
    comparison = payload["comparison_to_selected_single_checkpoint"]
    assert comparison["solved_count"] == 178
    assert comparison["ensemble_minus_selected_solved_cases"] == 1
    assert comparison["parity_receipt_sha256"] == (
        ensemble.SHIPPED_PARITY_SHA256
    )
    assert comparison["parity_rows_sha256"] == (
        ensemble.SHIPPED_PARITY_ROWS_SHA256
    )
    assert comparison["matched_contingency"] == {
        "both_solved": 176,
        "ensemble_only": 3,
        "selected_only": 2,
        "neither_solved": 19,
        "ensemble_only_row_ids": comparison["matched_contingency"][
            "ensemble_only_row_ids"
        ],
        "selected_only_row_ids": comparison["matched_contingency"][
            "selected_only_row_ids"
        ],
    }
    assert len(comparison["matched_contingency"]["ensemble_only_row_ids"]) == 3
    assert len(comparison["matched_contingency"]["selected_only_row_ids"]) == 2
    assert payload["decision"] == {
        "candidate_count": 1,
        "candidate_kept": True,
        "candidate_promoted": False,
        "configuration_search_performed": False,
        "reason": payload["decision"]["reason"],
    }
    assert payload["deployment_status"]["shipped_artifact_changed"] is False
    assert payload["deployment_status"]["exported"] is False

    for path, expected_sha256 in payload["source_files"].items():
        assert file_sha256(ensemble.ROOT / path) == expected_sha256
