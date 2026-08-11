"""Synthetic tests for the irreversible v4 final publication runner.

These tests never import the evaluator or build the reserved roster. Every
execution path receives injected synthetic cases, evidence, and git state.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from types import SimpleNamespace
from typing import Any

import pytest

import scripts.publish_final_evaluation_v4 as publication
from backend.app.shared_evidence import canonical_hash, file_sha256


FIXED_TIME = "2026-08-12T00:00:00+00:00"
FIXED_COMMIT = "a" * 40


def _paths(root: Path) -> publication.PublicationPaths:
    return publication.PublicationPaths(
        claim=root / "claim.json",
        success=root / "success.json",
        failure=root / "failure.json",
        markdown=root / "final.md",
    )


def _git_probe() -> dict[str, Any]:
    return {"commit": FIXED_COMMIT, "clean": True}


def _evidence_probe() -> dict[str, Any]:
    return {
        "artifact": publication.EXPECTED_ARTIFACT_SHA256,
        "artifact_manifest": publication.ARTIFACT_MANIFEST_SHA256,
        "dev_parity_receipt": publication.DEV_PARITY_RECEIPT_SHA256,
        "oracle_receipt": publication.ORACLE_RECEIPT_SHA256,
        "regression_gate": publication.REGRESSION_GATE_SHA256,
        "legacy_fixture": publication.LEGACY_FIXTURE_SHA256,
        "core_sources": dict(publication.CORE_SOURCE_SHA256),
    }


def _identity(index: int) -> dict[str, Any]:
    family_id = publication.EXPECTED_FINAL_SPLIT_CONTRACT["family_ids"][index // 40]
    case_seed = 830_000 + index % 40
    row_id = f"{family_id}:{case_seed}"
    return {
        "row_id": row_id,
        "family_id": family_id,
        "case_seed": case_seed,
        "tape_seed": 200_000 + index,
        "tape_sha256": hashlib.sha256(f"tape:{index}".encode()).hexdigest(),
    }


def _model_rows(solved_count: int = 163) -> list[dict[str, Any]]:
    solved_indices = set(range(min(solved_count, 162)))
    solved_indices.update(range(162, solved_count))
    if solved_count == 163:
        solved_indices = set(range(162)) | {190}
    rows = []
    for index in range(200):
        solved = index in solved_indices
        rows.append(
            {
                **_identity(index),
                "solved": solved,
                "status": "solved" if solved else "failed",
                "reason_codes": [] if solved else ["resilience_auc_met"],
                "resilience_auc": 0.45 + index / 100_000,
                "minimum_tail_margin": 0.02 if solved else -0.02,
                "critical_service_days": index % 10,
                "hard_violation_count": 0,
                "max_conservation_residual": 0.0,
                "trajectory_sha256": hashlib.sha256(
                    f"trajectory:{index}".encode()
                ).hexdigest(),
            }
        )
    return rows


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reasons = Counter(
        reason
        for row in rows
        if not row["solved"]
        for reason in row["reason_codes"]
    )
    solved = sum(row["solved"] for row in rows)
    return {
        "case_count": len(rows),
        "solved_count": solved,
        "solve_rate": solved / len(rows),
        "mean_resilience_auc": round(fmean(row["resilience_auc"] for row in rows), 10),
        "mean_minimum_tail_margin": round(
            fmean(row["minimum_tail_margin"] for row in rows), 10
        ),
        "hard_violation_count": 0,
        "maximum_conservation_residual": 0.0,
        "failure_reason_code_histogram": dict(sorted(reasons.items())),
    }


def _result(solved_count: int = 163) -> dict[str, Any]:
    rows = _model_rows(solved_count)
    label = publication._policy_label()
    return {
        "schema_version": 1,
        "tool": "evaluate",
        "authorizing": False,
        "split": publication.FINAL_SPLIT_ID,
        "case_count": 200,
        "same_tapes": True,
        "policies": {label: _aggregate(rows)},
        "paired_comparisons": {},
        "rows": {label: rows},
        "rollout_count": 200,
    }


def _reference() -> tuple[dict[str, Any], dict[str, Any]]:
    rows = []
    for index in range(200):
        rows.append(
            {
                **_identity(index),
                "clairvoyant_oracle_cem": {
                    "solved": index < 182,
                    "hard_violation_count": 0,
                    "maximum_conservation_residual": 0.0,
                },
                "oracle_search_wide_invariants": {
                    "maximum_hard_violation_count": 0,
                    "maximum_conservation_residual": 0.0,
                },
            }
        )
    receipt = {
        "rows": rows,
        "rows_sha256": canonical_hash(rows),
        "split_contract": dict(publication.EXPECTED_FINAL_SPLIT_CONTRACT),
    }
    binding = {
        "artifact_manifest": {
            "path": "synthetic/manifest.json",
            "sha256": publication.ARTIFACT_MANIFEST_SHA256,
        },
        "development_parity_receipt": {
            "path": "synthetic/parity.json",
            "sha256": publication.DEV_PARITY_RECEIPT_SHA256,
        },
        "oracle_receipt": {
            "path": "synthetic/oracle.json",
            "sha256": publication.ORACLE_RECEIPT_SHA256,
        },
        "regression_gate": {
            "path": "synthetic/gate.py",
            "sha256": publication.REGRESSION_GATE_SHA256,
        },
        "legacy_fixture": {
            "path": "synthetic/legacy.onnx",
            "sha256": publication.LEGACY_FIXTURE_SHA256,
        },
        "core_sources": dict(publication.CORE_SOURCE_SHA256),
    }
    return receipt, binding


def test_preflight_is_repeatable_and_does_not_import_or_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)

    def forbidden_import(name: str) -> Any:
        raise AssertionError(f"preflight attempted import: {name}")

    monkeypatch.setattr(publication.importlib, "import_module", forbidden_import)
    first = publication.preflight(
        paths=paths, git_probe=_git_probe, evidence_probe=_evidence_probe
    )
    second = publication.preflight(
        paths=paths, git_probe=_git_probe, evidence_probe=_evidence_probe
    )

    assert first == second
    assert first["reserved_split_imported_or_built"] is False
    assert first["filesystem_written"] is False
    assert not any(path.exists() for path in paths.__dict__.values())


def test_claim_exists_before_synthetic_runner_and_success_is_terminal(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    calls = 0

    def runner(_artifact: Path) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        assert paths.claim.is_file()
        assert not paths.success.exists()
        assert not paths.failure.exists()
        return _result()

    summary = publication.execute_once(
        paths=paths,
        git_probe=_git_probe,
        evidence_probe=_evidence_probe,
        reference_loader=_reference,
        runner=runner,
        created_at_utc=FIXED_TIME,
    )

    assert calls == 1
    assert summary["solved_count"] == 163
    assert summary["wilson_95"] == [0.7554293724, 0.862698072]
    assert paths.claim.is_file() and paths.success.is_file()
    assert not paths.failure.exists() and not paths.markdown.exists()

    receipt = publication.load_json_object(paths.success, "test success")
    assert receipt["claim"]["sha256"] == file_sha256(paths.claim)
    assert receipt["claim"]["git_commit"] == FIXED_COMMIT
    assert receipt["authorization"] == publication.AUTHORIZATION
    assert receipt["timing"]["started_at_utc"] == FIXED_TIME
    assert receipt["timing"]["elapsed_seconds"] >= 0.0
    assert receipt["split_contract"] == publication.EXPECTED_FINAL_SPLIT_CONTRACT
    assert receipt["split_contract_sha256"] == canonical_hash(
        publication.EXPECTED_FINAL_SPLIT_CONTRACT
    )
    assert receipt["rows_sha256"] == canonical_hash(receipt["rows"])
    assert receipt["ordered_split_identity_sha256"] == canonical_hash(
        receipt["ordered_split_identity"]
    )
    assert [row["solved_count"] for row in receipt["per_family"]] == [40, 40, 40, 40, 3]
    assert receipt["oracle_comparison"]["pairing"] == {
        "both": 162,
        "policy_only": 1,
        "oracle_only": 20,
        "neither": 17,
    }
    assert receipt["oracle_comparison"]["known_feasible_union_count"] == 183

    with pytest.raises(publication.FinalEvaluationError, match="claim output already exists"):
        publication.execute_once(
            paths=paths,
            git_probe=_git_probe,
            evidence_probe=_evidence_probe,
            reference_loader=_reference,
            runner=runner,
            created_at_utc=FIXED_TIME,
        )
    assert calls == 1


def test_post_claim_runner_failure_writes_terminal_failure(tmp_path: Path) -> None:
    paths = _paths(tmp_path)

    def failed_runner(_artifact: Path) -> dict[str, Any]:
        assert paths.claim.is_file()
        raise RuntimeError("synthetic runner failed")

    with pytest.raises(publication.FinalEvaluationError, match="terminal failure recorded"):
        publication.execute_once(
            paths=paths,
            git_probe=_git_probe,
            evidence_probe=_evidence_probe,
            reference_loader=_reference,
            runner=failed_runner,
            created_at_utc=FIXED_TIME,
        )

    assert paths.claim.is_file() and paths.failure.is_file()
    assert not paths.success.exists()
    failure = publication.load_json_object(paths.failure, "test failure")
    assert failure["retry_permitted"] is False
    assert failure["failed_stage"] == "lazy_import_and_exactly_200_rollouts"
    assert failure["claim"]["sha256"] == file_sha256(paths.claim)


def test_wrong_solve_count_consumes_claim_and_fails_closed(tmp_path: Path) -> None:
    paths = _paths(tmp_path)

    with pytest.raises(publication.FinalEvaluationError, match="terminal failure recorded"):
        publication.execute_once(
            paths=paths,
            git_probe=_git_probe,
            evidence_probe=_evidence_probe,
            reference_loader=_reference,
            runner=lambda _artifact: _result(162),
            created_at_utc=FIXED_TIME,
        )

    assert paths.claim.is_file() and paths.failure.is_file()
    assert not paths.success.exists()
    failure = publication.load_json_object(paths.failure, "test failure")
    assert failure["failed_stage"] == "result_validation_and_oracle_join"
    assert "exactly 163" in failure["error_message"]
    assert failure["observed_result_evidence"]["observed_solved_count"] == 162
    assert failure["observed_result_evidence"]["observed_row_count"] == 200
    assert len(failure["observed_result_evidence"]["observed_rows_sha256"]) == 64


def test_unclean_preflight_does_not_consume_claim(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    with pytest.raises(publication.FinalEvaluationError, match="clean worktree"):
        publication.preflight(
            paths=paths,
            git_probe=lambda: {"commit": FIXED_COMMIT, "clean": False},
            evidence_probe=_evidence_probe,
        )
    assert not paths.claim.exists()


def test_preflight_rejects_a_broken_output_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    original = Path.is_symlink

    def synthetic_is_symlink(path: Path) -> bool:
        return path == paths.claim or original(path)

    monkeypatch.setattr(Path, "is_symlink", synthetic_is_symlink)
    with pytest.raises(publication.FinalEvaluationError, match="exists or is a symlink"):
        publication.preflight(
            paths=paths,
            git_probe=_git_probe,
            evidence_probe=_evidence_probe,
        )
    assert not paths.claim.exists()


def test_markdown_is_published_from_success_without_a_runner(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    publication.execute_once(
        paths=paths,
        git_probe=_git_probe,
        evidence_probe=_evidence_probe,
        reference_loader=_reference,
        runner=lambda _artifact: _result(),
        created_at_utc=FIXED_TIME,
    )

    summary = publication.publish_markdown_from_success(paths=paths)
    markdown = paths.markdown.read_text(encoding="utf-8")
    assert summary["status"] == "published"
    assert "| Privileged clairvoyant CEM | 182/200 | 0.910 |" in markdown
    assert "| Shipped v4 PPO | **163/200** | **0.815** |" in markdown
    assert "| Tuned constant rule | 147/200 | 0.735 |" in markdown
    assert "| Preparedness teacher | 139/200 | 0.695 |" in markdown
    assert "| Selected causal MPC, k=5 | 135/200 | 0.675 |" in markdown
    assert "| Legacy ONNX regression fixture | 125/200 | 0.625 |" in markdown
    assert "| Reactive heuristic | 72/200 | 0.360 |" in markdown
    assert "| 162 | 1 | 20 | 17 | 183/200 |" in markdown
    assert "## Shipped v4 results by scenario family" in markdown
    assert "40/40" in markdown and "3/40" in markdown
    assert "clustered within five fixed scenario families" in markdown
    assert "precision is slightly overstated" in markdown
    assert "16 solved cases ahead" in markdown


@dataclass(frozen=True)
class _SyntheticShock:
    value: int


@dataclass(frozen=True)
class _SyntheticProbeRow:
    row_id: str
    family_id: str
    case_seed: int
    tape_seed: int
    solved: bool
    status: str
    reason_codes: tuple[str, ...]
    resilience_auc: float
    minimum_tail_margin: float
    critical_service_days: int
    hard_violation_count: int
    max_conservation_residual: float
    trajectory_sha256: str


def test_production_adapter_requests_one_synthetic_rollout_per_case(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    cases = [
        SimpleNamespace(
            row_id=f"synthetic:{index}",
            family_id="synthetic",
            case_seed=index,
            tape_seed=1000 + index,
            schedule=(_SyntheticShock(index),),
        )
        for index in range(200)
    ]
    policy = SimpleNamespace(kind="onnx")

    def rollout(case: Any, received_policy: Any) -> _SyntheticProbeRow:
        assert received_policy is policy
        calls.append(case.case_seed)
        return _SyntheticProbeRow(
            row_id=case.row_id,
            family_id=case.family_id,
            case_seed=case.case_seed,
            tape_seed=case.tape_seed,
            solved=case.case_seed < 163,
            status="solved" if case.case_seed < 163 else "failed",
            reason_codes=() if case.case_seed < 163 else ("resilience_auc_met",),
            resilience_auc=0.5,
            minimum_tail_margin=0.01,
            critical_service_days=0,
            hard_violation_count=0,
            max_conservation_residual=0.0,
            trajectory_sha256=hashlib.sha256(str(case.case_seed).encode()).hexdigest(),
        )

    fake_evaluator = SimpleNamespace(
        resolve_policy=lambda specification: policy,
        build_cases=lambda split: cases,
        rollout=rollout,
        aggregate=lambda rows: {"synthetic_count": len(rows)},
    )
    monkeypatch.setattr(
        publication.importlib,
        "import_module",
        lambda name: fake_evaluator,
    )

    result = publication._production_runner(publication.ARTIFACT)
    assert calls == list(range(200))
    assert result["rollout_count"] == 200
    assert len(result["rows"][publication._policy_label()]) == 200
