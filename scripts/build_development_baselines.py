#!/usr/bin/env python3
"""Build the current 200-case cheap-planner development evidence table."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.city.scenarios import (  # noqa: E402
    DEVELOPMENT_FAMILIES,
    DEVELOPMENT_SEEDS,
)
from backend.app.shared_evidence import (  # noqa: E402
    canonical_bytes,
    canonical_hash,
    file_sha256,
    fsync_parent,
    load_json_object,
    split_contract,
    wilson_interval,
)
from scripts.evaluate import (  # noqa: E402
    DEFAULT_ONNX_PATH,
    ProbeRow,
    resolve_policy,
    run_probe,
)

DEFAULT_RECEIPT = (
    ROOT / "internal" / "developmental_runs" / "v4" / "development-baselines-200.json"
)
DEFAULT_MARKDOWN = ROOT / "benchmarks" / "v4" / "development-baselines-200.md"
HISTORICAL_RECEIPT = (
    ROOT / "internal" / "developmental_runs" / "v4" / "step6-dev-baseline-table.json"
)
HISTORICAL_MARKDOWN = ROOT / "benchmarks" / "v4" / "development-baselines.md"
EXPECTED_CASES = 200
POLICY_SPECS = (
    "heuristic",
    "teacher",
    "tuned",
    f"onnx:{DEFAULT_ONNX_PATH.relative_to(ROOT).as_posix()}",
)
DISPLAY_LABELS = {
    "heuristic": "Reactive heuristic",
    "teacher": "Preparedness teacher",
    "tuned": "Tuned constant rule",
    f"onnx:{DEFAULT_ONNX_PATH.relative_to(ROOT).as_posix()}": (
        "Legacy ONNX regression fixture"
    ),
}


class BaselineError(RuntimeError):
    """Raised when development evidence cannot be assembled honestly."""


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _source_identity() -> dict[str, str]:
    paths = (
        ROOT / "scripts" / "build_development_baselines.py",
        ROOT / "scripts" / "evaluate.py",
        ROOT / "backend" / "app" / "city" / "scenarios.py",
        ROOT / "backend" / "app" / "city" / "environment.py",
        ROOT / "backend" / "app" / "city" / "outcome.py",
        ROOT / "backend" / "app" / "city" / "planners.py",
        ROOT / "model" / "policy.py",
        DEFAULT_ONNX_PATH,
    )
    return {_relative(path): file_sha256(path) for path in paths}


def _historical_reference() -> dict[str, Any]:
    receipt = load_json_object(
        HISTORICAL_RECEIPT,
        "historical 40-case development receipt",
        error_type=BaselineError,
    )
    planners = receipt.get("planners")
    if not isinstance(planners, list):
        raise BaselineError("historical planner evidence is malformed")
    oracle = next(
        (
            planner
            for planner in planners
            if planner.get("label") == "Clairvoyant CEM oracle (privileged)"
        ),
        None,
    )
    if not isinstance(oracle, dict) or oracle.get("solved_count") != 37:
        raise BaselineError("historical oracle evidence is missing or drifted")
    return {
        "scope": "original_40_case_development_subset",
        "subset_seed_interval": {"first": 820000, "last": 820007, "count": 8},
        "case_count": 40,
        "receipt": {
            "path": _relative(HISTORICAL_RECEIPT),
            "sha256": file_sha256(HISTORICAL_RECEIPT),
        },
        "markdown": {
            "path": _relative(HISTORICAL_MARKDOWN),
            "sha256": file_sha256(HISTORICAL_MARKDOWN),
        },
        "privileged_oracle": {
            "solved_count": 37,
            "case_count": 40,
            "mean_resilience_auc": oracle["mean_resilience_auc"],
            "mean_minimum_tail_margin": oracle["mean_minimum_tail_margin"],
            "disclosure": (
                "Clairvoyant CEM used future-tape information on only the original "
                "40-case subset. It is a headroom diagnostic, not a submission "
                "baseline and not a 200-case ceiling estimate."
            ),
        },
    }


def _row_payload(row: ProbeRow) -> dict[str, Any]:
    value = asdict(row)
    value["reason_codes"] = list(row.reason_codes)
    return value


def build_payload() -> dict[str, Any]:
    """Run the four cheap planners on the shared 200-case development split."""

    policies = [resolve_policy(specification) for specification in POLICY_SPECS]
    result = run_probe("dev", policies)
    if result.get("case_count") != EXPECTED_CASES:
        raise BaselineError("development evaluator did not return 200 cases")

    ordered_ids: list[str] | None = None
    ordered_tapes: list[int] | None = None
    policy_records: list[dict[str, Any]] = []
    row_records: dict[str, list[dict[str, Any]]] = {}
    for policy in policies:
        rows: list[ProbeRow] = result["rows"][policy.label]
        row_ids = [row.row_id for row in rows]
        tape_seeds = [row.tape_seed for row in rows]
        if ordered_ids is None:
            ordered_ids = row_ids
            ordered_tapes = tape_seeds
        elif row_ids != ordered_ids or tape_seeds != ordered_tapes:
            raise BaselineError("policies did not run on identical ordered tapes")
        metrics = dict(result["policies"][policy.label])
        if (
            metrics["case_count"] != EXPECTED_CASES
            or metrics["hard_violation_count"] != 0
            or metrics["maximum_conservation_residual"] != 0.0
        ):
            raise BaselineError("development baseline invariant failed")
        metrics.update(
            {
                "id": policy.label,
                "label": DISPLAY_LABELS[policy.label],
                "wilson_95": wilson_interval(
                    metrics["solved_count"], metrics["case_count"], digits=10
                ),
            }
        )
        policy_records.append(metrics)
        row_records[policy.label] = [_row_payload(row) for row in rows]

    if ordered_ids is None or len(set(ordered_ids)) != EXPECTED_CASES:
        raise BaselineError("development row identities are not unique")

    return {
        "schema_version": 1,
        "tool": "build_development_baselines",
        "status": "development_baselines_200_nonauthorizing",
        "authorizing": False,
        "training_performed": False,
        "split": "dev",
        "final_split_used": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "case_count": EXPECTED_CASES,
        "same_ordered_tapes": True,
        "split_contract": split_contract(
            "development", DEVELOPMENT_FAMILIES, DEVELOPMENT_SEEDS
        ),
        "policy_order": [policy.label for policy in policies],
        "policies": policy_records,
        "paired_comparisons": result["paired_comparisons"],
        "rows": row_records,
        "rows_sha256": canonical_hash(row_records),
        "historical_40_case_evidence": _historical_reference(),
        "source_identity": _source_identity(),
        "invariants": {
            "development_case_count_exactly_200": len(ordered_ids) == 200,
            "development_row_ids_unique": len(set(ordered_ids)) == 200,
            "same_ordered_tapes_for_every_policy": True,
            "all_hard_violation_counts_zero": all(
                policy["hard_violation_count"] == 0 for policy in policy_records
            ),
            "all_conservation_residuals_exactly_zero": all(
                policy["maximum_conservation_residual"] == 0.0
                for policy in policy_records
            ),
            "oracle_scope_limited_to_historical_40_case_subset": True,
        },
    }


def render_markdown(payload: dict[str, Any]) -> str:
    """Render a compact judge-facing table from the machine receipt."""

    lines = [
        "# Development baselines — 200 cases",
        "",
        (
            "All four cheap planners below ran on the same 200 development tapes "
            "(5 unchanged families × 40 seeds). This evidence is development-only, "
            "nonauthorizing, and contains no learned-v4 final result."
        ),
        "",
        "| Planner | Solved / 200 | Wilson 95% CI | Mean resilience AUC | Mean minimum tail margin | Hard violations | Max conservation residual |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for policy in payload["policies"]:
        lower, upper = policy["wilson_95"]
        lines.append(
            f"| {policy['label']} | **{policy['solved_count']}/200** | "
            f"[{lower:.3f}, {upper:.3f}] | {policy['mean_resilience_auc']:.6f} | "
            f"{policy['mean_minimum_tail_margin']:+.6f} | "
            f"{policy['hard_violation_count']} | "
            f"{policy['maximum_conservation_residual']:.1e} |"
        )
    lines.extend(
        [
            "",
            "## Historical 40-case evidence",
            "",
            (
                "The earlier table at `benchmarks/v4/development-baselines.md` "
                "remains byte-identical, receipt-bound historical evidence from "
                "the original eight-seed subset. Its PPO, BC, MPC, and rule scores "
                "must not be compared numerically with the 200-case results above."
            ),
            "",
            (
                "The privileged clairvoyant CEM result remains **37/40 on that "
                "original subset only**. It establishes constructive headroom; it "
                "is not a submission baseline, a 200-case result, or a mathematical "
                "upper bound."
            ),
            "",
            "Machine receipt: `internal/developmental_runs/v4/development-baselines-200.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_new(path: Path, payload: bytes) -> None:
    path = path.resolve()
    if path.exists() or path.is_symlink():
        raise BaselineError(f"refusing to overwrite evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists() or path.is_symlink():
            raise BaselineError(f"refusing to overwrite evidence: {path}")
        os.rename(temporary, path)
        fsync_parent(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_payload()
    markdown = render_markdown(payload)
    _write_new(args.receipt, canonical_bytes(payload))
    try:
        _write_new(args.markdown, markdown.encode("utf-8"))
    except Exception:
        # A lone machine receipt is still complete evidence; report the partial
        # publication rather than overwriting or silently deleting it.
        raise
    print(
        "development baselines: "
        + ", ".join(
            f"{policy['id']}={policy['solved_count']}/200"
            for policy in payload["policies"]
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
