#!/usr/bin/env python3
"""Assemble the canonical development-only v4 baseline evidence table.

This tool performs no training and cannot access the final split.  It combines
the hash-pinned Step 3e and Step 3.5 receipts with fresh deterministic replay of
the small public v3 baselines on the same 40 development tapes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate import (  # noqa: E402
    DEFAULT_ONNX_PATH,
    ProbeRow,
    build_cases,
    exact_mcnemar_p,
    resolve_policy,
    rollout,
)
from backend.app.shared_evidence import (  # noqa: E402
    file_sha256,
    load_json_object,
    wilson_interval,
)

TOOL_ID = "assemble_dev_baselines_v4.py"
SCHEMA_VERSION = 1
HEADROOM_PATH = (
    ROOT / "internal" / "developmental_runs" / "v4" / "headroom-probe-v4-dev.json"
)
STEP3E_PATH = (
    ROOT
    / "internal"
    / "developmental_runs"
    / "v4"
    / "step3e-matched-reward-1m-seed-37017-attempt-02.json"
)
DEFAULT_RECEIPT = (
    ROOT
    / "internal"
    / "developmental_runs"
    / "v4"
    / "step6-dev-baseline-table.json"
)
DEFAULT_MARKDOWN = ROOT / "benchmarks" / "v4" / "development-baselines.md"
EXPECTED_HEADROOM_SHA256 = (
    "f037c98d8fec483dfa6b5c9c1691842597a4163c7d1ee6f3e72618f987d671b9"
)
EXPECTED_STEP3E_SHA256 = (
    "5b2071cd79a92143bc7a07262e7eeb1dbee83db63f8755d21f7167aa41ec065b"
)


class AssemblyError(RuntimeError):
    """Raised when the canonical development table cannot be proved."""


def load_receipt(path: Path, expected_sha256: str) -> dict[str, Any]:
    return load_json_object(
        path,
        f"required receipt {path}",
        expected_sha256=expected_sha256,
        error_type=AssemblyError,
    )


def compact_probe_row(row: ProbeRow, tape_sha256: str) -> dict[str, Any]:
    value = asdict(row)
    value["reason_codes"] = list(row.reason_codes)
    value["tape_sha256"] = tape_sha256
    return value


def normalize_existing_result(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "solved": bool(value["solved"]),
        "reason_codes": list(value.get("reason_codes", [])),
        "resilience_auc": float(value["resilience_auc"]),
        "minimum_tail_margin": float(value["minimum_tail_margin"]),
        "hard_violation_count": int(value["hard_violation_count"]),
        "max_conservation_residual": float(
            value.get(
                "max_conservation_residual",
                value.get("maximum_conservation_residual"),
            )
        ),
    }


def aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 40:
        raise AssemblyError(f"planner has {len(rows)} rows, expected 40")
    solved = sum(bool(row["solved"]) for row in rows)
    lower, upper = wilson_interval(solved, 40, digits=10)
    return {
        "case_count": 40,
        "solved_count": solved,
        "solve_rate": round(solved / 40.0, 10),
        "wilson_95_ci": {"lower": lower, "upper": upper},
        "mean_resilience_auc": round(
            fmean(float(row["resilience_auc"]) for row in rows), 10
        ),
        "mean_minimum_tail_margin": round(
            fmean(float(row["minimum_tail_margin"]) for row in rows), 10
        ),
        "hard_violation_count": sum(
            int(row["hard_violation_count"]) for row in rows
        ),
        "maximum_conservation_residual": max(
            abs(float(row["max_conservation_residual"])) for row in rows
        ),
    }


def paired(left: Sequence[dict[str, Any]], right: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if [row["row_id"] for row in left] != [row["row_id"] for row in right]:
        raise AssemblyError("paired rows are not aligned")
    both = sum(a["solved"] and b["solved"] for a, b in zip(left, right))
    left_only = sum(a["solved"] and not b["solved"] for a, b in zip(left, right))
    right_only = sum(not a["solved"] and b["solved"] for a, b in zip(left, right))
    neither = len(left) - both - left_only - right_only
    return {
        "both_solved": both,
        "left_only": left_only,
        "right_only": right_only,
        "neither": neither,
        "discordant_count": left_only + right_only,
        "exact_mcnemar_p_two_sided": round(
            exact_mcnemar_p(left_only, right_only), 12
        ),
    }


def _assert_close(left: float, right: float, label: str) -> None:
    if not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9):
        raise AssemblyError(f"{label} drifted: {left} != {right}")


def replay_public_baselines(
    expected_rows: Sequence[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    cases = build_cases("dev")
    policies = {
        "reactive_heuristic": resolve_policy("heuristic"),
        "bc_teacher": resolve_policy("teacher"),
        "tuned_constant_rule": resolve_policy("tuned"),
        "shipped_v3_ppo_onnx": resolve_policy(f"onnx:{DEFAULT_ONNX_PATH}"),
    }
    if [case.row_id for case in cases] != [row["row_id"] for row in expected_rows]:
        raise AssemblyError("development case ordering drifted")
    output = {planner_id: [] for planner_id in policies}
    for case, expected in zip(cases, expected_rows):
        if case.tape_seed != int(expected["tape_seed"]):
            raise AssemblyError(f"tape seed drifted for {case.row_id}")
        for planner_id, policy in policies.items():
            row = compact_probe_row(
                rollout(case, policy), str(expected["tape_sha256"])
            )
            output[planner_id].append(row)
    return output


def build_payload() -> tuple[dict[str, Any], str]:
    headroom = load_receipt(HEADROOM_PATH, EXPECTED_HEADROOM_SHA256)
    step3e = load_receipt(STEP3E_PATH, EXPECTED_STEP3E_SHA256)
    for label, receipt in (("headroom", headroom), ("step3e", step3e)):
        if (
            receipt.get("split") != "dev"
            or receipt.get("authorizing") is not False
            or receipt.get("final_split_used") is not False
        ):
            raise AssemblyError(f"{label} receipt is not development-only evidence")

    headroom_rows = headroom["rows"]
    step3e_rows = step3e["profiles"]["v3_equivalent"]["development_curve"][
        "active_actor_critic_1000000_transitions"
    ]["rows"]
    if len(headroom_rows) != 40 or len(step3e_rows) != 40:
        raise AssemblyError("source receipts must each contain exactly 40 rows")
    row_ids = [row["row_id"] for row in headroom_rows]
    if row_ids != [row["row_id"] for row in step3e_rows] or len(set(row_ids)) != 40:
        raise AssemblyError("source receipt rows are not the same ordered 40 cases")
    for headroom_row, step3e_row in zip(headroom_rows, step3e_rows):
        if (
            headroom_row["tape_seed"] != step3e_row["tape_seed"]
            or headroom_row["tape_sha256"] != step3e_row["tape_sha256"]
        ):
            raise AssemblyError(f"tape identity mismatch for {headroom_row['row_id']}")

    rows_by_planner = replay_public_baselines(headroom_rows)
    replayed_tuned = rows_by_planner["tuned_constant_rule"]
    for replayed, source in zip(replayed_tuned, headroom_rows):
        recorded = normalize_existing_result(source["planners"]["tuned_rule"])
        if (
            replayed["solved"] != recorded["solved"]
            or replayed["hard_violation_count"] != recorded["hard_violation_count"]
        ):
            raise AssemblyError(f"tuned-rule replay drifted for {replayed['row_id']}")
        _assert_close(
            replayed["resilience_auc"], recorded["resilience_auc"], "tuned AUC"
        )
        _assert_close(
            replayed["minimum_tail_margin"],
            recorded["minimum_tail_margin"],
            "tuned tail margin",
        )
        _assert_close(
            replayed["max_conservation_residual"],
            recorded["max_conservation_residual"],
            "tuned conservation residual",
        )

    def source_rows(planner_key: str) -> list[dict[str, Any]]:
        return [
            {
                "row_id": row["row_id"],
                "tape_seed": row["tape_seed"],
                "tape_sha256": row["tape_sha256"],
                **normalize_existing_result(row["planners"][planner_key]),
            }
            for row in headroom_rows
        ]

    rows_by_planner["bc_initialization"] = source_rows("bc_initialization")
    rows_by_planner["tuned_constant_rule"] = source_rows("tuned_rule")
    rows_by_planner["v4_ppo_1m"] = [
        {
            "row_id": row["row_id"],
            "tape_seed": row["tape_seed"],
            "tape_sha256": row["tape_sha256"],
            **normalize_existing_result(row),
        }
        for row in step3e_rows
    ]
    for horizon in (1, 3, 5):
        rows_by_planner[f"causal_mpc_k{horizon}"] = [
            {
                "row_id": row["row_id"],
                "tape_seed": row["tape_seed"],
                "tape_sha256": row["tape_sha256"],
                **normalize_existing_result(
                    headroom["mpc"]["case_budgets"][row["row_id"]][str(horizon)][
                        "result"
                    ]
                ),
            }
            for row in headroom_rows
        ]
    rows_by_planner["clairvoyant_cem_oracle"] = source_rows("oracle")

    planner_specs = (
        ("reactive_heuristic", "Reactive heuristic", False, True),
        ("bc_teacher", "BC teacher", False, True),
        (
            "tuned_constant_rule",
            "Tuned constant rule (mult=10.0, cap=0.50)",
            False,
            True,
        ),
        ("bc_initialization", "BC initialization", False, True),
        ("shipped_v3_ppo_onnx", "Shipped v3 PPO ONNX", False, True),
        ("v4_ppo_1m", "v4 PPO at 1M active transitions", False, True),
        ("causal_mpc_k1", "Causal MPC (k=1)", False, True),
        ("causal_mpc_k3", "Causal MPC (k=3)", False, True),
        ("causal_mpc_k5", "Causal MPC (k=5)", False, True),
        (
            "clairvoyant_cem_oracle",
            "Clairvoyant CEM oracle (privileged)",
            True,
            False,
        ),
    )
    planners = []
    for planner_id, label, privileged, submission_baseline in planner_specs:
        rows = rows_by_planner[planner_id]
        if [row["row_id"] for row in rows] != row_ids:
            raise AssemblyError(f"row alignment drifted for {planner_id}")
        planners.append(
            {
                "planner_id": planner_id,
                "label": label,
                "privileged": privileged,
                "submission_baseline": submission_baseline,
                **aggregate(rows),
                "rows": rows,
            }
        )

    v4_rows = rows_by_planner["v4_ppo_1m"]
    comparison_targets = (
        "tuned_constant_rule",
        "bc_teacher",
        "shipped_v3_ppo_onnx",
        "clairvoyant_cem_oracle",
    )
    comparisons = {
        f"v4_ppo_1m_vs_{target}": {
            "left": "v4_ppo_1m",
            "right": target,
            **paired(v4_rows, rows_by_planner[target]),
        }
        for target in comparison_targets
    }
    if any(
        planner["hard_violation_count"] != 0
        or planner["maximum_conservation_residual"] != 0.0
        for planner in planners
    ):
        raise AssemblyError("a planner violates the frozen physics invariants")

    markdown = render_markdown(planners, comparisons)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL_ID,
        "status": "development_baseline_table_nonauthorizing",
        "authorizing": False,
        "split": "dev",
        "final_split_used": False,
        "uses_final_split": False,
        "training_performed": False,
        "case_count": 40,
        "same_ordered_tapes": True,
        "oracle_disclosure": (
            "The CEM oracle sees the complete future tape. It is a privileged "
            "achievable-ceiling diagnostic, not a submission baseline."
        ),
        "source_receipts": {
            "headroom": {
                "path": HEADROOM_PATH.relative_to(ROOT).as_posix(),
                "sha256": EXPECTED_HEADROOM_SHA256,
            },
            "step3e": {
                "path": STEP3E_PATH.relative_to(ROOT).as_posix(),
                "sha256": EXPECTED_STEP3E_SHA256,
            },
        },
        "source_identity": {
            "git_commit_before_step": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "assembler_sha256": file_sha256(Path(__file__).resolve()),
            "shared_evidence_sha256": file_sha256(
                ROOT / "backend" / "app" / "shared_evidence.py"
            ),
            "evaluator_sha256": file_sha256(ROOT / "scripts" / "evaluate.py"),
            "scenarios_v3_sha256": file_sha256(
                ROOT / "backend" / "app" / "scenarios_v3.py"
            ),
            "simulator_core_sha256": file_sha256(
                ROOT / "backend" / "app" / "simulator_core.py"
            ),
            "simulator_v2_sha256": file_sha256(
                ROOT / "backend" / "app" / "simulator_v2.py"
            ),
            "simulator_v3_sha256": file_sha256(
                ROOT / "backend" / "app" / "simulator_v3.py"
            ),
            "shipped_v3_onnx_sha256": file_sha256(DEFAULT_ONNX_PATH),
        },
        "v4_ppo_evidence": {
            "policy_seed": int(step3e["config"]["policy_seed"]),
            "reward_profile": "v3_equivalent",
            "active_actor_critic_transitions": 1_000_000,
            "checkpoint_persisted": bool(
                step3e["diagnostic_checkpoints_persisted"]
            ),
            "resumable_from_receipt": bool(step3e["resumable_from_receipt"]),
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "planners": planners,
        "paired_comparisons": comparisons,
        "invariants": {
            "all_ten_planners_present": len(planners) == 10,
            "all_rows_development_only": all(
                row_id.startswith("v3_dev_") for row_id in row_ids
            ),
            "all_planners_share_ordered_tapes": True,
            "tuned_rule_replay_matches_headroom_receipt": True,
            "all_hard_violations_zero": True,
            "all_conservation_residuals_exactly_zero": True,
            "oracle_marked_privileged_and_not_submission_baseline": True,
        },
    }
    return payload, markdown


def render_markdown(
    planners: Sequence[dict[str, Any]], comparisons: dict[str, dict[str, Any]]
) -> str:
    lines = [
        "# Development baseline table",
        "",
        "All planners use the same 40 development tapes. This table is diagnostic and nonauthorizing; it does not use the final split.",
        "",
        "> **Oracle disclosure:** the CEM oracle is privileged and clairvoyant. It sees the complete future shock tape and is **not a submission baseline**.",
        "",
        "| Planner | Solved / 40 | Wilson 95% CI | Mean resilience AUC | Mean minimum tail margin | Hard violations | Max conservation residual |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for planner in planners:
        ci = planner["wilson_95_ci"]
        lines.append(
            "| {label} | {solved}/40 | [{lower:.3f}, {upper:.3f}] | {auc:.6f} | {margin:+.6f} | {hard} | {residual:.1e} |".format(
                label=planner["label"],
                solved=planner["solved_count"],
                lower=ci["lower"],
                upper=ci["upper"],
                auc=planner["mean_resilience_auc"],
                margin=planner["mean_minimum_tail_margin"],
                hard=planner["hard_violation_count"],
                residual=planner["maximum_conservation_residual"],
            )
        )
    lines.extend(
        [
            "",
            "## Paired exact McNemar comparisons",
            "",
            "The v4 PPO is the left planner in every comparison.",
            "",
            "| Pair | Both | v4 only | Other only | Neither | Exact two-sided p |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    labels = {
        "v4_ppo_1m_vs_tuned_constant_rule": "v4 PPO vs tuned rule",
        "v4_ppo_1m_vs_bc_teacher": "v4 PPO vs BC teacher",
        "v4_ppo_1m_vs_shipped_v3_ppo_onnx": "v4 PPO vs shipped v3 PPO",
        "v4_ppo_1m_vs_clairvoyant_cem_oracle": "v4 PPO vs clairvoyant oracle",
    }
    for key, label in labels.items():
        values = comparisons[key]
        lines.append(
            f"| {label} | {values['both_solved']} | {values['left_only']} | "
            f"{values['right_only']} | {values['neither']} | "
            f"{values['exact_mcnemar_p_two_sided']:.12g} |"
        )
    lines.extend(
        [
            "",
            "The receipt at `internal/developmental_runs/v4/step6-dev-baseline-table.json` contains the complete paired rows and source hashes.",
            "",
        ]
    )
    return "\n".join(lines)


def write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--developmental-nonauthorizing", action="store_true")
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.developmental_nonauthorizing:
        print(
            f"{TOOL_ID}: error: --developmental-nonauthorizing is required",
            file=sys.stderr,
        )
        return 2
    receipt = args.receipt.resolve()
    markdown_path = args.markdown.resolve()
    if receipt.exists() or markdown_path.exists():
        print(f"{TOOL_ID}: error: outputs are create-new", file=sys.stderr)
        return 2
    try:
        payload, markdown = build_payload()
        payload["markdown"] = {
            "path": markdown_path.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(markdown.encode("utf-8")).hexdigest(),
        }
        write_new(markdown_path, markdown.encode("utf-8"))
        try:
            write_new(
                receipt,
                (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
                    "utf-8"
                ),
            )
        except BaseException:
            markdown_path.unlink(missing_ok=True)
            raise
    except (AssemblyError, OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        print(f"{TOOL_ID}: error: {exc}", file=sys.stderr)
        return 2
    print(
        f"{TOOL_ID}: wrote {receipt.relative_to(ROOT)} and "
        f"{markdown_path.relative_to(ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
