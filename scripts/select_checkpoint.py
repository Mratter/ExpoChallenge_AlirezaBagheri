#!/usr/bin/env python3
"""Select a durable development checkpoint strictly by solved-case count."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.shared_evidence import file_sha256, fsync_parent  # noqa: E402
from scripts.training_artifacts import verify_checkpoint_bundle  # noqa: E402


class SelectionError(RuntimeError):
    """Raised when checkpoint-selection evidence is incomplete or inconsistent."""


def runtime_versions() -> dict[str, str]:
    """Record the environment that performed development selection."""

    packages = (
        "numpy",
        "torch",
        "stable-baselines3",
        "gymnasium",
        "onnx",
        "onnxruntime",
    )
    return {
        "python": platform.python_version(),
        "operating_system": platform.platform(),
        **{name: importlib.metadata.version(name) for name in packages},
    }


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SelectionError(f"{label} is missing or invalid: {path}") from exc
    if not isinstance(value, dict):
        raise SelectionError(f"{label} must be a JSON object")
    return value


def rank_candidates(candidates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank only by solves, then neutral deterministic tie-breakers."""

    if len(candidates) < 2:
        raise SelectionError("selection requires at least two checkpoints")
    required = {
        "policy_seed",
        "active_actor_critic_transitions",
        "development",
    }
    if any(not required <= set(candidate) for candidate in candidates):
        raise SelectionError("checkpoint candidate fields are incomplete")
    return sorted(
        (dict(candidate) for candidate in candidates),
        key=lambda candidate: (
            -int(candidate["development"]["solved_count"]),
            int(candidate["active_actor_critic_transitions"]),
            int(candidate["policy_seed"]),
        ),
    )


def _candidate(
    *,
    receipt_path: Path,
    receipt: dict[str, Any],
    milestone: int,
) -> dict[str, Any]:
    config = receipt.get("config", {})
    seed = config.get("policy_seed")
    curve_key = f"ppo_{milestone}_transitions"
    development = receipt.get("development_curve", {}).get(curve_key)
    reference = receipt.get("checkpoint_bundles", {}).get(str(milestone))
    if (
        not isinstance(seed, int)
        or not isinstance(development, dict)
        or not isinstance(reference, dict)
    ):
        raise SelectionError(
            f"training receipt lacks checkpoint evidence for {milestone}: {receipt_path}"
        )
    manifest_path = Path(reference.get("manifest_path", ""))
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    bundle_path = manifest_path.parent
    verified = verify_checkpoint_bundle(bundle_path)
    manifest = verified.manifest
    model_record = manifest["checkpoint"]["file"]
    normalization_record = manifest["normalization"]
    expected = {
        "model_sha256": model_record["sha256"],
        "normalization_sha256": normalization_record["file"]["sha256"],
        "obs_rms_sha256": normalization_record[
            "observation_rms_sha256"
        ],
    }
    if any(reference.get(key) != value for key, value in expected.items()):
        raise SelectionError("training receipt checkpoint reference hash mismatch")
    solved_count = development.get("solved_count")
    if not isinstance(solved_count, int) or development.get("case_count") != 40:
        raise SelectionError("checkpoint development evaluation is incomplete")
    checkpoint_id = str(reference.get("checkpoint_id", "")).strip()
    if not checkpoint_id:
        raise SelectionError("checkpoint id is missing")
    return {
        "id": checkpoint_id,
        "policy_seed": seed,
        "active_actor_critic_transitions": milestone,
        "training_receipt_path": str(receipt_path.resolve()),
        "training_receipt_sha256": file_sha256(receipt_path),
        "bundle_path": str(verified.root),
        "bundle_manifest_path": str(verified.manifest_path),
        "bundle_manifest_sha256": file_sha256(verified.manifest_path),
        "checkpoint_path": str(verified.model_path),
        "checkpoint_sha256": model_record["sha256"],
        "normalization_path": str(verified.normalization_path),
        "normalization_file_sha256": normalization_record["file"]["sha256"],
        "observation_rms_sha256": normalization_record[
            "observation_rms_sha256"
        ],
        "development": {
            "solved_count": solved_count,
            "solve_rate": development["solve_rate"],
            "mean_resilience_auc": development["mean_resilience_auc"],
            "mean_minimum_tail_margin": development[
                "mean_minimum_tail_margin"
            ],
        },
    }


def build_selection(seed_sweep_summary_path: Path) -> dict[str, Any]:
    """Validate every baseline checkpoint and build the selection payload."""

    summary = _load_object(seed_sweep_summary_path, "seed-sweep summary")
    if (
        summary.get("phase") != "seed_sweep"
        or summary.get("split") != "dev"
        or summary.get("final_split_used") is not False
        or len(summary.get("rows", [])) != 5
    ):
        raise SelectionError("seed-sweep summary contract drifted")
    candidates: list[dict[str, Any]] = []
    for summary_row in summary["rows"]:
        receipt_path = Path(summary_row["receipt"])
        if not receipt_path.is_absolute():
            receipt_path = ROOT / receipt_path
        receipt = _load_object(receipt_path, "training receipt")
        if (
            receipt.get("status") != "complete"
            or receipt.get("final_split_used") is not False
        ):
            raise SelectionError("training receipt is incomplete or used final split")
        milestones = receipt.get("config", {}).get("evaluation_milestones", [])
        for value in milestones:
            candidates.append(
                _candidate(
                    receipt_path=receipt_path,
                    receipt=receipt,
                    milestone=int(value),
                )
            )
    ranked = rank_candidates(candidates)
    winner, runner_up = ranked[:2]
    winner_result = dict(winner["development"])
    runner_result = dict(runner_up["development"])
    solved_margin = (
        winner_result["solved_count"] - runner_result["solved_count"]
    )
    if solved_margin:
        tie_break = {"used": False, "level": None}
    elif (
        winner["active_actor_critic_transitions"]
        != runner_up["active_actor_critic_transitions"]
    ):
        tie_break = {
            "used": True,
            "level": "earlier_active_actor_critic_transitions",
        }
    else:
        tie_break = {"used": True, "level": "lower_policy_seed"}
    return {
        "schema_version": "city-recovery-checkpoint-selection-v1",
        "tool": "select_checkpoint.py",
        "split": "dev",
        "final_split_used": False,
        "source_seed_sweep_summary": {
            "path": str(seed_sweep_summary_path.resolve()),
            "sha256": file_sha256(seed_sweep_summary_path),
        },
        "ranking": {
            "primary_metric": "solved_count",
            "resilience_auc_used_for_selection": False,
            "tie_break_order": [
                "earlier_active_actor_critic_transitions",
                "lower_policy_seed",
            ],
        },
        "candidate_count": len(ranked),
        "candidates": ranked,
        "selected_checkpoint": {
            "id": winner["id"],
            "path": winner["checkpoint_path"],
            "sha256": winner["checkpoint_sha256"],
            "policy_seed": winner["policy_seed"],
            "active_actor_critic_transitions": winner[
                "active_actor_critic_transitions"
            ],
            "normalization_path": winner["normalization_path"],
            "normalization_file_sha256": winner[
                "normalization_file_sha256"
            ],
            "observation_rms_sha256": winner["observation_rms_sha256"],
            "training_receipt_path": winner["training_receipt_path"],
            "training_receipt_sha256": winner["training_receipt_sha256"],
        },
        "winner": winner_result,
        "runner_up": runner_result,
        "margin": {
            "solved_cases": solved_margin,
            "percentage_points": 100.0
            * (winner_result["solve_rate"] - runner_result["solve_rate"]),
        },
        "tie_break": tie_break,
        "runtime_versions": runtime_versions(),
    }


def write_new_json(path: Path, payload: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    encoded = (
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if path.exists():
        raise SelectionError(f"refusing to overwrite selection receipt: {path}")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise SelectionError(
                f"refusing to overwrite selection receipt: {path}"
            )
        os.replace(temporary, path)
        fsync_parent(path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-sweep-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_selection(args.seed_sweep_summary.resolve())
    write_new_json(args.output, payload)
    print(
        json.dumps(
            {
                "selected_checkpoint": payload["selected_checkpoint"],
                "winner": payload["winner"],
                "runner_up": payload["runner_up"],
                "margin": payload["margin"],
                "tie_break": payload["tie_break"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SelectionError as error:
        raise SystemExit(f"error: {error}") from error
