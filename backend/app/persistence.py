from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Any

from backend.app.simulator_core import canonical_hash, canonical_json_bytes

RESULT_ID_PATTERN = re.compile(r"[0-9a-f]{64}")
_WRITE_LOCK = threading.Lock()


class PersistenceError(RuntimeError):
    pass


def default_state_directory() -> Path:
    explicit = os.environ.get("INNOVERSE_STATE_DIR")
    if explicit:
        return Path(explicit).expanduser().resolve()
    local = os.environ.get("LOCALAPPDATA")
    if local:
        return Path(local) / "Innoverse" / "ai17-city-recovery"
    return Path.home() / "AppData" / "Local" / "Innoverse" / "ai17-city-recovery"


def result_identity(result: dict[str, Any]) -> str:
    if result.get("engine_version") == "city-recovery-env-v3":
        return canonical_hash(
            {
                "schema_version": result["schema_version"],
                "engine_version": result["engine_version"],
                "engine_spec_sha256": result["engine_spec_sha256"],
                "outcome_definition_sha256": result[
                    "outcome_definition_sha256"
                ],
                "seed": result["seed"],
                "scenario": result["scenario"],
                "policy_sha256": result["policy"]["sha256"],
                "baseline_id": result["baseline_spec"]["id"],
                "baseline_version": result["baseline_spec"]["version"],
            }
        )
    # Preserve the shipped V2 identity formula byte-for-byte so old canonical
    # results remain independently readable after V3 becomes primary.
    return canonical_hash(
        {
            "schema_version": result["schema_version"],
            "seed": result["seed"],
            "scenario": result["scenario"],
            "policy_sha256": result["policy"]["sha256"],
            "baseline_id": result["baseline_spec"]["id"],
        }
    )


class RunStore:
    def __init__(self, root: Path | None = None):
        self.root = root or default_state_directory()
        self.runs = self.root / "runs"

    def save(self, result: dict[str, Any]) -> dict[str, Any]:
        persisted = dict(result)
        result_id = result_identity(persisted)
        persisted["result_id"] = result_id
        persisted["persistence"] = {
            "format": "canonical-json-v1",
            "idempotent": True,
            "result_id": result_id,
        }
        payload = canonical_json_bytes(persisted)
        destination = self.runs / f"{result_id}.json"
        temporary = self.runs / f".{result_id}.tmp"
        try:
            with _WRITE_LOCK:
                self.runs.mkdir(parents=True, exist_ok=True)
                temporary.write_bytes(payload)
                os.replace(temporary, destination)
        except OSError as exc:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise PersistenceError("result could not be written to local persistence") from exc
        return persisted

    def _validate_result_id(self, result_id: str) -> None:
        if not RESULT_ID_PATTERN.fullmatch(result_id):
            raise PersistenceError("result id must be 64 lowercase hexadecimal characters")

    def load(self, result_id: str) -> dict[str, Any]:
        self._validate_result_id(result_id)
        path = self.runs / f"{result_id}.json"
        try:
            payload = path.read_bytes()
            result = json.loads(payload.decode("utf-8"))
        except FileNotFoundError as exc:
            raise PersistenceError("persisted result was not found") from exc
        except (OSError, UnicodeError, ValueError) as exc:
            raise PersistenceError("persisted result is corrupt or unreadable") from exc
        if not isinstance(result, dict) or result.get("result_id") != result_id:
            raise PersistenceError("persisted result identity is invalid")
        if result_identity(result) != result_id:
            raise PersistenceError("persisted result content does not match its identity")
        if canonical_json_bytes(result) != payload:
            raise PersistenceError("persisted result is not canonical JSON")
        return result

    def list_summaries(
        self, *, engine_version: str | None = None
    ) -> list[dict[str, Any]]:
        if not self.runs.exists():
            return []
        try:
            paths = sorted(self.runs.glob("[0-9a-f]*.json"), key=lambda item: item.name)
        except OSError as exc:
            raise PersistenceError("persisted result index is unreadable") from exc
        summaries = []
        for path in paths:
            result = self.load(path.stem)
            stored_engine = result.get("engine_version")
            if not isinstance(stored_engine, str):
                environment = result.get("environment")
                environment_id = (
                    environment.get("id") if isinstance(environment, dict) else None
                )
                stored_engine = (
                    "city-recovery-env-v3"
                    if environment_id == "CityRecoveryEnv-v3"
                    else "city-recovery-env-v2"
                )
            if engine_version is not None and stored_engine != engine_version:
                continue
            comparison = result.get("comparison")
            candidate = result.get("candidate")
            baseline = result.get("baseline")
            if (
                not isinstance(comparison, dict)
                or not isinstance(candidate, dict)
                or not isinstance(baseline, dict)
            ):
                raise PersistenceError("persisted result summary contract is invalid")
            outcome = comparison.get("absolute_outcome_pair", comparison.get("outcome"))
            if not isinstance(outcome, str):
                raise PersistenceError("persisted result outcome is invalid")
            candidate_solved = comparison.get("candidate_solved")
            baseline_solved = comparison.get("baseline_solved")
            if candidate_solved is None:
                absolute = candidate.get("absolute_outcome")
                candidate_solved = (
                    absolute.get("solved") if isinstance(absolute, dict) else None
                )
            if baseline_solved is None:
                absolute = baseline.get("absolute_outcome")
                baseline_solved = (
                    absolute.get("solved") if isinstance(absolute, dict) else None
                )
            summaries.append(
                {
                    "result_id": result["result_id"],
                    "schema_version": result["schema_version"],
                    "engine_version": stored_engine,
                    "seed": result["seed"],
                    "scenario_name": result["scenario"]["name"],
                    "horizon_days": result["scenario"]["horizon_days"],
                    "primary_metric": comparison.get("primary_metric"),
                    "candidate_solved": candidate_solved,
                    "baseline_solved": baseline_solved,
                    "candidate_rauc": candidate["rauc"],
                    "baseline_rauc": baseline["rauc"],
                    "outcome": outcome,
                    "policy_sha256": result["policy"]["sha256"],
                }
            )
        return summaries
