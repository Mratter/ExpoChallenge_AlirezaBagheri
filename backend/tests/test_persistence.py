from copy import deepcopy
from pathlib import Path

from backend.app.artifact import load_policy_bundle
from backend.app.models import Scenario
from backend.app.persistence import RunStore, result_identity
from backend.app.simulator import canonical_json_bytes, compare


def test_result_identity_is_idempotent_and_restores_across_store_instances(
    tmp_path: Path,
) -> None:
    result = compare(Scenario(name="Persisted authored scenario"), 314159, load_policy_bundle())
    first = RunStore(tmp_path).save(result)
    second = RunStore(tmp_path).save(result)
    restored = RunStore(tmp_path).load(first["result_id"])

    assert first["result_id"] == second["result_id"]
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert canonical_json_bytes(first) == canonical_json_bytes(restored)
    assert RunStore(tmp_path).list_summaries() == [
        {
            "baseline_rauc": first["baseline"]["rauc"],
            "candidate_rauc": first["candidate"]["rauc"],
            "horizon_days": first["scenario"]["horizon_days"],
            "outcome": first["comparison"]["outcome"],
            "policy_sha256": first["policy"]["sha256"],
            "result_id": first["result_id"],
            "scenario_name": "Persisted authored scenario",
            "seed": 314159,
        }
    ]


def test_v2_result_without_forced_shocks_restores_exact_canonical_bytes(
    tmp_path: Path,
) -> None:
    legacy_result = deepcopy(
        compare(Scenario(name="Legacy v2 persisted scenario"), 271828, load_policy_bundle())
    )
    legacy_result["schema_version"] = "2.0.0"
    assert legacy_result["scenario"].pop("forced_shocks") == []

    result_id = result_identity(legacy_result)
    legacy_result["result_id"] = result_id
    legacy_result["persistence"] = {
        "format": "canonical-json-v1",
        "idempotent": True,
        "result_id": result_id,
    }
    legacy_bytes = canonical_json_bytes(legacy_result)
    persisted_path = tmp_path / "runs" / f"{result_id}.json"
    persisted_path.parent.mkdir(parents=True)
    persisted_path.write_bytes(legacy_bytes)

    restored = RunStore(tmp_path).load(result_id)

    assert restored["schema_version"] == "2.0.0"
    assert "forced_shocks" not in restored["scenario"]
    assert canonical_json_bytes(restored) == legacy_bytes
    assert persisted_path.read_bytes() == legacy_bytes
