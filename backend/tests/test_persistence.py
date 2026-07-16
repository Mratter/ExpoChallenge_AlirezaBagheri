from pathlib import Path

from backend.app.artifact import load_policy_bundle
from backend.app.models import Scenario
from backend.app.persistence import RunStore
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
