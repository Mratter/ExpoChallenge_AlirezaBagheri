from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.artifact import load_policy  # noqa: E402
from backend.app.models import Scenario  # noqa: E402
from backend.app.simulator import compare  # noqa: E402


def main() -> None:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"Python 3.12 required, found {sys.version.split()[0]}")
    policy, checksum = load_policy()
    result = compare(Scenario(), 424242, policy, checksum)
    for planner_name in ("baseline", "candidate"):
        planner = result[planner_name]
        if len(planner["trajectory"]) != 14 or planner["constraint_violations"] != 0:
            raise RuntimeError(f"{planner_name} smoke trajectory failed")
        for day in planner["trajectory"]:
            if abs(sum(day["allocation"]) - day["available_budget"]) > 1e-7:
                raise RuntimeError(f"{planner_name} allocation sum failed on day {day['day']}")
    if result["shock_schedule"][4]["type"] != "utility":
        raise RuntimeError("forced fixture shock is missing")
    print(
        json.dumps(
            {
                "candidate_rauc": result["candidate"]["rauc"],
                "policy_sha256": checksum,
                "schedule_sha256": result["shock_schedule_sha256"],
                "status": "preflight-smoke-passed",
                "urgency_rauc": result["baseline"]["rauc"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

