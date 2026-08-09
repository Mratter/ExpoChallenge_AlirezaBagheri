from __future__ import annotations

from typing import Any

CRITICAL_THRESHOLD = 0.12
STRAINED_THRESHOLD = 0.30
DISTRICT_DARK_STREAK = 3
FALL_STREAK = 4

SERVICE_LABELS = {
    "transport": "transport",
    "housing": "housing",
    "food": "food",
    "healthcare": "healthcare",
    "public_services": "public services",
}

SHOCK_LABELS = {
    "aftershock": "aftershock",
    "supply": "supply disruption",
    "epidemic": "epidemic",
    "utility": "utility failure",
    "weather": "weather event",
}


def _service_label(service: str) -> str:
    return SERVICE_LABELS.get(service, service)


def _shock_label(shock_type: str | None) -> str:
    if shock_type is None:
        return "no shock"
    return SHOCK_LABELS.get(shock_type, shock_type)


def daily_recommendations(
    trajectory: list[dict[str, Any]],
    shock_schedule: list[dict[str, Any]],
    priorities: list[float],
    services: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Produce a deterministic per-day recommendation for each day of a trajectory."""
    recommendations: list[dict[str, Any]] = []
    below_critical_streak = [0] * len(services)
    for index, day in enumerate(trajectory):
        services_end = day["services_end"]
        allocation = day["allocation"]
        shock = shock_schedule[index] if index < len(shock_schedule) else {}

        priority_scores = [
            (1.0 - services_end[i]) * priorities[i] for i in range(len(services))
        ]
        priority_index = max(range(len(services)), key=lambda i: priority_scores[i])
        priority_service = services[priority_index]

        risk_alerts: list[dict[str, Any]] = []
        for i, service in enumerate(services):
            if services_end[i] < CRITICAL_THRESHOLD:
                below_critical_streak[i] += 1
            else:
                below_critical_streak[i] = 0
            if services_end[i] < CRITICAL_THRESHOLD:
                risk_alerts.append({
                    "service": service,
                    "level": "critical",
                    "detail": f"{_service_label(service)} below {CRITICAL_THRESHOLD:.2f} recovery floor",
                })
            elif services_end[i] < STRAINED_THRESHOLD:
                risk_alerts.append({
                    "service": service,
                    "level": "strained",
                    "detail": f"{_service_label(service)} below {STRAINED_THRESHOLD:.2f} stability band",
                })
            if below_critical_streak[i] >= DISTRICT_DARK_STREAK - 1:
                risk_alerts.append({
                    "service": service,
                    "level": "district_dark",
                    "detail": (
                        f"{_service_label(service)} critical for "
                        f"{below_critical_streak[i]} consecutive day(s)"
                    ),
                })

        allocation_index = max(range(len(services)), key=lambda i: allocation[i])
        allocation_focus = services[allocation_index]
        allocation_share = allocation[allocation_index] / max(day["available_budget"], 1e-9)

        rationale = (
            f"Prioritize {_service_label(priority_service)}: lowest condition relative "
            f"to weight ({services_end[priority_index]:.2f} state, "
            f"{priorities[priority_index]:.1f} weight)."
        )
        if shock.get("type"):
            rationale = (
                f"{_shock_label(shock['type'])} shock on day {day['day']} "
                f"({shock['severity']:.2f} severity). {rationale}"
            )

        recommendations.append({
            "day": day["day"],
            "priority_service": priority_service,
            "priority_rationale": rationale,
            "risk_alerts": risk_alerts,
            "allocation_focus": allocation_focus,
            "allocation_focus_share": round(float(allocation_share), 8),
        })
    return recommendations


def run_recommendations(result: dict[str, Any], services: tuple[str, ...]) -> dict[str, Any]:
    """Produce a deterministic end-of-run recommendation summary from a comparison result."""
    candidate = result["candidate"]
    baseline = result["baseline"]
    comparison = result["comparison"]
    scenario = result["scenario"]
    shock_schedule = result["shock_schedule"]
    priorities = scenario["priorities"]

    outcome = comparison["outcome"]
    margin_pp = round(comparison["candidate_minus_baseline"] * 100, 2)

    if outcome == "candidate_higher_rauc":
        winner = "candidate"
        winner_label = "SB3 PPO / ONNX"
        winner_rationale = (
            f"The learned policy outperformed the conventional planner by "
            f"{margin_pp:.2f} resilience AUC percentage points."
        )
    elif outcome == "baseline_higher_rauc":
        winner = "baseline"
        winner_label = "OR-Tools GLOP"
        winner_rationale = (
            f"The conventional planner outperformed the learned policy by "
            f"{abs(margin_pp):.2f} resilience AUC percentage points."
        )
    else:
        winner = "tie"
        winner_label = "neither planner"
        winner_rationale = (
            "Both planners produced equivalent resilience AUC under this scenario."
        )

    candidate_traj = candidate["trajectory"]
    resilience_values = [day["resilience"] for day in candidate_traj]
    critical_day_index = min(
        range(len(resilience_values)), key=lambda i: resilience_values[i]
    )
    critical_day = candidate_traj[critical_day_index]
    critical_shock = shock_schedule[critical_day_index]
    critical_description = (
        f"Day {critical_day['day']} recorded the lowest resilience "
        f"({critical_day['resilience']:.2f})"
    )
    if critical_shock.get("type"):
        critical_description += (
            f" following a {_shock_label(critical_shock['type'])} shock "
            f"at {critical_shock['severity']:.2f} severity"
        )
    critical_description += "."

    fragile_counts = [0] * len(services)
    fragile_sum = [0.0] * len(services)
    for day in candidate_traj:
        for i, value in enumerate(day["services_end"]):
            if value < STRAINED_THRESHOLD:
                fragile_counts[i] += 1
                fragile_sum[i] += 1.0 - value
    fragile_index = max(
        range(len(services)),
        key=lambda i: fragile_counts[i] * 100 + fragile_sum[i],
    )
    most_fragile_service = services[fragile_index]
    fragile_days = fragile_counts[fragile_index]

    shock_loss_days: list[tuple[int, str, float]] = []
    for i, day in enumerate(candidate_traj):
        before = day["services_before"]
        after = day["services_after_shock"]
        loss = sum(before[j] - after[j] for j in range(len(services)))
        if loss > 0.01:
            shock = shock_schedule[i]
            shock_loss_days.append((day["day"], shock.get("type") or "ambient", float(loss)))
    shock_loss_days.sort(key=lambda item: item[2], reverse=True)
    worst_shock_type = shock_loss_days[0][1] if shock_loss_days else "ambient"

    actionable: list[str] = []
    actionable.append(
        f"Adopt the {winner_label} allocation strategy for this scenario family; "
        f"it yields the higher resilience trajectory."
    )
    if fragile_days > 0:
        actionable.append(
            f"Reinforce {_service_label(most_fragile_service)} capacity early: it spent "
            f"{fragile_days} day(s) below the {STRAINED_THRESHOLD:.2f} stability band."
        )
    if worst_shock_type != "ambient":
        actionable.append(
            f"Maintain reserve units for {_shock_label(worst_shock_type)} events: "
            f"they caused the largest single-day service losses."
        )
    if candidate["constraint_violations"] == 0 and baseline["constraint_violations"] == 0:
        actionable.append(
            "Both planners satisfied all hard constraints; the recommendation rests on "
            "resilience quality, not feasibility."
        )
    actionable.append(
        "Re-run with additional forced shocks to stress-test the recommendation before "
        "operational use; this is synthetic local evidence, not a real-city forecast."
    )

    strategy_summary = (
        f"Across {scenario['horizon_days']} days with {scenario['daily_budget']:.0f} daily units, "
        f"the {winner_label} strategy is recommended. "
        f"Resilience AUC: candidate {candidate['rauc']:.4f} vs baseline {baseline['rauc']:.4f}. "
        f"Most fragile service: {_service_label(most_fragile_service)}. "
        f"Critical moment: day {critical_day['day']}."
    )

    return {
        "winner": winner,
        "winner_label": winner_label,
        "winner_margin_pp": margin_pp,
        "winner_rationale": winner_rationale,
        "critical_moment": {
            "day": critical_day["day"],
            "resilience": critical_day["resilience"],
            "description": critical_description,
        },
        "most_fragile_service": most_fragile_service,
        "most_fragile_days_below_threshold": fragile_days,
        "worst_shock_type": worst_shock_type,
        "strategy_summary": strategy_summary,
        "actionable_recommendations": actionable,
        "daily": daily_recommendations(candidate_traj, shock_schedule, priorities, services),
    }
