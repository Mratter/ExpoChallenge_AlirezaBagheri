"""Create deterministic, dependency-free recovery-plan CSV and PDF exports."""

from __future__ import annotations

import csv
import io
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np

from backend.app.city.outcome import CRITICAL_SERVICE_FLOOR
from backend.app.city.physics import SERVICES
from backend.app.shared_evidence import canonical_hash

PlannerName = Literal["candidate", "baseline"]
ExportFormat = Literal["csv", "pdf"]


class ExportError(RuntimeError):
    """Raised when persisted evidence cannot support a recovery-plan export."""


CSV_FIELDS = (
    "result_id",
    "planner",
    "planner_id",
    "seed",
    "scenario_name",
    "day",
    "service",
    "policy_sha256",
    "shock_schedule_sha256",
    "shock_type",
    "shock_severity",
    "shock_forced",
    "assessment_tail_active",
    "service_before",
    "service_after_shock",
    "service_target",
    "service_end",
    "material_allocation",
    "crew_allocation",
    "stock_release",
    "preparedness_investment",
    "preparedness_before",
    "preparedness_end",
    "resilience",
    "reward",
    "target_met_at_end",
    "below_critical_service_floor",
    "hard_violation_count",
    "conservation_residual",
)

_SPREADSHEET_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r", "\n")


def _planner_evidence(
    result: dict[str, Any], planner: PlannerName
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        summary = result[planner]
        trajectory = summary["trajectory"]
    except (KeyError, TypeError) as exc:
        raise ExportError(f"persisted {planner} trajectory is unavailable") from exc
    if not isinstance(summary, dict) or not isinstance(trajectory, list):
        raise ExportError(f"persisted {planner} trajectory is invalid")
    try:
        horizon = int(result["scenario"]["horizon_days"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ExportError("persisted scenario horizon is invalid") from exc
    if len(trajectory) != horizon:
        raise ExportError(f"persisted {planner} trajectory length is invalid")
    try:
        schedule = result["shock_schedule"]
        schedule_sha256 = result["shock_schedule_sha256"]
        trajectory_sha256 = summary["trajectory_sha256"]
    except (KeyError, TypeError) as exc:
        raise ExportError("persisted export hashes are incomplete") from exc
    if not isinstance(schedule, list) or len(schedule) != horizon:
        raise ExportError("persisted disaster tape is invalid")
    if canonical_hash(schedule) != schedule_sha256:
        raise ExportError("persisted disaster tape hash does not match")
    if canonical_hash(trajectory) != trajectory_sha256:
        raise ExportError(f"persisted {planner} trajectory hash does not match")
    try:
        if any(day["shock"] != schedule[index] for index, day in enumerate(trajectory)):
            raise ExportError(f"persisted {planner} trajectory does not match the disaster tape")
    except (KeyError, TypeError) as exc:
        raise ExportError(f"persisted {planner} shock evidence is incomplete") from exc
    return summary, trajectory


def _csv_value(value: Any) -> Any:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, str) and value.startswith(_SPREADSHEET_FORMULA_PREFIXES):
        return f"'{value}"
    return value


def recovery_plan_csv(result: dict[str, Any], planner: PlannerName) -> bytes:
    """Return one stable row for every day/service pair in a planner run."""

    summary, trajectory = _planner_evidence(result, planner)
    try:
        targets = result["scenario"]["recovery_targets"]
        common = {
            "result_id": result["result_id"],
            "planner": planner,
            "planner_id": summary["planner"],
            "seed": result["seed"],
            "scenario_name": result["scenario"]["name"],
            "policy_sha256": result["policy"]["sha256"],
            "shock_schedule_sha256": result["shock_schedule_sha256"],
        }
    except (KeyError, TypeError) as exc:
        raise ExportError("persisted export identity is incomplete") from exc
    if not isinstance(targets, list) or len(targets) != len(SERVICES):
        raise ExportError("persisted recovery targets are invalid")

    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output, fieldnames=CSV_FIELDS, lineterminator="\n", extrasaction="raise"
    )
    writer.writeheader()
    try:
        for day in trajectory:
            shock = day["shock"]
            residual = day["logistics"]["conservation_residual"]
            for index, service in enumerate(SERVICES):
                end = day["services_end"][index]
                row = {
                    **common,
                    "day": day["day"],
                    "service": service,
                    "shock_type": shock["type"],
                    "shock_severity": shock["severity"],
                    "shock_forced": shock["forced"],
                    "assessment_tail_active": shock["assessment_tail"],
                    "service_before": day["services_before"][index],
                    "service_after_shock": day["services_after_shock"][index],
                    "service_target": targets[index],
                    "service_end": end,
                    "material_allocation": day["material_allocation"][index],
                    "crew_allocation": day["crew_allocation"][index],
                    "stock_release": day["stock_release"][index],
                    "preparedness_investment": day["preparedness_investment"][
                        index
                    ],
                    "preparedness_before": day["preparedness_before"][index],
                    "preparedness_end": day["preparedness_end"][index],
                    "resilience": day["resilience"],
                    "reward": day["reward"],
                    "target_met_at_end": end >= targets[index],
                    "below_critical_service_floor": end < CRITICAL_SERVICE_FLOOR,
                    "hard_violation_count": day["hard_violation_count"],
                    "conservation_residual": residual[index],
                }
                writer.writerow({name: _csv_value(row[name]) for name in CSV_FIELDS})
    except (KeyError, IndexError, TypeError) as exc:
        raise ExportError(f"persisted {planner} day evidence is incomplete") from exc
    return output.getvalue().encode("utf-8")


def _pdf_escape(value: Any) -> str:
    text = str(value).encode("ascii", "replace").decode("ascii")
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _content_stream(lines: Sequence[str]) -> bytes:
    commands = ["BT", "/F1 8 Tf", "10 TL", "36 756 Td"]
    for line in lines:
        commands.append(f"({_pdf_escape(line)}) Tj")
        commands.append("T*")
    commands.append("ET")
    return ("\n".join(commands) + "\n").encode("ascii")


def _minimal_pdf(pages: Sequence[Sequence[str]]) -> bytes:
    """Build a small deterministic PDF 1.4 document using built-in Helvetica."""

    if not pages:
        raise ValueError("PDF requires at least one page")
    page_object_numbers = [4 + 2 * index for index in range(len(pages))]
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        (
            "<< /Type /Pages /Kids ["
            + " ".join(f"{number} 0 R" for number in page_object_numbers)
            + f"] /Count {len(pages)} >>"
        ).encode("ascii"),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for index, lines in enumerate(pages):
        page_number = page_object_numbers[index]
        content_number = page_number + 1
        objects.append(
            (
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_number} 0 R >>"
            ).encode("ascii")
        )
        stream = _content_stream(lines)
        objects.append(
            f"<< /Length {len(stream)} >>\nstream\n".encode("ascii")
            + stream
            + b"endstream"
        )

    payload = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(payload))
        payload.extend(f"{number} 0 obj\n".encode("ascii"))
        payload.extend(obj)
        payload.extend(b"\nendobj\n")
    xref_offset = len(payload)
    payload.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    payload.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        payload.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    payload.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    return bytes(payload)


def recovery_plan_pdf(result: dict[str, Any], planner: PlannerName) -> bytes:
    """Return a two-page headline, daily, and sector-endpoint briefing."""

    summary, trajectory = _planner_evidence(result, planner)
    try:
        outcome = summary["absolute_outcome"]
        targets = np.asarray(result["scenario"]["recovery_targets"], dtype=np.float64)
        initial = np.asarray(trajectory[0]["services_before"], dtype=np.float64)
        final = np.asarray(trajectory[-1]["services_end"], dtype=np.float64)
        tail = np.asarray(
            [day["services_end"] for day in trajectory[-3:]], dtype=np.float64
        )
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ExportError(f"persisted {planner} summary is incomplete") from exc
    if targets.shape != (5,) or initial.shape != (5,) or final.shape != (5,):
        raise ExportError("persisted service endpoint vectors are invalid")

    title = "CITY RECOVERY PLAN / 30-DAY EVIDENCE BRIEF"
    identity_lines = [
        title,
        f"Result: {result['result_id']}",
        f"Planner: {planner} ({summary['planner']})",
        f"Scenario: {result['scenario']['name']} | seed {result['seed']}",
        f"Policy SHA-256: {result['policy']['sha256']}",
        f"Tape SHA-256: {result['shock_schedule_sha256']}",
        "",
        (
            f"Headline: {'SOLVED' if outcome['solved'] else 'NOT SOLVED'} | "
            f"RAUC {summary['rauc']:.8f} | final resilience "
            f"{summary['final_resilience']:.8f} | minimum {summary['minimum_resilience']:.8f}"
        ),
        (
            f"Safety: hard violations {summary['hard_violation_count']} | "
            f"critical service-days {summary['critical_service_days']} | "
            f"max conservation residual {summary['max_logistics_conservation_residual']:.10f}"
        ),
        "",
        "DAY  SHOCK        SEVERITY  RESILIENCE  REWARD       MIN SERVICE  HARD",
    ]

    day_lines: list[str] = []
    for day in trajectory:
        shock_type = day["shock"]["type"] or "none"
        day_lines.append(
            f"{day['day']:>3}  {shock_type:<12} {day['shock']['severity']:>8.4f}  "
            f"{day['resilience']:>10.6f}  {day['reward']:>10.6f}  "
            f"{min(day['services_end']):>11.6f}  {day['hard_violation_count']:>4}"
        )

    endpoint_lines = [
        "",
        "SECTOR ENDPOINTS / initial -> final | target | three-day tail minimum",
    ]
    tail_minimum = np.min(tail, axis=0)
    for index, service in enumerate(SERVICES):
        endpoint_lines.append(
            f"{service:<16} {initial[index]:.6f} -> {final[index]:.6f} | "
            f"{targets[index]:.6f} | {tail_minimum[index]:.6f}"
        )
    endpoint_lines.extend(
        [
            "",
            "This export reports persisted evidence; it does not rerun or alter the plan.",
        ]
    )

    pages = [
        [*identity_lines, *day_lines[:15], "", "Continued on page 2."],
        [*identity_lines[:7], "DAILY EVIDENCE / days 16-30", "", *day_lines[15:], *endpoint_lines],
    ]
    return _minimal_pdf(pages)


def recovery_plan_export(
    result: dict[str, Any], planner: PlannerName, format: ExportFormat
) -> tuple[bytes, str, str]:
    """Return payload, media type, and a safe deterministic download filename."""

    stem = f"recovery-plan-{result['result_id'][:12]}-{planner}"
    if format == "csv":
        return recovery_plan_csv(result, planner), "text/csv; charset=utf-8", f"{stem}.csv"
    if format == "pdf":
        return recovery_plan_pdf(result, planner), "application/pdf", f"{stem}.pdf"
    raise ExportError("unsupported recovery-plan export format")


__all__ = (
    "CSV_FIELDS",
    "ExportError",
    "ExportFormat",
    "PlannerName",
    "recovery_plan_csv",
    "recovery_plan_export",
    "recovery_plan_pdf",
)
