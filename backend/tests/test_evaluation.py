import hashlib
import json
from pathlib import Path

from backend.app.artifact import load_policy_bundle
from backend.app.preflight import validate_evaluation

ROOT = Path(__file__).resolve().parents[2]


def test_pytorch_onnx_parity_evidence_covers_pre_and_post_projector() -> None:
    bundle = load_policy_bundle()
    report = json.loads(
        (ROOT / "evaluation" / "policy_parity.v1.json").read_text(encoding="utf-8")
    )
    assert report["passed"] is True
    assert report["cases"] == 32
    assert report["max_action_abs_error"] <= report["action_tolerance"]
    assert report["max_pre_projector_proposal_abs_error"] >= 0
    assert report["max_projected_allocation_abs_error"] <= report[
        "projected_allocation_tolerance"
    ]
    assert report["onnx_sha256"] == bundle.onnx_sha256
    assert report["sb3_checkpoint_sha256"] == bundle.sb3_sha256


def test_preregistered_heldout_report_matches_frozen_policy_and_invariants() -> None:
    bundle = load_policy_bundle()
    report = validate_evaluation(bundle)
    protocol_payload = (ROOT / "evaluation" / "protocol.v1.json").read_bytes()
    assert report["protocol_sha256"] == hashlib.sha256(protocol_payload).hexdigest()
    assert report["evaluation_case_count"] == 40
    assert report["determinism"]["repeats_per_case"] == 3
    assert report["determinism"]["mismatches"] == 0
    assert report["resilience_case_counts"] == {
        "baseline_higher": 0,
        "candidate_higher": 40,
        "ties": 0,
    }
    assert report["aggregate"]["rauc"]["candidate_minus_baseline"] == 0.04729588
    assert report["synthetic_only"] is True
    assert all(
        value == 0
        for planner in report["violation_totals"].values()
        for value in planner.values()
    )
