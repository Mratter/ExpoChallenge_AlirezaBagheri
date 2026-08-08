"""Fail-closed loader for the model-focused workbench evidence bundle."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[2]
WORKBENCH_EVIDENCE_ROOT: Final = REPOSITORY_ROOT / "artifacts" / "workbench"
WORKBENCH_MANIFEST_PATH: Final = WORKBENCH_EVIDENCE_ROOT / "manifest.v1.json"
EXPECTED_WORKBENCH_MANIFEST_SHA256: Final = (
    "c8498925aae87b99041b89d36b42e3de5dbd70bb543783ece4d0432bdbaec529"
)


@dataclass(frozen=True)
class ShowcaseEvidenceSpec:
    benchmark_id: str
    model_id: str
    model_file_stem: str
    track_id: str
    expected_manifest_sha256: str
    expected_result_sha256: str
    expected_candidate_manifest_sha256: str
    parameter_count: int
    training_scenarios: int = 800
    training_windows: int = 9600
    manifest_path_override: Path | None = None
    onnx_path_override: Path | None = None

    @property
    def evidence_root(self) -> Path:
        return WORKBENCH_EVIDENCE_ROOT / "benchmarks" / self.benchmark_id

    @property
    def final_root(self) -> Path:
        return self.evidence_root / "final"

    @property
    def manifest_path(self) -> Path:
        return self.manifest_path_override or self.final_root / "manifest.json"

    @property
    def candidate_root(self) -> Path:
        return self.evidence_root / "candidate"

    @property
    def candidate_manifest_path(self) -> Path:
        return self.candidate_root / "candidate-manifest.json"

    @property
    def training_receipt_path(self) -> Path:
        return self.candidate_root / "training-receipt.json"

    @property
    def onnx_path(self) -> Path:
        return self.onnx_path_override or self.candidate_root / f"{self.model_file_stem}.onnx"

    @property
    def checkpoint_path(self) -> Path:
        return self.candidate_root / f"{self.model_file_stem}.pt"


ARCHIVED_SHOWCASE_V1_SPEC: Final = ShowcaseEvidenceSpec(
    benchmark_id="adaptive-cascades-showcase-v1",
    model_id="adaptive-cascade-mlp-v1",
    model_file_stem="adaptive-cascade-mlp-v1",
    track_id="showcase-adaptive-v1",
    expected_manifest_sha256=("55f7430d1561a4d387b78990328b9b59b67f3479944a80a60825c53206a7b6a4"),
    expected_result_sha256=("2b5eec6657c0bd21825bd7f9bccfe274f72d9ba28a5b2b69cc10950e2f2bb13f"),
    expected_candidate_manifest_sha256=(
        "4bcbefe2c1497635b46d87b791faae262b7a09bc914bce1726d6e583c5a6de90"
    ),
    parameter_count=5893,
)
ACTIVE_SHOWCASE_SPEC: Final = ShowcaseEvidenceSpec(
    benchmark_id="adaptive-cascades-showcase-v2",
    model_id="adaptive-cascade-mlp-v2-300k",
    model_file_stem="adaptive-cascade-mlp-v2-300k",
    track_id="showcase-adaptive-v2",
    expected_manifest_sha256=("02dcc47c2389690d90f4f846e2d07ecd5e9fceba935404771e4751e758883640"),
    expected_result_sha256=("a69fbb96087298abaec35bae2a2797cca24c696ff8ee1463913d6e1d84cd5a5b"),
    expected_candidate_manifest_sha256=(
        "e1181f4cc999f44f756468809d6ae44c357a81faaaf7fd2b2bce5b11cbdadac5"
    ),
    parameter_count=300113,
)

# Active aliases keep the minimal runtime/preflight contract compact.
SHOWCASE_BENCHMARK_ID: Final = ACTIVE_SHOWCASE_SPEC.benchmark_id
SHOWCASE_EVIDENCE_ROOT: Final = ACTIVE_SHOWCASE_SPEC.evidence_root
SHOWCASE_FINAL_ROOT: Final = ACTIVE_SHOWCASE_SPEC.final_root
SHOWCASE_MANIFEST_PATH: Final = ACTIVE_SHOWCASE_SPEC.manifest_path
SHOWCASE_CANDIDATE_ROOT: Final = ACTIVE_SHOWCASE_SPEC.candidate_root
SHOWCASE_CANDIDATE_MANIFEST_PATH: Final = ACTIVE_SHOWCASE_SPEC.candidate_manifest_path
SHOWCASE_TRAINING_RECEIPT_PATH: Final = ACTIVE_SHOWCASE_SPEC.training_receipt_path
SHOWCASE_ONNX_PATH: Final = ACTIVE_SHOWCASE_SPEC.onnx_path
SHOWCASE_CHECKPOINT_PATH: Final = ACTIVE_SHOWCASE_SPEC.checkpoint_path
EXPECTED_SHOWCASE_MANIFEST_SHA256: Final = ACTIVE_SHOWCASE_SPEC.expected_manifest_sha256
EXPECTED_SHOWCASE_RESULT_SHA256: Final = ACTIVE_SHOWCASE_SPEC.expected_result_sha256
EXPECTED_SHOWCASE_CANDIDATE_MANIFEST_SHA256: Final = (
    ACTIVE_SHOWCASE_SPEC.expected_candidate_manifest_sha256
)
WORKBENCH_SCHEMA_VERSION: Final = "model-workbench-v1"
WORKBENCH_MANIFEST_SCHEMA_VERSION: Final = "model-workbench-manifest-v1"
EXPECTED_TRACK_IDS: Final = (
    "production-v2",
    ACTIVE_SHOWCASE_SPEC.track_id,
    "scientific-v4",
    "pilot-r9",
    "architecture-r22-v10",
)
SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")


class WorkbenchEvidenceError(RuntimeError):
    """The checked-in workbench evidence is absent, malformed, or altered."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise WorkbenchEvidenceError("workbench evidence is not canonicalizable") from exc
    return _sha256_bytes(payload)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise WorkbenchEvidenceError(f"workbench evidence repeats JSON key {key!r}")
        document[key] = value
    return document


def _read_json_object(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise WorkbenchEvidenceError(f"{label} is missing or unsafe")
    try:
        payload = path.read_bytes()
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except WorkbenchEvidenceError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WorkbenchEvidenceError(f"{label} is unreadable JSON") from exc
    if not isinstance(value, dict):
        raise WorkbenchEvidenceError(f"{label} must contain a JSON object")
    _canonical_sha256(value)
    return value, payload


def _read_safe_bytes(path: Path, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise WorkbenchEvidenceError(f"{label} is missing or unsafe")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise WorkbenchEvidenceError(f"{label} is unreadable") from exc


def _required_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise WorkbenchEvidenceError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _required_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise WorkbenchEvidenceError(f"{label} must be an object")
    return value


def _required_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise WorkbenchEvidenceError(f"{label} must be a list")
    return value


def _required_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise WorkbenchEvidenceError(f"{label} must be an integer")
    return value


def _bound_manifest_path(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str):
        raise WorkbenchEvidenceError(f"{label} must be a relative POSIX path")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or relative.as_posix() != value
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise WorkbenchEvidenceError(f"{label} must be a safe relative POSIX path")

    candidate = root.joinpath(*relative.parts)
    try:
        resolved_root = root.resolve(strict=True)
        resolved_candidate = candidate.resolve(strict=True)
        resolved_candidate.relative_to(resolved_root)
    except (OSError, ValueError) as exc:
        raise WorkbenchEvidenceError(
            f"{label} escapes or is absent from its evidence root"
        ) from exc
    for parent in (candidate, *candidate.parents):
        if parent == root.parent:
            break
        if parent.is_symlink():
            raise WorkbenchEvidenceError(f"{label} traverses a symlink")
    return candidate


def _expected_showcase_paths() -> set[str]:
    paths = {
        "anti-gaming-report.json",
        "preregistration.json",
        "replay-report.json",
        "result.json",
        "source-seal.json",
        "split-commitment.json",
        "terminal.json",
    }
    paths.update(f"rows/{index:03d}.json" for index in range(40))
    return paths


def _verify_showcase_final_bundle(
    spec: ShowcaseEvidenceSpec | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    if spec is None:
        spec = ACTIVE_SHOWCASE_SPEC
    manifest, manifest_bytes = _read_json_object(
        spec.manifest_path,
        "showcase final manifest",
    )
    if _sha256_bytes(manifest_bytes) != spec.expected_manifest_sha256:
        raise WorkbenchEvidenceError("showcase final manifest digest differs")
    if manifest.get("schema_version") != "adaptive-cascade-final-manifest-v1":
        raise WorkbenchEvidenceError("showcase final manifest schema differs")
    if manifest.get("benchmark_id") != spec.benchmark_id:
        raise WorkbenchEvidenceError("showcase final manifest benchmark differs")

    files = _required_list(manifest.get("files"), "showcase final manifest files")
    if manifest.get("file_count") != len(files) or len(files) != 47:
        raise WorkbenchEvidenceError("showcase final manifest file count differs")

    records: dict[str, tuple[dict[str, Any], dict[str, Any], bytes]] = {}
    for raw_record in files:
        record = _required_object(raw_record, "showcase final manifest file record")
        path_value = record.get("path")
        path = _bound_manifest_path(
            spec.final_root,
            path_value,
            "showcase final manifest file path",
        )
        if not isinstance(path_value, str):
            raise WorkbenchEvidenceError("showcase final manifest file path is absent")
        if path_value in records:
            raise WorkbenchEvidenceError("showcase final manifest repeats a file path")
        document, payload = _read_json_object(path, f"showcase evidence {path_value}")
        if _sha256_bytes(payload) != _required_sha256(
            record.get("sha256"),
            f"showcase evidence {path_value} file sha256",
        ):
            raise WorkbenchEvidenceError(f"showcase evidence {path_value} file digest differs")
        if _canonical_sha256(document) != _required_sha256(
            record.get("semantic_sha256"),
            f"showcase evidence {path_value} semantic sha256",
        ):
            raise WorkbenchEvidenceError(f"showcase evidence {path_value} semantic digest differs")
        records[path_value] = (record, document, payload)

    if set(records) != _expected_showcase_paths():
        raise WorkbenchEvidenceError("showcase final manifest roster differs")

    result = records["result.json"][1]
    if _sha256_bytes(records["result.json"][2]) != spec.expected_result_sha256:
        raise WorkbenchEvidenceError("showcase result digest differs")
    if result.get("benchmark_id") != spec.benchmark_id:
        raise WorkbenchEvidenceError("showcase result benchmark differs")
    if result.get("evidence_status") != "measured":
        raise WorkbenchEvidenceError("showcase result is not measured evidence")
    scenario_total = _required_int(result.get("scenario_total"), "showcase scenario total")
    if scenario_total != 40:
        raise WorkbenchEvidenceError("showcase scenario total differs")

    objective = _required_object(result.get("objective"), "showcase objective")
    if objective.get("counts_are_independent_not_complementary") is not True:
        raise WorkbenchEvidenceError("showcase objective counts are ambiguously labeled")
    for policy_id in ("learned_policy", "static_heuristic"):
        policy = _required_object(objective.get(policy_id), f"showcase {policy_id}")
        if (
            _required_int(policy.get("passes"), f"showcase {policy_id} passes")
            + _required_int(policy.get("misses"), f"showcase {policy_id} misses")
            != scenario_total
        ):
            raise WorkbenchEvidenceError(f"showcase {policy_id} counts do not total 40")

    head_to_head = _required_object(result.get("head_to_head"), "showcase head to head")
    head_to_head_total = sum(
        _required_int(head_to_head.get(key), f"showcase {key}")
        for key in ("learned_wins", "heuristic_wins", "ties")
    )
    if head_to_head_total != scenario_total:
        raise WorkbenchEvidenceError("showcase head-to-head counts do not total 40")

    invariants = _required_object(result.get("invariants"), "showcase invariants")
    expected_invariants: dict[str, Any] = {
        "all_rows_present": True,
        "exact_replay": True,
        "hard_violations": 0,
        "same_tapes": True,
        "unique_scenario_ids": True,
    }
    if invariants != expected_invariants:
        raise WorkbenchEvidenceError("showcase result invariants do not pass")

    replay = records["replay-report.json"][1]
    replay_rows = _required_list(replay.get("rows"), "showcase replay rows")
    if replay.get("all_exact") is not True or len(replay_rows) != scenario_total:
        raise WorkbenchEvidenceError("showcase replay report is incomplete")
    scenario_ids: set[str] = set()
    replay_row_hashes: set[str] = set()
    for raw_row in replay_rows:
        row = _required_object(raw_row, "showcase replay row")
        scenario_id = row.get("scenario_id")
        if not isinstance(scenario_id, str) or not scenario_id or scenario_id in scenario_ids:
            raise WorkbenchEvidenceError("showcase replay scenario identifiers are invalid")
        scenario_ids.add(scenario_id)
        replay_row_hashes.add(_required_sha256(row.get("row_sha256"), "replay row sha256"))
        if row.get("learned_exact") is not True or row.get("heuristic_exact") is not True:
            raise WorkbenchEvidenceError("showcase replay row is not exact")
    manifest_row_hashes = {
        _required_sha256(records[f"rows/{index:03d}.json"][0].get("sha256"), "row sha256")
        for index in range(scenario_total)
    }
    if replay_row_hashes != manifest_row_hashes or len(replay_row_hashes) != scenario_total:
        raise WorkbenchEvidenceError("showcase replay rows differ from the final manifest")

    terminal = records["terminal.json"][1]
    if (
        terminal.get("status") != "verified_complete"
        or terminal.get("result_sha256") != spec.expected_result_sha256
        or terminal.get("replay_report_sha256") != records["replay-report.json"][0].get("sha256")
        or terminal.get("anti_gaming_report_sha256")
        != records["anti-gaming-report.json"][0].get("sha256")
    ):
        raise WorkbenchEvidenceError("showcase terminal receipt bindings differ")

    anti_gaming = records["anti-gaming-report.json"][1]
    expected_anti_gaming: dict[str, Any] = {
        "duplicate_scenario_ids": 0,
        "forbidden_fields_absent": True,
        "future_tape_access": False,
        "observation_size": 21,
        "policy_blind_same_state_action_transition": True,
        "row_exclusions": 0,
        "same_tape_for_both_policies": True,
        "seed_conditioned_policy_logic": False,
        "training_development_final_seeds_disjoint": True,
    }
    if any(anti_gaming.get(key) != value for key, value in expected_anti_gaming.items()):
        raise WorkbenchEvidenceError("showcase anti-gaming checks do not pass")

    artifact_hashes = {
        path: _required_sha256(record[0].get("sha256"), f"showcase {path} sha256")
        for path, record in records.items()
    }
    return result, records["preregistration.json"][1], artifact_hashes


def _verify_showcase_candidate(
    preregistration: dict[str, Any],
    spec: ShowcaseEvidenceSpec | None = None,
) -> dict[str, Any]:
    if spec is None:
        spec = ACTIVE_SHOWCASE_SPEC
    candidate_manifest, candidate_manifest_bytes = _read_json_object(
        spec.candidate_manifest_path,
        "showcase candidate manifest",
    )
    if _sha256_bytes(candidate_manifest_bytes) != spec.expected_candidate_manifest_sha256:
        raise WorkbenchEvidenceError("showcase candidate manifest digest differs")
    if candidate_manifest != preregistration.get("candidate_manifest"):
        raise WorkbenchEvidenceError("showcase candidate differs from preregistration")
    if (
        candidate_manifest.get("benchmark_id") != spec.benchmark_id
        or candidate_manifest.get("model_id") != spec.model_id
    ):
        raise WorkbenchEvidenceError("showcase candidate benchmark differs")

    receipt, receipt_bytes = _read_json_object(
        spec.training_receipt_path,
        "showcase training receipt",
    )
    if _sha256_bytes(receipt_bytes) != _required_sha256(
        candidate_manifest.get("training_receipt_sha256"),
        "showcase training receipt sha256",
    ):
        raise WorkbenchEvidenceError("showcase training receipt digest differs")
    if receipt.get("benchmark_id") != spec.benchmark_id:
        raise WorkbenchEvidenceError("showcase training receipt benchmark differs")

    onnx_bytes = _read_safe_bytes(spec.onnx_path, "showcase ONNX model")
    checkpoint_bytes = _read_safe_bytes(spec.checkpoint_path, "showcase checkpoint")
    onnx_sha256 = _required_sha256(candidate_manifest.get("onnx_sha256"), "showcase ONNX sha256")
    checkpoint_sha256 = _required_sha256(
        candidate_manifest.get("checkpoint_sha256"),
        "showcase checkpoint sha256",
    )
    if _sha256_bytes(onnx_bytes) != onnx_sha256 or receipt.get("onnx_sha256") != onnx_sha256:
        raise WorkbenchEvidenceError("showcase ONNX model digest differs")
    if (
        _sha256_bytes(checkpoint_bytes) != checkpoint_sha256
        or receipt.get("checkpoint_sha256") != checkpoint_sha256
    ):
        raise WorkbenchEvidenceError("showcase checkpoint digest differs")

    expected_receipt_values: dict[str, Any] = {
        "algorithm": "supervised_multiclass_imitation",
        "device": "cpu",
        "onnx_action_parity": True,
        "parameter_count": spec.parameter_count,
        "training_scenarios": spec.training_scenarios,
        "training_windows": spec.training_windows,
    }
    if any(receipt.get(key) != value for key, value in expected_receipt_values.items()):
        raise WorkbenchEvidenceError("showcase training receipt summary differs")
    return receipt


def _expected_showcase_benchmark(
    result: dict[str, Any],
    artifact_hashes: dict[str, str],
    spec: ShowcaseEvidenceSpec | None = None,
) -> dict[str, Any]:
    if spec is None:
        spec = ACTIVE_SHOWCASE_SPEC
    objective = _required_object(result.get("objective"), "showcase result objective")
    learned_policy = _required_object(
        objective.get("learned_policy"),
        "showcase result learned policy",
    )
    static_heuristic = _required_object(
        objective.get("static_heuristic"),
        "showcase result static heuristic",
    )
    head_to_head = copy.deepcopy(
        _required_object(result.get("head_to_head"), "showcase result head to head")
    )
    secondary = _required_object(result.get("secondary"), "showcase result secondary")
    limitations = copy.deepcopy(_required_list(result.get("limitations"), "showcase limitations"))
    limitations.append(
        "Version 1 remains preserved as immutable archived predecessor evidence; its rows "
        "and results are not reused as Version 2 evidence."
    )
    scenario_total = _required_int(result.get("scenario_total"), "showcase result scenario total")
    learned_passes = _required_int(learned_policy.get("passes"), "showcase learned passes")
    heuristic_passes = _required_int(
        static_heuristic.get("passes"),
        "showcase heuristic passes",
    )
    learned_wins = _required_int(head_to_head.get("learned_wins"), "showcase learned wins")
    heuristic_wins = _required_int(
        head_to_head.get("heuristic_wins"),
        "showcase heuristic wins",
    )
    ties = _required_int(head_to_head.get("ties"), "showcase ties")
    return {
        "status": "measured",
        "benchmark_id": result.get("benchmark_id"),
        "name": "Adaptive Cascades Synthetic Showcase v2",
        "evidence_class": result.get("evidence_class"),
        "model_track_id": spec.track_id,
        "scenario_total": scenario_total,
        "objective": {
            "label": "Contain at least 10 of 12 cascade windows",
            "definition": objective.get("definition"),
            "success_threshold": objective.get("success_threshold"),
            "learned_policy": {
                "label": "Adaptive Cascade MLP v2 (300k)",
                "passes": learned_passes,
                "misses": learned_policy.get("misses"),
            },
            "static_heuristic": {
                "label": "Static visible-need heuristic",
                "passes": static_heuristic.get("passes"),
                "misses": static_heuristic.get("misses"),
            },
            "counts_are_independent_not_complementary": objective.get(
                "counts_are_independent_not_complementary"
            ),
        },
        "head_to_head": head_to_head,
        "secondary": {
            "metric": {
                "id": secondary.get("metric"),
                "label": "Critical service deficit AUC",
                "direction": secondary.get("direction"),
            },
            "learned_mean": secondary.get("learned_mean"),
            "heuristic_mean": secondary.get("heuristic_mean"),
        },
        "synthetic_disclosure": result.get("synthetic_disclosure"),
        "limitations": limitations,
        "note": (
            f"The {learned_passes}/{scenario_total} learned-policy and "
            f"{heuristic_passes}/{scenario_total} static-heuristic figures are independent "
            "objective pass counts. The complementary matched head-to-head scoreline is "
            f"{learned_wins} learned wins, {heuristic_wins} heuristic wins, and {ties} ties."
        ),
        "provenance": [
            {
                "label": "Sealed Version 2 result",
                "source_repository": "city-model-workbench",
                "path": (f"artifacts/workbench/benchmarks/{spec.benchmark_id}/final/result.json"),
                "sha256": spec.expected_result_sha256,
            },
            {
                "label": "Version 2 write-once final manifest",
                "source_repository": "city-model-workbench",
                "path": (f"artifacts/workbench/benchmarks/{spec.benchmark_id}/final/manifest.json"),
                "sha256": spec.expected_manifest_sha256,
            },
            {
                "label": "Version 2 exact replay report",
                "source_repository": "city-model-workbench",
                "path": (
                    f"artifacts/workbench/benchmarks/{spec.benchmark_id}/final/replay-report.json"
                ),
                "sha256": artifact_hashes["replay-report.json"],
            },
            {
                "label": "Version 2 anti-gaming report",
                "source_repository": "city-model-workbench",
                "path": (
                    f"artifacts/workbench/benchmarks/{spec.benchmark_id}/final/"
                    "anti-gaming-report.json"
                ),
                "sha256": artifact_hashes["anti-gaming-report.json"],
            },
            {
                "label": "Archived Version 1 result",
                "source_repository": "city-model-workbench",
                "path": (
                    "artifacts/workbench/benchmarks/adaptive-cascades-showcase-v1/final/result.json"
                ),
                "sha256": ARCHIVED_SHOWCASE_V1_SPEC.expected_result_sha256,
            },
            {
                "label": "Archived Version 1 final manifest",
                "source_repository": "city-model-workbench",
                "path": (
                    "artifacts/workbench/benchmarks/adaptive-cascades-showcase-v1/final/"
                    "manifest.json"
                ),
                "sha256": ARCHIVED_SHOWCASE_V1_SPEC.expected_manifest_sha256,
            },
        ],
    }


def _validate_overview(document: dict[str, Any]) -> None:
    if document.get("schema_version") != WORKBENCH_SCHEMA_VERSION:
        raise WorkbenchEvidenceError("workbench overview schema version differs")
    _required_object(document.get("project"), "workbench project")
    _required_list(document.get("pipeline"), "workbench pipeline")
    benchmark = _required_object(document.get("benchmark"), "workbench benchmark")

    archived_result, archived_preregistration, _ = _verify_showcase_final_bundle(
        ARCHIVED_SHOWCASE_V1_SPEC
    )
    _verify_showcase_candidate(archived_preregistration, ARCHIVED_SHOWCASE_V1_SPEC)
    if archived_result.get("benchmark_id") != ARCHIVED_SHOWCASE_V1_SPEC.benchmark_id:
        raise WorkbenchEvidenceError("archived showcase Version 1 identity differs")

    showcase_result, preregistration, artifact_hashes = _verify_showcase_final_bundle()
    showcase_receipt = _verify_showcase_candidate(preregistration)
    if benchmark != _expected_showcase_benchmark(showcase_result, artifact_hashes):
        raise WorkbenchEvidenceError("workbench benchmark differs from sealed showcase result")

    tracks = _required_list(document.get("tracks"), "workbench tracks")
    track_ids = tuple(track.get("id") if isinstance(track, dict) else None for track in tracks)
    if track_ids != EXPECTED_TRACK_IDS:
        raise WorkbenchEvidenceError("workbench track roster or order differs")

    evidence_classes: set[str] = set()
    for track in tracks:
        typed_track = _required_object(track, "workbench track")
        evidence_class = typed_track.get("evidence_class")
        if not isinstance(evidence_class, str) or not evidence_class:
            raise WorkbenchEvidenceError("workbench track evidence class is absent")
        if evidence_class in evidence_classes:
            raise WorkbenchEvidenceError("workbench evidence classes must be distinct")
        evidence_classes.add(evidence_class)
        _required_object(typed_track.get("architecture"), "workbench architecture")
        training = _required_object(typed_track.get("training"), "workbench training")
        expected_unit = (
            "labeled windows"
            if typed_track["id"] == ACTIVE_SHOWCASE_SPEC.track_id
            else "transitions"
        )
        if training.get("unit") != expected_unit:
            raise WorkbenchEvidenceError(f"workbench {typed_track['id']} training unit differs")
        _required_object(typed_track.get("evaluation"), "workbench evaluation")
        _required_object(typed_track.get("safety"), "workbench safety")
        _required_list(typed_track.get("limitations"), "workbench limitations")
        provenance = _required_list(typed_track.get("provenance"), "workbench provenance")
        if not provenance:
            raise WorkbenchEvidenceError("every workbench track requires provenance")
        for source in provenance:
            typed_source = _required_object(source, "workbench provenance source")
            _required_sha256(typed_source.get("sha256"), "provenance sha256")

        if typed_track["id"] == ACTIVE_SHOWCASE_SPEC.track_id:
            architecture = _required_object(
                typed_track.get("architecture"),
                "showcase architecture",
            )
            if (
                typed_track.get("status") != "trained_evaluated"
                or typed_track.get("claim_eligible") is not True
                or evidence_class != showcase_result.get("evidence_class")
                or architecture.get("parameters") != showcase_receipt.get("parameter_count")
                or training.get("started") is not True
                or training.get("transitions") != showcase_receipt.get("training_windows")
                or training.get("hardware") != "CPU"
            ):
                raise WorkbenchEvidenceError("showcase track differs from sealed model evidence")
            expected_training_note = (
                f"{showcase_receipt['training_windows']:,} labeled windows from "
                f"{showcase_receipt['training_scenarios']:,} synthetic training scenarios; "
                f"{showcase_receipt['epochs']:,} supervised epochs completed in "
                f"{showcase_receipt['duration_seconds']:.2f} seconds."
            )
            if training.get("note") != expected_training_note:
                raise WorkbenchEvidenceError("showcase training summary differs from receipt")

            candidate_manifest = _required_object(
                preregistration.get("candidate_manifest"),
                "showcase preregistered candidate",
            )
            expected_provenance = {
                (
                    f"artifacts/workbench/benchmarks/{ACTIVE_SHOWCASE_SPEC.benchmark_id}/"
                    "final/result.json"
                ): ACTIVE_SHOWCASE_SPEC.expected_result_sha256,
                (
                    f"artifacts/workbench/benchmarks/{ACTIVE_SHOWCASE_SPEC.benchmark_id}/"
                    "final/manifest.json"
                ): ACTIVE_SHOWCASE_SPEC.expected_manifest_sha256,
                (
                    f"artifacts/workbench/benchmarks/{ACTIVE_SHOWCASE_SPEC.benchmark_id}/"
                    f"candidate/{ACTIVE_SHOWCASE_SPEC.model_file_stem}.onnx"
                ): candidate_manifest.get("onnx_sha256"),
                (
                    f"artifacts/workbench/benchmarks/{ACTIVE_SHOWCASE_SPEC.benchmark_id}/"
                    f"candidate/{ACTIVE_SHOWCASE_SPEC.model_file_stem}.pt"
                ): candidate_manifest.get("checkpoint_sha256"),
                (
                    f"artifacts/workbench/benchmarks/{ACTIVE_SHOWCASE_SPEC.benchmark_id}/"
                    "candidate/candidate-manifest.json"
                ): ACTIVE_SHOWCASE_SPEC.expected_candidate_manifest_sha256,
                (
                    f"artifacts/workbench/benchmarks/{ACTIVE_SHOWCASE_SPEC.benchmark_id}/"
                    "candidate/training-receipt.json"
                ): candidate_manifest.get("training_receipt_sha256"),
                (
                    "artifacts/workbench/benchmarks/adaptive-cascades-showcase-v1/final/result.json"
                ): ARCHIVED_SHOWCASE_V1_SPEC.expected_result_sha256,
                (
                    "artifacts/workbench/benchmarks/adaptive-cascades-showcase-v1/"
                    "final/manifest.json"
                ): ARCHIVED_SHOWCASE_V1_SPEC.expected_manifest_sha256,
            }
            actual_provenance = {
                source.get("path"): source.get("sha256")
                for source in provenance
                if isinstance(source, dict)
            }
            if actual_provenance != expected_provenance:
                raise WorkbenchEvidenceError(
                    "showcase track provenance differs from sealed evidence"
                )
            evaluation = _required_object(typed_track.get("evaluation"), "showcase evaluation")
            metrics = _required_list(evaluation.get("metrics"), "showcase evaluation metrics")
            metric_values = {
                metric.get("id"): metric.get("value")
                for metric in metrics
                if isinstance(metric, dict)
            }
            objective = _required_object(
                showcase_result.get("objective"),
                "showcase result objective",
            )
            learned_policy = _required_object(
                objective.get("learned_policy"),
                "showcase result learned policy",
            )
            static_heuristic = _required_object(
                objective.get("static_heuristic"),
                "showcase result static heuristic",
            )
            head_to_head = _required_object(
                showcase_result.get("head_to_head"),
                "showcase result head to head",
            )
            expected_limitations = copy.deepcopy(
                _required_list(showcase_result.get("limitations"), "showcase limitations")
            )
            expected_limitations.append(
                "Version 1 remains preserved as immutable archived predecessor evidence; its "
                "rows and results are not reused as Version 2 evidence."
            )
            if typed_track.get("limitations") != expected_limitations:
                raise WorkbenchEvidenceError("showcase track limitations differ")
            expected_metric_values = {
                "objective_learned_passes": learned_policy.get("passes"),
                "objective_static_heuristic_passes": static_heuristic.get("passes"),
                "head_to_head_learned_wins": head_to_head.get("learned_wins"),
                "head_to_head_heuristic_wins": head_to_head.get("heuristic_wins"),
                "head_to_head_ties": head_to_head.get("ties"),
                "paired_mean_difference": head_to_head.get("paired_mean_difference"),
            }
            if metric_values != expected_metric_values:
                raise WorkbenchEvidenceError("showcase track metrics differ from sealed result")
            safety = _required_object(typed_track.get("safety"), "showcase safety")
            if safety.get("hard_violations") != 0 or safety.get("replay_verified") is not True:
                raise WorkbenchEvidenceError("showcase track safety differs from sealed result")

        if typed_track["id"] == "architecture-r22-v10":
            if training.get("started") is not False or training.get("transitions") != 0:
                raise WorkbenchEvidenceError("R22 must remain explicitly untrained")
            if typed_track.get("claim_eligible") is not False:
                raise WorkbenchEvidenceError("R22 diagnostic must remain non-claim-eligible")
            if evidence_class != "privileged_developmental_reachability_diagnostic":
                raise WorkbenchEvidenceError("R22 diagnostic evidence class differs")


def load_workbench_overview() -> dict[str, Any]:
    """Load and verify the compact, checked-in workbench evidence document."""

    manifest, manifest_bytes = _read_json_object(
        WORKBENCH_MANIFEST_PATH,
        "workbench evidence manifest",
    )
    if _sha256_bytes(manifest_bytes) != EXPECTED_WORKBENCH_MANIFEST_SHA256:
        raise WorkbenchEvidenceError("workbench evidence manifest digest differs")
    if manifest.get("schema_version") != WORKBENCH_MANIFEST_SCHEMA_VERSION:
        raise WorkbenchEvidenceError("workbench evidence manifest schema differs")
    if manifest.get("artifact_path") != "overview.v1.json":
        raise WorkbenchEvidenceError("workbench evidence artifact path differs")

    artifact_path = WORKBENCH_MANIFEST_PATH.parent / "overview.v1.json"
    overview, overview_bytes = _read_json_object(artifact_path, "workbench overview")
    expected_file_sha256 = _required_sha256(
        manifest.get("artifact_sha256"),
        "workbench overview file sha256",
    )
    if _sha256_bytes(overview_bytes) != expected_file_sha256:
        raise WorkbenchEvidenceError("workbench overview file digest differs")
    expected_semantic_sha256 = _required_sha256(
        manifest.get("semantic_sha256"),
        "workbench overview semantic sha256",
    )
    if _canonical_sha256(overview) != expected_semantic_sha256:
        raise WorkbenchEvidenceError("workbench overview semantic digest differs")

    _validate_overview(overview)
    return copy.deepcopy(overview)


__all__ = [
    "EXPECTED_TRACK_IDS",
    "EXPECTED_WORKBENCH_MANIFEST_SHA256",
    "WORKBENCH_MANIFEST_PATH",
    "WorkbenchEvidenceError",
    "load_workbench_overview",
]
