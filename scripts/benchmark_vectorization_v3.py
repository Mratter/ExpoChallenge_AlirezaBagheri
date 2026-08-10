"""Training-only V4 vectorization benchmark; never evaluation evidence.

This tool compares Stable-Baselines3 ``DummyVecEnv`` with Windows-safe
``SubprocVecEnv(..., start_method="spawn")`` execution.  It deliberately imports
only the authored training split, applies the same fixed action tensor to both
vectorizers, and requires byte-identical observation/reward/done digests before
calling a multiprocessing result usable.

The report is a local engineering diagnostic.  It does not train, select,
evaluate, authorize, or publish a policy, and it never reads development/final
scenario constants or model artifacts.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import multiprocessing
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

# These limits must be applied before NumPy, Torch, or SB3 is imported.  A
# spawned worker imports this module afresh on Windows; leaving the common
# defaults at 12 would create up to 12 x 12 native worker threads.
for _thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import numpy as np  # noqa: E402
from stable_baselines3.common.vec_env import (  # noqa: E402
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Import only the training split by name.  Do not broaden this import to the
# development or final constants.
from backend.app.scenarios_v3 import (  # noqa: E402
    TRAINING_FAMILIES_V3,
    TRAINING_SEEDS_V3,
)
from backend.app.simulator_v4 import (  # noqa: E402
    ACTION_SIZE_V4,
    CyclingScenarioEnvV4,
)

ACTION_STREAM_SEED = 0xB3C7_0A11
DEFAULT_LANES = (4, 8, 12, 20)
DEFAULT_WARMUP_STEPS = 60
DEFAULT_MEASURED_STEPS = 600
DEFAULT_MEMORY_SAMPLE_STEPS = 60


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _training_scenarios() -> tuple[tuple[Any, int], ...]:
    scenarios = tuple(
        (family.build(seed), family.tape_seed(seed))
        for family in TRAINING_FAMILIES_V3
        for seed in TRAINING_SEEDS_V3
    )
    if not scenarios:
        raise RuntimeError("the V3 training split is empty")
    if any(not family.id.startswith("v3_train_") for family in TRAINING_FAMILIES_V3):
        raise RuntimeError("a non-training family entered the vectorization benchmark")
    return scenarios


@dataclass(frozen=True)
class _TrainingLaneFactory:
    """Pickle-safe factory that reconstructs one rotated training lane."""

    lane: int

    def __call__(self) -> CyclingScenarioEnvV4:
        scenarios = _training_scenarios()
        offset = self.lane % len(scenarios)
        rotated = list(scenarios[offset:] + scenarios[:offset])
        return CyclingScenarioEnvV4(rotated, collect_evidence=False)


def _make_factories(lanes: int) -> list[Callable[[], CyclingScenarioEnvV4]]:
    return [_TrainingLaneFactory(lane) for lane in range(lanes)]


def _fixed_actions(total_steps: int, lanes: int) -> np.ndarray:
    # A distinct deterministic stream is registered for each lane geometry;
    # the identical tensor is then replayed by both vectorizers.
    generator = np.random.Generator(
        np.random.PCG64(ACTION_STREAM_SEED ^ (lanes << 17) ^ total_steps)
    )
    return generator.uniform(
        -1.0,
        1.0,
        size=(total_steps, lanes, ACTION_SIZE_V4),
    ).astype(np.float32)


def _update_array_digest(digest: Any, value: Any) -> None:
    array = np.ascontiguousarray(np.asarray(value))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())


def _descendant_rss_windows(root_pid: int) -> tuple[int, int]:
    from ctypes import wintypes

    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * 260),
        ]

    class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel32.Process32FirstW.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(PROCESSENTRY32W),
    ]
    kernel32.Process32FirstW.restype = wintypes.BOOL
    kernel32.Process32NextW.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(PROCESSENTRY32W),
    ]
    kernel32.Process32NextW.restype = wintypes.BOOL
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

    snapshot = kernel32.CreateToolhelp32Snapshot(0x00000002, 0)
    invalid_handle = ctypes.c_void_p(-1).value
    if snapshot in (None, 0, invalid_handle):
        return 0, 0
    parents: dict[int, int] = {}
    try:
        entry = PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
        success = bool(kernel32.Process32FirstW(snapshot, ctypes.byref(entry)))
        while success:
            parents[int(entry.th32ProcessID)] = int(entry.th32ParentProcessID)
            success = bool(kernel32.Process32NextW(snapshot, ctypes.byref(entry)))
    finally:
        kernel32.CloseHandle(snapshot)

    process_ids = {root_pid}
    changed = True
    while changed:
        changed = False
        for process_id, parent_id in parents.items():
            if parent_id in process_ids and process_id not in process_ids:
                process_ids.add(process_id)
                changed = True

    total = 0
    readable = 0
    for process_id in process_ids:
        # PROCESS_QUERY_INFORMATION | PROCESS_VM_READ.  All benchmark workers
        # belong to this user, but a raced process exit is harmless.
        handle = kernel32.OpenProcess(0x0400 | 0x0010, False, process_id)
        if not handle:
            continue
        try:
            counters = PROCESS_MEMORY_COUNTERS_EX()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
            if psapi.GetProcessMemoryInfo(
                handle, ctypes.byref(counters), counters.cb
            ):
                total += int(counters.WorkingSetSize)
                readable += 1
        finally:
            kernel32.CloseHandle(handle)
    return total, readable


def _descendant_rss_procfs(root_pid: int) -> tuple[int, int]:
    proc = Path("/proc")
    if not proc.is_dir():
        return 0, 0
    parents: dict[int, int] = {}
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="utf-8")
            tail = stat[stat.rfind(")") + 2 :].split()
            parents[int(entry.name)] = int(tail[1])
        except (OSError, ValueError, IndexError):
            continue
    process_ids = {root_pid}
    changed = True
    while changed:
        changed = False
        for process_id, parent_id in parents.items():
            if parent_id in process_ids and process_id not in process_ids:
                process_ids.add(process_id)
                changed = True
    total = 0
    readable = 0
    for process_id in process_ids:
        try:
            status = (proc / str(process_id) / "status").read_text(encoding="utf-8")
            rss_line = next(line for line in status.splitlines() if line.startswith("VmRSS:"))
            total += int(rss_line.split()[1]) * 1024
            readable += 1
        except (OSError, StopIteration, ValueError, IndexError):
            continue
    return total, readable


def _process_tree_rss() -> tuple[int, int]:
    if os.name == "nt":
        return _descendant_rss_windows(os.getpid())
    return _descendant_rss_procfs(os.getpid())


def _total_physical_memory() -> int:
    if os.name == "nt":
        from ctypes import wintypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", wintypes.DWORD),
                ("dwMemoryLoad", wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys)
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return 0


@dataclass(frozen=True)
class BenchmarkResult:
    mode: str
    lanes: int
    warmup_steps: int
    measured_steps: int
    measured_transitions: int
    startup_seconds: float
    measured_wall_seconds: float
    monitoring_seconds: float
    effective_seconds: float
    transitions_per_second: float
    rss_after_reset_mib: float
    peak_process_tree_rss_mib: float
    peak_process_count: int
    observation_sha256: str
    reward_sha256: str
    done_sha256: str
    terminal_observation_sha256: str
    terminal_count: int
    hard_violation_count: int
    max_conservation_residual: float


def _build_environment(mode: str, lanes: int) -> VecEnv:
    factories = _make_factories(lanes)
    if mode == "dummy":
        return DummyVecEnv(factories)
    if mode == "subproc_spawn":
        return SubprocVecEnv(factories, start_method="spawn")
    raise ValueError(f"unknown vectorization mode: {mode}")


def _run_one(
    mode: str,
    lanes: int,
    actions: np.ndarray,
    warmup_steps: int,
    measured_steps: int,
    memory_sample_steps: int,
) -> BenchmarkResult:
    observation_digest = hashlib.sha256()
    reward_digest = hashlib.sha256()
    done_digest = hashlib.sha256()
    terminal_digest = hashlib.sha256()
    terminal_count = 0
    hard_violations = 0
    max_residual = 0.0
    peak_rss = 0
    peak_process_count = 0
    monitoring_seconds = 0.0
    monitoring_at_measurement_start = 0.0
    monitoring_at_measurement_end = 0.0
    environment: VecEnv | None = None

    def sample_memory() -> None:
        nonlocal peak_rss, peak_process_count, monitoring_seconds
        started = time.perf_counter()
        rss, process_count = _process_tree_rss()
        monitoring_seconds += time.perf_counter() - started
        peak_rss = max(peak_rss, rss)
        peak_process_count = max(peak_process_count, process_count)

    startup_started = time.perf_counter()
    try:
        environment = _build_environment(mode, lanes)
        observation = environment.reset()
        startup_seconds = time.perf_counter() - startup_started
        sample_memory()
        rss_after_reset = peak_rss
        _update_array_digest(observation_digest, observation)

        total_steps = warmup_steps + measured_steps
        measured_started = 0.0
        measured_finished = 0.0
        for step in range(total_steps):
            if step == warmup_steps:
                gc.collect()
                sample_memory()
                monitoring_at_measurement_start = monitoring_seconds
                measured_started = time.perf_counter()
            observation, rewards, dones, infos = environment.step(actions[step])
            _update_array_digest(observation_digest, observation)
            # DummyVecEnv stores rewards in a float32 buffer while SubprocVecEnv
            # stacks the Python floats it receives as float64.  PPO consumes
            # both through its float32 rollout buffer, so normalize to that
            # actual learner contract before comparing execution modes.
            _update_array_digest(reward_digest, np.asarray(rewards, dtype=np.float32))
            _update_array_digest(done_digest, np.asarray(dones, dtype=np.bool_))
            for lane, info in enumerate(infos):
                terminal = info.get("terminal_observation")
                terminal_digest.update(bytes([1 if terminal is not None else 0]))
                terminal_digest.update(np.asarray([lane], dtype=np.int16).tobytes())
                if terminal is not None:
                    _update_array_digest(terminal_digest, terminal)
                    terminal_count += 1
                day = info.get("day")
                if isinstance(day, dict):
                    hard_violations += int(day.get("hard_violation_count", 0))
                    logistics = day.get("logistics")
                    if isinstance(logistics, dict):
                        residual = logistics.get("conservation_residual", ())
                        if residual:
                            max_residual = max(
                                max_residual,
                                max(abs(float(value)) for value in residual),
                            )
            if (
                step >= warmup_steps
                and (step - warmup_steps + 1) % memory_sample_steps == 0
            ):
                sample_memory()
        measured_finished = time.perf_counter()
        monitoring_at_measurement_end = monitoring_seconds
        sample_memory()
    finally:
        if environment is not None:
            environment.close()

    measured_wall = measured_finished - measured_started
    measured_monitoring = (
        monitoring_at_measurement_end - monitoring_at_measurement_start
    )
    effective_seconds = max(measured_wall - measured_monitoring, 1e-9)
    measured_transitions = measured_steps * lanes
    return BenchmarkResult(
        mode=mode,
        lanes=lanes,
        warmup_steps=warmup_steps,
        measured_steps=measured_steps,
        measured_transitions=measured_transitions,
        startup_seconds=round(startup_seconds, 6),
        measured_wall_seconds=round(measured_wall, 6),
        monitoring_seconds=round(measured_monitoring, 6),
        effective_seconds=round(effective_seconds, 6),
        transitions_per_second=round(measured_transitions / effective_seconds, 3),
        rss_after_reset_mib=round(rss_after_reset / (1024 * 1024), 3),
        peak_process_tree_rss_mib=round(peak_rss / (1024 * 1024), 3),
        peak_process_count=peak_process_count,
        observation_sha256=observation_digest.hexdigest(),
        reward_sha256=reward_digest.hexdigest(),
        done_sha256=done_digest.hexdigest(),
        terminal_observation_sha256=terminal_digest.hexdigest(),
        terminal_count=terminal_count,
        hard_violation_count=hard_violations,
        max_conservation_residual=round(max_residual, 12),
    )


def _digest_match(left: BenchmarkResult, right: BenchmarkResult) -> bool:
    return (
        left.observation_sha256 == right.observation_sha256
        and left.reward_sha256 == right.reward_sha256
        and left.done_sha256 == right.done_sha256
        and left.terminal_observation_sha256 == right.terminal_observation_sha256
        and left.terminal_count == right.terminal_count
        and left.hard_violation_count == right.hard_violation_count
        and left.max_conservation_residual == right.max_conservation_residual
    )


def _recommend(
    results: list[BenchmarkResult],
    total_memory: int,
    safe_fraction: float,
    min_parallel_speedup: float,
) -> dict[str, Any]:
    by_key = {(result.lanes, result.mode): result for result in results}
    comparisons: list[dict[str, Any]] = []
    safe_results: list[BenchmarkResult] = []
    memory_limit_mib = total_memory * safe_fraction / (1024 * 1024) if total_memory else 0.0
    for lanes in sorted({result.lanes for result in results}):
        dummy = by_key[(lanes, "dummy")]
        subproc = by_key[(lanes, "subproc_spawn")]
        digests_match = _digest_match(dummy, subproc)
        subproc_memory_safe = not memory_limit_mib or (
            subproc.peak_process_tree_rss_mib <= memory_limit_mib
        )
        speedup = subproc.transitions_per_second / max(
            dummy.transitions_per_second, 1e-9
        )
        comparisons.append(
            {
                "lanes": lanes,
                "digests_match": digests_match,
                "subproc_spawn_speedup_over_dummy": round(speedup, 4),
                "subproc_spawn_memory_safe": subproc_memory_safe,
            }
        )
        safe_results.append(dummy)
        if digests_match and subproc_memory_safe:
            safe_results.append(subproc)

    fastest = max(safe_results, key=lambda item: item.transitions_per_second)
    target_lanes = 20
    target_dummy = by_key.get((target_lanes, "dummy"))
    target_subproc = by_key.get((target_lanes, "subproc_spawn"))
    registered_recommendation: dict[str, Any] | None = None
    if target_dummy is not None and target_subproc is not None:
        exact = _digest_match(target_dummy, target_subproc)
        memory_safe = not memory_limit_mib or (
            target_subproc.peak_process_tree_rss_mib <= memory_limit_mib
        )
        speedup = target_subproc.transitions_per_second / max(
            target_dummy.transitions_per_second, 1e-9
        )
        use_subproc = exact and memory_safe and speedup >= min_parallel_speedup
        registered_recommendation = {
            "lanes": target_lanes,
            "mode": "subproc_spawn" if use_subproc else "dummy",
            "reason": (
                f"spawn passed exact digests and achieved {speedup:.3f}x speedup"
                if use_subproc
                else (
                    "keep DummyVecEnv: spawn failed exact digest or memory checks"
                    if not exact or not memory_safe
                    else (
                        f"keep DummyVecEnv: spawn speedup {speedup:.3f}x is below "
                        f"the {min_parallel_speedup:.3f}x adoption floor"
                    )
                )
            ),
        }
    return {
        "rss_safety_fraction_of_physical_memory": safe_fraction,
        "rss_safety_limit_mib": round(memory_limit_mib, 3),
        "minimum_parallel_speedup_for_adoption": min_parallel_speedup,
        "comparisons": comparisons,
        "fastest_safe_measured_configuration": {
            "mode": fastest.mode,
            "lanes": fastest.lanes,
            "transitions_per_second": fastest.transitions_per_second,
            "note": "changing lane count changes training geometry and requires a new seal",
        },
        "v4_20_lane_recommendation": registered_recommendation,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lanes",
        nargs="+",
        type=int,
        default=list(DEFAULT_LANES),
        help="lane counts to compare (default: 4 8 12 20)",
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS
    )
    parser.add_argument("--steps", type=int, default=DEFAULT_MEASURED_STEPS)
    parser.add_argument(
        "--memory-sample-steps",
        type=int,
        default=DEFAULT_MEMORY_SAMPLE_STEPS,
    )
    parser.add_argument("--rss-safe-fraction", type=float, default=0.80)
    parser.add_argument("--parallel-min-speedup", type=float, default=1.10)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="optional diagnostic JSON path; no file is written by default",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    lanes = tuple(dict.fromkeys(args.lanes))
    if any(value <= 0 for value in lanes):
        raise ValueError("all lane counts must be positive")
    if args.warmup_steps < 0 or args.steps <= 0 or args.memory_sample_steps <= 0:
        raise ValueError("warmup must be nonnegative; step counts must be positive")
    if not 0.0 < args.rss_safe_fraction <= 1.0:
        raise ValueError("--rss-safe-fraction must be in (0, 1]")
    if args.parallel_min_speedup <= 0.0:
        raise ValueError("--parallel-min-speedup must be positive")

    total_memory = _total_physical_memory()
    results: list[BenchmarkResult] = []
    for lane_count in lanes:
        actions = _fixed_actions(
            args.warmup_steps + args.steps,
            lane_count,
        )
        for mode in ("dummy", "subproc_spawn"):
            result = _run_one(
                mode,
                lane_count,
                actions,
                args.warmup_steps,
                args.steps,
                args.memory_sample_steps,
            )
            results.append(result)
            print(
                f"{mode:13s} lanes={lane_count:2d} "
                f"tps={result.transitions_per_second:9.2f} "
                f"rss={result.peak_process_tree_rss_mib:8.1f} MiB "
                f"obs={result.observation_sha256[:12]}",
                flush=True,
            )

    report = {
        "schema_version": 1,
        "status": "developmental_vectorization_diagnostic_nonauthorizing",
        "scope": "training_split_only",
        "uses_development_split": False,
        "uses_final_split": False,
        "trains_or_selects_policy": False,
        "collect_evidence": False,
        "thread_caps": {
            name: os.environ[name]
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "action_stream_seed": ACTION_STREAM_SEED,
        "training_family_ids": [family.id for family in TRAINING_FAMILIES_V3],
        "training_seed_count": len(TRAINING_SEEDS_V3),
        "physical_memory_mib": round(total_memory / (1024 * 1024), 3),
        "source_hashes": {
            "benchmark_script_sha256": _sha256_file(Path(__file__)),
            "simulator_v4_sha256": _sha256_file(
                ROOT / "backend" / "app" / "simulator_v4.py"
            ),
            "simulator_core_v4_sha256": _sha256_file(
                ROOT / "backend" / "app" / "simulator_core_v4.py"
            ),
            "scenarios_v3_sha256": _sha256_file(
                ROOT / "backend" / "app" / "scenarios_v3.py"
            ),
        },
        "results": [asdict(result) for result in results],
        "recommendation": _recommend(
            results,
            total_memory,
            args.rss_safe_fraction,
            args.parallel_min_speedup,
        ),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json_output is not None:
        output = args.json_output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(rendered + "\n", encoding="utf-8", newline="\n")
        os.replace(temporary, output)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
