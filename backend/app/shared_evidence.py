"""Shared evidence helpers for mutable v4 and development tooling.

The shipped v3 release cryptographically pins several modules that contain
older local copies of these helpers.  Those copies are intentional immutable
validation boundaries and must not import this module.  All non-frozen code
uses the implementations here instead.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, NoReturn, Sequence

__all__ = (
    "canonical_bytes",
    "canonical_hash",
    "file_sha256",
    "fsync_parent",
    "function_source_sha256",
    "load_json_object",
    "split_contract",
    "wilson_interval",
)


def _raise(error_type: type[Exception], message: str, cause: Exception | None = None) -> NoReturn:
    error = error_type(message)
    if cause is None:
        raise error
    raise error from cause


def canonical_bytes(value: Any) -> bytes:
    """Serialize JSON-compatible evidence using the repository canonical form."""

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_hash(value: Any) -> str:
    """Return the SHA-256 digest of canonical JSON evidence."""

    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(
    path: Path,
    *,
    label: str | None = None,
    error_type: type[Exception] = OSError,
) -> str:
    """Hash a file without loading the complete artifact into memory."""

    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        _raise(
            error_type,
            f"{label or str(path)} is missing or unreadable",
            exc,
        )
    return digest.hexdigest()


def load_json_object(
    path: Path,
    label: str,
    *,
    expected_sha256: str | None = None,
    error_type: type[Exception] = ValueError,
) -> dict[str, Any]:
    """Load a JSON object, optionally requiring an exact byte digest."""

    if expected_sha256 is not None:
        actual_sha256 = file_sha256(path, label=label, error_type=error_type)
        if actual_sha256 != expected_sha256:
            _raise(
                error_type,
                f"{label} hash mismatch: expected {expected_sha256}, got {actual_sha256}",
            )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        _raise(error_type, f"{label} is missing or invalid: {path}", exc)
    if not isinstance(value, dict):
        _raise(error_type, f"{label} root must be an object")
    return value


def wilson_interval(
    successes: int,
    total: int,
    *,
    digits: int = 8,
) -> list[float]:
    """Return the two-sided 95% Wilson score interval."""

    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("Wilson interval requires 0 <= successes <= total and total > 0")
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [
        round(max(0.0, center - margin), digits),
        round(min(1.0, center + margin), digits),
    ]


def split_contract(
    split_id: str,
    families: Sequence[Any],
    seeds: Sequence[int],
) -> dict[str, Any]:
    """Describe a deterministic family-by-seed Cartesian split."""

    seed_tuple = tuple(int(seed) for seed in seeds)
    if not seed_tuple:
        raise ValueError(f"{split_id} seeds are empty")
    if seed_tuple != tuple(range(seed_tuple[0], seed_tuple[-1] + 1)):
        raise ValueError(f"{split_id} seeds must be one contiguous closed interval")
    return {
        "id": split_id,
        "family_count": len(families),
        "family_ids": [family.id for family in families],
        "seed_interval": {
            "first": seed_tuple[0],
            "last": seed_tuple[-1],
            "count": len(seed_tuple),
        },
        "cartesian_case_count": len(families) * len(seed_tuple),
        "iteration_order": "family_order_then_ascending_seed",
    }


def function_source_sha256(
    root: Path,
    relative_path: str,
    function_name: str,
    *,
    error_type: type[Exception] = ValueError,
) -> str:
    """Hash one top-level function's normalized source text."""

    path = root / relative_path
    try:
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as exc:
        _raise(error_type, f"source is unreadable: {relative_path}", exc)
    node = next(
        (
            item
            for item in tree.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            and item.name == function_name
        ),
        None,
    )
    if node is None or node.end_lineno is None:
        _raise(error_type, f"bound function is missing: {relative_path}:{function_name}")
    lines = text.replace("\r\n", "\n").replace("\r", "\n").splitlines(keepends=True)
    payload = "".join(lines[node.lineno - 1 : node.end_lineno]).rstrip("\n") + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def fsync_parent(path: Path) -> None:
    """Best-effort fsync of the directory containing a durable write."""

    try:
        descriptor = os.open(path.parent, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)
