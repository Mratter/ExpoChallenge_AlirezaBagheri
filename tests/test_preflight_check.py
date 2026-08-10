"""Focused tests for explicit policy selection in portable preflight."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import preflight_check


def test_preflight_requires_an_explicit_policy_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(preflight_check.POLICY_PATH_ENV, raising=False)

    with pytest.raises(RuntimeError, match="INNOVERSE_POLICY_PATH is required"):
        preflight_check._load_configured_policy()


def test_preflight_forwards_the_selected_path_and_optional_hash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    selected = tmp_path / "candidate.onnx"
    expected_hash = "a" * 64
    sentinel = object()
    received: dict[str, object] = {}

    def fake_load_policy(
        path: str | Path,
        expected_sha256: str | None = None,
    ) -> object:
        received.update(path=path, expected_sha256=expected_sha256)
        return sentinel

    monkeypatch.setenv(preflight_check.POLICY_PATH_ENV, str(selected))
    monkeypatch.setenv(preflight_check.POLICY_SHA256_ENV, expected_hash)
    monkeypatch.setattr(preflight_check, "load_policy", fake_load_policy)

    assert preflight_check._load_configured_policy() is sentinel
    assert received == {
        "path": str(selected),
        "expected_sha256": expected_hash,
    }
