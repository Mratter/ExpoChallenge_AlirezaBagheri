"""Focused tests for bundled and explicit policy selection in preflight."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import preflight_check


def test_preflight_uses_the_bundled_policy_without_an_environment_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv(preflight_check.POLICY_PATH_ENV, raising=False)
    monkeypatch.delenv(preflight_check.POLICY_SHA256_ENV, raising=False)
    bundled = tmp_path / "artifacts" / "city_recovery_ppo.v4.onnx"
    sentinel = object()
    received: dict[str, object] = {}

    def fake_load_policy(
        path: str | Path,
        expected_sha256: str | None = None,
    ) -> object:
        received.update(path=path, expected_sha256=expected_sha256)
        return sentinel

    monkeypatch.setattr(preflight_check, "DEFAULT_POLICY_PATH", bundled)
    monkeypatch.setattr(preflight_check, "load_policy", fake_load_policy)

    assert preflight_check._load_configured_policy() is sentinel
    assert received == {"path": bundled, "expected_sha256": None}


def test_preflight_environment_path_and_hash_override_the_bundled_default(
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
    monkeypatch.setattr(
        preflight_check,
        "DEFAULT_POLICY_PATH",
        tmp_path / "artifacts" / "city_recovery_ppo.v4.onnx",
    )
    monkeypatch.setattr(preflight_check, "load_policy", fake_load_policy)

    assert preflight_check._load_configured_policy() is sentinel
    assert received == {
        "path": str(selected),
        "expected_sha256": expected_hash,
    }
