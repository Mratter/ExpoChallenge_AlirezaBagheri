"""Behavior locks for the optional causal allocation optimizer."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend.app.city.optimizer import ortools_proposal
from backend.app.shared_evidence import canonical_hash

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class _Context:
    shocked: np.ndarray
    support: np.ndarray
    available_budget: float
    lower: np.ndarray
    upper: np.ndarray
    stock_ready: np.ndarray
    throughput: np.ndarray


def test_import_does_not_load_ortools() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, sys; "
                "import backend.app.city.optimizer; "
                "print(json.dumps(sorted(name for name in sys.modules "
                "if name == 'ortools' or name.startswith('ortools.'))))"
            ),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == []


def test_default_public_state_matches_optimizer_golden() -> None:
    context = _Context(
        shocked=np.array([0.34, 0.26, 0.41, 0.38, 0.30], dtype=np.float64),
        support=np.array(
            [0.69535, 0.701425, 0.6967, 0.6985, 0.705925],
            dtype=np.float64,
        ),
        available_budget=180.0,
        lower=np.array([0.0, 7.2, 0.0, 0.0, 0.0], dtype=np.float64),
        upper=np.full(5, 90.0, dtype=np.float64),
        stock_ready=np.array([136.0, 104.0, 164.0, 152.0, 120.0]),
        throughput=np.array([1.0, 0.604, 0.604, 0.604, 0.604]),
    )

    proposal, evidence = ortools_proposal(
        context,
        np.array([1.0, 1.1, 1.2, 1.4, 1.0], dtype=np.float64),
    )

    np.testing.assert_allclose(
        proposal,
        np.array([82.458, 11.982, 6.042, 76.386, 3.132]),
        rtol=0.0,
        atol=1e-12,
    )
    assert float(proposal.sum()) == 180.0
    assert evidence["baseline_id"] == "ortools-glop-visible-v2"
    assert evidence["baseline_version"] == "2.1.0"
    assert evidence["future_shocks_visible"] is False
    assert canonical_hash(
        {
            "proposal": [round(float(value), 8) for value in proposal],
            "evidence": evidence,
        }
    ) == "661fd7ef4983dd4f037e44543138bcd1a63c4cb177e28a9f926ae6256bd976ff"
