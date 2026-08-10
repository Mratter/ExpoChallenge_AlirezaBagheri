"""Compatibility exports for the consolidated city physics module."""

from backend.app.city.physics import (
    BASE_OBSERVATION_ORDER,
    CONSTRAINT_TOLERANCE,
    DELTA,
    DEPENDENCIES,
    ETA,
    SERVICES,
    SHOCK_BUDGET_FACTORS,
    SHOCK_IMPACTS,
    SHOCK_TYPE_PROBABILITIES,
    SHOCKS,
    action_to_proposal,
    measure_constraints,
    project_capped_simplex,
    round_vector as _round_vector,
)
from backend.app.shared_evidence import (
    canonical_bytes as canonical_json_bytes,
    canonical_hash,
)

__all__ = (
    "BASE_OBSERVATION_ORDER",
    "CONSTRAINT_TOLERANCE",
    "DELTA",
    "DEPENDENCIES",
    "ETA",
    "SERVICES",
    "SHOCK_BUDGET_FACTORS",
    "SHOCK_IMPACTS",
    "SHOCK_TYPE_PROBABILITIES",
    "SHOCKS",
    "_round_vector",
    "action_to_proposal",
    "canonical_hash",
    "canonical_json_bytes",
    "measure_constraints",
    "project_capped_simplex",
)
