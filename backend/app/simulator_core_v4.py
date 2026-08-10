"""Compatibility exports for the consolidated city allocation physics."""

from backend.app.city.physics import (
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
    "_round_vector",
    "action_to_proposal",
    "canonical_hash",
    "canonical_json_bytes",
    "measure_constraints",
    "project_capped_simplex",
)
