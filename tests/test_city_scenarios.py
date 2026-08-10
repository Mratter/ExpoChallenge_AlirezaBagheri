from __future__ import annotations

from dataclasses import asdict

from backend.app.city.scenarios import (
    DEVELOPMENT_FAMILIES,
    DEVELOPMENT_SEEDS,
    FINAL_FAMILIES,
    FINAL_SEEDS,
    TRAINING_FAMILIES,
    TRAINING_SEEDS,
    generate_disaster_tape,
)
from backend.app.models import ForcedShock
from backend.app.shared_evidence import canonical_hash


def test_family_ids_and_tape_salt_remain_stable() -> None:
    assert tuple(family.id for family in TRAINING_FAMILIES) == (
        "v3_train_transit_nexus",
        "v3_train_displacement",
        "v3_train_supply_chain",
        "v3_train_health_surge",
        "v3_train_grid_failure",
        "v3_train_weather_isolation",
    )
    assert tuple(family.id for family in DEVELOPMENT_FAMILIES) == (
        "v3_dev_river_flood",
        "v3_dev_industrial_outage",
        "v3_dev_logistics_strike",
        "v3_dev_seismic_cluster",
        "v3_dev_health_compound",
    )
    assert tuple(family.id for family in FINAL_FAMILIES) == (
        "v3_final_coastal_isolation",
        "v3_final_grid_cascade",
        "v3_final_food_access",
        "v3_final_aftershock_corridor",
        "v3_final_public_health",
    )
    assert TRAINING_FAMILIES[0].tape_seed(TRAINING_SEEDS[0]) == 960163098


def test_canonical_tape_matches_the_frozen_golden_hash() -> None:
    seed = TRAINING_SEEDS[0]
    scenario = TRAINING_FAMILIES[0].build(seed)
    tape = generate_disaster_tape(scenario, seed)
    assert canonical_hash([asdict(shock) for shock in tape]) == (
        "cdade263357aeebff3d9c9274e04b14306c941efca579b3c79b6c73ba79511ae"
    )


def test_split_seed_ranges_are_disjoint_and_tapes_are_independent() -> None:
    assert not set(TRAINING_SEEDS) & set(DEVELOPMENT_SEEDS)
    assert not set(TRAINING_SEEDS) & set(FINAL_SEEDS)
    assert not set(DEVELOPMENT_SEEDS) & set(FINAL_SEEDS)

    for families, seeds in (
        (DEVELOPMENT_FAMILIES, DEVELOPMENT_SEEDS),
        (FINAL_FAMILIES, FINAL_SEEDS),
    ):
        derived = {family.tape_seed(seed) for family in families for seed in seeds}
        assert len(derived) == len(families) * len(seeds)


def test_assessment_tail_is_shock_free() -> None:
    seed = TRAINING_SEEDS[0]
    scenario = TRAINING_FAMILIES[0].build(seed)
    tape = generate_disaster_tape(scenario, seed)
    tail = tape[-scenario.assessment_tail_days :]
    assert all(item.assessment_tail for item in tail)
    assert all(item.type is None and item.severity == 0.0 for item in tail)


def test_future_forced_type_cannot_change_prior_public_tape() -> None:
    seed = TRAINING_SEEDS[0]
    scenario = TRAINING_FAMILIES[0].build(seed)
    original_forced = scenario.forced_shocks[0]
    alternate_type = "epidemic" if original_forced.type != "epidemic" else "utility"
    changed = scenario.model_copy(
        update={
            "forced_shocks": [
                ForcedShock(
                    day=original_forced.day,
                    type=alternate_type,
                    severity=original_forced.severity,
                )
            ]
        }
    )
    original = generate_disaster_tape(scenario, seed)
    alternate = generate_disaster_tape(changed, seed)
    prefix_length = original_forced.day - 1
    assert [item.public_risk_next for item in original[:prefix_length]] == [
        item.public_risk_next for item in alternate[:prefix_length]
    ]
    assert [(item.type, item.severity) for item in original[:prefix_length]] == [
        (item.type, item.severity) for item in alternate[:prefix_length]
    ]
