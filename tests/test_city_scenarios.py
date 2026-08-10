from __future__ import annotations

from dataclasses import asdict

from backend.app.city.scenarios import (
    DEVELOPMENT_FAMILIES,
    DEVELOPMENT_SEEDS,
    FINAL_FAMILIES,
    FINAL_SEEDS,
    TRAINING_FAMILIES,
    TRAINING_SEEDS,
    ScenarioFamily,
    generate_disaster_tape,
)
from backend.app.scenarios_v3 import (
    DEVELOPMENT_FAMILIES_V3,
    DEVELOPMENT_SEEDS_V3,
    FINAL_FAMILIES_V3,
    FINAL_SEEDS_V3,
    TRAINING_FAMILIES_V3,
    TRAINING_SEEDS_V3,
    ScenarioFamilyV3,
)
from backend.app.shared_evidence import canonical_hash
from backend.app.simulator_v3 import generate_disaster_tape_v3


def test_legacy_scenario_exports_are_identity_aliases() -> None:
    assert ScenarioFamilyV3 is ScenarioFamily
    assert TRAINING_FAMILIES_V3 is TRAINING_FAMILIES
    assert DEVELOPMENT_FAMILIES_V3 is DEVELOPMENT_FAMILIES
    assert FINAL_FAMILIES_V3 is FINAL_FAMILIES
    assert TRAINING_SEEDS_V3 is TRAINING_SEEDS
    assert DEVELOPMENT_SEEDS_V3 is DEVELOPMENT_SEEDS
    assert FINAL_SEEDS_V3 is FINAL_SEEDS
    assert generate_disaster_tape_v3 is generate_disaster_tape


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
