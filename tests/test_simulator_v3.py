from __future__ import annotations

import unittest

import numpy as np

from backend.app.city.outcome import absolute_outcome
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
from backend.app.simulator_core import canonical_hash
from backend.app.simulator_v3 import (
    ACTION_SIZE_V3,
    OBSERVATION_SIZE_V3,
    CityRecoveryEnvV3,
)


class SimulatorV3ContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.seed = TRAINING_SEEDS[0]
        self.scenario = TRAINING_FAMILIES[0].build(self.seed)

    def test_split_seed_ranges_are_disjoint(self) -> None:
        self.assertFalse(set(TRAINING_SEEDS) & set(DEVELOPMENT_SEEDS))
        self.assertFalse(set(TRAINING_SEEDS) & set(FINAL_SEEDS))
        self.assertFalse(set(DEVELOPMENT_SEEDS) & set(FINAL_SEEDS))

    def test_each_registered_case_has_an_independent_tape_seed(self) -> None:
        for families, seeds in (
            (DEVELOPMENT_FAMILIES, DEVELOPMENT_SEEDS),
            (FINAL_FAMILIES, FINAL_SEEDS),
        ):
            derived = {family.tape_seed(seed) for family in families for seed in seeds}
            self.assertEqual(len(derived), len(families) * len(seeds))

    def test_observation_and_action_contract(self) -> None:
        self.assertEqual(OBSERVATION_SIZE_V3, 73)
        self.assertEqual(ACTION_SIZE_V3, 22)
        env = CityRecoveryEnvV3(self.scenario, self.seed)
        observation, _ = env.reset(seed=self.seed)
        self.assertEqual(observation.shape, (OBSERVATION_SIZE_V3,))
        self.assertEqual(observation.dtype, np.float32)
        self.assertTrue(np.all(observation >= 0.0))
        self.assertTrue(np.all(observation <= 1.0))
        self.assertEqual(env.action_space.shape, (ACTION_SIZE_V3,))
        with self.assertRaises(ValueError):
            env.step(np.zeros(ACTION_SIZE_V3 - 1, dtype=np.float32))
        with self.assertRaises(ValueError):
            bad = np.zeros(ACTION_SIZE_V3, dtype=np.float32)
            bad[0] = np.nan
            env.step(bad)

    def test_assessment_tail_is_shock_free(self) -> None:
        tape = generate_disaster_tape(self.scenario, self.seed)
        tail = tape[-self.scenario.assessment_tail_days :]
        self.assertTrue(all(item.assessment_tail for item in tail))
        self.assertTrue(
            all(item.type is None and item.severity == 0.0 for item in tail)
        )

    def test_future_forced_type_cannot_change_prior_public_tape(self) -> None:
        original_forced = self.scenario.forced_shocks[0]
        alternate_type = "epidemic" if original_forced.type != "epidemic" else "utility"
        changed = self.scenario.model_copy(
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
        original = generate_disaster_tape(self.scenario, self.seed)
        alternate = generate_disaster_tape(changed, self.seed)
        prefix_length = original_forced.day - 1
        self.assertEqual(
            [item.public_risk_next for item in original[:prefix_length]],
            [item.public_risk_next for item in alternate[:prefix_length]],
        )
        self.assertEqual(
            [(item.type, item.severity) for item in original[:prefix_length]],
            [(item.type, item.severity) for item in alternate[:prefix_length]],
        )

    def test_same_actions_and_tape_replay_exactly(self) -> None:
        actions = [np.linspace(-1.0, 1.0, ACTION_SIZE_V3) for _ in range(30)]
        hashes: list[str] = []
        for _ in range(2):
            env = CityRecoveryEnvV3(self.scenario, self.seed)
            env.reset(seed=self.seed)
            terminated = False
            index = 0
            while not terminated:
                _, _, terminated, _, _ = env.step(actions[index])
                index += 1
            hashes.append(canonical_hash(env.trajectory))
        self.assertEqual(hashes[0], hashes[1])

    def test_common_decoder_conserves_material_and_crew(self) -> None:
        rng = np.random.Generator(np.random.PCG64(918273))
        env = CityRecoveryEnvV3(self.scenario, self.seed)
        env.reset(seed=self.seed)
        terminated = False
        while not terminated:
            action = rng.uniform(-1.0, 1.0, size=ACTION_SIZE_V3)
            _, _, terminated, _, info = env.step(action)
            day = info["day"]
            self.assertAlmostEqual(
                sum(day["material_allocation"]), day["material_used"], 6
            )
            self.assertAlmostEqual(
                day["material_used"] + day["material_unspent"],
                day["available_budget"],
                6,
            )
            self.assertAlmostEqual(sum(day["crew_allocation"]), day["crew_used"], 6)
            self.assertAlmostEqual(
                day["crew_used"] + day["crew_idle"], day["available_crew"], 6
            )
            self.assertEqual(day["hard_violation_count"], 0)
            self.assertLessEqual(
                max(abs(value) for value in day["logistics"]["conservation_residual"]),
                1e-6,
            )

    def test_utilization_gates_can_leave_resources_idle(self) -> None:
        env = CityRecoveryEnvV3(self.scenario, self.seed)
        env.reset(seed=self.seed)
        action = np.zeros(ACTION_SIZE_V3, dtype=np.float64)
        action[5] = -1.0
        action[11] = -1.0
        _, _, _, _, info = env.step(action)
        day = info["day"]
        self.assertGreater(day["material_unspent"], 0.0)
        self.assertGreater(day["crew_idle"], 0.0)
        self.assertEqual(day["hard_violation_count"], 0)

    def test_overordering_without_release_cannot_count_as_solved(self) -> None:
        env = CityRecoveryEnvV3(self.scenario, self.seed)
        env.reset(seed=self.seed)
        action = np.zeros(ACTION_SIZE_V3, dtype=np.float64)
        action[5] = 1.0
        action[11] = 1.0
        action[12:17] = -1.0
        action[17:22] = -1.0
        terminated = False
        while not terminated:
            _, _, terminated, _, _ = env.step(action)
        outcome = env.trajectory[-1]["absolute_outcome"]
        self.assertFalse(outcome["solved"])
        self.assertFalse(outcome["checks"]["terminal_pending_within_capacity"])

    def test_preparedness_consumes_real_resources_and_builds_state(self) -> None:
        env = CityRecoveryEnvV3(self.scenario, self.seed)
        env.reset(seed=self.seed)
        action = np.zeros(ACTION_SIZE_V3, dtype=np.float64)
        action[17:22] = 1.0
        _, _, _, _, info = env.step(action)
        day = info["day"]
        self.assertTrue(all(value > 0.0 for value in day["preparedness_gain"]))
        self.assertTrue(
            all(
                value > 0.0
                for value in day["logistics"]["preparedness_material_consumed"]
            )
        )
        self.assertTrue(
            all(
                committed + 1e-7 >= lower
                for committed, lower in zip(
                    day["logistics"]["repair_material_committed"],
                    day["lower_bounds"],
                    strict=True,
                )
            )
        )
        self.assertTrue(
            all(
                assigned + 1e-7 >= lower
                for assigned, lower in zip(
                    day["logistics"]["repair_crew_assigned"],
                    day["crew_lower_bounds"],
                    strict=True,
                )
            )
        )
        self.assertTrue(
            all(
                realized <= requested + 1e-8
                for realized, requested in zip(
                    day["preparedness_gain"],
                    day["preparedness_gain_requested"],
                    strict=True,
                )
            )
        )
        self.assertEqual(day["hard_violation_count"], 0)
        self.assertLessEqual(
            max(abs(value) for value in day["logistics"]["conservation_residual"]),
            1e-6,
        )

    def test_zero_stock_release_blocks_all_inventory_dispatch(self) -> None:
        env = CityRecoveryEnvV3(self.scenario, self.seed)
        env.reset(seed=self.seed)
        action = np.zeros(ACTION_SIZE_V3, dtype=np.float64)
        action[12:17] = -1.0
        action[17:22] = 1.0
        _, _, _, _, info = env.step(action)
        day = info["day"]
        self.assertTrue(
            all(
                abs(value) <= 1e-9
                for value in day["logistics"]["preparedness_material_consumed"]
            )
        )
        self.assertTrue(
            all(abs(value) <= 1e-9 for value in day["logistics"]["repair_dispatch"])
        )
        self.assertTrue(all(abs(value) <= 1e-9 for value in day["preparedness_gain"]))
        self.assertEqual(day["hard_violation_count"], 0)

    def test_absolute_outcome_recomputes_from_trajectory(self) -> None:
        env = CityRecoveryEnvV3(self.scenario, self.seed)
        env.reset(seed=self.seed)
        terminated = False
        while not terminated:
            _, _, terminated, _, _ = env.step(np.zeros(ACTION_SIZE_V3))
        outcome = absolute_outcome(
            env.trajectory,
            self.scenario.recovery_targets,
            self.scenario.assessment_tail_days,
        )
        self.assertEqual(outcome, env.trajectory[-1]["absolute_outcome"])


if __name__ == "__main__":
    unittest.main()
