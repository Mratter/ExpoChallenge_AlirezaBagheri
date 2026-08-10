"""Behavior checks for consolidated city allocation and logistics physics."""

from types import SimpleNamespace

import numpy as np

from backend.app.city.physics import (
    Transfer,
    apply_depot_damage,
    deterministic_transfer,
    land_capped,
    throughput_factors,
)


def test_depot_damage_and_throughput_preserve_registered_physics() -> None:
    shock = SimpleNamespace(
        type="utility",
        severity=0.4,
        impact=[1.0, 0.5, 0.0, 0.25, 0.1],
    )
    peaks, durations, remaining, penalty = apply_depot_damage(
        shock,
        np.zeros(5),
        np.zeros(5, dtype=np.int64),
        np.zeros(5, dtype=np.int64),
    )

    np.testing.assert_allclose(
        peaks,
        [0.6, 0.3, 0.0, 0.15, 0.06],
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_array_equal(durations, [6, 4, 0, 3, 3])
    np.testing.assert_array_equal(remaining, durations)
    np.testing.assert_array_equal(penalty, peaks)

    depot, road, throughput = throughput_factors(
        np.array([0.5, 1.0, 1.0, 1.0, 1.0]),
        np.array([0.2, 0.0, 0.8, 0.0, 0.5]),
    )
    np.testing.assert_array_equal(depot, [0.8, 1.0, 0.3, 1.0, 0.5])
    assert road == 0.7
    np.testing.assert_allclose(
        throughput,
        [0.8, 0.7, 0.21, 0.7, 0.35],
        rtol=0.0,
        atol=1e-15,
    )


def test_transfer_and_capacity_landing_conserve_inventory_exactly() -> None:
    stock = np.array([300.0, 20.0, 200.0, 100.0, 100.0])
    adjusted, net, transfers = deterministic_transfer(stock, np.ones(5))

    np.testing.assert_array_equal(adjusted, [276.0, 44.0, 200.0, 100.0, 100.0])
    np.testing.assert_array_equal(net, [-24.0, 24.0, 0.0, 0.0, 0.0])
    assert transfers == (
        Transfer(
            from_service="transport",
            to_service="housing",
            units=24.0,
            donor_stock_fraction_before=0.75,
            receiver_stock_fraction_before=0.05,
        ),
    )
    assert float(adjusted.sum()) == float(stock.sum())
    assert float(net.sum()) == 0.0

    landed_stock, landed, held = land_capped(
        np.array([390.0, 100.0, 400.0, 0.0, 399.0]),
        np.array([20.0, 10.0, 8.0, 5.0, 2.0]),
    )
    np.testing.assert_array_equal(landed_stock, [400.0, 110.0, 400.0, 5.0, 400.0])
    np.testing.assert_array_equal(landed, [10.0, 10.0, 0.0, 5.0, 1.0])
    np.testing.assert_array_equal(held, [10.0, 0.0, 8.0, 0.0, 1.0])
    np.testing.assert_array_equal(landed + held, [20.0, 10.0, 8.0, 5.0, 2.0])
