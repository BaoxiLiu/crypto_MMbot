"""Independent numerical examples for the shared finite-horizon AS math."""
import numpy as np
from app.market_making.as_math import ASConfig, Variance, as_prices


def test_inventory_symmetry_zero_variance_and_volatility_response():
    gamma = .01
    kappa = gamma / np.expm1(gamma * 2.)
    for inventory, expected in ((0., (100., 2.5)), (2., (98., 2.5)), (-2., (102., 2.5))):
        np.testing.assert_allclose(as_prices(100., inventory, 10., gamma, kappa, 10.), expected)
    np.testing.assert_allclose(as_prices(100., 3., 0., gamma, kappa, 10.), (100., 2.))
    assert as_prices(100., 0., 20., gamma, kappa, 10.)[1] > 2.5


def test_variance_rate_uses_elapsed_time_and_normalizes_startup():
    for cadence, expected in ((500_000_000, 8.), (1_000_000_000, 4.)):
        v = Variance(ASConfig(sample_ns=cadence, warmup_samples=1,
                             stale_ns=10_000_000_000, volatility_tau_s=60.))
        v.update(1_000_000_000, 100.)
        v.update(1_000_000_000 + cadence, 102.)
        assert np.isclose(v.value, expected)
        v.update(1_000_000_000 + 2 * cadence, 102.)
        assert 0 < v.value < expected


def test_zero_gamma_and_small_gamma_have_finite_liquidity_limit():
    for gamma in (0., 1e-12):
        reservation, half = as_prices(100., 0., 1., gamma, .2, 5.)
        assert np.isfinite(reservation) and np.isfinite(half)
        np.testing.assert_allclose((reservation, half), (100., 5.))
