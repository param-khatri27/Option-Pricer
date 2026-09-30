"""Tests. Run with:  pytest -q

Strategy: the closed-form Black-Scholes result is the ground truth. Each test
checks that some other piece of code agrees with it, or that a known
mathematical identity holds.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import bs  # noqa: E402
import mc  # noqa: E402

S, K, T, R, SIGMA = 100.0, 105.0, 1.0, 0.05, 0.20


def test_put_call_parity():
    """C - P = S - K e^{-rT} must hold exactly for the closed form."""
    c = bs.price(S, K, T, R, SIGMA, "call")
    p = bs.price(S, K, T, R, SIGMA, "put")
    assert c - p == pytest.approx(S - K * np.exp(-R * T), abs=1e-10)


@pytest.mark.parametrize("option", ["call", "put"])
@pytest.mark.parametrize("antithetic", [False, True])
def test_mc_price_within_confidence_interval(option, antithetic):
    """MC price should land within 4 standard errors of Black-Scholes."""
    exact = bs.price(S, K, T, R, SIGMA, option)
    p, se = mc.mc_price(S, K, T, R, SIGMA, 400_000, option, antithetic, seed=123)
    assert abs(p - exact) < 4 * se


def test_antithetic_reduces_standard_error():
    _, se_plain = mc.mc_price(S, K, T, R, SIGMA, 200_000, "call", False, seed=1)
    _, se_anti = mc.mc_price(S, K, T, R, SIGMA, 200_000, "call", True, seed=1)
    assert se_anti < se_plain


@pytest.mark.parametrize("option", ["call", "put"])
def test_analytical_greeks_match_bumped_prices(option):
    """Each closed-form Greek should equal a numerical derivative of the closed-form price."""
    g = bs.greeks(S, K, T, R, SIGMA, option)
    h = 1e-4
    p = lambda **kw: bs.price(**{**dict(S=S, K=K, T=T, r=R, sigma=SIGMA, option=option), **kw})

    assert g["delta"] == pytest.approx((p(S=S + h) - p(S=S - h)) / (2 * h), abs=1e-6)
    assert g["gamma"] == pytest.approx((p(S=S + h) - 2 * p() + p(S=S - h)) / h**2, abs=1e-4)
    assert g["vega"] == pytest.approx((p(sigma=SIGMA + h) - p(sigma=SIGMA - h)) / (2 * h), abs=1e-4)
    assert g["rho"] == pytest.approx((p(r=R + h) - p(r=R - h)) / (2 * h), abs=1e-3)
    # theta is the derivative w.r.t. calendar time, i.e. minus the derivative w.r.t. T
    assert g["theta"] == pytest.approx(-(p(T=T + h) - p(T=T - h)) / (2 * h), abs=1e-4)


def test_pathwise_greeks_match_analytical():
    exact = bs.greeks(S, K, T, R, SIGMA, "call")
    est = mc.greeks_pathwise(S, K, T, R, SIGMA, 500_000, "call", seed=5)
    for name in ("delta", "vega"):
        value, se = est[name]
        assert abs(value - exact[name]) < 4 * se


def test_likelihood_ratio_greeks_match_analytical():
    exact = bs.greeks(S, K, T, R, SIGMA, "call")
    est = mc.greeks_likelihood_ratio(S, K, T, R, SIGMA, 1_000_000, "call", seed=6)
    for name in ("delta", "gamma"):
        value, se = est[name]
        assert abs(value - exact[name]) < 4 * se


def test_finite_difference_greeks_close_to_analytical():
    exact = bs.greeks(S, K, T, R, SIGMA, "call")
    est = mc.greeks_finite_difference(S, K, T, R, SIGMA, 500_000, "call", seed=7)
    assert est["delta"] == pytest.approx(exact["delta"], abs=5e-3)
    assert est["gamma"] == pytest.approx(exact["gamma"], abs=1e-3)
    assert est["vega"] == pytest.approx(exact["vega"], rel=0.02)


def test_paths_start_at_spot_and_have_right_shape():
    paths = mc.simulate_paths(S, T, R, SIGMA, n_paths=10, n_steps=50, seed=0)
    assert paths.shape == (10, 51)
    assert np.all(paths[:, 0] == S)
    assert np.all(paths > 0)
