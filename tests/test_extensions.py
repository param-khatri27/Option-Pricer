"""Tests for the extensions (Asian options, implied volatility). Run with:  pytest -q"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import asian  # noqa: E402
import bs  # noqa: E402
import iv  # noqa: E402

S, K, T, R, SIGMA = 100.0, 105.0, 1.0, 0.05, 0.20
N_STEPS = 50


# ---------------------------- Asian options ---------------------------------
@pytest.mark.parametrize("option", ["call", "put"])
def test_geometric_asian_closed_form_matches_simulation(option):
    """Validates the closed-form formula by simulating the geometric payoff directly."""
    exact = asian.geometric_asian_price(S, K, T, R, SIGMA, N_STEPS, option)
    p, se = asian.asian_mc(S, K, T, R, SIGMA, N_STEPS, 300_000, option, geometric=True, seed=21)
    assert abs(p - exact) < 4 * se


def test_control_variate_is_consistent_with_plain_mc():
    """Both estimators target the same price, so they must agree within noise."""
    plain, se_plain = asian.asian_mc(S, K, T, R, SIGMA, N_STEPS, 300_000, "call", seed=22)
    cv, se_cv = asian.asian_mc(S, K, T, R, SIGMA, N_STEPS, 300_000, "call", control_variate=True, seed=22)
    assert abs(plain - cv) < 4 * se_plain
    assert se_cv < se_plain


def test_control_variate_gives_large_variance_reduction():
    _, se_plain = asian.asian_mc(S, K, T, R, SIGMA, N_STEPS, 100_000, "call", seed=23)
    _, se_cv = asian.asian_mc(S, K, T, R, SIGMA, N_STEPS, 100_000, "call", control_variate=True, seed=23)
    assert (se_plain / se_cv) ** 2 > 100  # observed ~900x, so this is a safe lower bound


def test_price_ordering_geometric_arithmetic_european():
    """AM-GM gives geometric <= arithmetic, and averaging lowers volatility so Asian <= European."""
    geo = asian.geometric_asian_price(S, K, T, R, SIGMA, N_STEPS, "call")
    arith, _ = asian.asian_mc(S, K, T, R, SIGMA, N_STEPS, 200_000, "call", control_variate=True, seed=24)
    european = bs.price(S, K, T, R, SIGMA, "call")
    assert geo < arith < european


def test_single_monitoring_date_reduces_to_vanilla():
    """With one monitoring date at expiry the Asian option IS the European option."""
    geo = asian.geometric_asian_price(S, K, T, R, SIGMA, 1, "call")
    assert geo == pytest.approx(bs.price(S, K, T, R, SIGMA, "call"), abs=1e-10)


# --------------------------- Implied volatility -----------------------------
@pytest.mark.parametrize("option", ["call", "put"])
@pytest.mark.parametrize("sigma", [0.05, 0.2, 0.6, 1.5])
@pytest.mark.parametrize("strike", [80.0, 100.0, 125.0])
def test_implied_vol_round_trip(sigma, strike, option):
    p = bs.price(S, strike, T, R, sigma, option)
    assert iv.implied_vol(p, S, strike, T, R, option) == pytest.approx(sigma, abs=1e-7)


def test_implied_vol_rejects_arbitrage_prices():
    assert np.isnan(iv.implied_vol(0.0, S, K, T, R, "call"))  # zero price
    assert np.isnan(iv.implied_vol(S + 1.0, S, K, T, R, "call"))  # call above the spot
    intrinsic = S - K * np.exp(-R * T) if S > K * np.exp(-R * T) else 0.0
    assert np.isnan(iv.implied_vol(intrinsic - 0.5, S, 80.0, T, R, "call"))  # below intrinsic


def test_implied_vol_chain_recovers_smile():
    strikes = np.linspace(80, 120, 9)
    x = np.log(strikes / S)
    smile = 0.18 - 0.10 * x + 1.2 * x**2
    prices = [bs.price(S, k, T, R, s, "call") for k, s in zip(strikes, smile)]
    assert iv.implied_vol_chain(prices, S, strikes, T, R, "call") == pytest.approx(smile, abs=1e-7)
