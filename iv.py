"""Implied volatility: invert Black-Scholes to find the sigma that reproduces a market price.

Black-Scholes price is strictly increasing in sigma, so for any price inside the
no-arbitrage bounds there is exactly one implied vol. That makes a bracketing
root-finder (Brent) reliable: it cannot diverge the way Newton's method can when
vega is tiny (deep in/out of the money).
"""

import numpy as np
from scipy.optimize import brentq

import bs


def implied_vol(price, S, K, T, r, option="call", lo=1e-6, hi=5.0):
    """Implied volatility of one option. Returns nan if no valid answer exists.

    No-arbitrage bounds for a European option (no dividends):
        call:  max(S - K e^{-rT}, 0) < price < S
        put :  max(K e^{-rT} - S, 0) < price < K e^{-rT}
    A price outside these bounds cannot come from any sigma, so we return nan
    instead of raising, which keeps a whole option chain from crashing on one
    bad quote.
    """
    disc = np.exp(-r * T)
    if option == "call":
        lower, upper = max(S - K * disc, 0.0), S
    elif option == "put":
        lower, upper = max(K * disc - S, 0.0), K * disc
    else:
        raise ValueError("option must be 'call' or 'put'")

    if not (lower < price < upper):
        return np.nan

    def f(sigma):
        return bs.price(S, K, T, r, sigma, option) - price

    if f(lo) > 0 or f(hi) < 0:  # target not bracketed inside [lo, hi]
        return np.nan
    return brentq(f, lo, hi, xtol=1e-12)


def implied_vol_chain(prices, S, strikes, T, r, option="call"):
    """Implied vols for a whole option chain (one expiry). Returns an array, nan where invalid."""
    return np.array(
        [implied_vol(p, S, k, T, r, option) for p, k in zip(prices, strikes)]
    )
