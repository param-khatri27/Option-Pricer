"""Closed-form Black-Scholes pricing and Greeks for European options.

Assumptions (the standard ones): the stock follows geometric Brownian motion
with constant volatility, constant risk-free rate, no dividends, no
transaction costs.

Notation used everywhere in this project
----------------------------------------
S      spot price today
K      strike
T      time to expiry in years
r      continuously compounded risk-free rate
sigma  annualised volatility
"""

import numpy as np
from scipy.stats import norm


def _d1_d2(S, K, T, r, sigma):
    """The two standard-normal arguments that appear in every BS formula."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def price(S, K, T, r, sigma, option="call"):
    """Black-Scholes price of a European call or put.

    call = S * N(d1) - K * e^{-rT} * N(d2)
    put  = K * e^{-rT} * N(-d2) - S * N(-d1)
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    if option == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    if option == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    raise ValueError("option must be 'call' or 'put'")


def greeks(S, K, T, r, sigma, option="call"):
    """Analytical Greeks.

    Units (these matter when you compare against Monte Carlo):
      delta  d price / d S          (dimensionless)
      gamma  d^2 price / d S^2
      vega   d price / d sigma      (per 1.00 = 100 vol points, not per 1%)
      theta  d price / d t          (per YEAR, and negative for long options)
      rho    d price / d r          (per 1.00 change in r)
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    pdf_d1 = norm.pdf(d1)
    disc = np.exp(-r * T)
    sqrtT = np.sqrt(T)

    gamma = pdf_d1 / (S * sigma * sqrtT)  # same for call and put
    vega = S * pdf_d1 * sqrtT  # same for call and put
    decay = -S * pdf_d1 * sigma / (2 * sqrtT)  # the part of theta common to both

    if option == "call":
        delta = norm.cdf(d1)
        theta = decay - r * K * disc * norm.cdf(d2)
        rho = K * T * disc * norm.cdf(d2)
    elif option == "put":
        delta = norm.cdf(d1) - 1.0
        theta = decay + r * K * disc * norm.cdf(-d2)
        rho = -K * T * disc * norm.cdf(-d2)
    else:
        raise ValueError("option must be 'call' or 'put'")

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}
