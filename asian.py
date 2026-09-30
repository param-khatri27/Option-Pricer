"""Arithmetic-average Asian options: Monte Carlo with a geometric-Asian control variate.

Payoff of an Asian call: max(A - K, 0), where A is the average of the stock price
over n monitoring dates t_i = i * T / n, i = 1..n.

There is no closed form when A is the ARITHMETIC average, so Monte Carlo is
genuinely needed here (unlike the vanilla option, where it was only a check).

Control variate idea
--------------------
The GEOMETRIC average G = (S_1 * ... * S_n)^(1/n) is almost perfectly correlated
with the arithmetic average A, and G is lognormal, so E[payoff(G)] has a closed
form. Simulate both payoffs on the same paths, then correct the noisy estimate:

    estimate = mean(Y) - b * (mean(X) - E[X])

with Y = discounted arithmetic payoff, X = discounted geometric payoff, E[X] the
known closed-form value, and b = Cov(Y, X) / Var(X). Since X's simulation error is
known exactly (we know E[X]), we subtract the part of Y's error that moves with it.

Reference: Kemna and Vorst (1990), "A pricing method for options based on
average asset values".
"""

import numpy as np
from scipy.stats import norm

import mc


def geometric_asian_price(S, K, T, r, sigma, n_steps, option="call"):
    """Closed-form price of a discretely monitored GEOMETRIC Asian option.

    ln G is normal:
        mean  m = ln S + (r - sigma^2/2) * T * (n + 1) / (2n)
        var   v = sigma^2 * T * (n + 1)(2n + 1) / (6 n^2)
    so the price is a Black-Scholes-style formula with forward e^{m + v/2}.
    """
    n = n_steps
    m = np.log(S) + (r - 0.5 * sigma**2) * T * (n + 1) / (2 * n)
    v = sigma**2 * T * (n + 1) * (2 * n + 1) / (6 * n**2)
    sd = np.sqrt(v)
    d1 = (m + v - np.log(K)) / sd
    d2 = d1 - sd
    disc = np.exp(-r * T)
    fwd = np.exp(m + 0.5 * v)
    if option == "call":
        return disc * (fwd * norm.cdf(d1) - K * norm.cdf(d2))
    if option == "put":
        return disc * (K * norm.cdf(-d2) - fwd * norm.cdf(-d1))
    raise ValueError("option must be 'call' or 'put'")


def asian_mc(
    S,
    K,
    T,
    r,
    sigma,
    n_steps=50,
    n_paths=100_000,
    option="call",
    control_variate=False,
    geometric=False,
    seed=None,
):
    """Monte Carlo price and standard error of an Asian option.

    geometric=True prices the geometric-average option by simulation (used only
    to validate the closed-form formula). control_variate=True applies the
    correction described in the module docstring.
    """
    paths = mc.simulate_paths(S, T, r, sigma, n_paths, n_steps, seed)
    obs = paths[:, 1:]  # monitoring dates t_1..t_n (drop the t_0 = today column)
    disc = np.exp(-r * T)

    geo_avg = np.exp(np.log(obs).mean(axis=1))
    if geometric:
        return mc._mean_and_se(disc * mc.payoff(geo_avg, K, option))

    arith_avg = obs.mean(axis=1)
    y = disc * mc.payoff(arith_avg, K, option)
    if not control_variate:
        return mc._mean_and_se(y)

    x = disc * mc.payoff(geo_avg, K, option)
    expected_x = geometric_asian_price(S, K, T, r, sigma, n_steps, option)
    # b is estimated from the same sample, which adds a bias of order 1/n_paths.
    # That is negligible next to the standard error, so we ignore it.
    b = np.cov(y, x)[0, 1] / x.var(ddof=1)
    return mc._mean_and_se(y - b * (x - expected_x))
