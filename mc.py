"""Monte Carlo pricing and Greeks for European options under GBM.

Core idea: under the risk-neutral measure the stock at expiry is

    S_T = S * exp((r - sigma^2/2) * T + sigma * sqrt(T) * Z),   Z ~ N(0, 1)

so the option price is the discounted expected payoff

    price = e^{-rT} * E[payoff(S_T)]

and we estimate the expectation with a sample average. The standard error of
that average shrinks like 1/sqrt(N).
"""

import numpy as np


# ----------------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------------
def terminal_price(S, T, r, sigma, z):
    """Exact GBM terminal price for standard-normal draws z (no time-stepping error)."""
    return S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)


def payoff(ST, K, option="call"):
    if option == "call":
        return np.maximum(ST - K, 0.0)
    if option == "put":
        return np.maximum(K - ST, 0.0)
    raise ValueError("option must be 'call' or 'put'")


def _mean_and_se(x):
    """Sample mean and its standard error."""
    return x.mean(), x.std(ddof=1) / np.sqrt(len(x))


def simulate_paths(S, T, r, sigma, n_paths, n_steps, seed=None):
    """Full GBM paths on a time grid (used for plotting; pricing only needs S_T).

    Uses the exact log-normal step, so there is no discretisation error even
    with few steps. Returns an array of shape (n_paths, n_steps + 1).
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    log_steps = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_paths = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(log_steps, axis=1)], axis=1
    )
    return S * np.exp(log_paths)


# ----------------------------------------------------------------------------
# Pricing
# ----------------------------------------------------------------------------
def mc_price(S, K, T, r, sigma, n=100_000, option="call", antithetic=False, seed=None):
    """Monte Carlo price with its standard error.

    antithetic=True uses the variance-reduction trick: for every draw z also
    use -z. The two payoffs are negatively correlated, so their average has a
    lower variance. We average each pair first, so the standard error is
    computed on n/2 independent pair-averages (the honest way).
    """
    rng = np.random.default_rng(seed)
    disc = np.exp(-r * T)

    if antithetic:
        z = rng.standard_normal(n // 2)
        pay = 0.5 * (
            payoff(terminal_price(S, T, r, sigma, z), K, option)
            + payoff(terminal_price(S, T, r, sigma, -z), K, option)
        )
    else:
        z = rng.standard_normal(n)
        pay = payoff(terminal_price(S, T, r, sigma, z), K, option)

    return _mean_and_se(disc * pay)


# ----------------------------------------------------------------------------
# Greeks: three different estimators
# ----------------------------------------------------------------------------
def greeks_pathwise(S, K, T, r, sigma, n=200_000, option="call", seed=None):
    """Pathwise (a.k.a. infinitesimal perturbation) estimators: delta and vega.

    Differentiate the payoff *inside* the expectation. For a call:
        d payoff / d S     = 1{S_T > K} * S_T / S
        d payoff / d sigma = 1{S_T > K} * S_T * (sqrt(T) * z - sigma * T)
    Low variance, but needs a payoff that is differentiable almost everywhere,
    which is why it cannot give gamma (the call payoff has a kink, so its
    second derivative is a spike).
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    ST = terminal_price(S, T, r, sigma, z)
    disc = np.exp(-r * T)
    sign = 1.0 if option == "call" else -1.0
    in_the_money = (ST > K) if option == "call" else (ST < K)

    delta_samples = disc * sign * in_the_money * ST / S
    vega_samples = disc * sign * in_the_money * ST * (np.sqrt(T) * z - sigma * T)
    return {
        "delta": _mean_and_se(delta_samples),
        "vega": _mean_and_se(vega_samples),
    }


def greeks_likelihood_ratio(S, K, T, r, sigma, n=200_000, option="call", seed=None):
    """Likelihood-ratio (score function) estimators: delta and gamma.

    Differentiate the *density* of S_T instead of the payoff, so the payoff can
    be as ugly as you like. Estimator: e^{-rT} * payoff * score, where the
    score is the derivative of log-density w.r.t. the parameter.
        delta score = z / (S * sigma * sqrt(T))
        gamma score = (z^2 - 1 - z * sigma * sqrt(T)) / (S^2 * sigma^2 * T)
    Works for gamma, but the variance is much higher than pathwise, so it
    needs many more paths for the same accuracy.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    ST = terminal_price(S, T, r, sigma, z)
    disc_pay = np.exp(-r * T) * payoff(ST, K, option)

    sqrtT = np.sqrt(T)
    delta_samples = disc_pay * z / (S * sigma * sqrtT)
    gamma_samples = disc_pay * (z**2 - 1.0 - z * sigma * sqrtT) / (S**2 * sigma**2 * T)
    return {
        "delta": _mean_and_se(delta_samples),
        "gamma": _mean_and_se(gamma_samples),
    }


def greeks_finite_difference(S, K, T, r, sigma, n=200_000, option="call", seed=0):
    """Bump-and-revalue with common random numbers.

    Central differences on the MC price. The crucial detail is re-using the
    *same seed* for the up and down bumps: the random noise then cancels in the
    difference. With different seeds the noise would be divided by a tiny bump
    and swamp the answer. Returns plain floats (no standard error).
    """
    def p(S_=S, sigma_=sigma):
        return mc_price(S_, K, T, r, sigma_, n, option, antithetic=False, seed=seed)[0]

    hS = 0.01 * S  # 1% spot bump
    hv = 0.01  # 1 vol point bump
    p0 = p()
    return {
        "delta": (p(S_=S + hS) - p(S_=S - hS)) / (2 * hS),
        "gamma": (p(S_=S + hS) - 2 * p0 + p(S_=S - hS)) / hS**2,
        "vega": (p(sigma_=sigma + hv) - p(sigma_=sigma - hv)) / (2 * hv),
    }
