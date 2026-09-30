"""Extensions: (1) Asian option with control variate, (2) implied volatility solver.

    python run_extension.py

Seeded and reproducible. Outputs go to ./outputs/.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import asian
import bs
import iv

S, K, T, r, SIGMA = 100.0, 105.0, 1.0, 0.05, 0.20
N_STEPS = 50  # monitoring dates for the Asian option
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)


def header(text):
    print("\n" + text)
    print("-" * len(text))


# ----------------------------------------------------------------------------
# 1. Asian option: plain MC vs control variate
# ----------------------------------------------------------------------------
def asian_experiment():
    header(f"1. Arithmetic Asian call (S={S}, K={K}, T={T}, r={r}, sigma={SIGMA}, {N_STEPS} monitoring dates)")
    n = 200_000
    european = bs.price(S, K, T, r, SIGMA, "call")
    geo = asian.geometric_asian_price(S, K, T, r, SIGMA, N_STEPS, "call")
    plain, se_plain = asian.asian_mc(S, K, T, r, SIGMA, N_STEPS, n, "call", seed=11)
    cv, se_cv = asian.asian_mc(S, K, T, r, SIGMA, N_STEPS, n, "call", control_variate=True, seed=11)

    print(f"European call (Black-Scholes)        : {european:.4f}")
    print(f"Geometric Asian call (closed form)   : {geo:.4f}")
    print(f"Arithmetic Asian, plain MC   N={n:,}: {plain:.4f} +/- {1.96 * se_plain:.4f} (95% CI)")
    print(f"Arithmetic Asian, control variate    : {cv:.4f} +/- {1.96 * se_cv:.4f} (95% CI)")
    print(f"Standard error reduced {se_plain / se_cv:.1f}x  ->  variance reduced {(se_plain / se_cv) ** 2:.0f}x")
    print("Sanity: geometric <= arithmetic <= European ->", geo <= cv <= european)

    # Standard error vs number of paths
    Ns = [1_000, 3_000, 10_000, 30_000, 100_000]
    se_p, se_c = [], []
    for i, m in enumerate(Ns):
        se_p.append(asian.asian_mc(S, K, T, r, SIGMA, N_STEPS, m, "call", seed=100 + i)[1])
        se_c.append(asian.asian_mc(S, K, T, r, SIGMA, N_STEPS, m, "call", control_variate=True, seed=100 + i)[1])

    plt.figure(figsize=(7, 4.5))
    plt.loglog(Ns, se_p, "o-", label="plain MC")
    plt.loglog(Ns, se_c, "s-", label="control variate")
    plt.xlabel("Number of paths N")
    plt.ylabel("Standard error of the price")
    plt.title("Asian call: control variate slashes the standard error")
    plt.legend()
    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(f"{OUT}/asian_control_variate.png", dpi=150)
    plt.close()


# ----------------------------------------------------------------------------
# 2. Implied volatility
# ----------------------------------------------------------------------------
def iv_experiment():
    header("2. Implied volatility solver")

    # (a) Round trip on a grid: price with a known sigma, then recover it
    worst = 0.0
    for sigma in [0.05, 0.10, 0.20, 0.40, 0.80, 1.50]:
        for k in [70, 85, 100, 115, 130]:
            for opt in ["call", "put"]:
                p = bs.price(S, k, T, r, sigma, opt)
                rec = iv.implied_vol(p, S, k, T, r, opt)
                if np.isfinite(rec):
                    worst = max(worst, abs(rec - sigma))
    print(f"Round trip over 60 (sigma, strike, type) combinations: max |error| = {worst:.2e}")

    # (b) Recover a smile. The market prices below come from a SYNTHETIC smile
    # (equity-like skew), NOT real market data. Replace with a real option chain
    # (strike, mid price) to get a real smile.
    strikes = np.linspace(75, 125, 26)
    x = np.log(strikes / S)
    true_smile = 0.18 - 0.10 * x + 1.2 * x**2
    prices = np.array([bs.price(S, k, T, r, s, "call") for k, s in zip(strikes, true_smile)])
    recovered = iv.implied_vol_chain(prices, S, strikes, T, r, "call")
    print(f"Synthetic smile recovered: max |error| = {np.nanmax(np.abs(recovered - true_smile)):.2e}")

    plt.figure(figsize=(7, 4.5))
    plt.plot(strikes, true_smile * 100, "k-", lw=2, label="true (synthetic) smile")
    plt.plot(strikes, recovered * 100, "o", ms=5, label="implied vol from prices")
    plt.xlabel("Strike K")
    plt.ylabel("Implied volatility (%)")
    plt.title("Implied vol solver recovers a volatility smile (synthetic data)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/implied_vol_smile.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    asian_experiment()
    iv_experiment()
    print(f"\nSaved plots to ./{OUT}/")
