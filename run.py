"""Run the full experiment: MC vs Black-Scholes, convergence, variance reduction, Greeks.

    python run.py

Everything is seeded, so the numbers and plots are reproducible.
Outputs go to ./outputs/.
"""

import csv
import os

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np

import bs
import mc

# ---- Contract and market parameters (change these to experiment) -----------
S, K, T, r, SIGMA = 100.0, 105.0, 1.0, 0.05, 0.20
OPTION = "call"
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)


def header(text):
    print("\n" + text)
    print("-" * len(text))


# ----------------------------------------------------------------------------
# 1. Price: Monte Carlo vs closed form
# ----------------------------------------------------------------------------
def price_comparison():
    header(f"1. Price of a European {OPTION} (S={S}, K={K}, T={T}, r={r}, sigma={SIGMA})")
    exact = bs.price(S, K, T, r, SIGMA, OPTION)
    print(f"Black-Scholes (exact): {exact:.4f}")
    n = 1_000_000
    for label, anti in [("MC plain     ", False), ("MC antithetic", True)]:
        p, se = mc.mc_price(S, K, T, r, SIGMA, n, OPTION, antithetic=anti, seed=42)
        print(
            f"{label} N={n:,}: {p:.4f} +/- {1.96 * se:.4f} (95% CI)"
            f"   error vs BS = {p - exact:+.4f}"
        )
    return exact


# ----------------------------------------------------------------------------
# 2. Convergence plot: one long run, estimate after N paths for growing N
# ----------------------------------------------------------------------------
def convergence_plot(exact):
    n_max = 1_000_000
    rng = np.random.default_rng(7)
    z = rng.standard_normal(n_max)
    samples = np.exp(-r * T) * mc.payoff(mc.terminal_price(S, T, r, SIGMA, z), K, OPTION)

    idx = np.unique(np.logspace(2, np.log10(n_max), 200).astype(int))
    csum = np.cumsum(samples)
    csum2 = np.cumsum(samples**2)
    mean = csum[idx - 1] / idx
    var = (csum2[idx - 1] / idx - mean**2) * idx / (idx - 1)
    se = np.sqrt(var / idx)

    plt.figure(figsize=(8, 4.5))
    plt.plot(idx, mean, lw=1.2, label="Monte Carlo estimate")
    plt.fill_between(idx, mean - 1.96 * se, mean + 1.96 * se, alpha=0.25, label="95% CI")
    plt.axhline(exact, color="crimson", ls="--", label=f"Black-Scholes = {exact:.4f}")
    plt.xscale("log")
    plt.xlabel("Number of simulated paths N")
    plt.ylabel("Option price")
    plt.title("Monte Carlo price converges to the Black-Scholes price")
    plt.ylim(exact - 1.5, exact + 1.5)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/convergence.png", dpi=150)
    plt.close()


# ----------------------------------------------------------------------------
# 3. Error vs N (checks the 1/sqrt(N) rate) and antithetic variance reduction
# ----------------------------------------------------------------------------
def error_vs_n(exact):
    header("2. Root-mean-square error vs number of paths (200 repeats each)")
    Ns = [1_000, 3_000, 10_000, 30_000, 100_000, 300_000]
    reps = 200
    rmse = {False: [], True: []}
    for n in Ns:
        for anti in (False, True):
            errs = [
                mc.mc_price(S, K, T, r, SIGMA, n, OPTION, antithetic=anti, seed=1000 + i)[0] - exact
                for i in range(reps)
            ]
            rmse[anti].append(np.sqrt(np.mean(np.square(errs))))
    print(f"{'N':>9} {'plain RMSE':>12} {'antithetic RMSE':>16} {'variance reduction':>19}")
    for i, n in enumerate(Ns):
        vr = (rmse[False][i] / rmse[True][i]) ** 2
        print(f"{n:>9,} {rmse[False][i]:>12.4f} {rmse[True][i]:>16.4f} {vr:>18.2f}x")

    plt.figure(figsize=(7, 4.5))
    plt.loglog(Ns, rmse[False], "o-", label="plain MC")
    plt.loglog(Ns, rmse[True], "s-", label="antithetic MC")
    ref = rmse[False][0] * np.sqrt(Ns[0] / np.array(Ns))
    plt.loglog(Ns, ref, "k--", lw=1, label=r"$1/\sqrt{N}$ reference")
    plt.xlabel("Number of paths N")
    plt.ylabel("RMSE vs Black-Scholes")
    plt.title("MC error shrinks like 1/sqrt(N); antithetic shifts it down")
    plt.legend()
    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(f"{OUT}/error_vs_n.png", dpi=150)
    plt.close()


# ----------------------------------------------------------------------------
# 4. Greeks: analytical vs three MC estimators
# ----------------------------------------------------------------------------
def greeks_comparison():
    header("3. Greeks (call): Black-Scholes vs Monte Carlo estimators")
    exact = bs.greeks(S, K, T, r, SIGMA, OPTION)
    pw = mc.greeks_pathwise(S, K, T, r, SIGMA, 1_000_000, OPTION, seed=1)
    lr = mc.greeks_likelihood_ratio(S, K, T, r, SIGMA, 2_000_000, OPTION, seed=2)
    fd = mc.greeks_finite_difference(S, K, T, r, SIGMA, 1_000_000, OPTION, seed=3)

    def fmt(entry):
        return f"{entry[0]:.4f} +/-{1.96 * entry[1]:.4f}" if entry else "n/a"

    rows = []
    print(f"{'Greek':<7}{'Black-Scholes':>15}{'Pathwise':>20}{'Likelihood ratio':>22}{'Finite diff':>13}")
    for g in ["delta", "gamma", "vega"]:
        row = [
            g,
            f"{exact[g]:.4f}",
            fmt(pw.get(g)),
            fmt(lr.get(g)),
            f"{fd[g]:.4f}",
        ]
        rows.append(row)
        print(f"{row[0]:<7}{row[1]:>15}{row[2]:>20}{row[3]:>22}{row[4]:>13}")
    print("(theta and rho are shown analytically only)")
    print(f"theta = {exact['theta']:.4f}/yr, rho = {exact['rho']:.4f}")

    with open(f"{OUT}/greeks_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["greek", "black_scholes", "pathwise", "likelihood_ratio", "finite_difference"])
        w.writerows(rows)


# ----------------------------------------------------------------------------
# 5. Illustrative plots
# ----------------------------------------------------------------------------
def paths_plot():
    paths = mc.simulate_paths(S, T, r, SIGMA, n_paths=30, n_steps=252, seed=5)
    t = np.linspace(0, T, paths.shape[1])
    plt.figure(figsize=(8, 4.5))
    plt.plot(t, paths.T, lw=0.8, alpha=0.8)
    plt.axhline(K, color="k", ls="--", lw=1, label=f"Strike K = {K:.0f}")
    plt.xlabel("Time (years)")
    plt.ylabel("Stock price")
    plt.title("Simulated GBM paths")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/paths.png", dpi=150)
    plt.close()


def greeks_vs_spot_plot():
    spots = np.linspace(60, 150, 200)
    g = bs.greeks(spots, K, T, r, SIGMA, OPTION)
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
    for ax, name in zip(axes.ravel(), ["delta", "gamma", "vega", "theta"]):
        ax.plot(spots, g[name])
        ax.axvline(K, color="gray", ls=":", lw=1)
        ax.set_title(name)
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("Spot price S")
    fig.suptitle(f"Black-Scholes Greeks of a European {OPTION} (K={K:.0f})")
    fig.tight_layout()
    fig.savefig(f"{OUT}/greeks_vs_spot.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    exact_price = price_comparison()
    convergence_plot(exact_price)
    error_vs_n(exact_price)
    greeks_comparison()
    paths_plot()
    greeks_vs_spot_plot()
    print(f"\nSaved plots and tables to ./{OUT}/")
