"""
poolsize_robustness.py — Pool-size externality robustness check.

The main model's shared-pool term S_φ is a ratio (composition effect only):
it is invariant to a uniform rescaling of adoption shares. To test whether the
qualitative conclusions survive the addition of a genuine pool-SIZE externality,
we multiply the shared-pool benefit by g(N_a) = N_a / (N_a + N_0), where
N_a = Σ_ψ w_ψ ρ_ψ is the total adoption share and N_0 ≥ 0 controls the
saturation. With N_0 = 0, g(N_a) = 1 and the baseline model is recovered
(a continuity check). For N_0 > 0, a larger adopting pool raises everyone's
shared-pool benefit, independent of composition.

This script is SELF-CONTAINED (does not modify model/*.py) to avoid perturbing
the validated baseline: it reimplements the best-response loop with the size
multiplier applied to the shared-pool term.

Output: data/poolsize_robustness.csv
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import (RANDOM_SEED, ALPHA, BETA, LAMBDA, GAMMA, THETA, DELTA,
                          FIRM_TYPE_SHARES)
from model.firm_generator import generate_firms
from model.distance import generate_distance_matrix
from model.accuracy import precompute_type_stats, shared_pool_and_centroid

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
W = FIRM_TYPE_SHARES


def _premiums_with_size(firms, stats, rho, N0, gamma, theta):
    S, d = shared_pool_and_centroid(stats, rho, theta)
    N_a = float(np.dot(W, np.clip(rho, 0, 1)))
    g = N_a / (N_a + N0) if (N_a + N0) > 1e-12 else 0.0
    types = firms['types']
    I = firms['I']
    V = np.zeros(firms['n_firms'])
    for t in range(3):
        mask = types == t
        V[mask] = DELTA * (ALPHA * I[mask] + (1 - LAMBDA) * BETA * S[t] * g
                           - gamma * d[t])
    return V


def _solve_with_size(firms, stats, N0, gamma=GAMMA, theta=THETA, dt=0.1,
                     t_max=50.0, tol=1e-4):
    rho = np.array([0.1, 0.02, 0.0], dtype=float)
    F = firms['F']
    types = firms['types']
    for _ in range(int(t_max / dt)):
        V = _premiums_with_size(firms, stats, rho, N0, gamma, theta)
        rho_star = np.array([
            (F[types == t] < V[types == t]).mean() if (types == t).any() else 0.0
            for t in range(3)])
        drift = 0.2 * (rho_star - rho)
        rho_new = np.clip(rho + dt * drift, 0, 1)
        if np.max(np.abs(rho_star - rho)) < tol:
            rho = rho_new
            break
        rho = rho_new
    return float(np.dot(W, rho))


def baseline_scan(N0_values=(0.0, 0.05, 0.15, 0.30), n=1000, mc=200, seed=RANDOM_SEED):
    rows = []
    for N0 in N0_values:
        T100s = []
        for r in range(mc):
            f = generate_firms(n, seed=seed + r)
            D, _ = generate_distance_matrix(f, seed=seed + r)
            stats = precompute_type_stats(f, D)
            T100s.append(_solve_with_size(f, stats, N0))
        rows.append({'N_0': N0, 'T100_mean': float(np.mean(T100s)),
                     'T100_std': float(np.std(T100s))})
        print(f"  N_0={N0}: T100={np.mean(T100s):.4f} +- {np.std(T100s):.4f}")
    return rows


def channel_check(N0=0.15, n=1000, mc=100, seed=RANDOM_SEED):
    """At N_0=0.15, verify the trap is still driven by gamma, not theta."""
    specs = {'both': (GAMMA, THETA), 'gamma_only': (GAMMA, 1e-6),
             'theta_only': (1e-6, THETA)}
    out = {}
    for name, (g, th) in specs.items():
        T100s = []
        for r in range(mc):
            f = generate_firms(n, seed=seed + r)
            D, _ = generate_distance_matrix(f, seed=seed + r)
            stats = precompute_type_stats(f, D)
            T100s.append(_solve_with_size(f, stats, N0, gamma=g, theta=th))
        out[name] = float(np.mean(T100s))
        print(f"  N_0={N0} {name}: T100={np.mean(T100s):.4f}")
    return out


if __name__ == '__main__':
    print("Pool-size externality robustness (P0 baseline scan):")
    rows = baseline_scan()
    ch = channel_check()
    df = pd.DataFrame(rows)
    df['gamma_only_T100'] = ch['gamma_only']
    df['theta_only_T100'] = ch['theta_only']
    # channel checks apply to N_0=0.15 only
    df.loc[df['N_0'] != 0.15, ['gamma_only_T100', 'theta_only_T100']] = np.nan
    df.to_csv(os.path.join(OUT, 'poolsize_robustness.csv'), index=False)
    print(f"\nsaved data/poolsize_robustness.csv")
    print(df.round(4).to_string(index=False))
