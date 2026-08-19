"""
Agent-based model with continuous heterogeneity (robustness check).

Verifies that the three-type discrete approximation does not alter the
qualitative conclusion. Each firm has:
  - continuous data quality I_i ~ Beta mixture matching the empirical R&D skew
  - continuous genetic position x_i ~ Uniform[0,1], with D_ij = |x_i - x_j|
  - adoption cost F_i ~ Uniform, positively correlated with I_i

Best-response dynamics are iterated at the individual level to the fixed point.

Multi-start analysis (run_abm_multistart): to test whether a SINGLE realization
admits multiple equilibria, we fix one population (I, x, F) and iterate the
best-response map from several distinct initial adoption states, recording the
terminal T100 for each. If the terminal T100 depends on the initial state, the
realization admits multiple equilibria; if all initial states converge to the
same T100, the fixed point is unique for that realization.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from model.config import ALPHA, BETA, LAMBDA, GAMMA, THETA, DELTA


def _generate_realization(n_firms, seed):
    rng = np.random.default_rng(seed)

    # Continuous data quality: mixture of Beta(8,2), Beta(5,5), Beta(2,8)
    # weighted by the empirical firm-type shares (0.03, 0.15, 0.82)
    types = rng.choice(3, size=n_firms, p=[0.03, 0.15, 0.82])
    I = np.zeros(n_firms)
    for t, (a, b) in enumerate([(8, 2), (5, 5), (2, 8)]):
        m = types == t
        I[m] = rng.beta(a, b, size=m.sum())

    # Continuous genetic position on [0,1]; D_ij = |x_i - x_j|
    x = rng.uniform(0, 1, size=n_firms)

    # Adoption cost correlated with data quality (higher quality → higher cost)
    F = 0.03 + 0.37 * I + rng.uniform(0, 0.05, size=n_firms)
    return I, x, F


def _iterate_adoption(I, x, F, adopt0, alpha=ALPHA, beta=BETA, lambda_=LAMBDA,
                      gamma=GAMMA, theta=THETA, delta=DELTA, max_iter=500, tol=1e-4):
    """Iterate best-response dynamics to a fixed point from adopt0.

    Returns the terminal boolean adoption array.
    """
    n_firms = len(I)
    adopt = adopt0.copy()
    for it in range(max_iter):
        adopt_old = adopt.copy()
        I_adopt = I[adopt]
        x_adopt = x[adopt]
        n_a = adopt.sum()

        if n_a == 0:
            V = delta * alpha * I
            new_adopt = V > F
        else:
            S = np.zeros(n_firms)
            d = np.zeros(n_firms)
            for i in range(n_firms):
                w = np.exp(-theta * np.abs(x[i] - x_adopt))
                den = w.sum()
                S[i] = (w * I_adopt).sum() / den if den > 1e-12 else 0.0
                d[i] = np.abs(x[i] - x_adopt).mean()
            V = delta * (alpha * I + (1 - lambda_) * beta * S - gamma * d)
            new_adopt = V > F

        if (new_adopt == adopt).all():
            adopt = new_adopt
            break
        adopt = new_adopt
    return adopt


def run_abm(n_firms=1000, seed=42, alpha=ALPHA, beta=BETA, lambda_=LAMBDA,
            gamma=GAMMA, theta=THETA, delta=DELTA, max_iter=500, tol=1e-4,
            F_scale=1.0):
    rng = np.random.default_rng(seed)
    I, x, F = _generate_realization(n_firms, seed)
    F = F * F_scale

    # Initial adoption: highest-quality 10% adopt
    adopt0 = I > np.quantile(I, 0.9)
    adopt = _iterate_adoption(I, x, F, adopt0, alpha, beta, lambda_, gamma, theta,
                              delta, max_iter, tol)

    # Aggregate: adoption rate by quality tercile
    terciles = np.quantile(I, [1/3, 2/3])
    low = I < terciles[0]
    mid = (I >= terciles[0]) & (I < terciles[1])
    high = I >= terciles[1]

    return {
        'T100': float(adopt.mean()),
        'adopt_high': float(adopt[high].mean()),
        'adopt_mid': float(adopt[mid].mean()),
        'adopt_low': float(adopt[low].mean()),
        'n_firms': n_firms,
    }


def _initial_adoption(I, x, mode, rng):
    """Distinct initial adoption states for the multi-start experiment."""
    n = len(I)
    if mode == 'top10':
        return I > np.quantile(I, 0.90)
    if mode == 'top5':
        return I > np.quantile(I, 0.95)
    if mode == 'top25':
        return I > np.quantile(I, 0.75)
    if mode == 'all':
        return np.ones(n, dtype=bool)
    if mode == 'none':
        return np.zeros(n, dtype=bool)
    if mode == 'random10':
        return rng.random(n) < 0.10
    if mode == 'left_half':
        return x < 0.5
    if mode == 'right_half':
        return x >= 0.5
    if mode == 'bottom10':
        return I < np.quantile(I, 0.10)
    raise ValueError(f'unknown mode {mode}')


INIT_MODES = ['top10', 'top5', 'top25', 'all', 'none', 'random10',
              'left_half', 'right_half', 'bottom10']


def run_abm_multistart(n_firms=1000, realizations=(42, 43, 44),
                       init_modes=INIT_MODES, alpha=ALPHA, beta=BETA,
                       lambda_=LAMBDA, gamma=GAMMA, theta=THETA, delta=DELTA,
                       max_iter=500, tol=1e-4):
    """Run the best-response map from multiple initial states on FIXED
    realizations. Returns a list of dict rows {seed, init, T100}."""
    rows = []
    for seed in realizations:
        I, x, F = _generate_realization(n_firms, seed)
        rng = np.random.default_rng(seed + 10_000)
        for mode in init_modes:
            adopt0 = _initial_adoption(I, x, mode, rng)
            adopt = _iterate_adoption(I, x, F, adopt0, alpha, beta, lambda_,
                                      gamma, theta, delta, max_iter, tol)
            rows.append({'seed': seed, 'init': mode, 'T100': float(adopt.mean())})
    return rows


def run_abm_monte_carlo(n_firms=1000, n_mc=100, seed=42, **kwargs):
    results = []
    for r in range(n_mc):
        results.append(run_abm(n_firms=n_firms, seed=seed + r, **kwargs))
    out = {}
    for k in ['T100', 'adopt_high', 'adopt_mid', 'adopt_low']:
        vals = [res[k] for res in results]
        out[f'{k}_mean'] = float(np.mean(vals))
        out[f'{k}_std'] = float(np.std(vals))
    return out


if __name__ == '__main__':
    import time
    t0 = time.time()
    res = run_abm(n_firms=1000, seed=42)
    print(f"ABM single run: {time.time()-t0:.2f}s")
    print(f"T100={res['T100']:.4f}  high={res['adopt_high']:.3f} mid={res['adopt_mid']:.3f} low={res['adopt_low']:.3f}")

    print("\nMulti-start experiment (3 fixed realizations x 9 initial states):")
    rows = run_abm_multistart(realizations=(42, 43, 44))
    import pandas as pd
    df = pd.DataFrame(rows)
    piv = df.pivot(index='seed', columns='init', values='T100')
    print(piv.round(4).to_string())
    # within-realization spread of terminal T100
    spread = df.groupby('seed')['T100'].agg(lambda s: s.max() - s.min())
    print("\nWithin-realization terminal T100 spread (max - min):")
    print(spread.round(4).to_string())
