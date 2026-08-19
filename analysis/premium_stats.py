"""
premium_stats.py — Compute firm-level premium statistics at equilibrium.

Reports the equilibrium technical premium V at the FIRM level (individual
I_i), its type-level mean ± std across Monte Carlo replicates, the fraction
of L firms with a positive premium, and the premium decomposition per type.

Output: data/premium_stats.csv
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import RANDOM_SEED, ALPHA, BETA, LAMBDA, GAMMA
from model.firm_generator import generate_firms
from model.distance import generate_distance_matrix
from model.accuracy import precompute_type_stats, shared_pool_and_centroid, firm_premiums
from model.dynamics import solve_dynamics

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')


def run(n=1000, mc=300, seed=RANDOM_SEED):
    type_names = ['H', 'M', 'L']
    # accumulate per-replicate statistics
    V_mean = {t: [] for t in range(3)}      # type-mean V at equilibrium
    V_min = {t: [] for t in range(3)}       # min individual V per type
    V_max = {t: [] for t in range(3)}       # max individual V per type
    frac_pos = {t: [] for t in range(3)}    # fraction of firms with V_i > 0
    own = {t: [] for t in range(3)}         # alpha * mean I_t
    net = {t: [] for t in range(3)}         # (1-lambda)*beta*S_t
    pen = {t: [] for t in range(3)}         # gamma * d_t
    S_of_t = {t: [] for t in range(3)}
    d_of_t = {t: [] for t in range(3)}
    rho_star_check = {t: [] for t in range(3)}  # empirical P(F_i < V_i)

    for r in range(mc):
        f = generate_firms(n, seed=seed + r)
        D, _ = generate_distance_matrix(f, seed=seed + r)
        stats = precompute_type_stats(f, D)
        sol = solve_dynamics(f, stats)
        rho = sol.y[:, -1]
        S, d = shared_pool_and_centroid(stats, rho)
        V = firm_premiums(f, S, d)
        types, I, F = f['types'], f['I'], f['F']
        for t in range(3):
            mask = types == t
            Vt = V[mask]
            V_mean[t].append(Vt.mean())
            V_min[t].append(Vt.min())
            V_max[t].append(Vt.max())
            frac_pos[t].append((Vt > 0).mean())
            rho_star_check[t].append((F[mask] < Vt).mean())
            own[t].append(ALPHA * I[mask].mean())
            net[t].append((1 - LAMBDA) * BETA * S[t])
            pen[t].append(GAMMA * d[t])
            S_of_t[t].append(S[t])
            d_of_t[t].append(d[t])

    rows = []
    for t, name in enumerate(type_names):
        rows.append({
            'type': name,
            'V_mean': np.mean(V_mean[t]),
            'V_std': np.std(V_mean[t]),
            'V_min': np.mean(V_min[t]),
            'V_max': np.mean(V_max[t]),
            'frac_pos': np.mean(frac_pos[t]),
            'own_term': np.mean(own[t]),
            'network_term': np.mean(net[t]),
            'penalty_term': np.mean(pen[t]),
            'S_mean': np.mean(S_of_t[t]),
            'd_mean': np.mean(d_of_t[t]),
            'rho_star_check': np.mean(rho_star_check[t]),
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'premium_stats.csv'), index=False)
    print(f"saved data/premium_stats.csv  (mc={mc})")
    print(df.round(4).to_string(index=False))
    return df


if __name__ == '__main__':
    run()
