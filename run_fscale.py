"""
F-scale cost robustness: T100 under F x {0.5, 0.75, 1.0, 1.25, 1.5}.

Generates data/fscale_robustness.csv (Supplementary Table S11).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from model.firm_generator import generate_firms
from model.distance import generate_distance_matrix
from model.accuracy import precompute_type_stats
from model.dynamics import solve_dynamics, T100
from model.config import RANDOM_SEED

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def run_fscale_robustness(n_firms=1000, n_mc=100, seed=RANDOM_SEED):
    rows = []
    for fscale in [0.5, 0.75, 1.0, 1.25, 1.5]:
        T100s = []
        for r in range(n_mc):
            f = generate_firms(n_firms, seed=seed + r)
            D, _ = generate_distance_matrix(f, seed=seed + r)
            s = precompute_type_stats(f, D)
            sol = solve_dynamics(f, s, F_scale=fscale)
            T100s.append(T100(sol))
        rows.append({'F_scale': fscale, 'T100_mean': float(np.mean(T100s)),
                     'T100_std': float(np.std(T100s))})
        print(f"  Fx{fscale:.2f}: T100={rows[-1]['T100_mean']:.4f}")
    return pd.DataFrame(rows)


if __name__ == '__main__':
    print("F-scale robustness (n=1000, MC=100)...")
    df = run_fscale_robustness()
    df.to_csv(os.path.join(DATA, 'fscale_robustness.csv'), index=False)
    print("saved data/fscale_robustness.csv")
