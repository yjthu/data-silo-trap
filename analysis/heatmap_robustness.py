"""
heatmap_robustness.py — Joint parameter-range sensitivity (Supplementary Fig S1, Table S17).

Evaluates T100 on a 9x9 grid of (gamma, beta) and (gamma, lambda) to show that
the composition trap is a property of a REGION of parameter space, not of a
single calibration point.

Outputs:
  data/heatmap_gamma_beta.csv    (gamma, beta, T100_mean)
  data/heatmap_gamma_lambda.csv  (gamma, lambda, T100_mean)
"""
import os
import sys
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import RANDOM_SEED
from model.firm_generator import generate_firms
from model.distance import generate_distance_matrix
from model.accuracy import precompute_type_stats
from model.dynamics import solve_dynamics, T100

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')


def grid_run(n=1000, mc=100, seed=RANDOM_SEED,
             gamma_vals=np.linspace(0.2, 0.6, 9),
             beta_vals=np.linspace(0.3, 0.7, 9),
             lambda_vals=np.linspace(0.0, 0.8, 9)):
    print('generating firm pools...')
    pools = []
    for r in range(mc):
        f = generate_firms(n, seed=seed + r)
        D, _ = generate_distance_matrix(f, seed=seed + r)
        pools.append((f, precompute_type_stats(f, D)))

    rows_gb, rows_gl = [], []
    n_grid = len(gamma_vals) * len(beta_vals)
    done = 0
    for g in gamma_vals:
        for b in beta_vals:
            T100s = [T100(solve_dynamics(f, s, gamma=g, beta=b)) for f, s in pools]
            rows_gb.append({'gamma': g, 'beta': b,
                            'T100_mean': float(np.mean(T100s))})
            done += 1
            if done % 9 == 0:
                print(f'  gamma-beta grid: {done}/{n_grid}')
    for g in gamma_vals:
        for lam in lambda_vals:
            T100s = [T100(solve_dynamics(f, s, gamma=g, lambda_=lam)) for f, s in pools]
            rows_gl.append({'gamma': g, 'lambda': lam,
                            'T100_mean': float(np.mean(T100s))})
    df_gb = pd.DataFrame(rows_gb)
    df_gl = pd.DataFrame(rows_gl)
    df_gb.to_csv(os.path.join(OUT, 'heatmap_gamma_beta.csv'), index=False)
    df_gl.to_csv(os.path.join(OUT, 'heatmap_gamma_lambda.csv'), index=False)
    print('gamma-beta grid (T100), rows=gamma, cols=beta:')
    print(df_gb.pivot(index='gamma', columns='beta', values='T100_mean').round(2).to_string())
    print('gamma-lambda grid (T100), rows=gamma, cols=lambda:')
    print(df_gl.pivot(index='gamma', columns='lambda', values='T100_mean').round(2).to_string())
    print(f'saved heatmap CSVs ({len(rows_gb) + len(rows_gl)} cells)')


if __name__ == '__main__':
    import time
    t0 = time.time()
    grid_run()
    print(f'\nheatmap done ({time.time()-t0:.0f}s)')
