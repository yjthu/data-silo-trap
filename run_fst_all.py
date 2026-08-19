"""Run F_ST sensitivity for ALL 6 countries (complete version)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from model.firm_generator import generate_firms
from model.distance import generate_distance_matrix
from model.accuracy import precompute_type_stats
from model.dynamics import solve_dynamics, T100
from model.config import RANDOM_SEED, COUNTRY_PARAMS

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

print("F_ST sensitivity for all 6 countries (n=1000, MC=100)...")
rows = []
for country, params in COUNTRY_PARAMS.items():
    shares = np.array([params['w_H'], params['w_M'], params['w_L']])
    for dscale in [0.8, 1.0, 1.2]:
        T100s = []
        for r in range(100):
            f = generate_firms(1000, shares=shares, seed=RANDOM_SEED + r)
            D, _ = generate_distance_matrix(f, seed=RANDOM_SEED + r)
            D = np.clip(D * dscale, 0, 1)
            s = precompute_type_stats(f, D)
            sol = solve_dynamics(f, s)
            T100s.append(T100(sol))
        rows.append({'country': country, 'D_scale': dscale,
                     'T100_mean': float(np.mean(T100s)),
                     'T100_std': float(np.std(T100s))})
        print(f"  {country} (D×{dscale}): T100={rows[-1]['T100_mean']:.4f}")
pd.DataFrame(rows).to_csv(os.path.join(DATA, 'cross_country_fst.csv'), index=False)
print("Done.")
