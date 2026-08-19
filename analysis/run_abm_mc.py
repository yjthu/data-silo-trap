"""
Regenerate data/abm_monte_carlo.csv with n_mc=100 draws (Supplementary Table S14).

30 draws was too thin to support the tercile means and the bimodal histogram;
100 draws keeps runtime ~7 min while making the descriptive statistics solid.
"""
import os
import sys
import time
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.abm import run_abm

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')


def main(n=1000, n_mc=100, seed=42):
    t0 = time.time()
    rows = []
    for r in range(n_mc):
        res = run_abm(n_firms=n, seed=seed + r)
        rows.append(res)
        if (r + 1) % 20 == 0:
            print(f'  {r+1}/{n_mc} draws ({time.time()-t0:.0f}s)')
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'abm_monte_carlo.csv'), index=False)
    print(f'saved data/abm_monte_carlo.csv ({len(df)} rows, {time.time()-t0:.0f}s)')
    print('\nTercile means (mean +- std across draws):')
    for col in ['T100', 'adopt_high', 'adopt_mid', 'adopt_low']:
        print(f'  {col}: {df[col].mean():.4f} +- {df[col].std():.4f}')
    print('\nT100 distribution (10 bins):')
    import numpy as np
    h, e = np.histogram(df['T100'], bins=10)
    for i in range(10):
        print(f'  [{e[i]:.3f}, {e[i+1]:.3f}): {h[i]}')


if __name__ == '__main__':
    main()
