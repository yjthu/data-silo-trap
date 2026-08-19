"""
Save ABM multi-start results to data/abm_multistart.csv (Supplementary Table S15).

Tests whether a FIXED realization (same I_i, x_i, F_i) admits multiple equilibria
by iterating the best-response map from nine distinct initial adoption states.
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.abm import run_abm_multistart, INIT_MODES

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')


def main():
    rows = run_abm_multistart(n_firms=1000, realizations=(42, 43, 44))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'abm_multistart.csv'), index=False)
    print(f"saved data/abm_multistart.csv ({len(df)} rows)")
    piv = df.pivot(index='seed', columns='init', values='T100')
    print(piv.round(4).to_string())


if __name__ == '__main__':
    main()
