"""
Three-way decomposition of the two genetic-distance channels.

Separates the contributions of the two genetic-distance terms:
  θ (transferability decay in the shared pool) vs γ (misalignment penalty).

The model is estimated under three specifications (plus a "neither" control):
  1. θ-only  (γ = 0): only transferability decay
  2. γ-only  (θ → 0, i.e., ω ≈ 1): only misalignment penalty
  3. both     (baseline): both channels active
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from model.firm_generator import generate_firms
from model.distance import generate_distance_matrix
from model.accuracy import precompute_type_stats
from model.dynamics import solve_dynamics, T100

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

def run_spec(n_firms=1000, n_mc=200, seed=42, gamma=None, theta=None):
    """Run a channel specification. gamma=None→0, theta=None→tiny (ω≈1)."""
    T100s = []
    rho_Ls = []
    for r in range(n_mc):
        f = generate_firms(n_firms, seed=seed + r)
        D, _ = generate_distance_matrix(f, seed=seed + r)
        s = precompute_type_stats(f, D)
        sol = solve_dynamics(f, s, gamma=(0.0 if gamma is None else gamma),
                              theta=(0.01 if theta is None else theta))
        T100s.append(T100(sol))
        rho_Ls.append(sol.y[2, -1])
    return float(np.mean(T100s)), float(np.std(T100s)), float(np.mean(rho_Ls))


if __name__ == '__main__':
    print("Three-way genetic-distance channel decomposition (n=1000, MC=200)")
    print()

    # both (baseline)
    T_both, sd_both, L_both = run_spec(gamma=0.45, theta=2.0)
    print(f"both  (gamma=0.45, theta=2.0): T100={T_both:.4f}+-{sd_both:.4f}, L={L_both:.4f}")

    # theta-only (no misalignment penalty)
    T_theta, sd_theta, L_theta = run_spec(gamma=0.0, theta=2.0)
    print(f"theta-only (gamma=0):           T100={T_theta:.4f}+-{sd_theta:.4f}, L={L_theta:.4f}")

    # gamma-only (no transferability decay, ω≈1)
    T_gamma, sd_gamma, L_gamma = run_spec(gamma=0.45, theta=0.01)
    print(f"gamma-only (theta→0):           T100={T_gamma:.4f}+-{sd_gamma:.4f}, L={L_gamma:.4f}")

    # neither (both off)
    T_neither, sd_neither, L_neither = run_spec(gamma=0.0, theta=0.01)
    print(f"neither (gamma=0, theta→0):     T100={T_neither:.4f}+-{sd_neither:.4f}, L={L_neither:.4f}")

    df = pd.DataFrame([
        {'spec': 'both', 'T100': T_both, 'rho_L': L_both},
        {'spec': 'theta_only', 'T100': T_theta, 'rho_L': L_theta},
        {'spec': 'gamma_only', 'T100': T_gamma, 'rho_L': L_gamma},
        {'spec': 'neither', 'T100': T_neither, 'rho_L': L_neither},
    ])
    df.to_csv(os.path.join(DATA, 'channel_decomposition.csv'), index=False)
    print("\nsaved data/channel_decomposition.csv")
