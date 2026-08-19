"""
gap_items.py — Close the remaining reviewer gaps.

1. Cross-effect characterization (2.11 iii): numerical ∂V_φ/∂ρ_ψ at the
   baseline equilibrium (sign structure of strategic interdependence).
2. Local stability (2.10): eigenvalues of the best-response dynamics
   Jacobian k·(Dρ* − I) at the equilibrium.
3. Adjustment-speed neutrality (R3-3): equilibria across k ∈ {0.05..1.0}.
4. Trust-boost sensitivity (P3): T100 for beta boost ∈ {0.05, 0.10, 0.15}.
5. Cost-scaled cross-country (editor 3): T100 with country-specific
   infrastructure cost multipliers.

Outputs:
  data/cross_effect.csv       (3x3 numerical ∂V_φ/∂ρ_ψ)
  data/eigenvalues.csv        (eigenvalues of k(Dρ*−I) at equilibrium)
  data/k_neutrality.csv       (T100 vs k)
  data/trust_boost.csv        (T100 vs beta boost at fee=0.10)
  data/cross_country_cost.csv (T100 with cost multipliers)
"""
import os
import sys
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import (RANDOM_SEED, ALPHA, BETA, LAMBDA, GAMMA, THETA, DELTA,
                          FIRM_TYPE_SHARES, PLATFORM_BETA_BOOST)
from model.firm_generator import generate_firms
from model.distance import generate_distance_matrix
from model.accuracy import precompute_type_stats, shared_pool_and_centroid, firm_premiums
from model.dynamics import solve_dynamics, T100

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
W = FIRM_TYPE_SHARES


def _V_type_mean(firms, stats, rho):
    S, d = shared_pool_and_centroid(stats, rho)
    V = firm_premiums(firms, S, d)
    types = firms['types']
    return np.array([V[types == t].mean() if (types == t).any() else 0.0 for t in range(3)])


def _rho_star(firms, stats, rho):
    S, d = shared_pool_and_centroid(stats, rho)
    V = firm_premiums(firms, S, d)
    types = firms['types']
    F = firms['F']
    out = np.zeros(3)
    for t in range(3):
        m = types == t
        if m.any():
            out[t] = (F[m] < V[m]).mean()
    return np.clip(out, 0, 1)


def cross_effect_and_eigenvalues(n=1000, seed=RANDOM_SEED, h=0.01):
    f = generate_firms(n, seed=seed)
    D, _ = generate_distance_matrix(f, seed=seed)
    stats = precompute_type_stats(f, D)
    sol = solve_dynamics(f, stats)
    rho_eq = sol.y[:, -1]

    # cross-effect: dV_φ / dρ_ψ at equilibrium
    V0 = _V_type_mean(f, stats, rho_eq)
    JV = np.zeros((3, 3))
    for j in range(3):
        rp = rho_eq.copy(); rp[j] = min(1.0, rho_eq[j] + h)
        rm = rho_eq.copy(); rm[j] = max(0.0, rho_eq[j] - h)
        Vp = _V_type_mean(f, stats, rp)
        Vm = _V_type_mean(f, stats, rm)
        JV[:, j] = (Vp - Vm) / (rp[j] - rm[j])
    ce = pd.DataFrame(JV.round(4), index=['V_H', 'V_M', 'V_L'],
                      columns=['d/dρ_H', 'd/dρ_M', 'd/dρ_L'])
    ce.to_csv(os.path.join(OUT, 'cross_effect.csv'))
    print('cross-effect dV_φ/dρ_ψ at equilibrium:')
    print(ce.to_string())
    print(f'  (rho_eq = {np.round(rho_eq, 4)})')

    # local stability: eigenvalues of k·(Dρ* − I)
    rho_star0 = _rho_star(f, stats, rho_eq)
    J = np.zeros((3, 3))
    for j in range(3):
        rp = rho_eq.copy(); rp[j] = min(1.0, rho_eq[j] + h)
        rm = rho_eq.copy(); rm[j] = max(0.0, rho_eq[j] - h)
        J[:, j] = (_rho_star(f, stats, rp) - _rho_star(f, stats, rm)) / (rp[j] - rm[j])
    k = 0.2
    A = k * (J - np.eye(3))
    ev = np.linalg.eigvals(A)
    pd.DataFrame({'eigenvalue': [complex(e).real for e in ev],
                  'imag': [complex(e).imag for e in ev]}).to_csv(
        os.path.join(OUT, 'eigenvalues.csv'), index=False)
    print('\nlocal stability (eigenvalues of k(Dρ* − I)):')
    for e in ev:
        print(f'  {complex(e):.5f}  (Re = {complex(e).real:.4f})')
    print('  (all real parts must be negative for local stability)')


def k_neutrality(n=1000, seed=RANDOM_SEED):
    print('\nk-neutrality (T100 vs k):')
    f = generate_firms(n, seed=seed)
    D, _ = generate_distance_matrix(f, seed=seed)
    stats = precompute_type_stats(f, D)
    rows = []
    for k in [0.05, 0.1, 0.2, 0.5, 1.0]:
        sol = solve_dynamics(f, stats, k=k)
        rows.append({'k': k, 'T100': T100(sol),
                     'rho_H': sol.y[0, -1], 'rho_M': sol.y[1, -1], 'rho_L': sol.y[2, -1]})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'k_neutrality.csv'), index=False)
    print(df.round(6).to_string(index=False))


def trust_boost_sensitivity(n=1000, mc=200, seed=RANDOM_SEED):
    print('\ntrust-boost sensitivity (T100 at fee=0.10, boost 0.05/0.10/0.15):')
    rows = []
    for boost in [0.05, 0.10, 0.15]:
        T100s = []
        for r in range(mc):
            f = generate_firms(n, seed=seed + r)
            D, _ = generate_distance_matrix(f, seed=seed + r)
            stats = precompute_type_stats(f, D)
            sol = solve_dynamics(f, stats, fee=0.10, beta_boost=boost)
            T100s.append(T100(sol))
        rows.append({'beta_boost': boost, 'T100_mean': float(np.mean(T100s)),
                     'T100_std': float(np.std(T100s))})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'trust_boost.csv'), index=False)
    print(df.round(4).to_string(index=False))


COST_MULT = {'China': 1.0, 'India': 1.2, 'Africa': 1.5, 'Brazil': 0.8,
             'Indonesia': 1.2, 'Nigeria': 1.5}


def cost_scaled_cross_country(n=1000, mc=200, seed=RANDOM_SEED):
    """Cross-country with local infrastructure cost multipliers (editor item 3)."""
    print('\ncost-scaled cross-country:')
    from model.scenarios import run_cross_country  # not used; do it directly
    rows = []
    for country, mult in COST_MULT.items():
        shares = None
        # reuse COUNTRY_PARAMS shares
        from model.config import COUNTRY_PARAMS
        p = COUNTRY_PARAMS[country]
        shares = np.array([p['w_H'], p['w_M'], p['w_L']])
        T100s = []
        for r in range(mc):
            f = generate_firms(n, shares=shares, seed=seed + r)
            D, _ = generate_distance_matrix(f, seed=seed + r)
            stats = precompute_type_stats(f, D)
            sol = solve_dynamics(f, stats, F_scale=mult)
            T100s.append(T100(sol))
        rows.append({'country': country, 'F_scale': mult,
                     'T100_mean': float(np.mean(T100s)), 'T100_std': float(np.std(T100s))})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'cross_country_cost.csv'), index=False)
    print(df.round(4).to_string(index=False))


if __name__ == '__main__':
    import time
    t0 = time.time()
    cross_effect_and_eigenvalues()
    k_neutrality()
    trust_boost_sensitivity()
    cost_scaled_cross_country()
    print(f'\ngap items done ({time.time()-t0:.0f}s)')
