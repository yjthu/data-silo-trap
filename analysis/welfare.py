"""
welfare.py — Social welfare assessment.

Welfare of firm i at a given adoption state:
    W_i = 1 + δ·a_i − F_i   (adopter)
    W_i = 1 + δ·λ·β·S_φ    (non-adopter)
with a_i = α·I_i + β·S_φ(i) − γ·d_φ(i) (adopter accuracy).

Part A (type model):
  - W0  = welfare at the baseline equilibrium.
  - W_full = welfare under a forced full-adoption counterfactual (ρ = 1).
  - Net welfare of policies (gross welfare minus policy expenditure):
      P1 subsidy s: expenditure = Σ_adopters (F_i − F'_i), F' = max(0.05F, F(1−s))
      P3 platform fee: expenditure = Σ_adopters (F_i − F_plat)
      P2 quality bonus b: welfare change from the induced pool quality S(q)
          (trickle-down) minus bonus expenditure b × #contributors.
Part B (ABM):
  - For fixed realizations with multiple equilibria (multi-start), compare
    welfare at the LOW equilibrium vs the HIGH equilibrium (coordination
    failure test: same economy, two equilibria, which is better).

Output: data/welfare.csv (Part A) and data/welfare_abm.csv (Part B).
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import (RANDOM_SEED, ALPHA, BETA, LAMBDA, GAMMA, THETA, DELTA,
                          FIRM_TYPE_SHARES, PLATFORM_BETA_BOOST)
from model.firm_generator import generate_firms
from model.distance import generate_distance_matrix
from model.accuracy import precompute_type_stats, shared_pool_and_centroid, firm_premiums
from model.dynamics import solve_dynamics, T100

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
W = FIRM_TYPE_SHARES


def _welfare(firms, S, d, adopt, F_effective=None):
    """Total welfare at a given state (type-level pool terms, individual I, F)."""
    types, I = firms['types'], firms['I']
    F = F_effective if F_effective is not None else firms['F']
    S_i = np.array([S[t] for t in types])
    d_i = np.array([d[t] for t in types])
    adopter = 1.0 + DELTA * (ALPHA * I + BETA * S_i - GAMMA * d_i) - F
    non_adopter = 1.0 + DELTA * LAMBDA * BETA * S_i
    return float(np.where(adopt, adopter, non_adopter).sum())


def _equilibrium_state(firms, stats, subsidy=0.0, fee=0.0, beta_boost=0.0):
    """Solve to equilibrium and return (S, d, adopt, F_eff)."""
    import copy
    f = copy.deepcopy(firms)
    reduction = max(subsidy, fee)
    if reduction > 0:
        f['F'] = np.maximum(0.05 * f['F'], f['F'] * (1 - reduction))
    beta_eff = BETA + beta_boost
    sol = solve_dynamics(f, stats, subsidy=subsidy, fee=fee, beta_boost=beta_boost)
    rho = sol.y[:, -1]
    # recompute pool terms with beta_eff? solve_dynamics already used beta_eff
    S, d = shared_pool_and_centroid(stats, rho)
    V = firm_premiums(f, S, d, beta=beta_eff)
    adopt = f['F'] < V
    return S, d, adopt, f['F']


def part_a(n=1000, mc=300, seed=RANDOM_SEED):
    print('Part A: type-model welfare')
    rows = []
    for r in range(mc):
        f = generate_firms(n, seed=seed + r)
        D, _ = generate_distance_matrix(f, seed=seed + r)
        stats = precompute_type_stats(f, D)

        # baseline equilibrium
        S0, d0, adopt0, F0 = _equilibrium_state(f, stats)
        w0 = _welfare(f, S0, d0, adopt0)

        # forced full adoption (rho = 1): recompute S, d at full adoption
        S_full, d_full = shared_pool_and_centroid(stats, np.array([1.0, 1.0, 1.0]))
        w_full = _welfare(f, S_full, d_full, np.ones(n, dtype=bool))

        # P1 subsidy s = 0.5
        s = 0.5
        S_s, d_s, adopt_s, F_s = _equilibrium_state(f, stats, subsidy=s)
        w_s_gross = _welfare(f, S_s, d_s, adopt_s, F_s)
        expend_s = float((f['F'] - F_s)[adopt_s].sum())
        w_s_net = w_s_gross - expend_s

        # P3 platform fee = 0.10 + trust boost
        fee = 0.10
        S_p, d_p, adopt_p, F_p = _equilibrium_state(f, stats, fee=fee,
                                                    beta_boost=PLATFORM_BETA_BOOST)
        w_p_gross = _welfare(f, S_p, d_p, adopt_p, F_p)
        expend_p = float((f['F'] - F_p)[adopt_p].sum())
        w_p_net = w_p_gross - expend_p

        # P2 quality bonus b = 0.30: type-level pool quality from contributions q
        I = f['I']
        Istar = 0.5
        types = f['types']
        q = np.where(I >= Istar, Istar, 0.0)
        q_bar = np.array([q[types == t].mean() if (types == t).any() else 0.0
                          for t in range(3)])
        omega_t = stats['omega']  # type-level omega matrix
        S_q = np.zeros(3)
        for phi in range(3):
            num = float((W * q_bar * omega_t[phi]).sum())
            den = float((W * omega_t[phi]).sum())
            S_q[phi] = num / den if den > 1e-12 else 0.0
        delta_pool = np.array([S_q[t] for t in types]) - np.array([S0[t] for t in types])
        gain = DELTA * BETA * float(delta_pool[adopt0].sum())
        gain += DELTA * LAMBDA * BETA * float(delta_pool[~adopt0].sum())
        n_contrib = int((q > 0).sum())
        expend_b = 0.30 * n_contrib
        w_b_net = gain - expend_b

        rows.append({
            'W0': w0 / n, 'W_full': w_full / n,
            'W_subsidy_net': (w_s_net - w0) / n,
            'W_platform_net': (w_p_net - w0) / n,
            'W_bonus_net': w_b_net / n,
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'welfare.csv'), index=False)
    print(df.describe().loc[['mean', 'std']].round(4))
    print(f'saved data/welfare.csv (mc={mc})')
    return df


def part_b(realizations=(42, 43, 44), n=1000):
    """ABM welfare at the low vs high equilibrium of fixed realizations."""
    print('\nPart B: ABM low- vs high-equilibrium welfare')
    from analysis.abm import _generate_realization, _iterate_adoption, _initial_adoption
    rows = []
    for seed in realizations:
        I, x, F = _generate_realization(n, seed)
        rng = np.random.default_rng(seed + 10_000)
        for mode, tag in [('top10', 'low'), ('all', 'high')]:
            adopt0 = _initial_adoption(I, x, mode, rng)
            adopt = _iterate_adoption(I, x, F, adopt0)
            # recompute pool at this equilibrium
            n_a = adopt.sum()
            if n_a == 0:
                continue
            I_adopt, x_adopt = I[adopt], x[adopt]
            S = np.zeros(n)
            d = np.zeros(n)
            for i in range(n):
                w = np.exp(-THETA * np.abs(x[i] - x_adopt))
                den = w.sum()
                S[i] = (w * I_adopt).sum() / den if den > 1e-12 else 0.0
                d[i] = np.abs(x[i] - x_adopt).mean()
            # welfare
            welf = np.where(adopt, 1.0 + DELTA * (ALPHA * I + BETA * S - GAMMA * d) - F,
                            1.0 + DELTA * LAMBDA * BETA * S)
            rows.append({'seed': seed, 'equilibrium': tag,
                         'T100': float(adopt.mean()),
                         'W_per_firm': float(welf.mean())})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'welfare_abm.csv'), index=False)
    print(df.round(4).to_string(index=False))
    print('saved data/welfare_abm.csv')
    return df


def part_c(n=1000, mc=200, seed=RANDOM_SEED):
    """Policy welfare scans: net welfare per firm vs subsidy rate / fee / bonus."""
    print('\nPart C: policy welfare scans')
    import copy
    s_vals = np.arange(0, 1.05, 0.1)
    fee_vals = np.arange(0, 0.55, 0.1)
    b_vals = [0.26, 0.30, 0.40, 0.50]
    rows = []
    for r in range(mc):
        f = generate_firms(n, seed=seed + r)
        D, _ = generate_distance_matrix(f, seed=seed + r)
        stats = precompute_type_stats(f, D)
        S0, d0, adopt0, F0 = _equilibrium_state(f, stats)
        w0 = _welfare(f, S0, d0, adopt0)
        for s in s_vals:
            S_s, d_s, adopt_s, F_s = _equilibrium_state(f, stats, subsidy=s)
            wg = _welfare(f, S_s, d_s, adopt_s, F_s)
            exp = float((f['F'] - F_s)[adopt_s].sum())
            rows.append({'policy': 'subsidy', 'value': s,
                         'W_net': (wg - exp - w0) / n})
        for fee in fee_vals:
            S_p, d_p, adopt_p, F_p = _equilibrium_state(f, stats, fee=fee,
                                                        beta_boost=PLATFORM_BETA_BOOST)
            wg = _welfare(f, S_p, d_p, adopt_p, F_p)
            exp = float((f['F'] - F_p)[adopt_p].sum())
            rows.append({'policy': 'platform', 'value': fee,
                         'W_net': (wg - exp - w0) / n})
        # bonus: within-regime net benefit (pool from q vs zero pool)
        I = f['I']
        Istar = 0.5
        types = f['types']
        omega_t = stats['omega']
        for b in b_vals:
            q = np.where(I >= Istar, Istar, 0.0)
            q_bar = np.array([q[types == t].mean() if (types == t).any() else 0.0
                              for t in range(3)])
            S_q = np.zeros(3)
            for phi in range(3):
                num = float((W * q_bar * omega_t[phi]).sum())
                den = float((W * omega_t[phi]).sum())
                S_q[phi] = num / den if den > 1e-12 else 0.0
            gain = DELTA * BETA * float(sum(S_q[t] * (types == t).sum() for t in range(3)))
            n_contrib = int((q > 0).sum())
            rows.append({'policy': 'bonus', 'value': b,
                         'W_net': (gain - b * n_contrib) / n})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'welfare_scan.csv'), index=False)
    print(df.groupby(['policy', 'value'])['W_net'].mean().round(5).unstack().T.to_string())
    print('saved data/welfare_scan.csv')
    return df


if __name__ == '__main__':
    import time
    t0 = time.time()
    part_a()
    part_b()
    part_c()
    print(f'\nwelfare analysis done ({time.time()-t0:.0f}s)')
