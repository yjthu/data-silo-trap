"""
Policy scenarios (P0–P3).

P0/P1/P3: adoption = fraction of firms with F_i < V_i (empirical, finite-population).
P2:      mandatory sharing — all firms participate; outcome is measured by the
         induced pool quality S(q), a different construct from the adoption
         rates of P0/P1/P3 because participation is compulsory.
"""
import numpy as np
from .config import (RANDOM_SEED, SUBSIDY_RANGE, BONUS_LEVELS, QUALITY_THRESHOLD,
                      BETA, DELTA, ALPHA, LAMBDA, GAMMA, FIRM_TYPE_SHARES,
                      COUNTRY_PARAMS, PLATFORM_BETA_BOOST)
from .firm_generator import generate_firms
from .distance import generate_distance_matrix
from .accuracy import (precompute_type_stats, shared_pool_and_centroid,
                       firm_premiums, target_adoption)
from .dynamics import solve_dynamics, T100


def _summarize(arr):
    arr = np.array(arr)
    return float(np.mean(arr)), float(np.std(arr))


# ---------------------------------------------------------------------------
# P0 — Baseline
# ---------------------------------------------------------------------------
def run_baseline(n_firms=1000, n_mc=500, seed=None, **kwargs):
    if seed is None: seed = RANDOM_SEED
    results = []
    for r in range(n_mc):
        f = generate_firms(n_firms, seed=seed + r)
        D, _ = generate_distance_matrix(f, seed=seed + r)
        stats = precompute_type_stats(f, D)
        sol = solve_dynamics(f, stats, **kwargs)
        results.append({'rho_H': sol.y[0, -1], 'rho_M': sol.y[1, -1],
                        'rho_L': sol.y[2, -1], 'T100': T100(sol)})
    return results


# ---------------------------------------------------------------------------
# P1 — Cost subsidy scan
# ---------------------------------------------------------------------------
def run_subsidy_scan(s_values=None, n_firms=1000, n_mc=500, seed=None, **kwargs):
    if s_values is None: s_values = SUBSIDY_RANGE
    if seed is None: seed = RANDOM_SEED
    firms_pool = [generate_firms(n_firms, seed=seed + r) for r in range(n_mc)]
    D_pool = [generate_distance_matrix(f, seed=seed + r)[0] for r, f in enumerate(firms_pool)]
    all_results = []
    for s_val in s_values:
        T100s, rho_Ls = [], []
        for r in range(n_mc):
            stats = precompute_type_stats(firms_pool[r], D_pool[r])
            sol = solve_dynamics(firms_pool[r], stats, subsidy=s_val, **kwargs)
            T100s.append(T100(sol)); rho_Ls.append(sol.y[2, -1])
        m, std = _summarize(T100s)
        all_results.append({'s': s_val, 'T100_mean': m, 'T100_std': std,
                            'rho_L_mean': float(np.mean(rho_Ls))})
    return all_results


# ---------------------------------------------------------------------------
# P2 — Mandatory sharing with quality bonus (clean reconstruction)
# ---------------------------------------------------------------------------
def _nash_quality(firms, D, b, c0=1.0):
    """
    Nash quality contributions under mandatory sharing.

    Firm i chooses q_i ∈ [0, I_i] to maximize
        −c0·q_i² + 1[q_i ≥ I*]·b
    (the own-data, pool, penalty, and cost terms do not depend on q_i in the
    mean-field limit with self-exclusion).

    Best response: q_i = I* if (I_i ≥ I* and b > c0·I*²), else q_i = 0.
    """
    n = firms['n_firms']
    I = firms['I']
    Istar = QUALITY_THRESHOLD
    q = np.zeros(n)
    mask = (I >= Istar) & (b > c0 * Istar**2)
    q[mask] = Istar
    return q


def _pool_quality(firms, D, q):
    """Weighted-average pool quality S_φ induced by contributions q."""
    n = firms['n_firms']
    types = firms['types']
    # firm-level pool: weighted average of q_j exp(-θ D_ij)
    theta = 2.0
    omega = np.exp(-theta * D)
    S_firm = np.zeros(n)
    for i in range(n):
        den = omega[i].sum()
        if den > 1e-12:
            S_firm[i] = (q * omega[i]).sum() / den
    # type-level means
    S = np.array([S_firm[types == t].mean() if (types == t).any() else 0 for t in range(3)])
    return S


def run_mandatory_scan(b_values=None, n_firms=1000, n_mc=200, seed=None):
    if b_values is None: b_values = BONUS_LEVELS
    if seed is None: seed = RANDOM_SEED
    all_results = []
    for b_val in b_values:
        q_Hs, q_Ms, q_Ls, S_Hs, S_Ms, S_Ls = [], [], [], [], [], []
        for r in range(n_mc):
            f = generate_firms(n_firms, seed=seed + r)
            D, _ = generate_distance_matrix(f, seed=seed + r)
            q = _nash_quality(f, D, b_val)
            S = _pool_quality(f, D, q)
            for t, lst in enumerate([q_Hs, q_Ms, q_Ls]):
                m = f['types'] == t
                lst.append(q[m].mean() if m.any() else 0)
            for t, lst in enumerate([S_Hs, S_Ms, S_Ls]):
                lst.append(S[t])
        all_results.append({
            'b': b_val,
            'q_H_mean': float(np.mean(q_Hs)), 'q_M_mean': float(np.mean(q_Ms)),
            'q_L_mean': float(np.mean(q_Ls)),
            'S_H_mean': float(np.mean(S_Hs)), 'S_M_mean': float(np.mean(S_Ms)),
            'S_L_mean': float(np.mean(S_Ls)),
            # fraction of firms contributing at threshold (a clean participation measure)
            'frac_threshold': float((np.array(q_Hs) + np.array(q_Ms)).mean() > 0),
        })
    return all_results


# ---------------------------------------------------------------------------
# P3 — Platform model (cost reduction + trust enhancement)
# ---------------------------------------------------------------------------
def run_platform_scan(fee_values=None, n_firms=1000, n_mc=500, seed=None,
                      trust_boost=PLATFORM_BETA_BOOST, **kwargs):
    if fee_values is None: fee_values = np.linspace(0, 0.5, 11)
    if seed is None: seed = RANDOM_SEED
    firms_pool = [generate_firms(n_firms, seed=seed + r) for r in range(n_mc)]
    D_pool = [generate_distance_matrix(f, seed=seed + r)[0] for r, f in enumerate(firms_pool)]
    all_results = []
    for fee in fee_values:
        T100s, rho_Ls = [], []
        for r in range(n_mc):
            stats = precompute_type_stats(firms_pool[r], D_pool[r])
            sol = solve_dynamics(firms_pool[r], stats, fee=fee, beta_boost=trust_boost, **kwargs)
            T100s.append(T100(sol)); rho_Ls.append(sol.y[2, -1])
        m, std = _summarize(T100s)
        all_results.append({'fee': fee, 'T100_mean': m, 'T100_std': std,
                            'rho_L_mean': float(np.mean(rho_Ls))})
    return all_results


# ---------------------------------------------------------------------------
# Cross-country structural scenario analysis
# ---------------------------------------------------------------------------
def run_cross_country(n_firms=1000, n_mc=500, seed=None, **kwargs):
    if seed is None: seed = RANDOM_SEED
    all_results = []
    for country, params in COUNTRY_PARAMS.items():
        shares = np.array([params['w_H'], params['w_M'], params['w_L']])
        T100s = []
        for r in range(n_mc):
            f = generate_firms(n_firms, shares=shares, seed=seed + r)
            D, _ = generate_distance_matrix(f, seed=seed + r)
            stats = precompute_type_stats(f, D)
            sol = solve_dynamics(f, stats, **kwargs)
            T100s.append(T100(sol))
        m, std = _summarize(T100s)
        all_results.append({'country': country, 'T100_mean': m, 'T100_std': std,
                            'w_H': params['w_H'], 'w_M': params['w_M'], 'w_L': params['w_L']})
    return all_results
