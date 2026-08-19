"""
Full analysis runner for the reconstructed FS model.

Runs: P0-P3, cross-country, OAT sensitivity, λ=0 closed-data,
multi-initial-condition stability, hysteresis, P3 decomposition.
Saves all results to data/.
"""
import sys, os, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.scenarios import (run_baseline, run_subsidy_scan, run_mandatory_scan,
                              run_platform_scan, run_cross_country)
from model.config import (RANDOM_SEED, OAT_RANGES, COUNTRY_PARAMS,
                           PLATFORM_BETA_BOOST, FIRM_TYPE_SHARES)
from model.firm_generator import generate_firms
from model.distance import generate_distance_matrix
from model.accuracy import precompute_type_stats
from model.dynamics import solve_dynamics, T100

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
os.makedirs(OUT, exist_ok=True)


def save(df, name):
    df.to_csv(os.path.join(OUT, name), index=False)
    print(f"  saved data/{name}")


def summarize(res):
    T = np.array([r['T100'] for r in res])
    return {'T100_mean': T.mean(), 'T100_std': T.std(),
            'rho_H': np.mean([r['rho_H'] for r in res]),
            'rho_M': np.mean([r['rho_M'] for r in res]),
            'rho_L': np.mean([r['rho_L'] for r in res])}


def step_p0(n=1000, mc=500):
    print("\n[1] P0 Baseline")
    t0 = time.time()
    res = run_baseline(n_firms=n, n_mc=mc, seed=RANDOM_SEED)
    s = summarize(res)
    print(f"  T100={s['T100_mean']:.4f}+-{s['T100_std']:.4f}  H={s['rho_H']:.3f} M={s['rho_M']:.3f} L={s['rho_L']:.3f}")
    save(pd.DataFrame(res), 'p0_baseline.csv')
    print(f"  {time.time()-t0:.0f}s")


def step_p1(n=1000, mc=500):
    print("\n[2] P1 Subsidy scan")
    t0 = time.time()
    res = run_subsidy_scan(np.arange(0, 1.05, 0.05), n_firms=n, n_mc=mc, seed=RANDOM_SEED)
    save(pd.DataFrame(res), 'p1_subsidy.csv')
    for r in res:
        print(f"  s={r['s']:.2f}: T100={r['T100_mean']:.4f}")
    print(f"  {time.time()-t0:.0f}s")


def step_p2(n=1000, mc=200):
    print("\n[3] P2 Mandatory + bonus")
    t0 = time.time()
    res = run_mandatory_scan(n_firms=n, n_mc=mc, seed=RANDOM_SEED)
    save(pd.DataFrame(res), 'p2_mandatory.csv')
    for r in res:
        print(f"  b={r['b']:.2f}: qH={r['q_H_mean']:.3f} qM={r['q_M_mean']:.3f} qL={r['q_L_mean']:.3f} SH={r['S_H_mean']:.3f}")
    print(f"  {time.time()-t0:.0f}s")


def step_p3(n=1000, mc=500):
    print("\n[4] P3 Platform scan")
    t0 = time.time()
    res = run_platform_scan(np.linspace(0, 0.5, 11), n_firms=n, n_mc=mc, seed=RANDOM_SEED)
    save(pd.DataFrame(res), 'p3_platform.csv')
    for r in res:
        print(f"  fee={r['fee']:.2f}: T100={r['T100_mean']:.4f}")
    print(f"  {time.time()-t0:.0f}s")


def step_cc(n=1000, mc=500):
    print("\n[5] Cross-country structural analysis")
    t0 = time.time()
    res = run_cross_country(n_firms=n, n_mc=mc, seed=RANDOM_SEED)
    save(pd.DataFrame(res), 'cross_country.csv')
    for r in res:
        print(f"  {r['country']}: T100={r['T100_mean']:.4f}+-{r['T100_std']:.4f}")
    print(f"  {time.time()-t0:.0f}s")


def step_oat(n=1000, mc=200):
    print("\n[6] OAT sensitivity")
    t0 = time.time()
    firms = generate_firms(n, seed=RANDOM_SEED)
    D, _ = generate_distance_matrix(firms, seed=RANDOM_SEED)
    stats = precompute_type_stats(firms, D)
    all_rows = []
    for param, vals in OAT_RANGES.items():
        for v in vals:
            kwargs = {param if param != 'lambda' else 'lambda_': v}
            T100s = []
            for r in range(mc):
                f = generate_firms(n, seed=RANDOM_SEED + r)
                D2, _ = generate_distance_matrix(f, seed=RANDOM_SEED + r)
                s2 = precompute_type_stats(f, D2)
                sol = solve_dynamics(f, s2, **kwargs)
                T100s.append(T100(sol))
            all_rows.append({'param': param, 'value': v,
                             'T100_mean': float(np.mean(T100s)),
                             'T100_std': float(np.std(T100s))})
    save(pd.DataFrame(all_rows), 'sensitivity_oat.csv')
    print(f"  {time.time()-t0:.0f}s")


def step_lambda0(n=1000, mc=500):
    print("\n[7] Lambda=0 closed-data scenario")
    t0 = time.time()
    res = run_baseline(n_firms=n, n_mc=mc, seed=RANDOM_SEED, lambda_=0.0)
    s = summarize(res)
    print(f"  lambda=0: T100={s['T100_mean']:.4f}+-{s['T100_std']:.4f}  H={s['rho_H']:.3f} M={s['rho_M']:.3f} L={s['rho_L']:.3f}")
    save(pd.DataFrame(res), 'lambda0_baseline.csv')
    print(f"  {time.time()-t0:.0f}s")


def step_multistart(n=1000):
    print("\n[8] Multi-initial-condition convergence")
    t0 = time.time()
    firms = generate_firms(n, seed=RANDOM_SEED)
    D, _ = generate_distance_matrix(firms, seed=RANDOM_SEED)
    stats = precompute_type_stats(firms, D)
    init_conditions = [(0.0,0.0,0.0), (0.3,0.3,0.3), (0.6,0.6,0.6), (1.0,1.0,1.0),
                       (0.9,0.5,0.1), (0.1,0.9,0.5), (0.5,0.1,0.9), (0.8,0.8,0.8)]
    rows = []
    for rho0 in init_conditions:
        sol = solve_dynamics(firms, stats, rho0=rho0)
        rows.append({'rho0': str(rho0), 'rho_H': sol.y[0,-1], 'rho_M': sol.y[1,-1],
                     'rho_L': sol.y[2,-1], 'T100': T100(sol)})
    for r in rows:
        print(f"  rho0={r['rho0']}: -> T100={r['T100']:.4f}")
    save(pd.DataFrame(rows), 'stability_multistart.csv')
    print(f"  {time.time()-t0:.0f}s")


def step_hysteresis(n=1000):
    print("\n[9] Hysteresis (forward/backward subsidy scans)")
    t0 = time.time()
    firms = generate_firms(n, seed=RANDOM_SEED)
    D, _ = generate_distance_matrix(firms, seed=RANDOM_SEED)
    stats = precompute_type_stats(firms, D)
    s_range = np.arange(0, 1.01, 0.02)
    rows = []
    # forward
    rho = [0.1, 0.02, 0.0]
    for s in s_range:
        sol = solve_dynamics(firms, stats, rho0=rho, subsidy=s)
        rho = [sol.y[0,-1], sol.y[1,-1], sol.y[2,-1]]
        rows.append({'direction': 'forward', 's': s, 'T100': T100(sol)})
    # backward
    rho = [0.9, 0.9, 0.9]
    for s in s_range[::-1]:
        sol = solve_dynamics(firms, stats, rho0=rho, subsidy=s)
        rho = [sol.y[0,-1], sol.y[1,-1], sol.y[2,-1]]
        rows.append({'direction': 'backward', 's': s, 'T100': T100(sol)})
    save(pd.DataFrame(rows), 'hysteresis.csv')
    print(f"  {time.time()-t0:.0f}s")


def step_p3_decomp(n=1000, mc=200):
    print("\n[10] P3 four-quadrant decomposition")
    t0 = time.time()
    cases = {
        'neither': dict(fee=0.0, beta_boost=0.0),
        'cost_only': dict(fee=0.10, beta_boost=0.0),
        'trust_only': dict(fee=0.0, beta_boost=PLATFORM_BETA_BOOST),
        'both': dict(fee=0.10, beta_boost=PLATFORM_BETA_BOOST),
    }
    rows = []
    for name, kw in cases.items():
        T100s = []
        for r in range(mc):
            f = generate_firms(n, seed=RANDOM_SEED + r)
            D2, _ = generate_distance_matrix(f, seed=RANDOM_SEED + r)
            s2 = precompute_type_stats(f, D2)
            sol = solve_dynamics(f, s2, **kw)
            T100s.append(T100(sol))
        rows.append({'case': name, 'T100_mean': float(np.mean(T100s)),
                     'T100_std': float(np.std(T100s))})
        print(f"  {name}: T100={rows[-1]['T100_mean']:.4f}")
    save(pd.DataFrame(rows), 'p3_decomposition.csv')
    print(f"  {time.time()-t0:.0f}s")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--mc', type=int, default=500)
    p.add_argument('--n', type=int, default=1000)
    p.add_argument('--quick', action='store_true')
    args = p.parse_args()

    if args.quick:
        n, mc, mc2 = 500, 50, 20
    else:
        n, mc, mc2 = args.n, args.mc, 200

    t_all = time.time()
    step_p0(n, mc)
    step_p1(n, mc)
    step_p2(n, mc2)
    step_p3(n, mc)
    step_cc(n, mc)
    step_oat(n, mc2)
    step_lambda0(n, mc)
    step_multistart(n)
    step_hysteresis(n)
    step_p3_decomp(n, mc2)
    print(f"\n=== DONE in {(time.time()-t_all)/60:.1f} min ===")
