"""
run_all.py — One-command reproducible pipeline for the Data Silo Trap analysis.

Regenerates, in order and with fixed seeds (42), every number behind the
manuscript: simulation data (P0–P3, cross-country, F_ST, F-scale, OAT,
closed-data, stability, hysteresis, decomposition, channel decomposition,
premium statistics, ABM Monte Carlo, ABM multi-start, pool-size robustness),
the seven figures, and the consolidated supplementary-tables CSV — then
VERIFIES the key published values against the manuscript.

Usage (run from the repository root):
    python run_all.py --full          # regenerate everything (~60–90 min)
    python run_all.py --quick         # smoke test: compile + deterministic
                                      #   solve + regenerate artifacts from
                                      #   EXISTING data (~2 min, no data
                                      #   files overwritten)
    python run_all.py --verify-only   # only check data/ against the
                                      #   manuscript's published values

Each stage can also be run standalone (see README). All stages are
deterministic given the seeds in model/config.py (default 42).

Paper parameters: n=1,000 firms; MC=300 (baseline/subsidy/platform/cross-
country/closed-data), MC=200 (P2/OAT/decomposition/channel/pool-size),
ABM MC=100, ABM multi-start 3 realizations x 9 initial states.
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, 'data')

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ---------------------------------------------------------------------------
# Stages (full pipeline, in order)
# ---------------------------------------------------------------------------
DATA_STAGES = [
    ('P0–P3, cross-country, OAT, closed-data, stability, hysteresis, decomposition '
     '(n=1000, MC=300/200)',
     [sys.executable, 'run_analysis.py', '--mc', '300']),
    ('Channel decomposition (γ vs θ, MC=200)',
     [sys.executable, 'analysis', 'channel_decomposition.py']),
    ('F_ST sensitivity (6 countries x 3 scales)', [sys.executable, 'run_fst_all.py']),
    ('Cost-scale robustness (F x 0.5–1.5)', [sys.executable, 'run_fscale.py']),
    ('Firm-level premium statistics', [sys.executable, 'analysis', 'premium_stats.py']),
    ('ABM Monte Carlo (100 draws)', [sys.executable, 'analysis', 'run_abm_mc.py']),
    ('ABM multi-start (3 realizations x 9 initial states)',
     [sys.executable, 'analysis', 'run_abm_multistart.py']),
    ('Pool-size externality robustness', [sys.executable, 'analysis', 'poolsize_robustness.py']),
    ('Welfare assessment (type model + ABM equilibria)', [sys.executable, 'analysis', 'welfare.py']),
    ('Joint parameter-range sensitivity (heatmaps)', [sys.executable, 'analysis', 'heatmap_robustness.py']),
    ('Gap items (cross-effects, stability, k, trust boost, cost scaling)',
     [sys.executable, 'analysis', 'gap_items.py']),
]

ARTIFACT_STAGES = [
    ('Figures (Fig 1–5)', [sys.executable, 'make_figures.py']),
    ('Supplementary tables CSV', [sys.executable, 'export_tables.py']),
]


def run_stage(name, cmd):
    t0 = time.time()
    print(f'\n=== [{name}] ===', flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    dt = time.time() - t0
    status = 'OK' if r.returncode == 0 else f'FAILED (exit {r.returncode})'
    print(f'--- {name}: {status} ({dt:.0f}s)', flush=True)
    return r.returncode == 0


def sanity_solve():
    """Deterministic single-population solve; asserts the seed-42 baseline."""
    code = f"""
import sys; sys.path.insert(0, r'{ROOT}')
from model.firm_generator import generate_firms
from model.distance import generate_distance_matrix
from model.accuracy import precompute_type_stats
from model.dynamics import solve_dynamics, T100
f = generate_firms(1000, seed=42)
D, _ = generate_distance_matrix(f, seed=42)
s = precompute_type_stats(f, D)
sol = solve_dynamics(f, s)
t = T100(sol)
assert abs(t - 0.1505) < 1e-5, f'T100={{t}}'
print('deterministic solve OK: T100(seed42)=%.4f' % t)
"""
    t0 = time.time()
    print('\n=== [Deterministic sanity solve (no files written)] ===', flush=True)
    r = subprocess.run([sys.executable, '-c', code], cwd=ROOT)
    print(f'--- sanity solve: {"OK" if r.returncode == 0 else "FAILED"} '
          f'({time.time()-t0:.0f}s)', flush=True)
    return r.returncode == 0


def compile_all():
    t0 = time.time()
    print('\n=== [Compile check (all Python files)] ===', flush=True)
    r = subprocess.run([sys.executable, '-m', 'compileall', '-q',
                        'model', 'analysis', 'make_figures.py',
                        'export_tables.py', 'run_analysis.py',
                        'run_fst_all.py', 'run_fscale.py'], cwd=ROOT)
    print(f'--- compileall: {"OK" if r.returncode == 0 else "FAILED"} '
          f'({time.time()-t0:.0f}s)', flush=True)
    return r.returncode == 0


# ---------------------------------------------------------------------------
# Verification against the manuscript's published values
# ---------------------------------------------------------------------------
EXPECTED = [
    ('p0_baseline.csv', lambda df: df['T100'].mean(), 0.1700, 'baseline T100'),
    ('p0_baseline.csv', lambda df: df['rho_H'].mean(), 0.8350, 'baseline rho_H'),
    ('p0_baseline.csv', lambda df: df['rho_L'].mean(), 0.0613, 'baseline rho_L'),
    ('premium_stats.csv', lambda df: df[df['type'] == 'H']['V_mean'].iloc[0], 0.3845, 'V_H'),
    ('premium_stats.csv', lambda df: df[df['type'] == 'L']['V_mean'].iloc[0], -0.0530, 'V_L'),
    ('premium_stats.csv', lambda df: df[df['type'] == 'L']['frac_pos'].iloc[0], 0.1861,
     'frac L firms with V>0'),
    ('p1_subsidy.csv', lambda df: df[df['s'] == 1.0]['T100_mean'].iloc[0], 0.2807,
     'subsidy s=1 T100'),
    ('p2_mandatory.csv', lambda df: df[df['b'] > 0.25]['q_H_mean'].max(), 0.492, 'P2 q_H'),
    ('p2_mandatory.csv', lambda df: df[df['b'] > 0.25]['q_L_mean'].max(), 0.010, 'P2 q_L'),
    ('p3_platform.csv', lambda df: df[df['fee'] == 0.5]['T100_mean'].iloc[0], 0.2781,
     'platform fee=0.5'),
    ('p3_decomposition.csv',
     lambda df: df[df['case'] == 'both']['T100_mean'].iloc[0], 0.2243, 'decomp both (fee=0.1)'),
    ('p3_decomposition.csv',
     lambda df: df[df['case'] == 'cost_only']['T100_mean'].iloc[0], 0.1870, 'decomp cost-only'),
    ('channel_decomposition.csv',
     lambda df: df[df['spec'] == 'gamma_only']['T100'].iloc[0], 0.1553, 'gamma-only T100'),
    ('channel_decomposition.csv',
     lambda df: df[df['spec'] == 'theta_only']['T100'].iloc[0], 0.9991, 'theta-only T100'),
    ('lambda0_baseline.csv', lambda df: df['T100'].mean(), 0.2548, 'lambda=0 T100'),
    ('sensitivity_oat.csv',
     lambda df: df[(df['param'] == 'lambda') & (df['value'] == 0.8)]['T100_mean'].iloc[0],
     0.0555, 'OAT lambda=0.8'),
    ('sensitivity_oat.csv',
     lambda df: df[(df['param'] == 'gamma')]['T100_mean'].max(), 0.5425, 'OAT gamma max'),
    ('cross_country.csv',
     lambda df: df[df['country'] == 'China']['T100_mean'].iloc[0], 0.1700, 'China baseline'),
    ('cross_country_fst.csv',
     lambda df: df[df['D_scale'] == 0.8]['T100_mean'].mean(), 0.236, 'F_ST D x 0.8'),
    ('fscale_robustness.csv',
     lambda df: df[df['F_scale'] == 0.5]['T100_mean'].iloc[0], 0.2403, 'F x 0.5'),
    ('fscale_robustness.csv',
     lambda df: df[df['F_scale'] == 1.5]['T100_mean'].iloc[0], 0.0707, 'F x 1.5'),
    ('stability_multistart.csv', lambda df: df['T100'].mean(), 0.1505, 'multistart T100'),
    ('hysteresis.csv',
     lambda df: float(abs(df[df['direction'] == 'forward'].sort_values('s')['T100'].values
                          - df[df['direction'] == 'backward'].sort_values('s')['T100'].values).max()),
     0.0018, 'hysteresis max diff'),
    ('abm_monte_carlo.csv', lambda df: df['adopt_high'].mean(), 0.3181, 'ABM high tercile'),
    ('abm_monte_carlo.csv', lambda df: df['adopt_low'].mean(), 0.0858, 'ABM low tercile'),
    ('abm_multistart.csv',
     lambda df: df[(df['seed'] == 42) & (df['init'] == 'top10')]['T100'].iloc[0], 0.055,
     'ABM multistart top10'),
    ('abm_multistart.csv',
     lambda df: df[(df['seed'] == 42) & (df['init'] == 'all')]['T100'].iloc[0], 0.889,
     'ABM multistart all'),
    ('poolsize_robustness.csv',
     lambda df: df[df['N_0'] == 0.0]['T100_mean'].iloc[0], 0.1691, 'poolsize N0=0'),
    ('poolsize_robustness.csv',
     lambda df: df[df['N_0'] == 0.3]['T100_mean'].iloc[0], 0.0331, 'poolsize N0=0.3'),
    ('welfare.csv', lambda df: df['W0'].mean(), 1.0845, 'baseline welfare per firm'),
    ('welfare.csv', lambda df: df['W_full'].mean(), 0.8558, 'forced-adoption welfare'),
    ('welfare.csv', lambda df: df['W_subsidy_net'].mean(), -0.0085, 'subsidy net welfare'),
    ('welfare_abm.csv',
     lambda df: df[df['equilibrium'] == 'low']['W_per_firm'].mean(), 1.1124, 'ABM low-eq welfare'),
    ('welfare_abm.csv',
     lambda df: df[df['equilibrium'] == 'high']['W_per_firm'].mean(), 0.9869, 'ABM high-eq welfare'),
    ('welfare_lambda.csv',
     lambda df: df[df['lambda'] == 0.0]['W0'].iloc[0], 1.0194, 'closed-data baseline welfare'),
    ('cross_effect.csv', lambda df: df.iloc[0, 3], -0.8111, 'cross-effect dV_H/drho_L'),
    ('eigenvalues.csv', lambda df: df['eigenvalue'].max(), -0.2, 'Jacobian eigenvalue max (<0)'),
    ('k_neutrality.csv', lambda df: df[df['k'] >= 0.2]['T100'].max()
     - df[df['k'] >= 0.2]['T100'].min(), 0.0005, 'k neutrality span'),
    ('cross_country_cost.csv',
     lambda df: df[df['country'] == 'Africa']['T100_mean'].iloc[0], 0.0732, 'Africa cost-scaled'),
    ('cross_country_cost.csv',
     lambda df: df[df['country'] == 'Brazil']['T100_mean'].iloc[0], 0.2025, 'Brazil cost-scaled'),
    ('heatmap_gamma_beta.csv',
     lambda df: df[(abs(df['gamma'] - 0.45) < 1e-9) & (abs(df['beta'] - 0.5) < 1e-9)]
     ['T100_mean'].iloc[0], 0.17, 'heatmap (gamma,beta) baseline point'),
    ('heatmap_gamma_lambda.csv',
     lambda df: df[(abs(df['gamma'] - 0.45) < 1e-9) & (abs(df['lambda'] - 0.3) < 1e-9)]
     ['T100_mean'].iloc[0], 0.17, 'heatmap (gamma,lambda) baseline point'),
]

TOL = 1e-3


def verify():
    import pandas as pd
    print('\n' + '=' * 70)
    print('VERIFICATION: fresh data/ vs manuscript published values')
    print('=' * 70, flush=True)
    npass = nfail = 0
    for csv, extract, expected, label in EXPECTED:
        path = os.path.join(DATA, csv)
        if not os.path.exists(path):
            print(f'  [FAIL] {label}: {csv} missing')
            nfail += 1
            continue
        try:
            df = pd.read_csv(path)
            actual = float(extract(df))
        except Exception as e:
            print(f'  [FAIL] {label}: extraction error {e}')
            nfail += 1
            continue
        ok = abs(actual - expected) <= TOL
        npass += ok
        nfail += (not ok)
        print(f'  [{"PASS" if ok else "FAIL"}] {label}: got {actual:.4f}, '
              f'expected {expected:.4f}')
    print(f'\nVERIFICATION: {npass} passed, {nfail} failed')
    return nfail == 0


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--full', action='store_true', help='regenerate all data+artifacts (60-90 min)')
    ap.add_argument('--quick', action='store_true', help='compile+sanity+artifact regen from existing data')
    ap.add_argument('--verify-only', action='store_true', help='only verify data/ vs published values')
    args = ap.parse_args()

    t_all = time.time()
    if args.verify_only:
        ok = verify()
        sys.exit(0 if ok else 1)

    if args.quick:
        stages_ok = compile_all()
        stages_ok = sanity_solve() and stages_ok
        data_ok = True  # quick mode does not regenerate data
    elif args.full:
        stages_ok = True
        for name, cmd in DATA_STAGES:
            stages_ok = run_stage(name, cmd) and stages_ok
        data_ok = stages_ok
    else:
        ap.print_help()
        sys.exit(1)

    for name, cmd in ARTIFACT_STAGES:
        stages_ok = run_stage(name, cmd) and stages_ok

    verify_ok = verify() if data_ok else False

    print('\n' + '=' * 70)
    print(f'run_all: stages {"OK" if stages_ok else "FAILED"}, '
          f'verification {"OK" if verify_ok else "FAILED/skipped"}, '
          f'total {(time.time()-t_all)/60:.1f} min')
    print('=' * 70)
    sys.exit(0 if (stages_ok and verify_ok) else 1)


if __name__ == '__main__':
    main()
