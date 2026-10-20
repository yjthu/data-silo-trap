"""
welfare_corrected.py — recompute the quality-bonus welfare (P2) and the
forced-adoption counterfactual with a non-negative accuracy index.

This script DOES NOT modify any published file. data/welfare.csv,
data/welfare_scan.csv and data/p2_mandatory.csv are left exactly as they are,
so `run_all.py --verify-only` keeps reproducing the published numbers (42/42).
Corrected values are written to data/welfare_bonus_corrected.csv.

What it does, in order:

  1. Reproduces the two published bonus numbers from the same seeds, using the
     code path of analysis/welfare.py unchanged:
        Table S24  W_bonus_net   (part_a: difference against S0, type-level
                                  denominator over ALL types, net of b*n_contrib)
        Fig. 7 / Sec. 4.11       (part_c: LEVEL of the induced pool, full beta
                                  for every firm, net of b*n_contrib)
     Both are checked against the stored values at the end of the run.

  2. Recomputes the same object with the denominator restricted to firms that
     actually contribute. Every contributor supplies exactly q = I*, so the
     induced pool quality is exactly I* = 0.5, independent of type and of
     genetic distance. The welfare gain is then taken as a difference against
     the baseline pool S0 (adopters weighted beta, non-adopters lambda*beta).

  3. Recomputes the forced-adoption gap with the accuracy index floored at 0.

See KNOWN_ISSUES.md (KI-1, KI-2) for why.

Usage
-----
    python analysis/welfare_corrected.py             # MC = 300 = part_a's sample
    python analysis/welfare_corrected.py --mc 40     # quick look, ~5 s
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from model.config import (ALPHA, BETA, GAMMA, THETA, DELTA, LAMBDA,
                          FIRM_TYPE_SHARES, QUALITY_THRESHOLD, RANDOM_SEED)
from model.firm_generator import generate_firms
from model.distance import generate_distance_matrix
from model.accuracy import (precompute_type_stats, shared_pool_and_centroid,
                            firm_premiums)
from model.dynamics import solve_dynamics

W = FIRM_TYPE_SHARES
N = 1000
I_STAR = QUALITY_THRESHOLD
B_LO, B_HI = 0.26, 0.30          # the two bonuses quoted in the article
DATA = os.path.join(ROOT, 'data')

# Published values, for the reproduction check (Supplementary Table S24, part_a,
# 300 draws; welfare_scan.csv, part_c, 200 draws).
PUB = {'W0': 1.084479, 'W_full': 0.855783, 'W_bonus_net': -0.121890,
       'W_subsidy_net': -0.008546, 'W_platform_net': -0.008990,
       'bonus_scan_026': 0.005848, 'bonus_scan_030': 0.001033}


def _welfare(firms, S, d, adopt, floor_a=False):
    """Welfare per firm, optionally with the accuracy index floored at zero.

    Identical to analysis/welfare.py::_welfare except that it returns the mean
    and also hands back the accuracy index itself.
    """
    types, I = firms['types'], firms['I']
    S_i = np.array([S[t] for t in types])
    d_i = np.array([d[t] for t in types])
    a = ALPHA * I + BETA * S_i - GAMMA * d_i
    if floor_a:
        a = np.maximum(a, 0.0)
    adopter = 1.0 + DELTA * a - firms['F']
    non_adopter = 1.0 + DELTA * LAMBDA * BETA * S_i
    return float(np.where(adopt, adopter, non_adopter).mean()), a


def _paper_pool_quality(q_bar, omega):
    """Type-level pool quality exactly as part_a and part_c compute it:
    denominator over ALL types, regardless of whether they contribute."""
    S = np.zeros(3)
    for phi in range(3):
        num = float((W * q_bar * omega[phi]).sum())
        den = float((W * omega[phi]).sum())
        S[phi] = num / den if den > 1e-12 else 0.0
    return S


def _contributor_pool_quality(f, D, q):
    """Firm-level pool quality with the denominator restricted to contributors
    (the convention of accuracy.shared_pool_and_centroid)."""
    om = np.exp(-THETA * D)
    contrib = q > 0
    om_c = om[:, contrib]
    den = om_c.sum(axis=1)
    S_i = np.where(den > 1e-12, (om_c * q[contrib]).sum(axis=1) / den, 0.0)
    types = f['types']
    return S_i, np.array([S_i[types == t].mean() for t in range(3)])


def main(mc: int):
    rows = []
    for r in range(mc):
        seed = RANDOM_SEED + r
        f = generate_firms(N, seed=seed)
        D, _ = generate_distance_matrix(f, seed=seed)
        stats = precompute_type_stats(f, D)

        rho0 = solve_dynamics(f, stats).y[:, -1]
        S0, d0 = shared_pool_and_centroid(stats, rho0)
        V0 = firm_premiums(f, S0, d0)
        adopt0 = f['F'] < V0

        I, types = f['I'], f['types']
        q = np.where(I >= I_STAR, I_STAR, 0.0)
        q_bar = np.array([q[types == t].mean() if (types == t).any() else 0.0
                          for t in range(3)])
        share = float((q > 0).mean())          # contributors / N

        rec = {'seed': seed,
               'S0_mean': float(np.mean([S0[t] for t in types])),
               'contrib_share': share,
               'expend_b026': B_LO * share, 'expend_b030': B_HI * share}

        # --- (1) published arithmetic, reproduced ---
        S_paper = _paper_pool_quality(q_bar, stats['omega'])
        dS_paper = np.array([S_paper[t] for t in types]) - np.array([S0[t] for t in types])
        gain_a = (DELTA * BETA * dS_paper[adopt0].sum()
                  + DELTA * LAMBDA * BETA * dS_paper[~adopt0].sum()) / N
        gain_c = DELTA * BETA * float(
            sum(S_paper[t] * (types == t).sum() for t in range(3))) / N
        rec.update({'S_paper': float(np.mean([S_paper[t] for t in types])),
                    'dS_paper': float(dS_paper.mean()),
                    'gain_paper_dS': gain_a,
                    'net_paper_dS_b030': gain_a - rec['expend_b030'],
                    'gain_paper_level': gain_c,
                    'net_paper_level_b026': gain_c - rec['expend_b026'],
                    'net_paper_level_b030': gain_c - rec['expend_b030']})

        # --- (2) corrected: contributors only, difference against S0 ---
        S_i, S_fix = _contributor_pool_quality(f, D, q)
        dS_fix = S_i - np.array([S0[t] for t in types])
        gain_fix = (DELTA * BETA * dS_fix[adopt0].sum()
                    + DELTA * LAMBDA * BETA * dS_fix[~adopt0].sum()) / N
        rec.update({'S_fixed': float(np.mean([S_fix[t] for t in types])),
                    'max_dev_from_Istar': float(np.abs(S_i - I_STAR).max()),
                    'dS_fixed': float(dS_fix.mean()),
                    'gain_fixed': gain_fix,
                    'net_fixed_b026': gain_fix - rec['expend_b026'],
                    'net_fixed_b030': gain_fix - rec['expend_b030']})

        # --- (3) forced adoption, with and without the accuracy floor ---
        S_full, d_full = shared_pool_and_centroid(stats, np.array([1.0, 1.0, 1.0]))
        w_full_raw, a_full = _welfare(f, S_full, d_full, np.ones(N, bool))
        w_full_floored, _ = _welfare(f, S_full, d_full, np.ones(N, bool), floor_a=True)
        w0, a0 = _welfare(f, S0, d0, adopt0)
        w0_floored, _ = _welfare(f, S0, d0, adopt0, floor_a=True)
        rec.update({'W0': w0, 'W0_floored': w0_floored,
                    'W_full_raw': w_full_raw, 'W_full_floored': w_full_floored,
                    'gap_raw': w_full_raw / w0 - 1.0,
                    'gap_floored': w_full_floored / w0_floored - 1.0,
                    'share_a_neg_base': float((a0 < 0).mean()),
                    'share_a_neg_full': float((a_full < 0).mean()),
                    'share_a_neg_adopters': float((a0[adopt0] < 0).mean())})
        rows.append(rec)

    df = pd.DataFrame(rows)
    out = os.path.join(DATA, 'welfare_bonus_corrected.csv')
    df.to_csv(out, index=False)
    m = df.mean()

    print(f'\n=== 1. Reproduction of the published bonus numbers, MC = {mc} ===')
    print(f'  Table S24  W_bonus_net (= gain_a - 0.30*share)'
          f'   {m["net_paper_dS_b030"]:+.5f}   published {PUB["W_bonus_net"]:+.5f}')
    print(f'  Sec. 4.11  net at b = 0.26 (= level - 0.26*share)'
          f'   {m["net_paper_level_b026"]:+.5f}   published {PUB["bonus_scan_026"]:+.5f}')
    print(f'  scan b=0.30 net (= level - 0.30*share)'
          f'   {m["net_paper_level_b030"]:+.5f}   published {PUB["bonus_scan_030"]:+.5f}')
    if mc >= 200:                      # part_c used the first 200 draws
        sub = df.iloc[:200].mean()
        print(f'  ... first 200 draws, part_c\'s own sample:'
              f' {sub["net_paper_level_b026"]:+.5f} / {sub["net_paper_level_b030"]:+.5f}')

    print('\n=== 2. Where the numbers come from ===')
    print(f'  baseline pool quality S0                 {m["S0_mean"]:.5f}')
    print(f'  contributors / N                         {m["contrib_share"]:.5f}')
    print(f'  induced pool, paper denominator          {m["S_paper"]:.5f}'
          f'   (dS = {m["dS_paper"]:+.5f})')
    print(f'  induced pool, contributors only          {m["S_fixed"]:.5f}'
          f'   (max deviation from I*: {m["max_dev_from_Istar"]:.2e})')
    print(f'  dS against S0, corrected                 {m["dS_fixed"]:+.5f}')
    print(f'  gain, level, full beta to everyone       {m["gain_paper_level"]:+.5f}   <- what part_c did')
    print(f'  gain, dS, paper denominator              {m["gain_paper_dS"]:+.5f}   <- what part_a did')
    print(f'  gain, dS, contributors only              {m["gain_fixed"]:+.5f}   <- corrected')

    print('\n=== 3. Corrected net welfare of the bonus (per firm) ===')
    print(f'  b = {B_LO}   gain {m["gain_fixed"]:+.5f}  - bill {m["expend_b026"]:.5f}'
          f'  = {m["net_fixed_b026"]:+.5f}')
    print(f'  b = {B_HI}   gain {m["gain_fixed"]:+.5f}  - bill {m["expend_b030"]:.5f}'
          f'  = {m["net_fixed_b030"]:+.5f}')
    print(f'  break-even bonus b* = gain / share       {m["gain_fixed"]/m["contrib_share"]:.4f}'
          f'   (below the 0.25 at which the bonus starts to buy anything)')

    print('\n=== 4. Forced adoption, with and without the accuracy floor ===')
    print(f'  W0, unfloored / floored                  {m["W0"]:.4f} / {m["W0_floored"]:.4f}')
    print(f'  W_full, a unfloored                      {m["W_full_raw"]:.4f}'
          f'   (gap {m["gap_raw"]*100:+.1f}%)')
    print(f'  W_full, a floored at 0                   {m["W_full_floored"]:.4f}'
          f'   (gap {m["gap_floored"]*100:+.1f}%)')
    print(f'  overstatement factor                     {m["gap_raw"]/m["gap_floored"]:.2f}x')
    print(f'  a < 0: baseline {m["share_a_neg_base"]*100:.1f}%,'
          f' adopters at baseline {m["share_a_neg_adopters"]*100:.1f}%,'
          f' forced adoption {m["share_a_neg_full"]*100:.1f}%')

    print('\n=== 5. Reproduction check against the stored values ===')
    checks = [('Table S24 W0', m['W0'], PUB['W0']),
              ('Table S24 W_full', m['W_full_raw'], PUB['W_full']),
              ('Table S24 W_bonus_net', m['net_paper_dS_b030'], PUB['W_bonus_net'])]
    if mc >= 200:
        checks += [('welfare_scan bonus b=0.26', sub['net_paper_level_b026'], PUB['bonus_scan_026']),
                   ('welfare_scan bonus b=0.30', sub['net_paper_level_b030'], PUB['bonus_scan_030'])]
    ok = True
    for name, got, pub in checks:
        dev = abs(got - pub)
        flag = 'OK ' if dev < 5e-4 else 'DEV'
        ok = ok and dev < 5e-4
        print(f'  [{flag}] {name:28s} {got:+.5f} vs {pub:+.5f}   |dev| {dev:.1e}')
    print(f'  => {"all reproduced" if ok else "MISMATCH"}')

    print(f'\nsaved {out}')
    return df


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mc', type=int, default=300,
                    help='Monte Carlo draws (300 matches part_a\'s sample)')
    args = ap.parse_args()
    main(args.mc)
