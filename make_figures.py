"""
Generate 5 publication figures for the FS revision.

Fig 1: Baseline data-pool composition trap (3 panels)
Fig 2: Policy intervention effects (4 panels incl. platform decomposition)
Fig 3: What drives the trap — mechanism (3 panels)
Fig 4: Robustness (4 panels: OAT, cross-country+F_ST, F-scale, multistart trajectories)
Fig 5: Model validation — ABM multi-start and hysteresis (4 panels)
"""
import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from model.firm_generator import generate_firms
from model.distance import generate_distance_matrix
from model.accuracy import precompute_type_stats, shared_pool_and_centroid, firm_premiums
from model.dynamics import solve_dynamics, T100
from model.config import FIRM_TYPE_SHARES, FIRM_TYPE_LABELS, ALPHA, BETA, LAMBDA, GAMMA, DELTA

FIG = os.path.join(ROOT, 'figures')
DATA = os.path.join(ROOT, 'data')
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8, 'figure.dpi': 300, 'savefig.dpi': 300,
    'legend.frameon': False,
})
cH, cM, cL = '#D62728', '#1F77B4', '#7F7F7F'
cOrange, cPurple, cGreen = '#FF7F0E', '#9467BD', '#2CA02C'


def save(fig, name):
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIG, f'{name}.{ext}'), bbox_inches='tight', facecolor='white', dpi=300)
    plt.close(fig)
    print(f'saved {name}')


def label(ax, ch):
    # Panel letter outside the axes at the top-left, LEFT-ALIGNED with the
    # y-axis tick-label column (x = -0.18); the raised height (y = 1.08) keeps
    # the letter clear of the top tick label.
    ax.text(-0.18, 1.13, ch, transform=ax.transAxes, ha='left', va='top',
            fontsize=11, fontweight='bold')


# =====================================================================
# Fig 1: Baseline network externality trap
# =====================================================================
def fig1():
    firms = generate_firms(1000, seed=42)
    D, _ = generate_distance_matrix(firms, seed=42)
    stats = precompute_type_stats(firms, D)
    sol = solve_dynamics(firms, stats)
    t = sol.t

    fig = plt.figure(figsize=(7.5, 2.5))
    gs = gridspec.GridSpec(1, 3, wspace=0.35)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t, sol.y[2], color=cL, lw=1)
    ax.plot(t, sol.y[1], color=cM, lw=1)
    ax.plot(t, sol.y[0], color=cH, lw=1.5, ls='--')
    # direct end-of-curve labels (no legend box — avoids overlap with the M plateau)
    ax.text(t[-1], sol.y[0, -1] + 0.03, 'H (3%)', color=cH, fontsize=6.5,
            ha='right', va='center', fontweight='bold')
    ax.text(t[-1], sol.y[1, -1] + 0.03, 'M (15%)', color=cM, fontsize=6.5,
            ha='right', va='center', fontweight='bold')
    ax.text(t[-1], sol.y[2, -1] + 0.04, 'L (82%)', color=cL, fontsize=6.5,
            ha='right', va='center', fontweight='bold')
    ax.set_xlabel('Time'); ax.set_ylabel(r'Adoption $\rho_\phi$')
    ax.set_ylim(-0.05, 1.05)
    label(ax, 'a')

    ax = fig.add_subplot(gs[0, 1])
    ax.bar([0,1,2], FIRM_TYPE_SHARES, color=[cH,cM,cL], edgecolor='white', width=0.5)
    for i,s in enumerate(FIRM_TYPE_SHARES):
        ax.text(i, s+0.02, f'{s*100:.0f}%', ha='center', fontsize=7)
    mI = stats['mean_I']
    ax.annotate(f'$I_H$={mI[0]:.2f}, $I_M$={mI[1]:.2f}, $I_L$={mI[2]:.2f}',
                xy=(1.15, 0.95), ha='center', fontsize=7,
                bbox=dict(boxstyle='round', facecolor='#FFF8E1', alpha=0.8))
    ax.set_xticks([0,1,2]); ax.set_xticklabels(FIRM_TYPE_LABELS)
    ax.set_ylabel('Share'); ax.set_ylim(0, 1.1)
    label(ax, 'b')

    ax = fig.add_subplot(gs[0, 2])
    ps = pd.read_csv(os.path.join(DATA, 'premium_stats.csv'))
    V_type = [ps[ps['type'] == n]['V_mean'].values[0] for n in ['H', 'M', 'L']]
    V_std = [ps[ps['type'] == n]['V_std'].values[0] for n in ['H', 'M', 'L']]
    F_min = [firms['F'][firms['types'] == t].min() for t in range(3)]
    x = np.arange(3)
    ax.bar(x, V_type, yerr=V_std, color=[cH, cM, cL], edgecolor='white', width=0.5,
           alpha=0.8, capsize=2)
    # value labels on the bars (color-matched); the ylabel identifies the quantity
    for i, (v, col) in enumerate(zip(V_type, [cH, cM, cL])):
        dy = 0.018 if v >= 0 else -0.018
        va = 'bottom' if v >= 0 else 'top'
        ax.text(i, v + dy, f'{v:.3f}', ha='center', va=va, fontsize=6.5, color=col)
    ax.plot(x, F_min, 'k_', markersize=13, label=r'$F_{\phi,\min}$')
    ax.axhline(0, color='gray', lw=0.8, ls='--')
    # compact negative-premium annotation above the L bar (empty region),
    # arrow down to the L bar top
    ax.annotate(r'$V_L<0$',
                xy=(2.06, V_type[2] + 0.006), xytext=(2.06, 0.06),
                fontsize=7.5, color=cL, ha='center', fontweight='bold',
                arrowprops=dict(arrowstyle='-', lw=0.6, color=cL))
    ax.set_xticks(x); ax.set_xticklabels(FIRM_TYPE_LABELS)
    ax.set_ylabel(r'Premium $V_\phi$'); ax.set_ylim(-0.12, 0.45)
    ax.set_xlim(-0.55, 3.05)
    ax.legend(fontsize=6, loc='upper right')
    label(ax, 'c')

    save(fig, 'Fig1_baseline')


# =====================================================================
# Fig 2: Policy intervention effects (4 panels)
# =====================================================================
def fig2():
    p1 = pd.read_csv(os.path.join(DATA, 'p1_subsidy.csv'))
    p2 = pd.read_csv(os.path.join(DATA, 'p2_mandatory.csv'))
    p3 = pd.read_csv(os.path.join(DATA, 'p3_platform.csv'))
    p3d = pd.read_csv(os.path.join(DATA, 'p3_decomposition.csv'))

    fig = plt.figure(figsize=(7.5, 5.0))
    gs = gridspec.GridSpec(2, 2, wspace=0.35, hspace=0.45)

    ax = fig.add_subplot(gs[0, 0])
    ax.errorbar(p1['s'], p1['T100_mean'], yerr=p1['T100_std'], color=cOrange,
                lw=1.5, ms=3, fmt='o-', capsize=2)
    ax.set_xlabel('Subsidy rate s'); ax.set_ylabel(r'$T_{100}$')
    ax.set_ylim(0, 0.35)
    label(ax, 'a')

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(p2['b'], p2['q_H_mean'], 's-', color=cH, lw=1.2, ms=3, label='H')
    ax.plot(p2['b'], p2['q_M_mean'], 'o-', color=cM, lw=1.2, ms=3, label='M')
    ax.plot(p2['b'], p2['q_L_mean'], '^-', color=cL, lw=1.2, ms=3, label='L')
    ax.axvline(0.25, color='gray', ls=':', lw=0.8)
    ax.annotate(r'$b^*=0.25$', xy=(0.20, 0.42), fontsize=6.5, color='#555', ha='center')
    ax.set_xlabel('Quality bonus b'); ax.set_ylabel(r'Contribution $q_\phi$')
    ax.legend(fontsize=6.5)
    label(ax, 'b')

    ax = fig.add_subplot(gs[1, 0])
    ax.errorbar(p3['fee'], p3['T100_mean'], yerr=p3['T100_std'], color=cGreen,
                lw=1.5, ms=3, fmt='o-', capsize=2)
    ax.set_xlabel('Platform fee'); ax.set_ylabel(r'$T_{100}$')
    ax.set_ylim(0, 0.35)
    label(ax, 'c')

    ax = fig.add_subplot(gs[1, 1])
    # Marginal effects relative to "neither" (cost reduction alone, trust alone, both)
    base = p3d[p3d['case']=='neither']['T100_mean'].values[0]
    case_order = ['cost_only', 'trust_only', 'both']
    labels_map = {'cost_only': 'Cost\nonly', 'trust_only': 'Trust\nonly', 'both': 'Both'}
    vals = [p3d[p3d['case']==c]['T100_mean'].values[0] - base for c in case_order]
    errs = [p3d[p3d['case']==c]['T100_std'].values[0] for c in case_order]
    x = np.arange(3)
    ax.bar(x, vals, yerr=errs, color=[cOrange, cPurple, cGreen], edgecolor='white',
           width=0.5, capsize=2)
    for i, (v, e) in enumerate(zip(vals, errs)):
        ax.text(i, v + e + 0.004, f'+{v:.3f}', ha='center', fontsize=6.5)
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels([labels_map[c] for c in case_order], fontsize=6.5)
    ax.set_ylabel(r'$\Delta T_{100}$ (vs. neither)')
    ax.set_ylim(0, 0.085)
    ax.annotate('Trust > cost', xy=(1, 0.072), fontsize=5.5, color='#555', ha='center')
    label(ax, 'd')

    save(fig, 'Fig2_policy')


# =====================================================================
# Fig 3: What drives the trap — mechanism
# =====================================================================
def fig3():
    cd = pd.read_csv(os.path.join(DATA, 'channel_decomposition.csv'))
    oat = pd.read_csv(os.path.join(DATA, 'sensitivity_oat.csv'))

    firms = generate_firms(1000, seed=42)
    D, _ = generate_distance_matrix(firms, seed=42)
    stats = precompute_type_stats(firms, D)
    sol = solve_dynamics(firms, stats)
    S, d_c = shared_pool_and_centroid(stats, sol.y[:, -1])

    fig = plt.figure(figsize=(7.5, 2.5))
    gs = gridspec.GridSpec(1, 3, wspace=0.35)

    ax = fig.add_subplot(gs[0, 0])
    specs = ['both', 'theta_only', 'gamma_only', 'neither']
    labels = ['Both', r'$\theta$-only' + '\n' + r'($\gamma=0$)', r'$\gamma$-only' + '\n' + r'($\theta\to0$)', 'Neither']
    vals = [cd[cd['spec']==s]['T100'].values[0] for s in specs]
    ax.bar(range(4), vals, color=['#555', cGreen, cPurple, '#ccc'], edgecolor='white', width=0.5)
    for i,v in enumerate(vals):
        ax.text(i, v+0.03, f'{v:.3f}', ha='center', fontsize=7)
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel(r'$T_{100}$'); ax.set_ylim(0, 1.1)
    label(ax, 'a')

    ax = fig.add_subplot(gs[0, 1])
    ps = pd.read_csv(os.path.join(DATA, 'premium_stats.csv'))
    own = np.array([ps[ps['type'] == n]['own_term'].values[0] for n in ['H', 'M', 'L']])
    net = np.array([ps[ps['type'] == n]['network_term'].values[0] for n in ['H', 'M', 'L']])
    pen = np.array([-ps[ps['type'] == n]['penalty_term'].values[0] for n in ['H', 'M', 'L']])
    x = np.arange(3)
    ax.bar(x, own, 0.3, color=cH, alpha=0.6, label='Own data')
    ax.bar(x, net, 0.3, bottom=own, color=cGreen, alpha=0.6, label='Shared pool')
    ax.bar(x, pen, 0.3, bottom=own+net, color=cL, alpha=0.6, label=r'Misalignment ($-\gamma d$)')
    ax.axhline(0, color='gray', lw=0.5)
    # annotate L negative total (legend now sits top-right; bottom area is free)
    ax.annotate(r'$V_L<0$', xy=(2, -0.10), fontsize=7, color=cL, ha='center', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(FIRM_TYPE_LABELS)
    ax.set_ylabel('Premium components')
    ax.set_ylim(-0.15, 0.62)
    # stacked single-column legend fits inside the empty top-right corner
    ax.legend(fontsize=5.5, loc='upper right')
    label(ax, 'b')

    ax = fig.add_subplot(gs[0, 2])
    lam = oat[oat['param']=='lambda'].sort_values('value')
    ax.plot(lam['value'], lam['T100_mean'], 'o-', color=cM, lw=1.5, ms=3)
    ax.axvline(0.3, color='gray', ls=':', lw=0.8)
    ax.annotate(r'$\lambda_{base}=0.3$', xy=(0.3, 0.09), fontsize=6.5, color='#555', rotation=90)
    ax.set_xlabel(r'Free-riding $\lambda$'); ax.set_ylabel(r'$T_{100}$')
    ax.set_ylim(0, 0.3)
    label(ax, 'c')

    save(fig, 'Fig3_mechanism')


# =====================================================================
# Fig 4: Robustness (4 panels)
# =====================================================================
def fig4():
    oat = pd.read_csv(os.path.join(DATA, 'sensitivity_oat.csv'))
    cc = pd.read_csv(os.path.join(DATA, 'cross_country.csv'))
    fst = pd.read_csv(os.path.join(DATA, 'cross_country_fst.csv'))
    fs = pd.read_csv(os.path.join(DATA, 'fscale_robustness.csv'))

    fig = plt.figure(figsize=(7.5, 5.0))
    gs = gridspec.GridSpec(2, 2, wspace=0.35, hspace=0.45)

    # (a) OAT
    ax = fig.add_subplot(gs[0, 0])
    param_colors = {'gamma': cL, 'alpha': cH, 'delta': cOrange, 'lambda': cM,
                    'beta': cGreen, 'theta': cPurple, 'k': '#999999'}
    param_labels = {'gamma': r'$\gamma$', 'alpha': r'$\alpha$', 'delta': r'$\delta$',
                    'lambda': r'$\lambda$', 'beta': r'$\beta$', 'theta': r'$\theta$',
                    'k': r'$k$'}
    for p, grp in oat.groupby('param'):
        x = grp['value'].values
        xn = (x - x.min()) / (x.max() - x.min() + 1e-12)
        ax.plot(xn, grp['T100_mean'].values, 'o-', color=param_colors[p], lw=1.2, ms=3, label=param_labels[p])
    ax.set_xlabel('Normalized parameter'); ax.set_ylabel(r'$T_{100}$')
    ax.set_ylim(0, 0.60)
    ax.legend(fontsize=5.5, ncol=2)
    label(ax, 'a')

    # (b) cross-country with F_ST sensitivity (range bars)
    ax = fig.add_subplot(gs[0, 1])
    x = np.arange(len(cc))
    ax.bar(x, cc['T100_mean'].values, yerr=cc['T100_std'].values,
           color='#3498DB', edgecolor='white', width=0.5, capsize=2, label=r'Base $F_{ST}$')
    # F_ST sensitivity markers for each country (D×0.8 to D×1.2)
    for i, country in enumerate(cc['country'].values):
        cdata = fst[fst['country'] == country]
        lo = cdata[cdata['D_scale'] == 0.8]['T100_mean'].values[0]
        hi = cdata[cdata['D_scale'] == 1.2]['T100_mean'].values[0]
        ax.plot(i, lo, marker='o', color='black', ms=4, zorder=6)
        ax.plot(i, hi, marker='^', color='black', ms=4, zorder=6)
    ax.scatter([], [], marker='o', color='black', s=20, label=r'Low $F_{ST}$ (D$\times$0.8)')
    ax.scatter([], [], marker='^', color='black', s=20, label=r'High $F_{ST}$ (D$\times$1.2)')
    ax.set_xticks(range(len(cc)))
    ax.set_xticklabels(cc['country'].values, rotation=30, fontsize=6.5, ha='right')
    ax.set_ylabel(r'$T_{100}$'); ax.set_ylim(0, 0.34)
    ax.legend(fontsize=5, loc='upper left')
    label(ax, 'b')

    # (c) F-scale robustness
    ax = fig.add_subplot(gs[1, 0])
    ax.errorbar(fs['F_scale'], fs['T100_mean'], yerr=fs['T100_std'], color=cOrange,
                lw=1.5, ms=3, fmt='o-', capsize=2)
    ax.set_xlabel('Cost scale factor'); ax.set_ylabel(r'$T_{100}$')
    ax.set_ylim(0, 0.3)
    label(ax, 'c')

    # (d) multistart trajectories (unique equilibrium)
    ax = fig.add_subplot(gs[1, 1])
    firms = generate_firms(1000, seed=42)
    D, _ = generate_distance_matrix(firms, seed=42)
    stats = precompute_type_stats(firms, D)
    ics = [(0.0,0.0,0.0), (0.3,0.3,0.3), (0.6,0.6,0.6), (1.0,1.0,1.0),
           (0.9,0.5,0.1), (0.1,0.9,0.5), (0.5,0.1,0.9), (0.8,0.8,0.8)]
    for ic in ics:
        sol = solve_dynamics(firms, stats, rho0=ic)
        T100s = np.dot([0.03, 0.15, 0.82], sol.y)
        ax.plot(sol.t, T100s, color='#4A90D9', lw=0.8, alpha=0.35)
    ax.annotate('8 initial conditions\n→ same equilibrium', xy=(25, 0.20),
                fontsize=7, color='#555', ha='center')
    ax.set_xlabel('Time'); ax.set_ylabel(r'$T_{100}$')
    ax.set_ylim(0, 0.25)
    label(ax, 'd')

    save(fig, 'Fig4_robustness')


# =====================================================================
# Fig 5: Model validation — ABM and hysteresis
# =====================================================================
def fig5():
    abm = pd.read_csv(os.path.join(DATA, 'abm_monte_carlo.csv'))
    hy = pd.read_csv(os.path.join(DATA, 'hysteresis.csv'))

    fig = plt.figure(figsize=(7.5, 5.0))
    gs = gridspec.GridSpec(2, 2, wspace=0.35, hspace=0.45)

    # (a) hysteresis
    ax = fig.add_subplot(gs[0, 0])
    fwd = hy[hy['direction']=='forward'].sort_values('s')
    bwd = hy[hy['direction']=='backward'].sort_values('s')
    ax.plot(fwd['s'], fwd['T100'], '-', color=cH, lw=1.5, label='Forward')
    ax.plot(bwd['s'], bwd['T100'], '--', color=cM, lw=1.5, label='Backward',
            dashes=(4, 3))
    ax.annotate('No hysteresis\n(forward = backward)', xy=(0.5, 0.28),
                fontsize=7, color='#555', ha='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.set_xlabel('Subsidy rate s'); ax.set_ylabel(r'$T_{100}$')
    ax.set_ylim(0, 0.35)
    ax.legend(fontsize=6.5)
    label(ax, 'a')

    # (b) ABM adoption by quality tercile
    ax = fig.add_subplot(gs[0, 1])
    labels = ['High', 'Mid', 'Low']
    n_abm = len(abm)
    means = [abm['adopt_high'].mean(), abm['adopt_mid'].mean(), abm['adopt_low'].mean()]
    # standard error of the mean (not std) — avoids error bars extending below zero
    sems = [abm['adopt_high'].std()/np.sqrt(n_abm),
            abm['adopt_mid'].std()/np.sqrt(n_abm),
            abm['adopt_low'].std()/np.sqrt(n_abm)]
    ax.bar(range(3), means, yerr=sems, color=[cH, cM, cL], edgecolor='white', width=0.5, capsize=3)
    for i,v in enumerate(means):
        ax.text(i, v+0.04, f'{v:.2f}', ha='center', fontsize=7)
    ax.set_xticks(range(3)); ax.set_xticklabels(labels)
    ax.set_ylabel('ABM adoption'); ax.set_ylim(0, 0.45)
    ax.set_title('Quality tercile', fontsize=7)
    label(ax, 'b')

    # (c) ABM bimodal distribution
    ax = fig.add_subplot(gs[1, 0])
    ax.hist(abm['T100'], bins=10, color=cM, alpha=0.7, edgecolor='white')
    ax.axvline(0.170, color=cH, lw=1.5, ls='--', label='3-type model')
    ax.annotate('Low\n(spatially\nscattered)', xy=(0.06, 3.5), fontsize=5.5, color='#555', ha='center')
    ax.annotate('High\n(spatially\nclustered)', xy=(0.40, 3.5), fontsize=5.5, color='#555', ha='center')
    ax.set_xlabel(r'ABM $T_{100}$'); ax.set_ylabel('Frequency')
    ax.legend(fontsize=6.5)
    label(ax, 'c')

    # (d) ABM multi-start: fixed realization, multiple initial states
    ax = fig.add_subplot(gs[1, 1])
    ms = pd.read_csv(os.path.join(DATA, 'abm_multistart.csv'))
    ms42 = ms[ms['seed'] == 42].set_index('init')
    # order by terminal T100 (ascending) so the three equilibrium clusters
    # (low / intermediate / high) are visually apparent
    order = sorted(['top5', 'top10', 'top25', 'bottom10', 'none', 'left_half',
                    'right_half', 'random10', 'all'],
                   key=lambda o: ms42.loc[o, 'T100'])
    labels = {'top5': 'top 5%', 'top10': 'top 10%', 'top25': 'top 25%',
              'bottom10': 'bottom 10%', 'none': 'none', 'left_half': 'left half',
              'right_half': 'right half', 'random10': 'random 10%', 'all': 'all'}
    vals = [ms42.loc[o, 'T100'] for o in order]
    colors = [cL if v < 0.10 else (cOrange if v < 0.60 else cH) for v in vals]
    ax.bar(range(len(order)), vals, color=colors, edgecolor='white', width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=5.5)
    ax.axhline(0.17, color='grey', lw=0.8, ls=':')
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([labels[o] for o in order], rotation=40, ha='right',
                       rotation_mode='anchor', fontsize=5.5)
    ax.set_ylabel('Terminal $T_{100}$'); ax.set_ylim(0, 1.05)
    ax.annotate('Fixed realization (seed 42):\ninitial state selects equilibrium',
                xy=(3.0, 0.88), fontsize=6.5, color='#555', ha='center')
    label(ax, 'd')

    save(fig, 'Fig5_validation')


# =====================================================================
# Fig 6: Welfare assessment
# =====================================================================
def fig6():
    ws = pd.read_csv(os.path.join(DATA, 'welfare_scan.csv'))
    wa = pd.read_csv(os.path.join(DATA, 'welfare_abm.csv'))

    fig = plt.figure(figsize=(7.5, 2.5))
    gs = gridspec.GridSpec(1, 2, wspace=0.35)

    # (a) net welfare vs subsidy rate / platform fee
    ax = fig.add_subplot(gs[0, 0])
    sub = ws[ws['policy'] == 'subsidy'].groupby('value')['W_net'].mean()
    plat = ws[ws['policy'] == 'platform'].groupby('value')['W_net'].mean()
    ax.plot(sub.index, sub.values, 'o-', color=cOrange, lw=1.5, ms=3, label='Subsidy s')
    ax.plot(plat.index, plat.values, 's-', color=cGreen, lw=1.5, ms=3, label='Platform fee')
    ax.axhline(0, color='gray', lw=0.8, ls='--')
    ax.annotate('welfare-negative\nnet of costs', xy=(0.5, -0.010), fontsize=6.5,
                color='#555', ha='center')
    ax.set_xlabel('Intervention intensity'); ax.set_ylabel(r'$\Delta$W per firm')
    ax.set_ylim(-0.025, 0.005)
    ax.legend(fontsize=6.5)
    label(ax, 'a')

    # (b) ABM low- vs high-equilibrium welfare
    ax = fig.add_subplot(gs[0, 1])
    seeds = sorted(wa['seed'].unique())
    low = [wa[(wa['seed'] == s) & (wa['equilibrium'] == 'low')]['W_per_firm'].iloc[0]
           for s in seeds]
    high = [wa[(wa['seed'] == s) & (wa['equilibrium'] == 'high')]['W_per_firm'].iloc[0]
            for s in seeds]
    x = np.arange(len(seeds))
    ax.bar(x - 0.18, low, width=0.36, color=cL, edgecolor='white', label='Low equilibrium')
    ax.bar(x + 0.18, high, width=0.36, color=cH, edgecolor='white', label='High equilibrium')
    for i in range(len(seeds)):
        ax.text(i - 0.18, low[i] + 0.004, f'{low[i]:.2f}', ha='center', fontsize=6)
        ax.text(i + 0.18, high[i] + 0.004, f'{high[i]:.2f}', ha='center', fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels([f'seed {s}' for s in seeds])
    ax.set_ylabel('Welfare per firm'); ax.set_ylim(0, 1.2)
    ax.legend(fontsize=6.5)
    ax.annotate('low equilibrium is\nwelfare-superior', xy=(1, 0.35),
                fontsize=6.5, color='#555', ha='center')
    label(ax, 'b')

    save(fig, 'Fig6_welfare')


# =====================================================================
# Fig S1 (supplementary): joint parameter-range sensitivity heatmaps
# =====================================================================
def figS1():
    gb = pd.read_csv(os.path.join(DATA, 'heatmap_gamma_beta.csv'))
    gl = pd.read_csv(os.path.join(DATA, 'heatmap_gamma_lambda.csv'))

    fig = plt.figure(figsize=(7.5, 2.5))
    gs = gridspec.GridSpec(1, 2, wspace=0.35)

    ax = fig.add_subplot(gs[0, 0])
    piv = gb.pivot(index='gamma', columns='beta', values='T100_mean')
    im = ax.imshow(piv.values, origin='lower', aspect='auto', cmap='YlOrRd',
                   extent=[piv.columns.min(), piv.columns.max(), piv.index.min(), piv.index.max()])
    ax.scatter([0.5], [0.45], marker='o', facecolors='none', edgecolors='k', s=60, lw=1.2)
    ax.annotate('baseline', xy=(0.5, 0.45), xytext=(0.56, 0.52), fontsize=6, color='k')
    ax.set_xlabel(r'$\beta$ (sharing benefit)'); ax.set_ylabel(r'$\gamma$ (misalignment)')
    ax.set_title(r'$T_{100}$ over ($\gamma$, $\beta$)', fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    label(ax, 'a')

    ax = fig.add_subplot(gs[0, 1])
    piv2 = gl.pivot(index='gamma', columns='lambda', values='T100_mean')
    im2 = ax.imshow(piv2.values, origin='lower', aspect='auto', cmap='YlOrRd',
                    extent=[piv2.columns.min(), piv2.columns.max(), piv2.index.min(), piv2.index.max()])
    ax.scatter([0.3], [0.45], marker='o', facecolors='none', edgecolors='k', s=60, lw=1.2)
    ax.annotate('baseline', xy=(0.3, 0.45), xytext=(0.38, 0.52), fontsize=6, color='k')
    ax.set_xlabel(r'$\lambda$ (free-riding)'); ax.set_ylabel(r'$\gamma$ (misalignment)')
    ax.set_title(r'$T_{100}$ over ($\gamma$, $\lambda$)', fontsize=7)
    fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
    label(ax, 'b')

    save(fig, 'FigS1_heatmap')


if __name__ == '__main__':
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    figS1()
    print('All figures generated.')
