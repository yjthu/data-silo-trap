"""
Export supplementary tables in manuscript Results-section order (S1-S14).
"""
import os, pandas as pd

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'supplementary_tables.csv')

# Ordered to follow the manuscript Results subsections:
#   §4.1 baseline → S1
#   §4.2 subsidy → S2
#   §4.3 mandatory → S3
#   §4.4 platform → S4, S5 (decomposition)
#   §4.5 mechanism → S6 (channel), S7 (lambda=0)
#   §4.6 sensitivity → S8 (OAT), S9 (cross-country), S10 (F_ST), S11 (F-scale)
#   §4.7 equilibrium → S12 (multistart), S13 (hysteresis)
#   §4.8 ABM → S14
tables = {}

tables['S1_P0_baseline'] = pd.read_csv(os.path.join(DATA, 'p0_baseline.csv')).describe().T[['mean','std','min','max']]
tables['S2_premium_stats'] = pd.read_csv(os.path.join(DATA, 'premium_stats.csv'))
tables['S3_P1_subsidy'] = pd.read_csv(os.path.join(DATA, 'p1_subsidy.csv'))
tables['S4_P2_mandatory'] = pd.read_csv(os.path.join(DATA, 'p2_mandatory.csv'))
tables['S5_P3_platform'] = pd.read_csv(os.path.join(DATA, 'p3_platform.csv'))
tables['S6_P3_decomposition'] = pd.read_csv(os.path.join(DATA, 'p3_decomposition.csv'))
tables['S7_channel_decomposition'] = pd.read_csv(os.path.join(DATA, 'channel_decomposition.csv'))
tables['S8_lambda0'] = pd.read_csv(os.path.join(DATA, 'lambda0_baseline.csv')).describe().T[['mean','std']]
tables['S9_OAT_sensitivity'] = pd.read_csv(os.path.join(DATA, 'sensitivity_oat.csv'))
tables['S10_cross_country'] = pd.read_csv(os.path.join(DATA, 'cross_country.csv'))
tables['S12_cross_country_fst'] = pd.read_csv(os.path.join(DATA, 'cross_country_fst.csv'))
tables['S13_fscale_robustness'] = pd.read_csv(os.path.join(DATA, 'fscale_robustness.csv'))
tables['S14_multistart'] = pd.read_csv(os.path.join(DATA, 'stability_multistart.csv'))
tables['S18_hysteresis'] = pd.read_csv(os.path.join(DATA, 'hysteresis.csv'))
tables['S19_abm'] = pd.read_csv(os.path.join(DATA, 'abm_monte_carlo.csv'))
tables['S20_abm_multistart'] = pd.read_csv(os.path.join(DATA, 'abm_multistart.csv'))
tables['S21_poolsize'] = pd.read_csv(os.path.join(DATA, 'poolsize_robustness.csv'))
tables['S22_heatmap_gb'] = pd.read_csv(os.path.join(DATA, 'heatmap_gamma_beta.csv'))
tables['S23_heatmap_gl'] = pd.read_csv(os.path.join(DATA, 'heatmap_gamma_lambda.csv'))
tables['S24_welfare'] = pd.read_csv(os.path.join(DATA, 'welfare.csv'))
tables['S25_welfare_abm'] = pd.read_csv(os.path.join(DATA, 'welfare_abm.csv'))
tables['S26_welfare_lambda'] = pd.read_csv(os.path.join(DATA, 'welfare_lambda.csv'))
tables['S11_cost_scaled'] = pd.read_csv(os.path.join(DATA, 'cross_country_cost.csv'))
tables['S15_cross_effect'] = pd.read_csv(os.path.join(DATA, 'cross_effect.csv'))
tables['S16_eigenvalues'] = pd.read_csv(os.path.join(DATA, 'eigenvalues.csv'))
tables['S17_k_neutrality'] = pd.read_csv(os.path.join(DATA, 'k_neutrality.csv'))

rows = []
for name, df in tables.items():
    for idx, row in df.iterrows():
        d = {'table': name}
        d.update(row.to_dict())
        rows.append(d)

pd.DataFrame(rows).to_csv(OUT, index=False)
print(f"Exported {len(rows)} rows across {len(tables)} tables")
for name, df in tables.items():
    print(f"  {name}: {len(df)} rows")
