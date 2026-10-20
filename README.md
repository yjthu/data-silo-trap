# The Data Silo Trap

Code and data for **"The Data Silo Trap: A Coordination Game Analysis of AI-Driven Breeding Adoption in Fragmented Seed Markets"** (*Farming System*).

A coordination-game model in which **genetic distance** enters the payoff structure as a structural barrier to data sharing in fragmented seed markets. The model identifies a **data-pool composition trap**: at equilibrium, high- and medium-R&D firms adopt, while the low-R&D majority (82%) is structurally excluded from the data-sharing pool. The trap is driven specifically by the **germplasm misalignment penalty** (γ), not by transferability decay (θ).

> **Known issues.** Five unintentional errors in the published article and code are documented, with corrected values and a script that recomputes them, in [KNOWN_ISSUES.md](KNOWN_ISSUES.md). None of them reverses a headline result; `run_all.py --verify-only` still reproduces the published numbers exactly. The correction script first re-derives every published number it then corrects (deviation < 5×10⁻⁷), so the corrected values are the same arithmetic with the defect removed, not a different experiment.

## Key findings

| Result | Value |
|--------|-------|
| Baseline adoption T₁₀₀ | 0.170 ± 0.011 (H = 0.835, M = 0.631, L = 0.061) |
| Equilibrium type-mean premium | V_H = 0.385, V_M = 0.179, V_L = −0.053 |
| Quality-bonus phase transition | b\* = 0.25 (analytical, b\* = c₀·I\*²) |
| Dominant driver | Germplasm misalignment penalty γ |
| Weak driver | Cross-population transferability decay θ |
| Free-riding effect | λ = 0 → T₁₀₀ = 0.255 (+50% vs. baseline) |
| ABM multiple equilibria | multi-start: ≈0.06 / ≈0.37 / ≈0.89 for fixed realizations |
| Welfare | baseline welfare 1.0845/firm > forced adoption 0.8558 (−21%); subsidies/platform welfare-negative; low equilibrium welfare-superior in ABM |

## Repository structure

```
.
├── model/                  # Core model (best-response dynamics)
│   ├── config.py           #   All calibrated parameters (Table 1)
│   ├── firm_generator.py   #   Firm types, data quality, adoption costs
│   ├── distance.py         #   Genetic-distance matrix
│   ├── accuracy.py         #   Accuracy, payoff, adoption threshold
│   ├── dynamics.py         #   Best-response (myopic adjustment) dynamics
│   └── scenarios.py        #   Policy scenarios P0–P3
├── analysis/               # Extended analyses
│   ├── abm.py              #   Agent-based model (continuous heterogeneity)
│   ├── channel_decomposition.py  # γ vs. θ channel separation
│   ├── premium_stats.py    #   Firm-level premium statistics (Fig 1c)
│   ├── run_abm_mc.py       #   ABM Monte Carlo (S19)
│   ├── run_abm_multistart.py    # ABM multi-start experiment (S20)
│   ├── poolsize_robustness.py   # Pool-size externality robustness (S21)
│   ├── welfare.py          #   Welfare assessment (S24–S26, Fig 7)
│   ├── heatmap_robustness.py    # Joint parameter-range heatmaps (S22–S23, Fig 6)
│   └── gap_items.py        #   Cross-effects, stability, k, trust boost, cost scaling (S11, S15–S17)
├── run_all.py              # One-command full pipeline + verification
├── run_analysis.py         # Main runner: P0–P3, sensitivity, stability
├── run_fst_all.py          # F_ST sensitivity (6 countries × 3 scales)
├── run_fscale.py           # Cost-scale robustness (F × 0.5–1.5)
├── make_figures.py         # Generate Fig 1–7 (PDF + PNG)
├── export_tables.py        # Assemble consolidated supplementary tables
├── data/                   # All simulation outputs (28 CSVs, S1–S26)
├── figures/                # Publication figures (14 files: Fig 1–7)
├── supplementary_tables.csv  # Consolidated supplementary tables
├── requirements.txt
└── .gitignore
```

## Installation

Requires Python 3.10+.

```bash
pip install -r requirements.txt   # numpy, pandas, matplotlib
```

## Quick start

All results are deterministic given the random seed (default: 42).

```bash
# One-command full pipeline: all data -> figures -> tables -> 42-value
# verification against the manuscript (~60-90 min)
python run_all.py --full

# Quick smoke test (compile + deterministic solve + regenerate artifacts
# from existing data; no data files overwritten; ~2 min)
python run_all.py --quick

# Verify the committed data/ against the manuscript's published values
python run_all.py --verify-only
```

### Running individual stages

Every stage is a standalone script (run from the repository root):

```bash
python run_analysis.py --mc 300        # P0-P3, cross-country, OAT, stability
python analysis/channel_decomposition.py   # γ vs θ channel separation (S6)
python run_fst_all.py                  # F_ST sensitivity (S10)
python run_fscale.py                   # Cost-scale robustness (S11)
python analysis/premium_stats.py       # Firm-level premium statistics (Fig 1c)
python analysis/run_abm_mc.py          # ABM Monte Carlo (S19, ~6 min)
python analysis/run_abm_multistart.py  # ABM multi-start (S20)
python analysis/poolsize_robustness.py # Pool-size robustness (S21)
python analysis/welfare.py             # Welfare assessment (S24–S26, Fig 7)
python analysis/heatmap_robustness.py  # Joint parameter-range heatmaps (S22–S23, Fig 6)
python analysis/gap_items.py           # Cross-effects, stability, k, trust boost,
                                       #   infrastructure cost scaling (S9/S12)
python make_figures.py                 # Regenerate all figures (reads data/)
python export_tables.py                # Reassemble supplementary tables CSV
```

## Model

The population consists of *n* = 1,000 seed enterprises partitioned into three R&D types — high (3%), medium (15%), and low (82%) — matching the empirical R&D-intensity distribution in China's seed industry. Each firm *i* carries:

- **Data quality** Iᵢ ~ Beta, with type-specific means 0.80 (H), 0.50 (M), 0.20 (L) on the [0,1] scale;
- **Adoption cost** Fᵢ ~ Uniform, with type-specific ranges;
- **Genetic distance** Dᵢⱼ, Beta-drawn with type-pair-specific means set within the range of F_ST values reported in crop diversity studies.

The firm-level technical premium is

Vᵢ = δ · [ α·Iᵢ + (1−λ)·β·S_φ(i)(ρ) − γ·d_φ(i)(ρ) ],

where the misalignment penalty γ and the transferability decay θ act through distinct channels. Adoption evolves via **best-response (myopic adjustment) dynamics** to a fixed point; the type-averaged model has a unique equilibrium, while the agent-based model with continuous heterogeneity admits multiple equilibria (multi-start experiment, S15).

## Data

All simulation outputs are committed in `data/` (28 CSVs, backing supplementary tables S1–S26 and all seven figures). Re-running the pipeline with the default seed reproduces these files exactly, and `run_all.py --verify-only` cross-checks 42 key values against the published numbers.

## Citation

Zhou, M., Yan, J., Peng, J., Zhang, Y.,  Pei, S., He, Z. *The Data Silo Trap: A Coordination Game Analysis of AI-Driven Breeding Adoption in Fragmented Seed Markets*. Farming System.
