"""
Model configuration for the Data Silo Trap analysis.

1. Data quality I_i on the unified [0,1] scale.
   I_H ~ Beta(8,2) mean 0.80; I_M ~ Beta(5,5) mean 0.50; I_L ~ Beta(2,8) mean 0.20.
   Quality threshold I* = 0.5 is reached by ~98% of H firms and ~50% of M firms,
   and by ~2% of L firms (the untruncated Beta(2,8) tail) — the trickle-down
   mechanism operates through this tail, not through categorical exclusion.
2. Profit is absolute (private technical gains plus shared-pool spillovers),
   not relative to an industry average (ā cancels in the payoff difference).
3. Two distinct genetic-distance channels are separated:
   θ = cross-population transferability decay (value of others' data)
   γ = germplasm misalignment cost (penalty for sharing-pool mismatch)
"""
import numpy as np

# ---- Firm type distribution (China baseline) ----
W_H, W_M, W_L = 0.03, 0.15, 0.82
FIRM_TYPE_SHARES = np.array([W_H, W_M, W_L])
FIRM_TYPE_LABELS = ['H', 'M', 'L']

# ---- Data quality distributions I_i ~ Beta(a,b), on [0,1] ----
I_DIST_H = {'dist': 'beta', 'a': 8, 'b': 2}   # mean 0.80
I_DIST_M = {'dist': 'beta', 'a': 5, 'b': 5}   # mean 0.50
I_DIST_L = {'dist': 'beta', 'a': 2, 'b': 8}   # mean 0.20
I_DISTRIBUTIONS = [I_DIST_H, I_DIST_M, I_DIST_L]

# ---- Adoption cost distributions F_i ~ Uniform (million RMB) ----
F_DIST_H = {'dist': 'uniform', 'loc': 0.2, 'scale': 0.2}
F_DIST_M = {'dist': 'uniform', 'loc': 0.1, 'scale': 0.1}
F_DIST_L = {'dist': 'uniform', 'loc': 0.03, 'scale': 0.05}
F_DISTRIBUTIONS = [F_DIST_H, F_DIST_M, F_DIST_L]

# ---- Technical parameters (I on [0,1]) ----
ALPHA = 0.5    # Own-data benefit coefficient
               # α·I_φ: H=0.39, M=0.25, L=0.10 (own-data accuracy contribution)
BETA = 0.5     # Shared-data benefit coefficient
LAMBDA = 0.3   # Free-riding discount (non-adopter share of pool benefit)
GAMMA = 0.45   # Germplasm misalignment (centroid) penalty
               # Defensible: Fst>0.25 → 30-50% accuracy loss (Scutari et al. 2016)
THETA = 2.0    # Cross-population transferability decay in ω = exp(−θ·D)
DELTA = 1.0    # Profit gain per unit accuracy (million RMB), normalized
K_EVOL = 0.2   # Best-response adjustment speed

# ---- Genetic distance matrix (type-pair means, Beta-distributed) ----
DISTANCE_PARAMS = {
    ('H', 'H'): {'alpha': 8, 'beta': 32,  'mean': 0.20},
    ('H', 'M'): {'alpha': 6, 'beta': 9,   'mean': 0.40},
    ('H', 'L'): {'alpha': 4, 'beta': 2.67,'mean': 0.60},
    ('M', 'M'): {'alpha': 5, 'beta': 5,   'mean': 0.50},
    ('L', 'M'): {'alpha': 3, 'beta': 1.29,'mean': 0.70},
    ('L', 'L'): {'alpha': 2, 'beta': 0.5, 'mean': 0.80},
}

# ---- Simulation parameters ----
N_FIRMS = 1000
N_MONTE_CARLO = 500
TIME_SPAN = (0, 200)
TIME_EVAL = np.linspace(0, 200, 201)

ODE_METHOD = 'RK45'
ODE_RTOL = 1e-6
ODE_ATOL = 1e-8
ODE_MAX_STEP = 0.5

RHO_H0, RHO_M0, RHO_L0 = 0.10, 0.02, 0.00
RANDOM_SEED = 42

# ---- Policy parameters ----
SUBSIDY_RANGE = np.arange(0, 1.05, 0.05)
SUBSIDY_FLOOR = 0.05
QUALITY_THRESHOLD = 0.5   # I* : only H and some M can reach
BONUS_LEVELS = np.array([0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.25, 0.30, 0.40, 0.50])
PLATFORM_BETA_BOOST = 0.1

# ---- Cross-country parameters ----
COUNTRY_PARAMS = {
    'China':     {'w_H': 0.03, 'w_M': 0.15, 'w_L': 0.82},
    'India':     {'w_H': 0.02, 'w_M': 0.10, 'w_L': 0.88},
    'Africa':    {'w_H': 0.005,'w_M': 0.02, 'w_L': 0.975},
    'Brazil':    {'w_H': 0.08, 'w_M': 0.25, 'w_L': 0.67},
    'Indonesia': {'w_H': 0.04, 'w_M': 0.18, 'w_L': 0.78},
    'Nigeria':   {'w_H': 0.02, 'w_M': 0.12, 'w_L': 0.86},
}

# ---- OAT sensitivity ranges ----
OAT_RANGES = {
    'alpha':  np.linspace(0.3,  0.7,   9),
    'beta':   np.linspace(0.3,  0.7,   9),
    'lambda': np.linspace(0.0,  0.8,   9),
    'gamma':  np.linspace(0.2,  0.6,   9),
    'theta':  np.linspace(1.0,  3.0,   9),
    'delta':  np.linspace(0.5,  1.5,   7),
}
