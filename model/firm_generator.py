"""Stochastic agent population generation (I on [0,1] scale)."""
import numpy as np
from .config import FIRM_TYPE_SHARES, FIRM_TYPE_LABELS, I_DISTRIBUTIONS, F_DISTRIBUTIONS

def generate_firm_types(n_firms, shares=None, seed=None):
    if shares is None:
        shares = FIRM_TYPE_SHARES
    rng = np.random.default_rng(seed)
    types = rng.choice(len(shares), size=n_firms, p=shares)
    return types, np.array([FIRM_TYPE_LABELS[t] for t in types])

def generate_data_quality(types, distributions=None, seed=None):
    """I_i ~ Beta(a,b) on [0,1] (NO ×10 scaling)."""
    if distributions is None:
        distributions = I_DISTRIBUTIONS
    rng = np.random.default_rng(seed)
    I = np.zeros(len(types))
    for t, d in enumerate(distributions):
        mask = types == t
        if mask.sum() > 0:
            I[mask] = rng.beta(d['a'], d['b'], size=mask.sum())
    return np.clip(I, 0, 1)

def generate_fixed_costs(types, distributions=None, seed=None):
    if distributions is None:
        distributions = F_DISTRIBUTIONS
    rng = np.random.default_rng(seed)
    F = np.zeros(len(types))
    for t, d in enumerate(distributions):
        mask = types == t
        if mask.sum() > 0:
            F[mask] = rng.uniform(d['loc'], d['loc'] + d['scale'], size=mask.sum())
    return F

def generate_firms(n_firms, shares=None, i_dist=None, f_dist=None, seed=None):
    types, labels = generate_firm_types(n_firms, shares, seed)
    I = generate_data_quality(types, i_dist, seed)
    F = generate_fixed_costs(types, f_dist, seed)
    return {'types': types, 'type_labels': labels, 'I': I, 'F': F,
            'n_firms': n_firms, 'n_types': 3}
