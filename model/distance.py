"""Genetic distance matrix generation (vectorized)."""
import numpy as np
from .config import DISTANCE_PARAMS as DP, THETA, FIRM_TYPE_LABELS

def generate_distance_matrix(firms, seed=None):
    rng = np.random.default_rng(seed)
    n = firms['n_firms']
    types = firms['types']
    labels = FIRM_TYPE_LABELS
    D = np.zeros((n, n))

    for t1 in range(3):
        for t2 in range(t1, 3):
            m1 = types == t1
            m2 = types == t2
            n1, n2 = m1.sum(), m2.sum()
            if n1 == 0 or n2 == 0:
                continue
            tp = DP[tuple(sorted([labels[t1], labels[t2]]))]
            if t1 == t2:
                vals = rng.beta(tp['alpha'], tp['beta'], size=(n1, n1))
                vals = np.triu(vals, 1)
                vals = vals + vals.T
                D[np.ix_(m1, m1)] = np.clip(vals, 0, 1)
            else:
                vals = rng.beta(tp['alpha'], tp['beta'], size=(n1, n2))
                D[np.ix_(m1, m2)] = np.clip(vals, 0, 1)
                D[np.ix_(m2, m1)] = np.clip(vals.T, 0, 1)
    return D, np.exp(-THETA * D)
