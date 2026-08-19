"""
Accuracy, payoff, and adoption-threshold computation.

- Profit is ABSOLUTE (private technical gains plus shared-pool spillovers),
  not relative to an industry average (which cancels in the payoff difference).
- Data quality I_i is on [0,1].
- Two distinct genetic-distance channels: θ (transferability decay in the
  shared pool) and γ (germplasm misalignment penalty).
- Finite-population adoption: ρ* is the EMPIRICAL fraction of firms whose
  cost F_i falls below the firm-level premium V_i (not an analytical CDF).
"""
import numpy as np
from .config import (ALPHA, BETA, GAMMA, THETA, DELTA, LAMBDA, FIRM_TYPE_SHARES)


def precompute_type_stats(firms, D):
    types, I, F = firms['types'], firms['I'], firms['F']
    n_types = 3
    mean_I = np.array([I[types == t].mean() if (types == t).any() else 0 for t in range(n_types)])
    mean_D = np.zeros((n_types, n_types))
    for t1 in range(n_types):
        for t2 in range(n_types):
            m1, m2 = types == t1, types == t2
            if m1.any() and m2.any():
                mean_D[t1, t2] = D[np.ix_(m1, m2)].mean()
    return {'mean_I': mean_I, 'mean_D': mean_D,
            'omega': np.exp(-THETA * mean_D)}


def shared_pool_and_centroid(stats, rho, theta=None):
    """
    Compute type-level shared-pool quality S_φ and centroid distance d_φ.

    S_φ = Σ_ψ w_ψ ρ_ψ I_ψ ω_φψ / Σ_ψ w_ψ ρ_ψ ω_φψ
      — the distance-weighted AVERAGE quality of the sharing pool as
        accessible to type φ (on the same [0,1] scale as I_φ).
    d_φ = Σ_ψ w_ψ ρ_ψ D_φψ / Σ_ψ w_ψ ρ_ψ
      — the average genetic distance from type φ to the adopter centroid.
    """
    w = FIRM_TYPE_SHARES
    mI = stats['mean_I']
    mD = stats['mean_D']
    omega = stats['omega'] if theta is None else np.exp(-theta * mD)

    S = np.zeros(3)
    d_c = np.zeros(3)
    for phi in range(3):
        num = sum(w[j] * rho[j] * mI[j] * omega[phi, j] for j in range(3))
        den = sum(w[j] * rho[j] * omega[phi, j] for j in range(3))
        S[phi] = num / den if den > 1e-12 else 0.0

        d_num = sum(w[j] * rho[j] * mD[phi, j] for j in range(3))
        d_den = sum(w[j] * rho[j] for j in range(3))
        d_c[phi] = d_num / d_den if d_den > 1e-12 else 0.0
    return S, d_c


def firm_premiums(firms, S_by_type, d_by_type, alpha=None, beta=None,
                  lambda_=None, gamma=None, delta=None):
    """
    Firm-level technical premium:
    V_i = δ · (α·I_i + (1−λ)·β·S_φ(i) − γ·d_φ(i))
    """
    if alpha is None: alpha = ALPHA
    if beta is None: beta = BETA
    if lambda_ is None: lambda_ = LAMBDA
    if gamma is None: gamma = GAMMA
    if delta is None: delta = DELTA

    types = firms['types']
    I = firms['I']
    V = np.zeros(firms['n_firms'])
    for t in range(3):
        mask = types == t
        V[mask] = delta * (alpha * I[mask] + (1 - lambda_) * beta * S_by_type[t] - gamma * d_by_type[t])
    return V


def target_adoption(firms, V):
    """
    Finite-population empirical adoption: ρ*_φ = (1/n_φ) Σ 1[F_i < V_i].
    """
    types = firms['types']
    F = firms['F']
    rho_star = np.zeros(3)
    for t in range(3):
        mask = types == t
        if mask.sum() > 0:
            rho_star[t] = (F[mask] < V[mask]).mean()
    return np.clip(rho_star, 0, 1)
