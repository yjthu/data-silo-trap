"""
Best-response (myopic adjustment) dynamics — robust fixed-step integration.

The process dρ/dt = k(ρ* − ρ) always relaxes ρ toward the empirical adoption
target ρ*. Rather than an adaptive ODE solver (which struggles with the
piecewise-constant ρ*), we use a fixed-step Euler scheme that provably
converges to the fixed point ρ = ρ*(ρ) for the chosen step size.
"""
import numpy as np
from .config import (K_EVOL, ALPHA, BETA, LAMBDA, GAMMA, THETA, DELTA,
                      RHO_H0, RHO_M0, RHO_L0, FIRM_TYPE_SHARES)
from .accuracy import (shared_pool_and_centroid, firm_premiums, target_adoption)


def _rho_star(firms, stats, rho, alpha, beta, lambda_, gamma, theta, delta):
    S, d_c = shared_pool_and_centroid(stats, rho, theta)
    V = firm_premiums(firms, S, d_c, alpha, beta, lambda_, gamma, delta)
    return target_adoption(firms, V)


def solve_dynamics(firms, stats, rho0=None, alpha=None, beta=None, lambda_=None,
                   gamma=None, theta=None, delta=None, k=None,
                   F_scale=1.0, subsidy=0.0, fee=0.0, beta_boost=0.0,
                   dt=0.1, t_max=50.0, tol=1e-4):
    """
    Integrate best-response dynamics with fixed-step Euler to the fixed point.

    Returns a lightweight object with `.y` (3 × T array) and `.t` for
    compatibility with downstream T100().
    """
    if rho0 is None:
        rho0 = [RHO_H0, RHO_M0, RHO_L0]
    if k is None: k = K_EVOL
    if alpha is None: alpha = ALPHA
    if beta is None: beta = BETA
    if lambda_ is None: lambda_ = LAMBDA
    if gamma is None: gamma = GAMMA
    if theta is None: theta = THETA
    if delta is None: delta = DELTA

    import copy
    firms_adj = copy.deepcopy(firms)
    firms_adj['F'] = firms['F'] * F_scale
    reduction = max(subsidy, fee)
    if reduction > 0:
        firms_adj['F'] = np.maximum(0.05 * firms_adj['F'],
                                     firms_adj['F'] * (1 - reduction))
    beta_eff = beta + beta_boost

    rho = np.array(rho0, dtype=float)
    traj = [rho.copy()]
    ts = [0.0]
    t = 0.0
    n_steps = int(t_max / dt)

    for step in range(n_steps):
        rho_star = _rho_star(firms_adj, stats, rho, alpha, beta_eff,
                              lambda_, gamma, theta, delta)
        drift = k * (rho_star - rho)
        rho = np.clip(rho + dt * drift, 0.0, 1.0)
        t += dt
        ts.append(t)
        traj.append(rho.copy())
        if np.max(np.abs(rho_star - rho)) < tol:
            break

    traj = np.array(traj).T  # 3 × T
    ts = np.array(ts)

    class _Sol:
        pass
    sol = _Sol()
    sol.y = traj
    sol.t = ts
    return sol


def T100(sol):
    """Overall adoption at the final state."""
    return float(np.dot(FIRM_TYPE_SHARES, sol.y[:, -1]))
