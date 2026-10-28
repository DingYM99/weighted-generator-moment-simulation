"""
numerics.py — reproducible simulation engine.

Contents
  * Monte Carlo path simulation of the original SDEs (Milstein, diagonal noise)
    with per-time-point standard errors, so that every "reference" value carries
    an uncertainty band.
  * Integration of the closed mass-matrix moment systems.
  * Linear-noise-approximation (LNA) baseline.
  * Error metrics with an explicit Monte-Carlo resolution floor.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


# ----------------------------------------------------------------------
# Monte Carlo reference
# ----------------------------------------------------------------------
def simulate_mc(drift, sigma, dsigma, x0, T, dt, npaths, seed,
                out_times, indices, block=5000):
    """
    Milstein scheme for diagonal multiplicative noise:
        X += b(X) dt + s(X) dW + 0.5 s(X) s'(X) (dW^2 - dt)

    Returns
        mean : (nt, nmom)  empirical raw moments
        se   : (nt, nmom)  standard error of the mean at each output time
        nbad : number of paths lost to overflow / non-finite values
    """
    rng = np.random.default_rng(seed)
    n = len(x0)
    nsteps = int(round(T / dt))
    sqdt = np.sqrt(dt)
    X = np.tile(np.asarray(x0, dtype=float), (npaths, 1))
    nt = len(out_times)
    nm = len(indices)

    step_of = {int(round(out_times[k] / dt)): k for k in range(nt)}
    acc = np.zeros((nt, nm))
    acc2 = np.zeros((nt, nm))
    cnt = np.zeros(nt)
    nbad = 0

    def moments(X, out):
        good = np.isfinite(X).all(axis=1)
        Xg = X[good]
        ng = Xg.shape[0]
        for j, idx in enumerate(indices):
            v = np.ones(ng)
            for i in range(n):
                if idx[i] == 1:
                    v = v * Xg[:, i]
                elif idx[i] > 1:
                    v = v * Xg[:, i] ** idx[i]
            out[0][j] = v.sum()
            out[1][j] = (v * v).sum()
        return ng

    cnt[0] = moments(X, (acc[0], acc2[0]))

    for step in range(1, nsteps + 1):
        dW = rng.standard_normal((npaths, n)) * sqdt
        b = drift(X)
        s = sigma(X)
        sp_ = dsigma(X)
        X = X + b * dt + s * dW + 0.5 * s * sp_ * (dW * dW - dt)
        bad = ~np.isfinite(X).all(axis=1)
        if bad.any():
            nbad += int(bad.sum())
            X[bad] = np.nan
        k = step_of.get(step)
        if k is not None:
            cnt[k] = moments(X, (acc[k], acc2[k]))

    cnt = np.maximum(cnt, 1.0)
    mean = acc / cnt[:, None]
    var = np.maximum(acc2 / cnt[:, None] - mean ** 2, 0.0) * cnt[:, None] / np.maximum(cnt[:, None] - 1, 1)
    se = np.sqrt(np.maximum(var, 0.0)) / np.sqrt(cnt[:, None])
    return mean, se, nbad


# ----------------------------------------------------------------------
# closed moment system
# ----------------------------------------------------------------------
def integrate_closed(E0, E1, C, A, B, Phi, DPhi, u0, out_times,
                     rtol=1e-8, atol=1e-10, method="LSODA"):
    """Integrate  (E0 + E1 DPhi(u)) u' = C + A u + B Phi(u)."""
    E0 = np.asarray(E0, dtype=float)
    E1 = np.asarray(E1, dtype=float)
    Cv = np.asarray(C, dtype=float).reshape(-1)
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    conds = []
    nfallback = [0]

    def solve_M(M, g):
        """Solve M z = g with a rank-deficiency safeguard.

        The dense LU solve is used whenever M is far from singular.  If
        the estimated condition number exceeds 1e12, or M contains
        non-finite entries, the system is declared numerically singular
        and solved in the least-squares sense by the SVD-based
        pseudo-inverse, which is defined for every matrix.  Every use of
        the fallback is counted so that it can be reported.
        """
        if not np.all(np.isfinite(M)) or not np.all(np.isfinite(g)):
            nfallback[0] += 1
            return np.linalg.pinv(np.nan_to_num(M, nan=0.0,
                                                posinf=1e300, neginf=-1e300),
                                  rcond=1e-10) @ np.nan_to_num(g)
        kap = np.linalg.cond(M)
        if kap < 1e12:
            return np.linalg.solve(M, g)
        nfallback[0] += 1
        return np.linalg.lstsq(M, g, rcond=1e-10)[0]

    def rhs(t, u):
        rhs.ncalls += 1
        if rhs.ncalls > rhs.budget:
            raise RuntimeError("step budget %d exceeded; the mass matrix is "
                               "probably numerically singular" % rhs.budget)
        P = Phi(u)
        M = E0 + E1 @ DPhi(u)
        rhs.cond = np.linalg.cond(M)
        return solve_M(M, Cv + A @ u + B @ P)

    rhs.ncalls = 0
    rhs.budget = 150000

    sol = solve_ivp(rhs, (out_times[0], out_times[-1]), np.asarray(u0, float),
                    method=method, t_eval=out_times, rtol=rtol, atol=atol,
                    max_step=np.inf)
    if not sol.success:
        raise RuntimeError("ODE failure: " + sol.message)
    for k in range(sol.y.shape[1]):
        u = sol.y[:, k]
        conds.append(np.linalg.cond(E0 + E1 @ DPhi(u)))
    return sol.y.T, np.asarray(conds), {"n_fallback": nfallback[0],
                                        "sol": sol}


# ----------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------
def rmse(app, ref):
    return float(np.sqrt(np.mean((np.asarray(app) - np.asarray(ref)) ** 2)))


def nrmse(app, ref):
    app = np.asarray(app)
    ref = np.asarray(ref)
    return float(np.sqrt(np.mean((app - ref) ** 2)) / np.sqrt(np.mean(ref ** 2)))


def mc_floor(se, ref):
    """
    Contribution of Monte-Carlo sampling noise to the RMSE / NRMSE.
    sqrt( mean_t SE(t)^2 ) and its normalised counterpart.
    """
    se = np.asarray(se)
    ref = np.asarray(ref)
    return float(np.sqrt(np.mean(se ** 2))), \
        float(np.sqrt(np.mean(se ** 2)) / np.sqrt(np.mean(ref ** 2)))


# ----------------------------------------------------------------------
# linear noise approximation
# ----------------------------------------------------------------------
def lna(drift, jac, sigma_diag, x0, out_times, indices, rtol=1e-9, atol=1e-11):
    """
    LNA: mean follows the deterministic drift; covariance obeys
        S' = J S + S J^T + B B^T,   B = diag(sigma_i(m_i)).
    Raw moments are recovered from (m, S).
    """
    n = len(x0)
    nret = len(indices)

    def rhs(t, y):
        m = y[:n]
        S = y[n:].reshape(n, n)
        J = jac(m)
        Bm = np.diag(sigma_diag(m))
        dS = J @ S + S @ J.T + Bm @ Bm.T
        return np.concatenate([drift(m), dS.ravel()])

    y0 = np.concatenate([np.asarray(x0, float), np.zeros(n * n)])
    sol = solve_ivp(rhs, (out_times[0], out_times[-1]), y0, method="LSODA",
                    t_eval=out_times, rtol=rtol, atol=atol)
    out = np.zeros((len(out_times), nret))
    for k, t in enumerate(sol.t):
        m = sol.y[:n, k]
        S = sol.y[n:, k].reshape(n, n)
        vals = _raw_from_ms(m, S, indices)
        out[k] = vals
    return out


def _raw_from_ms(m, S, indices):
    """Raw moments of a Gaussian with mean m and covariance S (any order)."""
    n = len(m)
    zero = tuple(0 for _ in range(n))
    memo = {}

    def G(a):
        if a == zero:
            return 1.0
        if a in memo:
            return memo[a]
        r = next(i for i in range(n) if a[i] > 0)
        beta = list(a)
        beta[r] -= 1
        beta = tuple(beta)
        val = m[r] * G(beta)
        for s in range(n):
            if beta[s] > 0:
                gm = list(beta)
                gm[s] -= 1
                val += beta[s] * S[r, s] * G(tuple(gm))
        memo[a] = val
        return val

    return [G(tuple(idx)) for idx in indices]


def _gauss_raw(idx, m, S):
    """Raw moment E[prod x_i^idx_i] of a Gaussian (Isserlis recursion)."""
    return _raw_from_ms(np.asarray(m, float), np.asarray(S, float), [tuple(idx)])[0]
