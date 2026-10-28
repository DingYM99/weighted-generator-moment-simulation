"""
moment_tools.py
---------------
Symbolic generation of exact implicit raw-moment hierarchies for Itô systems
with rational drift and polynomial diffusion covariance, together with the
Gaussian and derivative-matching closure maps used in

  "Algebraic polynomialization and automated derivation of simulation-ready
   implicit raw-moment hierarchies for rational stochastic systems"
   (target: Mathematics and Computers in Simulation).

Everything here is pure symbolic/numeric scaffolding: the hierarchy is
generated from the model specification (drift, diffusion, denominator
factors) and is *exact* before closure.
"""
from __future__ import annotations

import itertools
import sympy as sp


# ----------------------------------------------------------------------
# multi-index helpers
# ----------------------------------------------------------------------
def multiindices(n, mindeg, maxdeg):
    """All multi-indices in N_0^n with mindeg <= |alpha| <= maxdeg."""
    out = []
    for d in range(mindeg, maxdeg + 1):
        for comb in itertools.combinations_with_replacement(range(n), d):
            idx = [0] * n
            for i in comb:
                idx[i] += 1
            out.append(tuple(idx))
    return out


def deg(a):
    return int(sum(a))


def add_idx(a, b):
    return tuple(x + y for x, y in zip(a, b))


def sub_idx(a, b):
    return tuple(x - y for x, y in zip(a, b))


def unit(n, r):
    v = [0] * n
    v[r] = 1
    return tuple(v)


def binom_multi(h, l):
    """Product of componentwise binomial coefficients."""
    p = 1
    for hr, lr in zip(h, l):
        if lr > hr:
            return 0
        p *= sp.binomial(hr, lr)
    return int(p)


# ----------------------------------------------------------------------
# polynomial handling
# ----------------------------------------------------------------------
def to_poly(expr, syms):
    """Expand after cancelling denominators; return {multi-index: coeff}."""
    expr = sp.cancel(sp.together(expr))
    expr = sp.expand(expr)
    if expr == 0:
        return {}
    p = sp.Poly(expr, *syms)
    return {tuple(m): sp.nsimplify(c, rational=False) for m, c in p.terms()}


def generator(f, syms, drift, a):
    """Backward Kolmogorov generator acting on f."""
    n = len(syms)
    out = 0
    for i in range(n):
        di = sp.diff(f, syms[i])
        if di != 0:
            out += sp.cancel(drift[i] * di)
    for i in range(n):
        for j in range(n):
            if a[i][j] == 0:
                continue
            dij = sp.diff(f, syms[i], syms[j])
            if dij != 0:
                out += sp.Rational(1, 2) * a[i][j] * dij
    return out


def diffusion_cov(sigma_cols, n):
    """a = sum_k sigma_k sigma_k^T from the columns of Sigma."""
    a = [[0] * n for _ in range(n)]
    for col in sigma_cols:
        for i in range(n):
            for j in range(n):
                a[i][j] += sp.expand(col[i] * col[j])
    return a


# ----------------------------------------------------------------------
# hierarchy generation
# ----------------------------------------------------------------------
def build_hierarchy(syms, drift, sigma_cols, R, A_eq):
    """
    For every alpha in A_eq return (alpha, lhs, rhs) with
        lhs : {alpha+beta : e_beta}   (coefficients of raw-moment derivatives)
        rhs : {nu : coeff}            (coefficients of raw moments)
    The identity is  sum_j lhs[j] * d/dt mu_j  =  sum_nu rhs[nu] * mu_nu.
    """
    n = len(syms)
    a = diffusion_cov(sigma_cols, n)
    Rd = to_poly(sp.expand(R), syms)
    out = []
    for alpha in A_eq:
        mono = 1
        for i in range(n):
            mono *= syms[i] ** alpha[i]
        T = sp.expand(mono * R)
        LT = generator(T, syms, drift, a)
        rhs = to_poly(LT, syms)
        lhs = {}
        for b, c in Rd.items():
            lhs[add_idx(alpha, b)] = c
        out.append((alpha, lhs, rhs))
    return out, Rd


def assemble(eqs, retained):
    """
    Split the selected identities into
        E0 d/dt mu + E1 d/dt mubar = C + A mu + B mubar .
    Returns E0, E1, C, A, B, unresolved (ordered list).
    """
    nret = len(retained)
    pos = {idx: k for k, idx in enumerate(retained)}
    zero = tuple(0 for _ in retained[0])

    seen = []
    for _, lhs, rhs in eqs:
        for idx in list(lhs) + list(rhs):
            if idx == zero or idx in pos:
                continue
            if idx not in seen:
                seen.append(idx)
    unresolved = sorted(seen, key=lambda t: (deg(t), t))
    upos = {idx: k for k, idx in enumerate(unresolved)}
    nun = len(unresolved)

    neq = len(eqs)
    E0 = sp.zeros(neq, nret)
    E1 = sp.zeros(neq, nun)
    C = sp.zeros(neq, 1)
    A = sp.zeros(neq, nret)
    B = sp.zeros(neq, nun)

    for r, (_, lhs, rhs) in enumerate(eqs):
        for idx, c in lhs.items():
            if idx in pos:
                E0[r, pos[idx]] += c
            else:
                E1[r, upos[idx]] += c
        for idx, c in rhs.items():
            if idx == zero:
                C[r, 0] += c
            elif idx in pos:
                A[r, pos[idx]] += c
            else:
                B[r, upos[idx]] += c
    return E0, E1, C, A, B, unresolved


# ----------------------------------------------------------------------
# Gaussian closure
# ----------------------------------------------------------------------
def gaussian_moment_polys(indices, n, retained, usyms):
    """
    Symbolic Gaussian closure functions for `indices`, expressed in the
    retained moment symbols usyms (ordered as `retained`).
    """
    pos = {idx: k for k, idx in enumerate(retained)}
    m = {}
    for r in range(n):
        m[r] = usyms[pos[unit(n, r)]]
    Cov = {}
    for r in range(n):
        for s in range(n):
            idx = add_idx(unit(n, r), unit(n, s))
            Cov[(r, s)] = usyms[pos[idx]] - m[r] * m[s]

    cache = {}
    zero = tuple(0 for _ in range(n))

    def G(alpha):
        if alpha == zero:
            return sp.Integer(1)
        if alpha in cache:
            return cache[alpha]
        r = next(i for i in range(n) if alpha[i] > 0)
        beta = list(alpha)
        beta[r] -= 1
        beta = tuple(beta)
        val = m[r] * G(beta)
        for s in range(n):
            if beta[s] > 0:
                gm = list(beta)
                gm[s] -= 1
                val += beta[s] * Cov[(r, s)] * G(tuple(gm))
        val = sp.expand(val)
        cache[alpha] = val
        return val

    return [sp.expand(G(a)) for a in indices]


# ----------------------------------------------------------------------
# derivative-matching closure
# ----------------------------------------------------------------------
def dm_exponents(index, retained):
    """
    Solve  C(index, alpha_s) = sum_p theta_p C(alpha_p, alpha_s)
    for the exponent vector theta (least squares; exact if consistent).
    """
    import numpy as np

    K = len(retained)
    M = np.zeros((K, K), dtype=float)
    rhs = np.zeros(K, dtype=float)
    for s, als in enumerate(retained):
        for p, alp in enumerate(retained):
            M[s, p] = binom_multi(alp, als)
        rhs[s] = binom_multi(index, als)
    theta, *_ = np.linalg.lstsq(M, rhs, rcond=None)
    resid = float(np.linalg.norm(M @ theta - rhs))
    return theta, resid


def dm_closure_polys(indices, retained, usyms):
    """Symbolic derivative-matching closure functions and their exponents."""
    import sympy as sp

    exprs, thetas = [], []
    for idx in indices:
        th, _ = dm_exponents(idx, retained)
        e = sp.Integer(1)
        for p, tp in enumerate(th):
            if abs(tp) > 1e-9:
                e *= usyms[p] ** sp.nsimplify(round(tp, 10))
        exprs.append(sp.expand(sp.powsimp(e)))
        thetas.append(th)
    return exprs, thetas


# ----------------------------------------------------------------------
# numeric callables
# ----------------------------------------------------------------------
def make_closure_callable(exprs, retained, usyms, params=None):
    """Return f(u)->Phi(u) and J(u)->DPhi(u) as numpy callables."""
    import sympy as sp
    import numpy as np

    nret = len(retained)
    J = sp.Matrix(len(exprs), nret,
                  lambda i, j: sp.diff(exprs[i], usyms[j]))
    f = sp.lambdify((usyms,), sp.Matrix(exprs), modules="numpy")
    Jf = sp.lambdify((usyms,), J, modules="numpy")

    def Phi(u):
        return np.asarray(f(list(u)), dtype=float).reshape(-1)

    def DPhi(u):
        return np.asarray(Jf(list(u)), dtype=float).reshape(len(exprs), nret)

    return Phi, DPhi
