"""
models.py — model specifications for the two numerical examples.

All parameters are supplied as exact rationals so that the symbolic
denominator cancellation in the generator action is exact.
"""
import sympy as sp


def ex1_SI(DS, DI, p=2, params=None):
    """
    Stochastic SI model with saturated incidence.
        dS = [A - beta S I/(1+h I) - k S] dt + DS S dW_S
        dI = [ beta S I/(1+h I) - k I] dt + DI I dW_I
    Weight R = (1 + h I)^p  (p = 2 is the componentwise-minimal choice).
    """
    P = dict(A=sp.Integer(1), beta=sp.Rational(1, 2), h=sp.Rational(1, 2),
             k=sp.Rational(1, 5))
    if params:
        P.update(params)
    S, I = sp.symbols("S I", positive=True)
    syms = (S, I)
    drift = [P["A"] - P["beta"] * S * I / (1 + P["h"] * I) - P["k"] * S,
             P["beta"] * S * I / (1 + P["h"] * I) - P["k"] * I]
    sig = [[DS * S, sp.Integer(0)], [sp.Integer(0), DI * I]]
    R = (1 + P["h"] * I) ** p
    return syms, drift, sig, R, P


def ex2_miRNA(D1, D2, D3, params=None):
    """
    Stochastic miRNA-mediated gene-regulatory model.
        dx = [gi + V1 z^2/(z^2+K1^2) - delta x y - (di+Lam) x] dt + D1 x dW1
        dy = [gm + V2 z^2/(z^2+K2^2) - dm y - delta x y] dt       + D2 y dW2
        dz = [gamma y + ks - dp z] dt                              + D3 z dW3
    Weight R = (z^2+K1^2)(z^2+K2^2).
    """
    P = dict(gi=sp.Rational(3, 4), gm=sp.Rational(3, 50), V1=sp.Integer(7),
             V2=sp.Integer(5), K1=sp.Integer(3), K2=sp.Integer(7),
             gamma=sp.Rational(1, 5), delta=sp.Rational(1, 50),
             di=sp.Rational(1, 50), dm=sp.Rational(1, 50),
             dp=sp.Rational(3, 50), Lam=sp.Integer(0), ks=sp.Integer(0))
    if params:
        P.update(params)
    x, y, z = sp.symbols("x y z", positive=True)
    syms = (x, y, z)
    drift = [P["gi"] + P["V1"] * z**2 / (z**2 + P["K1"]**2)
             - P["delta"] * x * y - (P["di"] + P["Lam"]) * x,
             P["gm"] + P["V2"] * z**2 / (z**2 + P["K2"]**2)
             - P["dm"] * y - P["delta"] * x * y,
             P["gamma"] * y + P["ks"] - P["dp"] * z]
    sig = [[D1 * x, sp.Integer(0), sp.Integer(0)],
           [sp.Integer(0), D2 * y, sp.Integer(0)],
           [sp.Integer(0), sp.Integer(0), D3 * z]]
    R = (z**2 + P["K1"]**2) * (z**2 + P["K2"]**2)
    return syms, drift, sig, R, P
