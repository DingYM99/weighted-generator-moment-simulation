"""
check_conditioning.py
---------------------
Re-integrates the closed moment systems with the nonsingularity safeguard
of numerics.integrate_closed and reports

  * the full time history range of kappa(M_Phi),
  * the smallest singular value of M_Phi along the trajectory,
  * the norm ||E0^{-1} E1 DPhi|| that appears in the sufficient
    nonsingularity criterion,
  * the number of times the least-squares fallback was triggered.

Only the deterministic closure integration is repeated, so this is cheap.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import sympy as sp

from moment_tools import (multiindices, build_hierarchy, assemble, deg,
                          gaussian_moment_polys, dm_closure_polys,
                          make_closure_callable)
import models
import numerics as nu

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")


def scan(syms, drift, sig, R, N, closure, x0, out_times, tag):
    ret = multiindices(len(syms), 1, N)
    eqs, _ = build_hierarchy(syms, drift, sig, R, list(ret))
    E0, E1, C, A, B, unres = assemble(eqs, ret)
    usyms = list(sp.symbols("u0:%d" % len(ret)))
    if closure == "DM":
        exprs, _th = dm_closure_polys(unres, ret, usyms)
    else:
        exprs = gaussian_moment_polys(unres, len(syms), ret, usyms)
    Phi, DPhi = make_closure_callable(exprs, ret, usyms)
    f = lambda M: np.array(M.tolist(), dtype=float)
    E0f, E1f, Cf, Af, Bf = f(E0), f(E1), f(C), f(A), f(B)
    u0 = np.array([np.prod([x0[i] ** a[i] for i in range(len(x0))]) for a in ret])

    traj, cond, info = nu.integrate_closed(E0f, E1f, Cf, Af, Bf, Phi, DPhi,
                                           u0, out_times)
    E0inv = np.linalg.inv(E0f)
    neumann, smin = [], []
    for k in range(traj.shape[0]):
        M = E0f + E1f @ DPhi(traj[k])
        neumann.append(np.linalg.norm(E0inv @ E1f @ DPhi(traj[k]), 2))
        smin.append(np.linalg.svd(M, compute_uv=False)[-1])
    print("%-26s %-9s kappa in [%.3g, %.3g]  sigma_min >= %.3g  "
          "||E0^-1E1 DPhi|| <= %.3g  fallback = %d"
          % (tag, closure, cond.min(), cond.max(), min(smin),
             max(neumann), info["n_fallback"]))
    return dict(tag=tag, closure=closure, kmin=float(cond.min()),
                kmax=float(cond.max()), smin=float(min(smin)),
                neumann=float(max(neumann)),
                fallback=int(info["n_fallback"]))


def main():
    out = []
    t1 = np.linspace(0.0, 15.0, 301)
    # ---------------- Example 1, minimal weight ----------------
    for (DS, DI) in [(0.05, 0.05), (0.15, 0.15), (0.40, 0.15)]:
        syms, drift, sig, R, _ = models.ex1_SI(
            sp.Rational(DS).limit_denominator(10 ** 6),
            sp.Rational(DI).limit_denominator(10 ** 6))
        for cl in ["DM", "Gaussian"]:
            out.append(scan(syms, drift, sig, R, 2, cl, (0.1, 0.1), t1,
                            "Ex1 (%.2f,%.2f)" % (DS, DI)))
    # ---------------- Example 1, weight-degree ablation --------
    for p in [2, 3, 4]:
        syms, drift, sig, R, _ = models.ex1_SI(sp.Rational(3, 20),
                                               sp.Rational(3, 20), p=p)
        for cl in ["DM", "Gaussian"]:
            out.append(scan(syms, drift, sig, R, 2, cl, (0.1, 0.1), t1,
                            "Ex1 p=%d" % p))
    # ---------------- Example 2 --------------------------------
    t2 = np.linspace(0.0, 60.0, 241)
    for D in [(0.01, 0.05, 0.02), (0.05, 0.25, 0.05)]:
        syms, drift, sig, R, _ = models.ex2_miRNA(
            *[sp.Rational(d).limit_denominator(10 ** 6) for d in D])
        for cl in ["DM", "Gaussian"]:
            out.append(scan(syms, drift, sig, R, 2, cl, (0.1, 0.1, 0.1), t2,
                            "Ex2 %s" % (str(D),)))

    json.dump(out, open(os.path.join(RES, "conditioning.json"), "w"), indent=1)
    print("\nwritten conditioning.json")

    # ---- LaTeX table -------------------------------------------------
    def sc(x):
        """Compact scientific notation: plain decimals for 10^-1..10^2."""
        if x == 0:
            return "$0$"
        e = int(np.floor(np.log10(abs(x))))
        if -1 <= e <= 2:
            return "$%.*f$" % (max(0, 2 - e), x)
        return "$%.2f\\times10^{%d}$" % (x / 10 ** e, e)

    keep = [r for r in out if r["tag"].startswith(("Ex1 (", "Ex2 "))]
    lines = [r"\begin{tabular}{@{}llccc@{}}", r"\toprule",
             "Case & Closure & $\\kappa(M_\\Phi)$ & "
             "$\\sigma_{\\min}$ & "
             "$\\sup\\|E_0^{-1}E_1D\\Phi\\|_2$\\\\", r"\midrule"]
    for r in keep:
        lines.append("%s & %s & %s--%s & %s & %s\\\\"
                     % (r["tag"], r["closure"], sc(r["kmin"]), sc(r["kmax"]),
                        sc(r["smin"]), sc(r["neumann"])))
    lines += [r"\bottomrule", r"\end{tabular}"]
    tab = os.path.join(HERE, "..", "tables")
    os.makedirs(tab, exist_ok=True)
    open(os.path.join(tab, "tab_cond.tex"), "w").write("\n".join(lines))
    print("written tables/tab_cond.tex")


if __name__ == "__main__":
    main()
