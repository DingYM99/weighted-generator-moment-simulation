"""Stage 1: Example 1 (stochastic SI model) — full numerical study."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import sympy as sp

from moment_tools import (multiindices, build_hierarchy, assemble, deg,
                          gaussian_moment_polys, dm_closure_polys,
                          make_closure_callable, dm_exponents)
import models
import numerics as nu

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(OUT, exist_ok=True)

A, beta, h, k = 1.0, 0.5, 0.5, 0.2
X0 = (0.1, 0.1)
T_END, DT, NPATHS, NSEEDS = 15.0, 0.005, 20000, 5
OUT_T = np.linspace(0.0, T_END, 301)
NOISE = [(0.05, 0.05), (0.15, 0.15), (0.40, 0.15)]
RET = multiindices(2, 1, 2)


# ---------------------------------------------------------------- closures
def build_system(DS, DI, p=2, N=2, closure="DM"):
    syms, drift, sig, R, P = models.ex1_SI(sp.Rational(DS).limit_denominator(10**6),
                                           sp.Rational(DI).limit_denominator(10**6), p=p)
    ret = multiindices(2, 1, N)
    eqs, Rd = build_hierarchy(syms, drift, sig, R, list(ret))
    E0, E1, C, A, B, unres = assemble(eqs, ret)
    usyms = list(sp.symbols("u0:%d" % len(ret)))
    if closure == "DM":
        exprs, th = dm_closure_polys(unres, ret, usyms)
    else:
        exprs = gaussian_moment_polys(unres, 2, ret, usyms)
    Phi, DPhi = make_closure_callable(exprs, ret, usyms)
    f = lambda M: np.array(M.tolist(), dtype=float)
    return (f(E0), f(E1), f(C), f(A), f(B), unres, ret, Phi, DPhi, exprs)


def dirac_ic(ret, x0):
    return np.array([np.prod([x0[i] ** a[i] for i in range(len(x0))]) for a in ret])


# ---------------------------------------------------------------- sanity
if __name__ == "__main__":
    t0 = time.time()
    # --- check DM closure against the published formulas (Appendix B.2) ---
    E0, E1, Cvec, Amat, Bmat, unres, ret, Phi, DPhi, exprs = build_system(0.15, 0.15, p=2)
    usyms = list(sp.symbols("u0:%d" % len(ret)))
    u_test = {usyms[0]: sp.Rational(11, 10), usyms[1]: sp.Rational(38, 10),
              usyms[2]: sp.Rational(15, 10), usyms[3]: sp.Rational(45, 10),
              usyms[4]: sp.Rational(150, 10)}
    ref = {(1, 2): usyms[4] * usyms[3] ** 2 / (usyms[0] * usyms[1] ** 2),
           (2, 1): usyms[2] * usyms[3] ** 2 / (usyms[0] ** 2 * usyms[1]),
           (0, 3): usyms[4] ** 3 / usyms[1] ** 3,
           (0, 4): usyms[4] ** 6 / usyms[1] ** 8,
           (2, 2): usyms[2] * usyms[4] * usyms[3] ** 4 / (usyms[0] ** 4 * usyms[1] ** 4),
           (3, 1): usyms[2] ** 3 * usyms[3] ** 3 / (usyms[0] ** 6 * usyms[1] ** 2),
           (1, 3): usyms[4] ** 3 * usyms[3] ** 3 / (usyms[0] ** 2 * usyms[1] ** 6)}
    print("--- DM closure validation (against Appendix B.2) ---")
    for idx, e in zip(unres, exprs):
        d = sp.simplify(sp.nsimplify(sp.expand(e - ref[idx])))
        print("  ", idx, "residual =", d)

    # --- Monte Carlo + closed systems ---
    res = {"settings": dict(T=T_END, dt=DT, npaths=NPATHS, nseeds=NSEEDS,
                            out_times=len(OUT_T), x0=list(X0)),
           "cases": {}}
    for (DS, DI) in NOISE:
        print("\n=== noise (%.2f, %.2f) ===" % (DS, DI))
        mc_means, mc_ses = [], []
        for s in range(NSEEDS):
            tm = time.time()
            def dr(X):
                S, I = X[:, 0], X[:, 1]
                q = 1.0 + h * I
                return np.column_stack([A - beta * S * I / q - k * S,
                                        beta * S * I / q - k * I])
            sg = lambda X: np.column_stack([DS * X[:, 0], DI * X[:, 1]])
            dsg = lambda X: np.tile([DS, DI], (X.shape[0], 1))
            m, se, nbad = nu.simulate_mc(dr, sg, dsg, X0, T_END, DT, NPATHS,
                                         20240 + 17 * s, OUT_T, RET)
            mc_means.append(m); mc_ses.append(se)
            print("   seed %d  (%.1fs)  bad paths=%d" % (s, time.time() - tm, nbad))
        mc_means = np.array(mc_means)
        mc_ses = np.array(mc_ses)
        mc_ref = mc_means.mean(axis=0)
        mc_se_comb = np.sqrt((mc_ses ** 2).mean(axis=0) / NSEEDS)
        seed_spread = mc_means.std(axis=0, ddof=1)

        # LNA
        def drift_m(m):
            S, I = m
            q = 1.0 + h * I
            return np.array([A - beta * S * I / q - k * S, beta * S * I / q - k * I])
        def jac_m(m):
            S, I = m
            q = 1.0 + h * I
            return np.array([[-beta * I / q - k, -beta * S / q ** 2],
                             [beta * I / q, beta * S / q ** 2 - k]])
        lna_traj = nu.lna(drift_m, jac_m, lambda m: np.array([DS * m[0], DI * m[1]]),
                          X0, OUT_T, RET)

        case = {"noise": [DS, DI], "methods": {}}
        for cl in ["DM", "Gaussian"]:
            E0f, E1f, Cf, Af, Bf, unres, ret, Phi, DPhi, _ = build_system(DS, DI, 2, 2, cl)
            u0 = dirac_ic(ret, X0)
            traj, cond, info = nu.integrate_closed(E0f, E1f, Cf, Af, Bf, Phi, DPhi,
                                                  u0, OUT_T)
            nr = [nu.nrmse(traj[:, j], mc_ref[:, j]) for j in range(len(RET))]
            fl = [nu.mc_floor(mc_se_comb[:, j], mc_ref[:, j])[1] for j in range(len(RET))]
            print("   %-8s MeanNRMSE=%.4e  floor=%.4e  cond(M) in [%.2f, %.2f]"
                  % (cl, np.mean(nr), np.mean(fl), cond.min(), cond.max()))
            case["methods"]["Generator-" + cl] = {
                "nrmse": nr, "mean_nrmse": float(np.mean(nr)),
                "traj": traj.tolist(), "cond": cond.tolist(),
                "Ncl": max(deg(u) for u in unres),
                "Nmu": len(unres),
                "Ndotmu": int(sum(1 for j in range(E1f.shape[1])
                                  if np.abs(E1f[:, j]).max() > 0))}
        nr = [nu.nrmse(lna_traj[:, j], mc_ref[:, j]) for j in range(len(RET))]
        print("   %-8s MeanNRMSE=%.4e" % ("LNA", np.mean(nr)))
        case["methods"]["LNA"] = {"nrmse": nr, "mean_nrmse": float(np.mean(nr)),
                                  "traj": lna_traj.tolist()}
        case["mc_ref"] = mc_ref.tolist()
        case["mc_se"] = mc_se_comb.tolist()
        case["mc_seed_spread"] = seed_spread.tolist()
        case["mc_floor"] = [nu.mc_floor(mc_se_comb[:, j], mc_ref[:, j])[1]
                            for j in range(len(RET))]
        case["t"] = OUT_T.tolist()
        res["cases"]["%.2f_%.2f" % (DS, DI)] = case

    with open(os.path.join(OUT, "ex1.json"), "w") as fh:
        json.dump(res, fh)
    print("\nwritten ex1.json  (%.1fs)" % (time.time() - t0))
