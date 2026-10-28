"""Stage 2: Example 2 (miRNA model), weight-degree ablation, N=3, cost study."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import sympy as sp

from moment_tools import (multiindices, build_hierarchy, assemble, deg,
                          gaussian_moment_polys, dm_closure_polys,
                          make_closure_callable)
import models
import numerics as nu

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- Ex1 helpers
def build_ex1(DS, DI, p=2, N=2, closure="DM"):
    syms, drift, sig, R, P = models.ex1_SI(sp.Rational(DS).limit_denominator(10**6),
                                           sp.Rational(DI).limit_denominator(10**6), p=p)
    ret = multiindices(2, 1, N)
    eqs, _ = build_hierarchy(syms, drift, sig, R, list(ret))
    E0, E1, C, A, B, unres = assemble(eqs, ret)
    usyms = list(sp.symbols("u0:%d" % len(ret)))
    exprs = (dm_closure_polys(unres, ret, usyms)[0] if closure == "DM"
             else gaussian_moment_polys(unres, 2, ret, usyms))
    Phi, DPhi = make_closure_callable(exprs, ret, usyms)
    f = lambda M: np.array(M.tolist(), dtype=float)
    return f(E0), f(E1), f(C), f(A), f(B), unres, ret, Phi, DPhi


def build_ex2(D1, D2, D3, N=2, closure="DM"):
    syms, drift, sig, R, P = models.ex2_miRNA(sp.Rational(D1).limit_denominator(10**6),
                                              sp.Rational(D2).limit_denominator(10**6),
                                              sp.Rational(D3).limit_denominator(10**6))
    ret = multiindices(3, 1, N)
    eqs, _ = build_hierarchy(syms, drift, sig, R, list(ret))
    E0, E1, C, A, B, unres = assemble(eqs, ret)
    usyms = list(sp.symbols("u0:%d" % len(ret)))
    exprs = (dm_closure_polys(unres, ret, usyms)[0] if closure == "DM"
             else gaussian_moment_polys(unres, 3, ret, usyms))
    Phi, DPhi = make_closure_callable(exprs, ret, usyms)
    f = lambda M: np.array(M.tolist(), dtype=float)
    return f(E0), f(E1), f(C), f(A), f(B), unres, ret, Phi, DPhi


def dirac_ic(ret, x0):
    return np.array([np.prod([x0[i] ** a[i] for i in range(len(x0))]) for a in ret])


P1 = dict(A=1.0, beta=0.5, h=0.5, k=0.2)
P2 = dict(gi=0.75, gm=0.06, V1=7.0, V2=5.0, K1=3.0, K2=7.0,
          gamma=0.2, delta=0.02, di=0.02, dm=0.02, dp=0.06)


def ex1_fields(DS, DI):
    def dr(X):
        S, I = X[:, 0], X[:, 1]
        q = 1.0 + P1["h"] * I
        return np.column_stack([P1["A"] - P1["beta"] * S * I / q - P1["k"] * S,
                                P1["beta"] * S * I / q - P1["k"] * I])
    return dr, lambda X: np.column_stack([DS * X[:, 0], DI * X[:, 1]]), \
        lambda X: np.tile([DS, DI], (X.shape[0], 1))


def ex2_fields(D1, D2, D3):
    def dr(X):
        x, y, z = X[:, 0], X[:, 1], X[:, 2]
        return np.column_stack([
            P2["gi"] + P2["V1"] * z**2 / (z**2 + P2["K1"]**2)
            - P2["delta"] * x * y - P2["di"] * x,
            P2["gm"] + P2["V2"] * z**2 / (z**2 + P2["K2"]**2)
            - P2["dm"] * y - P2["delta"] * x * y,
            P2["gamma"] * y - P2["dp"] * z])
    return dr, lambda X: np.column_stack([D1 * X[:, 0], D2 * X[:, 1], D3 * X[:, 2]]), \
        lambda X: np.tile([D1, D2, D3], (X.shape[0], 1))


if __name__ == "__main__":
    t00 = time.time()
    out = {}

    # =====================================================================
    # 1) Example 2
    # =====================================================================
    T2, DT2, NP2, NS2 = 60.0, 0.005, 20000, 3
    OT2 = np.linspace(0.0, T2, 241)
    RET2 = multiindices(3, 1, 2)
    X02 = (0.1, 0.1, 0.1)
    out["ex2"] = {"settings": dict(T=T2, dt=DT2, npaths=NP2, nseeds=NS2,
                                   x0=list(X02), out_times=len(OT2)),
                  "cases": {}}
    for (D1, D2, D3) in [(0.01, 0.05, 0.02), (0.05, 0.25, 0.05)]:
        print("=== Ex2 noise %s ===" % str((D1, D2, D3)), flush=True)
        dr, sg, dsg = ex2_fields(D1, D2, D3)
        ms, ses = [], []
        for s in range(NS2):
            t0 = time.time()
            m, se, nbad = nu.simulate_mc(dr, sg, dsg, X02, T2, DT2, NP2,
                                         700 + 31 * s, OT2, RET2)
            ms.append(m); ses.append(se)
            print("   seed %d (%.1fs) bad=%d" % (s, time.time() - t0, nbad), flush=True)
        ms = np.array(ms); ses = np.array(ses)
        ref = ms.mean(axis=0)
        sec = np.sqrt((ses ** 2).mean(axis=0) / NS2)
        spread = ms.std(axis=0, ddof=1)

        # LNA
        def dm_fun(m):
            x, y, z = m
            return np.array([P2["gi"] + P2["V1"] * z**2 / (z**2 + P2["K1"]**2)
                             - P2["delta"] * x * y - P2["di"] * x,
                             P2["gm"] + P2["V2"] * z**2 / (z**2 + P2["K2"]**2)
                             - P2["dm"] * y - P2["delta"] * x * y,
                             P2["gamma"] * y - P2["dp"] * z])
        def jac_fun(m):
            x, y, z = m
            d1 = P2["V1"] * 2 * z * P2["K1"]**2 / (z**2 + P2["K1"]**2) ** 2
            d2 = P2["V2"] * 2 * z * P2["K2"]**2 / (z**2 + P2["K2"]**2) ** 2
            return np.array([[-P2["delta"] * y - P2["di"], -P2["delta"] * x, d1],
                             [-P2["delta"] * y, -P2["dm"] - P2["delta"] * x, d2],
                             [0.0, P2["gamma"], -P2["dp"]]])
        lna_t = nu.lna(dm_fun, jac_fun, lambda m: np.array([D1 * m[0], D2 * m[1], D3 * m[2]]),
                       X02, OT2, RET2)

        case = {"noise": [D1, D2, D3], "methods": {}}
        for cl in ["DM", "Gaussian"]:
            t0 = time.time()
            E0, E1, C, A, B, unres, ret, Phi, DPhi = build_ex2(D1, D2, D3, 2, cl)
            traj, cond, info = nu.integrate_closed(E0, E1, C, A, B, Phi, DPhi,
                                                dirac_ic(ret, X02), OT2)
            nr = [nu.nrmse(traj[:, j], ref[:, j]) for j in range(len(RET2))]
            fl = [nu.mc_floor(sec[:, j], ref[:, j])[1] for j in range(len(RET2))]
            print("   %-8s mean NRMSE=%.4e floor=%.4e cond[%.1f,%.1f] (%.1fs)"
                  % (cl, np.mean(nr), np.mean(fl), cond.min(), cond.max(),
                     time.time() - t0), flush=True)
            case["methods"]["Generator-" + cl] = {
                "nrmse": nr, "mean_nrmse": float(np.mean(nr)),
                "traj": traj.tolist(), "cond": cond.tolist()}
        nr = [nu.nrmse(lna_t[:, j], ref[:, j]) for j in range(len(RET2))]
        print("   LNA      mean NRMSE=%.4e" % np.mean(nr), flush=True)
        case["methods"]["LNA"] = {"nrmse": nr, "mean_nrmse": float(np.mean(nr)),
                                  "traj": lna_t.tolist()}
        case["mc_ref"] = ref.tolist(); case["mc_se"] = sec.tolist()
        case["mc_seed_spread"] = spread.tolist()
        case["mc_floor"] = [nu.mc_floor(sec[:, j], ref[:, j])[1] for j in range(len(RET2))]
        case["t"] = OT2.tolist()
        out["ex2"]["cases"]["%g_%g_%g" % (D1, D2, D3)] = case
        with open(os.path.join(OUT, "ex2_and_extras.json"), "w") as fh:
            json.dump(out, fh)
        print("   [checkpoint written]", flush=True)

    # =====================================================================
    # 2) weight-degree ablation on Example 1 at (0.15,0.15) and (0.40,0.15)
    # =====================================================================
    T1, DT1, NP1 = 15.0, 0.005, 20000
    OT1 = np.linspace(0.0, T1, 301)
    RET1 = multiindices(2, 1, 2)
    X01 = (0.1, 0.1)
    out["ablation"] = {"cases": {}}
    for (DS, DI) in [(0.15, 0.15), (0.40, 0.15)]:
        dr, sg, dsg = ex1_fields(DS, DI)
        ref = None
        cache = os.path.join(OUT, "ex1.json")
        if os.path.exists(cache):
            j = json.load(open(cache))
            key = "%.2f_%.2f" % (DS, DI)
            if key in j["cases"]:
                ref = np.array(j["cases"][key]["mc_ref"])
                sec = np.array(j["cases"][key]["mc_se"])
        if ref is None:
            ms, ses = [], []
            for s in range(3):
                m, se, _ = nu.simulate_mc(dr, sg, dsg, X01, T1, DT1, NP1,
                                          900 + 13 * s, OT1, RET1)
                ms.append(m); ses.append(se)
            ref = np.mean(ms, axis=0)
            sec = np.sqrt((np.array(ses) ** 2).mean(axis=0) / 3)
        c = {"noise": [DS, DI], "rows": []}
        for p in [2, 3, 4]:
            for cl in ["DM", "Gaussian"]:
                E0, E1, Cc, A, B, unres, ret, Phi, DPhi = build_ex1(DS, DI, p, 2, cl)
                traj, cond, info = nu.integrate_closed(E0, E1, Cc, A, B, Phi, DPhi,
                                                    dirac_ic(ret, X01), OT1)
                nr = [nu.nrmse(traj[:, j], ref[:, j]) for j in range(len(RET1))]
                c["rows"].append(dict(p=p, closure=cl,
                                      Ncl=max(deg(u) for u in unres),
                                      Nmu=len(unres),
                                      Ndotmu=int(sum(1 for j in range(E1.shape[1])
                                                     if np.abs(E1[:, j]).max() > 0)),
                                      mean_nrmse=float(np.mean(nr)),
                                      nrmse=nr,
                                      cond_max=float(max(cond)),
                                      cond_min=float(min(cond))))
                print("   abl p=%d %-8s Ncl=%d Nmu=%d NRMSE=%.4e"
                      % (p, cl, max(deg(u) for u in unres), len(unres), np.mean(nr)),
                      flush=True)
        c["mc_floor"] = [nu.mc_floor(sec[:, j], ref[:, j])[1] for j in range(len(RET1))]
        out["ablation"]["cases"]["%.2f_%.2f" % (DS, DI)] = c
        with open(os.path.join(OUT, "ex2_and_extras.json"), "w") as fh:
            json.dump(out, fh)
        print("   [checkpoint written]", flush=True)

    # =====================================================================
    # 3) N = 3 on Example 1
    # =====================================================================
    out["N3"] = {"cases": {}}
    for (DS, DI) in [(0.15, 0.15), (0.40, 0.15)]:
        j = json.load(open(os.path.join(OUT, "ex1.json")))
        key = "%.2f_%.2f" % (DS, DI)
        ref5 = np.array(j["cases"][key]["mc_ref"])
        t1 = np.array(j["cases"][key]["t"])
        c = {"noise": [DS, DI], "rows": []}
        for cl in ["DM", "Gaussian"]:
            try:
                E0, E1, Cc, A, B, unres, ret, Phi, DPhi = build_ex1(DS, DI, 2, 3, cl)
                traj, cond, info = nu.integrate_closed(E0, E1, Cc, A, B, Phi, DPhi,
                                                       dirac_ic(ret, X01), t1)
                # compare the five first/second-order moments against the same MC ref
                idx5 = [ret.index(a) for a in multiindices(2, 1, 2)]
                nr = [nu.nrmse(traj[:, i], ref5[:, j]) for j, i in enumerate(idx5)]
                c["rows"].append(dict(closure=cl, Nret=len(ret),
                                      Ncl=max(deg(u) for u in unres),
                                      Nmu=len(unres),
                                      mean_nrmse_N2subset=float(np.mean(nr)),
                                      cond_max=float(max(cond)),
                                      cond_min=float(min(cond)),
                                      n_fallback=int(info["n_fallback"]),
                                      status="ok"))
                print("   N=3 %-8s Ncl=%d Nmu=%d  NRMSE(orders<=2)=%.4e"
                      % (cl, max(deg(u) for u in unres), len(unres), np.mean(nr)),
                      flush=True)
            except Exception as exc:
                c["rows"].append(dict(closure=cl, Nret=9, Ncl=-1, Nmu=-1,
                                      mean_nrmse_N2subset=float("nan"),
                                      n_fallback=-1,
                                      status="failed: " + str(exc)[:120]))
                print("   N=3 %-8s FAILED: %s" % (cl, str(exc)[:120]), flush=True)
        out["N3"]["cases"][key] = c
        with open(os.path.join(OUT, "ex2_and_extras.json"), "w") as fh:
            json.dump(out, fh)
        print("   [checkpoint written]", flush=True)

    # =====================================================================
    # 4) work--error (cost) study on Example 1 at (0.15,0.15)
    # =====================================================================
    DS, DI = 0.15, 0.15
    dr, sg, dsg = ex1_fields(DS, DI)
    print("=== cost study ===", flush=True)
    REF = []
    for s in range(5):
        m, se, _ = nu.simulate_mc(dr, sg, dsg, X01, T1, DT1, 20000, 3000 + 7 * s,
                                  OT1, RET1)
        REF.append(m)
    REF = np.mean(REF, axis=0)
    cost = {"reference_paths": 100000, "rows": []}
    for npth in [1000, 5000, 20000, 80000]:
        t0 = time.time()
        m, se, _ = nu.simulate_mc(dr, sg, dsg, X01, T1, DT1, npth, 4242, OT1, RET1)
        el = time.time() - t0
        nr = np.mean([nu.nrmse(m[:, j], REF[:, j]) for j in range(len(RET1))])
        cost["rows"].append(dict(kind="MC", n=npth, wall=el, mean_nrmse=float(nr)))
        print("   MC n=%-6d wall=%6.2fs NRMSE=%.3e" % (npth, el, nr), flush=True)
    for cl in ["DM", "Gaussian"]:
        E0, E1, Cc, A, B, unres, ret, Phi, DPhi = build_ex1(DS, DI, 2, 2, cl)
        for tol in [1e-3, 1e-5, 1e-7, 1e-9, 1e-11]:
            t0 = time.time()
            traj, cond, info = nu.integrate_closed(E0, E1, Cc, A, B, Phi, DPhi,
                                                  dirac_ic(ret, X01), OT1,
                                                  rtol=tol, atol=tol * 1e-2)
            el = time.time() - t0
            nr = np.mean([nu.nrmse(traj[:, j], REF[:, j]) for j in range(len(RET1))])
            cost["rows"].append(dict(kind="closure-" + cl, n=int(info['sol'].nfev), tol=tol,
                                     wall=el, mean_nrmse=float(nr)))
            print("   %-8s tol=%.0e wall=%6.3fs NRMSE=%.3e" % (cl, tol, el, nr), flush=True)
    # symbolic generation cost
    symcost = []
    for p in [2, 3, 4, 5, 6]:
        t0 = time.time()
        build_ex1(DS, DI, p, 2, "DM")
        symcost.append(dict(p=p, wall=time.time() - t0))
        print("   symbolic build p=%d  %.2fs" % (p, time.time() - t0), flush=True)
    for N in [2, 3, 4]:
        t0 = time.time()
        build_ex1(DS, DI, 2, N, "DM")
        symcost.append(dict(N=N, wall=time.time() - t0))
        print("   symbolic build N=%d  %.2fs" % (N, time.time() - t0), flush=True)
    cost["symbolic"] = symcost
    out["cost"] = cost

    with open(os.path.join(OUT, "ex2_and_extras.json"), "w") as fh:
        json.dump(out, fh)
    print("\nwritten ex2_and_extras.json (%.1fs)" % (time.time() - t00))
