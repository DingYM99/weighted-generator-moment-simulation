"""Check the derivative-matching residuals and the invertibility of E_0.

Both are claims made in the manuscript, and neither was stored before.
Writes results/structure_checks.json.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import sympy as sp

from moment_tools import (assemble, binom_multi, build_hierarchy, dm_exponents,
                          multiindices)
import models

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results")


def matching_matrix(retained):
    K = len(retained)
    M = np.zeros((K, K), dtype=float)
    for s, alpha_s in enumerate(retained):
        for p, alpha_p in enumerate(retained):
            M[s, p] = binom_multi(alpha_p, alpha_s)
    return M


def inspect(name, syms, drift, sigma, R, retained):
    eqs, _ = build_hierarchy(syms, drift, sigma, R, list(retained))
    E0, E1, C, A, B, unresolved = assemble(eqs, retained)
    residuals, ranks = [], []
    for idx in unresolved:
        _, resid = dm_exponents(idx, retained)
        residuals.append(float(resid))
        ranks.append(int(np.linalg.matrix_rank(matching_matrix(retained))))
    E0n = np.array(E0.tolist(), dtype=float)
    return dict(example=name, n_retained=len(retained),
                n_targets=len(unresolved),
                max_residual=float(max(residuals)),
                min_system_rank=int(min(ranks)),
                system_size=len(retained),
                E0_det=float(np.linalg.det(E0n)),
                E0_cond=float(np.linalg.cond(E0n)))


out = {}

syms1, drift1, sig1, R1, _ = models.ex1_SI(sp.Rational(3, 20),
                                           sp.Rational(3, 20), p=2)
out["Ex1"] = inspect("Ex1", syms1, drift1, sig1, R1, multiindices(2, 1, 2))
print(out["Ex1"], flush=True)

syms2, drift2, sig2, R2, _ = models.ex2_miRNA(sp.Rational(1, 20),
                                              sp.Rational(1, 4),
                                              sp.Rational(1, 20))
out["Ex2"] = inspect("Ex2", syms2, drift2, sig2, R2, multiindices(3, 1, 2))
print(out["Ex2"], flush=True)

with open(os.path.join(OUT, "structure_checks.json"), "w") as fh:
    json.dump(out, fh, indent=1)
print("structure_checks.json written")
