"""Structural validation of the auto-generated hierarchies (no closure yet)."""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import sympy as sp
from moment_tools import (multiindices, build_hierarchy, assemble, deg)
import models

t0 = time.time()

# ---------------- Example 1 ----------------
syms, drift, sig, R, P = models.ex1_SI(sp.Rational(3, 20), sp.Rational(3, 20), p=2)
n = 2
N = 2
retained = multiindices(n, 1, N)
Aeq = list(retained)
eqs, Rd = build_hierarchy(syms, drift, sig, R, Aeq)
E0, E1, C, A, B, unresolved = assemble(eqs, retained)
print("Ex1 weight R =", sp.expand(R))
print("Ex1 retained  :", retained)
print("Ex1 unresolved (%d):" % len(unresolved), unresolved)
print("Ex1 N_cl      :", max(deg(u) for u in unresolved))
nlhs = sum(1 for u in unresolved if any(E1[r, k] != 0 for r in range(E1.rows)
           for k in [unresolved.index(u)]))
nder = sum(1 for k in range(E1.cols) if any(E1[r, k] != 0 for r in range(E1.rows)))
print("Ex1 N_dotmu   :", nder)
print("Ex1 t=%.1fs" % (time.time() - t0))

# ---------------- Example 2 ----------------
t1 = time.time()
syms2, drift2, sig2, R2, P2 = models.ex2_miRNA(sp.Rational(1, 20), sp.Rational(1, 4),
                                               sp.Rational(1, 20))
n2 = 3
retained2 = multiindices(n2, 1, 2)
eqs2, Rd2 = build_hierarchy(syms2, drift2, sig2, R2, list(retained2))
E02, E12, C2, A2, B2, unresolved2 = assemble(eqs2, retained2)
print("\nEx2 weight R =", sp.expand(R2))
print("Ex2 unresolved count:", len(unresolved2))
print("Ex2 N_cl      :", max(deg(u) for u in unresolved2))
nder2 = sum(1 for k in range(E12.cols) if any(E12[r, k] != 0 for r in range(E12.rows)))
print("Ex2 N_dotmu   :", nder2)
print("Ex2 order breakdown:",
      sorted(set(deg(u) for u in unresolved2)))
print("Ex2 t=%.1fs" % (time.time() - t1))
