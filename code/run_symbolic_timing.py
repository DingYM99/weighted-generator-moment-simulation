"""Time the symbolic stage of both examples and write the record behind Table 5.

One build per row, no numerical integration.  The timings depend on the
machine, so the record is regenerated rather than stored as a constant.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_ex2_and_extras as rx

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results")

DS1, DI1 = 0.15, 0.15
D1, D2, D3 = 0.05, 0.25, 0.05


def timed_build(fn):
    t0 = time.time()
    system = fn()
    return time.time() - t0, system


rows = []
for N in [2, 3, 4]:
    elapsed, sysm = timed_build(
        lambda N=N: rx.build_ex1(DS1, DI1, p=2, N=N, closure="DM"))
    ret, unres = sysm[6], sysm[5]
    rows.append(dict(model="Ex1 (n=2)", N=N, nret=len(ret), nunres=len(unres),
                     Ncl=max(sum(u) for u in unres), wall=round(elapsed, 3)))
    print(rows[-1], flush=True)

for N in [2, 3]:
    elapsed, sysm = timed_build(
        lambda N=N: rx.build_ex2(D1, D2, D3, N=N, closure="DM"))
    ret, unres = sysm[6], sysm[5]
    rows.append(dict(model="Ex2 (n=3)", N=N, nret=len(ret), nunres=len(unres),
                     Ncl=max(sum(u) for u in unres), wall=round(elapsed, 3)))
    print(rows[-1], flush=True)

with open(os.path.join(OUT, "symbolic_timing.json"), "w") as fh:
    json.dump(rows, fh, indent=1)
print("symbolic_timing.json written")
