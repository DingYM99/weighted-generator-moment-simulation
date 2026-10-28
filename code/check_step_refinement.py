"""Pilot time-step refinement for the Milstein reference of both examples.

Two runs are compared at a reduced path count.  The first pair uses an
independent increment stream for each step size, so the measured difference
carries the sampling noise of both runs.  The second pair drives the coarse
and the fine path with the same Brownian increments, aggregated over pairs of
fine steps, which removes almost all of the sampling noise and leaves the
discretisation difference.  Writes results/step_refinement.json.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from moment_tools import multiindices
import numerics as nu
import run_ex2_and_extras as rx

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results")

NPILOT = 2000
DT = 5e-3
N_OUT = 121


def run(spec, x0, T, dt, seed, indices, n_out=N_OUT):
    drift, sigma, dsigma = spec
    t = np.linspace(0.0, T, n_out)
    mean, se, nbad = nu.simulate_mc(drift, sigma, dsigma, x0, T, dt, NPILOT,
                                    seed, t, indices)
    return mean, se, nbad


def coupled(spec, x0, T, seed, indices, n_out=N_OUT):
    """Coarse and fine Milstein paths driven by the same increments."""
    drift, sigma, dsigma = spec
    n = len(x0)
    h = DT / 2
    n_fine = int(round(T / h))
    out_times = np.linspace(0.0, T, n_out)
    step_of = {int(round(t / h)): k for k, t in enumerate(out_times)}
    rng = np.random.default_rng(seed)
    Xc = np.tile(np.asarray(x0, float), (NPILOT, 1))
    Xf = Xc.copy()
    mean_c = np.zeros((n_out, len(indices)))
    mean_f = np.zeros((n_out, len(indices)))
    se_f = np.zeros((n_out, len(indices)))

    def values(X):
        out = np.empty(len(indices))
        for j, idx in enumerate(indices):
            v = np.ones(X.shape[0])
            for i in range(n):
                if idx[i] == 1:
                    v = v * X[:, i]
                elif idx[i] > 1:
                    v = v * X[:, i] ** idx[i]
            out[j] = v.mean()
        return out

    def record(step):
        k = step_of.get(step)
        if k is None:
            return
        mean_c[k] = values(Xc)
        mean_f[k] = values(Xf)
        sq = values(Xf ** 2)
        se_f[k] = np.sqrt(np.maximum(sq - mean_f[k] ** 2, 0.0) / NPILOT)

    record(0)
    accum = np.zeros((NPILOT, n))
    for step in range(1, n_fine + 1):
        dW = rng.standard_normal((NPILOT, n)) * np.sqrt(h)
        Xf = (Xf + drift(Xf) * h + sigma(Xf) * dW
              + 0.5 * sigma(Xf) * dsigma(Xf) * (dW * dW - h))
        accum += dW
        if step % 2 == 0:
            Xc = (Xc + drift(Xc) * DT + sigma(Xc) * accum
                  + 0.5 * sigma(Xc) * dsigma(Xc) * (accum * accum - DT))
            accum = np.zeros((NPILOT, n))
        record(step)
    return mean_c, mean_f, se_f


def study(name, spec, x0, T, seed, indices):
    coarse, _, bad_c = run(spec, x0, T, DT, seed, indices)
    fine, se_ind, bad_f = run(spec, x0, T, DT / 2, seed, indices)
    second = [j for j, idx in enumerate(indices) if sum(idx) == 2]
    diff = np.abs(fine[:, second] - coarse[:, second])
    ratio = diff / np.maximum(se_ind[:, second], 1e-300)

    mean_c, mean_f, se_c = coupled(spec, x0, T, seed, indices)
    cdiff = np.abs(mean_f[:, second] - mean_c[:, second])
    crel = cdiff / np.maximum(np.abs(mean_f[:, second]), 1e-300)
    cratio = cdiff / np.maximum(se_c[:, second], 1e-300)

    return dict(example=name, n_paths=NPILOT, seed=seed, dt=DT,
                n_second_moments=len(second),
                independent_max_change_over_se=float(ratio.max()),
                coupled_max_change_over_se=float(cratio.max()),
                coupled_max_relative_change=float(crel.max()),
                coupled_max_abs_change=float(cdiff.max()),
                coupled_worst_time=float(np.linspace(0.0, T, N_OUT)[
                    int(np.unravel_index(np.argmax(cratio), cratio.shape)[0])]),
                bad_paths=[int(bad_c), int(bad_f)])


out = []

dr1, sg1, dsg1 = rx.ex1_fields(0.15, 0.15)
out.append(study("Ex1 (0.15,0.15)", (dr1, sg1, dsg1), (0.1, 0.1), 15.0, 20240,
                 multiindices(2, 1, 2)))
print(out[-1], flush=True)

dr2, sg2, dsg2 = rx.ex2_fields(0.05, 0.25, 0.05)
out.append(study("Ex2 (0.05,0.25,0.05)", (dr2, sg2, dsg2), (0.1, 0.1, 0.1),
                 60.0, 700, multiindices(3, 1, 2)))
print(out[-1], flush=True)

with open(os.path.join(OUT, "step_refinement.json"), "w") as fh:
    json.dump(out, fh, indent=1)
print("step_refinement.json written")
