"""Generate the manuscript figures and LaTeX tables from the result JSON files."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figures")
TAB = os.path.join(HERE, "..", "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

plt.rcParams.update({"font.size": 8, "axes.labelsize": 9, "axes.titlesize": 9,
                     "legend.fontsize": 7, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "figure.dpi": 200,
                     "mathtext.fontset": "cm", "font.family": "serif"})

RED, BLUE, GREEN, GREY = "#c0392b", "#2471a3", "#1e8449", "#555555"
IDX1 = [(1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]
# Panel titles must follow the order in which the moment arrays are stored,
# which is multiindices(3, 1, 2), not the caption order used previously.
IDX2 = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (2, 0, 0), (1, 1, 0), (1, 0, 1),
        (0, 2, 0), (0, 1, 1), (0, 0, 2)]


def lab(a):
    return "\\mu_{(" + ",".join(str(x) for x in a) + ")}"


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".eps"), bbox_inches="tight")
    plt.close(fig)


def band(ax, t, m, se, color=GREY, k=1.96, alpha=0.25, lw=1.4):
    ax.plot(t, m, color="k", lw=lw, zorder=3)
    ax.fill_between(t, m - k * se, m + k * se, color=color, alpha=alpha,
                    lw=0, zorder=1)


# ---------------------------------------------------------------- Fig 1
def fig_ex1_traj(ex1):
    noises = [(0.05, 0.05), (0.15, 0.15), (0.40, 0.15)]
    fig, axs = plt.subplots(3, 5, figsize=(11.5, 5.6), sharex=True)
    for r, (DS, DI) in enumerate(noises):
        c = ex1["cases"]["%.2f_%.2f" % (DS, DI)]
        t = np.array(c["t"]); ref = np.array(c["mc_ref"]); se = np.array(c["mc_se"])
        for j in range(5):
            ax = axs[r, j]
            band(ax, t, ref[:, j], se[:, j])
            ax.plot(t, np.array(c["methods"]["Generator-DM"]["traj"])[:, j],
                    color=RED, lw=1.2, ls="-")
            ax.plot(t, np.array(c["methods"]["Generator-Gaussian"]["traj"])[:, j],
                    color=BLUE, lw=1.2, ls="--")
            ax.plot(t, np.array(c["methods"]["LNA"]["traj"])[:, j],
                    color=GREEN, lw=1.0, ls=":")
            if r == 0:
                ax.set_title("$%s$" % lab(IDX1[j]), fontsize=9)
            if j == 0:
                ax.set_ylabel("$(D_S,D_I)=(%g,%g)$" % (DS, DI), fontsize=8)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))
            ax.grid(alpha=0.25, lw=0.4)
    for j in range(5):
        axs[-1, j].set_xlabel("$t$")
    from matplotlib.lines import Line2D
    leg = [Line2D([], [], color="k", lw=1.4),
           Line2D([], [], color=RED, lw=1.2),
           Line2D([], [], color=BLUE, lw=1.2, ls="--"),
           Line2D([], [], color=GREEN, lw=1.0, ls=":")]
    fig.legend(leg, ["Monte Carlo (95\\% CI)", "Generator--DM",
                     "Generator--Gaussian", "LNA"],
               loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=[0, 0.045, 1, 1])
    save(fig, "fig1_ex1_traj")


# ---------------------------------------------------------------- Fig 2
def fig_ex1_nrmse(ex1):
    noises = [(0.05, 0.05), (0.15, 0.15), (0.40, 0.15)]
    fig, axs = plt.subplots(1, 5, figsize=(11.5, 2.5))
    for j in range(5):
        ax = axs[j]
        w = 0.27
        for i, (DS, DI) in enumerate(noises):
            c = ex1["cases"]["%.2f_%.2f" % (DS, DI)]
            vals = [c["methods"][k]["nrmse"][j] for k in
                    ["Generator-DM", "Generator-Gaussian", "LNA"]]
            fl = c["mc_floor"][j]
            xs = np.arange(3) + i * w
            ax.bar(xs, vals, width=w, color=[RED, BLUE, GREEN],
                   edgecolor="k", lw=0.3)
            ax.plot([i * w - 0.5 * w, i * w + 2.5 * w], [np.sqrt(2) * fl] * 2,
                    color=GREY, lw=0.8, ls="--")
        ax.set_yscale("log")
        ax.set_title("$%s$" % lab(IDX1[j]), fontsize=9)
        ax.set_xticks(np.arange(3) + w)
        ax.set_xticklabels(["DM", "Gauss", "LNA"], fontsize=7)
        ax.grid(alpha=0.25, axis="y", lw=0.4)
    axs[0].set_ylabel("NRMSE")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig.legend([Patch(facecolor=GREY, alpha=0.6),
                Line2D([], [], color=GREY, ls="--", lw=0.8)],
               ["noise group: $(0.05,0.05)$, $(0.15,0.15)$, $(0.40,0.15)$",
                "$\\sqrt{2}\\times$ Monte Carlo resolution floor"],
               loc="lower center", ncol=2, frameon=False, fontsize=7)
    fig.tight_layout(rect=[0, 0.16, 1, 1])
    save(fig, "fig2_ex1_nrmse")


# ---------------------------------------------------------------- Fig 3/4
def fig_ex2_traj(ex2, key, name, title_noise):
    c = ex2["cases"][key]
    t = np.array(c["t"]); ref = np.array(c["mc_ref"]); se = np.array(c["mc_se"])
    fig, axs = plt.subplots(3, 3, figsize=(9.5, 6.2), sharex=True)
    for j in range(9):
        ax = axs[j // 3, j % 3]
        band(ax, t, ref[:, j], se[:, j])
        ax.plot(t, np.array(c["methods"]["Generator-DM"]["traj"])[:, j],
                color=RED, lw=1.2)
        ax.plot(t, np.array(c["methods"]["Generator-Gaussian"]["traj"])[:, j],
                color=BLUE, lw=1.2, ls="--")
        ax.set_title("$%s$" % lab(IDX2[j]), fontsize=9)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))
        ax.grid(alpha=0.25, lw=0.4)
    for j in range(3):
        axs[2, j].set_xlabel("$t$")
    from matplotlib.lines import Line2D
    fig.legend([Line2D([], [], color="k", lw=1.4),
                Line2D([], [], color=RED, lw=1.2),
                Line2D([], [], color=BLUE, lw=1.2, ls="--")],
               ["Monte Carlo (95\\% CI)", "Generator--DM", "Generator--Gaussian"],
               loc="lower center", ncol=3, frameon=False)
    fig.suptitle(title_noise, fontsize=9, y=0.99)
    fig.tight_layout(rect=[0, 0.045, 1, 0.975])
    save(fig, name)


# ---------------------------------------------------------------- Fig 5
def fig_ex2_nrmse(ex2):
    keys = ["0.01_0.05_0.02", "0.05_0.25_0.05"]
    fig, axs = plt.subplots(1, 9, figsize=(11.5, 2.5))
    for j in range(9):
        ax = axs[j]
        w = 0.35
        for i, k in enumerate(keys):
            c = ex2["cases"][k]
            vals = [c["methods"][m]["nrmse"][j] for m in
                    ["Generator-DM", "Generator-Gaussian"]]
            ax.bar(np.arange(2) + i * w, vals, width=w,
                   color=[RED, BLUE], edgecolor="k", lw=0.3)
        ax.set_yscale("log")
        ax.set_title("$%s$" % lab(IDX2[j]), fontsize=9)
        ax.set_xticks(np.arange(2) + w / 2)
        ax.set_xticklabels(["DM", "Gauss"], fontsize=7)
        ax.grid(alpha=0.25, axis="y", lw=0.4)
    axs[0].set_ylabel("NRMSE")
    fig.suptitle("left group: $(D_1,D_2,D_3)=(0.01,0.05,0.02)$; "
                 "right group: $(0.05,0.25,0.05)$", fontsize=8, y=1.02)
    fig.tight_layout()
    save(fig, "fig5_ex2_nrmse")


# ---------------------------------------------------------------- Fig 6
def fig_cost_abl(cost, abl):
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.4))
    ax = axs[0]
    mc = [r for r in cost["rows"] if r["kind"] == "MC"]
    ax.loglog([r["wall"] for r in mc], [r["mean_nrmse"] for r in mc], "o-",
              color=GREY, label="Monte Carlo (path count $10^3$--$8\\cdot10^4$)")
    for r in mc:
        ax.annotate("%d" % r["n"], (r["wall"], r["mean_nrmse"]),
                    textcoords="offset points", xytext=(4, -9), fontsize=6)
    for cl, col in [("DM", RED), ("Gaussian", BLUE)]:
        rows = [r for r in cost["rows"] if r["kind"] == "closure-" + cl]
        ax.loglog([r["wall"] for r in rows], [r["mean_nrmse"] for r in rows],
                  "s--", color=col, label="closed moment system, " + cl)
    ax.set_xlabel("wall-clock time (s)")
    ax.set_ylabel("mean NRMSE against the $10^5$-path reference")
    ax.grid(alpha=0.25, which="both", lw=0.4)
    ax.legend(frameon=False, fontsize=7)

    ax = axs[1]
    c = abl["cases"]["0.15_0.15"]
    ps = sorted(set(r["p"] for r in c["rows"]))
    w = 0.35
    for i, cl in enumerate(["DM", "Gaussian"]):
        rows = [r for r in c["rows"] if r["closure"] == cl]
        rows.sort(key=lambda r: r["p"])
        xs = np.arange(len(ps)) + i * w
        ax.bar(xs, [r["mean_nrmse"] for r in rows], width=w,
               color=[RED, BLUE][i], edgecolor="k", lw=0.3, label=cl)
        for x, r in zip(xs, rows):
            ax.text(x, r["mean_nrmse"], "$N_{\\rm cl}\\!=\\!%d$" % r["Ncl"],
                    ha="center", va="bottom", fontsize=6)
    ax.axhline(np.mean(c["mc_floor"]), color=GREY, ls="--", lw=0.8)
    ax.text(0.02, np.mean(c["mc_floor"]), " MC floor", fontsize=6, va="bottom")
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(ps)) + w / 2)
    ax.set_xticklabels(["$p=%d$" % p for p in ps])
    ax.set_xlabel("weight exponent $p$ in $R=(1+hI)^p$")
    ax.set_ylabel("mean NRMSE, $(D_S,D_I)=(0.15,0.15)$")
    ax.grid(alpha=0.25, axis="y", lw=0.4)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, "fig6_cost_ablation")


# ---------------------------------------------------------------- Fig 7
def fig_cond(ex1, ex2):
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.0))
    ax = axs[0]
    c = ex1["cases"]["0.15_0.15"]
    t = np.array(c["t"])
    for m, col, ls in [("Generator-DM", RED, "-"),
                       ("Generator-Gaussian", BLUE, "--")]:
        ax.semilogy(t, c["methods"][m]["cond"], color=col, ls=ls, lw=1.3, label=m)
    ax.set_xlabel("$t$"); ax.set_ylabel("$\\kappa(M_\\Phi)$")
    ax.set_title("Example 1, $(D_S,D_I)=(0.15,0.15)$", fontsize=9)
    ax.grid(alpha=0.25, which="both", lw=0.4); ax.legend(frameon=False, fontsize=7)
    ax = axs[1]
    c = ex2["cases"]["0.05_0.25_0.05"]
    t = np.array(c["t"])
    for m, col, ls in [("Generator-DM", RED, "-"),
                       ("Generator-Gaussian", BLUE, "--")]:
        ax.semilogy(t, c["methods"][m]["cond"], color=col, ls=ls, lw=1.3, label=m)
    ax.set_xlabel("$t$"); ax.set_ylabel("$\\kappa(M_\\Phi)$")
    ax.set_title("Example 2, $(D_1,D_2,D_3)=(0.05,0.25,0.05)$", fontsize=9)
    ax.grid(alpha=0.25, which="both", lw=0.4); ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, "fig7_cond")


# ---------------------------------------------------------------- tables
def sci(x, d=2):
    if x == 0:
        return "$0$"
    e = int(np.floor(np.log10(abs(x))))
    m = x / 10 ** e
    return "$%.*f\\times10^{%d}$" % (d, m, e)


def tab_ex1(ex1):
    noises = [(0.05, 0.05), (0.15, 0.15), (0.40, 0.15)]
    lines = []
    lines.append("\\begin{tabular}{@{}ll" + "c" * 3 + "@{}}")
    lines.append("\\toprule")
    lines.append("Method & $(D_S,D_I)$ & Mean NRMSE & MC floor "
                 "& Ratio\\\\")
    lines.append("\\midrule")
    for m, nm in [("Generator-DM", "Generator--DM"),
                  ("Generator-Gaussian", "Generator--Gaussian"),
                  ("LNA", "LNA")]:
        for i, (DS, DI) in enumerate(noises):
            c = ex1["cases"]["%.2f_%.2f" % (DS, DI)]
            v = c["methods"][m]["mean_nrmse"]
            f = float(np.mean(c["mc_floor"]))
            lines.append("%s & $(%.2f,%.2f)$ & %s & %s & %.1f\\\\"
                         % (nm if i == 0 else "", DS, DI, sci(v), sci(f), v / f))
        lines.append("\\midrule" if m != "LNA" else "")
    lines.pop()
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    open(os.path.join(TAB, "tab_ex1_nrmse.tex"), "w").write("\n".join(lines))


def tab_ablation(abl):
    lines = []
    lines.append("\\begin{tabular}{@{}ccccccc@{}}")
    lines.append("\\toprule")
    lines.append("$p$ & Closure & $N_{\\rm cl}$ & $N_\\mu$ & $N_{\\dot\\mu}$ "
                 "& \\multicolumn{2}{c}{Mean NRMSE}\\\\")
    lines.append("\\cmidrule(lr){6-7}")
    lines.append("&&&&& $(0.15,0.15)$ & $(0.40,0.15)$\\\\")
    lines.append("\\midrule")
    c1 = abl["cases"]["0.15_0.15"]
    c2 = abl["cases"]["0.40_0.15"]
    for p in [2, 3, 4]:
        for cl in ["DM", "Gaussian"]:
            r1 = [r for r in c1["rows"] if r["p"] == p and r["closure"] == cl][0]
            r2 = [r for r in c2["rows"] if r["p"] == p and r["closure"] == cl][0]
            lines.append("$%d$ & %s & $%d$ & $%d$ & $%d$ & %s & %s\\\\"
                         % (p, cl, r1["Ncl"], r1["Nmu"], r1["Ndotmu"],
                            sci(r1["mean_nrmse"]), sci(r2["mean_nrmse"])))
    lines.append("\\midrule")
    lines.append("\\multicolumn{5}{@{}l}{Monte Carlo resolution floor} & %s & %s\\\\"
                 % (sci(float(np.mean(c1["mc_floor"]))),
                    sci(float(np.mean(c2["mc_floor"])))))
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    open(os.path.join(TAB, "tab_ablation.tex"), "w").write("\n".join(lines))


def tab_ex2(ex2):
    keys = [(0.01, 0.05, 0.02), (0.05, 0.25, 0.05)]
    lines = []
    lines.append("\\begin{tabular}{@{}ll" + "c" * 3 + "@{}}")
    lines.append("\\toprule")
    lines.append("Method & $(D_1,D_2,D_3)$ & Mean NRMSE & MC floor "
                 "& Ratio\\\\")
    lines.append("\\midrule")
    for m, nm in [("Generator-DM", "Generator--DM"),
                  ("Generator-Gaussian", "Generator--Gaussian"),
                  ("LNA", "LNA")]:
        for i, D in enumerate(keys):
            c = ex2["cases"]["%g_%g_%g" % D]
            v = c["methods"][m]["mean_nrmse"]
            if "mc_floor" in c:
                f = float(np.mean(c["mc_floor"]))
            else:
                se = np.array(c["mc_se"])
                ref = np.array(c["mc_ref"])
                f = float(np.mean([
                    np.sqrt(np.mean(se[:, j] ** 2))
                    / np.sqrt(np.mean(ref[:, j] ** 2)) for j in range(9)]))
            lines.append("%s & $(%g,%g,%g)$ & %s & %s & %.1f\\\\"
                         % (nm if i == 0 else "", D[0], D[1], D[2],
                            sci(v), sci(f), v / f))
        lines.append("\\midrule")
    lines.pop()
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    open(os.path.join(TAB, "tab_ex2_nrmse.tex"), "w").write("\n".join(lines))


def tab_cost(cost):
    lines = []
    lines.append("\\begin{tabular}{@{}llcc@{}}")
    lines.append("\\toprule")
    lines.append("Solver & Setting & Wall clock (s) & Mean NRMSE\\\\")
    lines.append("\\midrule")
    for r in cost["rows"]:
        if r["kind"] == "MC":
            lines.append("Monte Carlo & $n_{\\rm path}=%d$ & %.2f & %s\\\\"
                         % (r["n"], r["wall"], sci(r["mean_nrmse"])))
        else:
            lines.append("Closed system (%s) & tol $=%.0e$ & %.3f & %s\\\\"
                         % (r["kind"].split("-")[1], r["tol"], r["wall"],
                            sci(r["mean_nrmse"])))
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    open(os.path.join(TAB, "tab_cost.tex"), "w").write("\n".join(lines))


def tab_n3(n3):
    lines = []
    lines.append("\\begin{tabular}{@{}llcccc@{}}")
    lines.append("\\toprule")
    lines.append("$(D_S,D_I)$ & Closure & Retained moments & "
                 "$N_{\\rm cl}$ & $N_\\mu$ & NRMSE (orders $\\leq2$)\\\\")
    lines.append("\\midrule")
    for key, c in n3["cases"].items():
        DS, DI = [float(x) for x in key.split("_")]
        for r in c["rows"]:
            if r.get("status", "ok") != "ok":
                lines.append("$(%.2f,%.2f)$ & %s & $%d$ & -- & -- & not completed\\\\"
                             % (DS, DI, r["closure"], r.get("Nret", 9)))
                continue
            lines.append("$(%.2f,%.2f)$ & %s & $%d$ & $%d$ & $%d$ & %s\\\\"
                         % (DS, DI, r["closure"], r["Nret"], r["Ncl"],
                            r["Nmu"], sci(r["mean_nrmse_N2subset"])))
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    open(os.path.join(TAB, "tab_n3.tex"), "w").write("\n".join(lines))


def tab_symbolic(rows):
    lines = []
    lines.append("\\begin{tabular}{@{}llcccc@{}}")
    lines.append("\\toprule")
    lines.append("Model & $N$ & Retained & Unresolved & "
                 "$N_{\\rm cl}$ & Build (s)\\\\")
    lines.append("\\midrule")
    for r in rows:
        lines.append("%s & $%d$ & $%d$ & $%d$ & $%d$ & %.2f\\\\"
                     % (r["model"], r["N"], r["nret"], r["nunres"],
                        r["Ncl"], r["wall"]))
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    open(os.path.join(TAB, "tab_symbolic.tex"), "w").write("\n".join(lines))


if __name__ == "__main__":
    ex1 = json.load(open(os.path.join(RES, "ex1.json")))
    extra = json.load(open(os.path.join(RES, "ex2_and_extras.json")))
    ex2 = extra["ex2"]
    fig_ex1_traj(ex1)
    fig_ex1_nrmse(ex1)
    fig_ex2_traj(ex2, "0.01_0.05_0.02", "fig3_ex2_traj_weak",
                 "$(D_1,D_2,D_3)=(0.01,0.05,0.02)$")
    fig_ex2_traj(ex2, "0.05_0.25_0.05", "fig4_ex2_traj_strong",
                 "$(D_1,D_2,D_3)=(0.05,0.25,0.05)$")
    fig_ex2_nrmse(ex2)
    fig_cost_abl(extra["cost"], extra["ablation"])
    fig_cond(ex1, ex2)
    tab_ex1(ex1); tab_ablation(extra["ablation"])
    tab_ex2(ex2); tab_cost(extra["cost"]); tab_n3(extra["N3"])
    tab_symbolic(json.load(open(os.path.join(RES, "symbolic_timing.json"))))
    print("figures and tables written")
