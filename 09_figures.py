"""Publication figures (static PNG, 300 dpi) from the saved results. Palette: reference categorical slots
(blue = observed data, orange = reference bound, gray = permutation null)."""
import pickle
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from common import FIGS, RESULTS

warnings.filterwarnings("ignore")
BLUE, ORANGE, GRAY, LGRAY, GRID, INK, INK2, AQUA = "#2a78d6", "#eb6834", "#898781", "#c3c2b7", "#e1e0d9", "#0b0b0b", "#52514e", "#1baf7a"
plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9, "axes.edgecolor": LGRAY, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.titlesize": 10, "axes.titleweight": "bold",
                     "axes.titlecolor": INK, "savefig.dpi": 300, "figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})
CURVE_X = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 350, 400, 450, 500]


def load(name):
    p = RESULTS / name
    return pickle.load(open(p, "rb")) if p.exists() else None


def tidy(ax, grid="y"):
    ax.grid(axis=grid, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(FIGS / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", name, flush=True)


def fig_repro():
    r = load("insample_null.pkl")
    if r is None:
        return
    obs, nul = r[0], [x for x in r if x["kind"] == "null"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 3.5), gridspec_kw={"width_ratios": [1.05, 1.2]})
    rows = [("multi_mt_ci", "Multivariate model,\nall MT columns"), ("multi_all_ci", "Multivariate model,\nall genes"),
            ("top10_ci_mt", "Top-10 univariate\ncolumns, MT genes"), ("top10_ci_all", "Top-10 univariate\ncolumns, all genes")]
    for i, (k, lab) in enumerate(rows):
        z = np.array([x[k] for x in nul], float)
        a.plot([np.nanquantile(z, .05), np.nanquantile(z, .95)], [i, i], color=LGRAY, lw=7, solid_capstyle="round", zorder=1)
        a.plot(np.nanmean(z), i, "|", color=GRAY, ms=14, mew=2, zorder=2)
        a.plot(obs[k], i, "o", color=BLUE, ms=7, zorder=3, mec="white", mew=1)
    a.set_yticks(range(len(rows)))
    a.set_yticklabels([x[1] for x in rows], fontsize=8)
    a.invert_yaxis()
    a.set_xlabel("In-sample concordance index")
    a.set_xlim(0.65, 1.0)
    tidy(a, "x")
    a.set_title("A  Permuted data give the same in-sample CIs", loc="left", fontsize=9)
    a.plot([], [], "o", color=BLUE, label="observed data")
    a.plot([], [], color=LGRAY, lw=7, label="permutation null, 5-95%")
    a.legend(frameon=False, fontsize=7.5, loc="center", bbox_to_anchor=(0.36, 0.5))
    xs = CURVE_X
    cur = np.array([x["curve"] for x in nul if "curve" in x], float)
    b.fill_between(xs, np.nanquantile(cur, .05, 0), np.nanquantile(cur, .95, 0), color=LGRAY, alpha=0.7, lw=0, label="permutation null, 5-95%")
    b.plot(xs, np.nanmean(cur, 0), color=GRAY, lw=1.6)
    b.plot(xs, obs["curve"], color=BLUE, lw=2, marker="o", ms=3.5, label="observed data")
    b.set_xlabel("Number of top univariate genes in the model")
    b.set_ylabel("In-sample CI")
    b.set_ylim(0.7, 1.0)
    tidy(b)
    b.legend(frameon=False, fontsize=7.5, loc="lower right")
    b.set_title("B  The same rise occurs on permuted data", loc="left", fontsize=9)
    fig.tight_layout()
    save(fig, "fig1_insample.png")


def fig_audit():
    a = load("audit.pkl")
    if a is None:
        return
    fig, (p, q) = plt.subplots(1, 2, figsize=(9.2, 3.3), gridspec_kw={"width_ratios": [1.25, 1]})
    p.hist(a["under"], bins=np.linspace(0, 1, 51), color=BLUE, alpha=0.9)
    p.set_yscale("log")
    p.axvline(0.25, color=ORANGE, lw=1.5)
    p.text(0.265, 6000, "25% bound for\nstandardised z-scores", color=INK2, fontsize=7.5, va="top")
    p.set_xlabel("Share of patients with z <= -2, per gene")
    p.set_ylabel("Genes (log scale)")
    tidy(p)
    p.set_title("A  %d genes exceed what a within-cohort z-score allows" % a["n_gt25"], loc="left", fontsize=9)
    parts = [("silent genes\n(median RSEM < 1)", a["elig_silent"], GRAY), ("low (1-10)", a["elig_mid"], LGRAY), ("expressed\n(>= 10)", a["elig_expr"], BLUE)]
    q.barh([0], [a["elig_silent"]], color=GRAY, height=0.5)
    q.barh([0], [a["elig_mid"]], left=[a["elig_silent"]], color=LGRAY, height=0.5)
    q.barh([0], [a["elig_expr"]], left=[a["elig_silent"] + a["elig_mid"]], color=BLUE, height=0.5)
    q.text(a["elig_silent"] / 2, 0, "silent genes\n%d (%.0f%%)" % (a["elig_silent"], 100 * a["elig_silent"] / a["n_elig"]), ha="center", va="center", color="white", fontsize=8)
    q.text(a["elig_silent"] + a["elig_mid"] / 2, 0.32, "low (median RSEM 1-10)\n%d (%.0f%%)" % (a["elig_mid"], 100 * a["elig_mid"] / a["n_elig"]), ha="center", va="bottom", color=INK2, fontsize=7.5)
    q.text(a["n_elig"], -0.32, "expressed (median RSEM >= 10)\n%d (%.1f%%)" % (a["elig_expr"], 100 * a["elig_expr"] / a["n_elig"]), ha="right", va="top", color=BLUE, fontsize=7.5)
    q.set_xlim(0, a["n_elig"])
    q.set_ylim(-0.75, 0.75)
    q.set_yticks([])
    q.set_xlabel("Binary columns that survive the variance filter (n = %d of %d)" % (a["n_elig"], a["ncols"]))
    q.spines["left"].set_visible(False)
    q.set_title("B  What the usable columns are", loc="left", fontsize=9)
    fig.tight_layout()
    save(fig, "fig2_features.png")


def fig_gate1():
    names = [("cols_mt", "Gene search, MT genes (k=10)"), ("cols_all", "Gene search, all genes (k=10)"),
             ("rows_mt", "Patient + gene search, MT genes"), ("rows_all", "Patient + gene search, all genes")]
    data = {k: load(f"gate1_{k}_union.pkl") for k, _ in names}
    if not any(data.values()):
        return
    fig, axes = plt.subplots(2, 4, figsize=(10.8, 5.0))
    for j, (k, title) in enumerate(names):
        res = data[k]
        top, bot = axes[0, j], axes[1, j]
        if res is None:
            top.set_title(title, loc="left", fontsize=8.2)
            continue
        obs = [r for r in res if r["kind"] == "obs" and r.get("ok")]
        nul = [r for r in res if r["kind"] == "null" and r.get("ok")]
        o = np.array([r["best_ci"] for r in obs]); z = np.array([r["best_ci"] for r in nul])
        p = (1 + (z >= o.mean()).sum()) / (1 + len(z))
        top.set_title("%s\nnull %.3f | observed %.3f | p = %.2f" % (title, z.mean(), o.mean(), p), loc="left", fontsize=7.8)
        top.hist(z, bins=np.linspace(min(z.min(), o.min()) - 0.01, 1.0, 26), color=LGRAY, alpha=0.9)
        for v in o:
            top.axvline(v, color=BLUE, lw=1.1, alpha=0.8)
        top.set_xlabel("Best CI in the GA run")
        tidy(top)
        if k.startswith("rows"):
            h = np.array([r["heldout_ci"] for r in obs], float); hz = np.array([r["heldout_ci"] for r in nul], float)
            bot.hist(hz[np.isfinite(hz)], bins=np.linspace(0.3, 0.8, 26), color=LGRAY, alpha=0.9)
            for v in h[np.isfinite(h)]:
                bot.axvline(v, color=BLUE, lw=1.1, alpha=0.8)
            bot.axvline(0.5, color=INK2, lw=1, ls=":")
            bot.set_xlabel("CI of subset model, other patients")
            bot.text(0.97, 0.95, "subset n = %.0f\nheld-out CI %.2f" % (np.mean([r["best_n"] for r in obs]), np.nanmean(h)), transform=bot.transAxes, va="top", ha="right", fontsize=7.5, color=INK2)
        else:
            t = np.array([r["top_ci"] for r in obs], float); tz = np.array([r["top_ci"] for r in nul], float)
            bot.hist(tz[np.isfinite(tz)], bins=np.linspace(0.55, 0.85, 26), color=LGRAY, alpha=0.9)
            for v in t[np.isfinite(t)]:
                bot.axvline(v, color=BLUE, lw=1.1, alpha=0.8)
            bot.set_xlabel("CI of the 10 most frequent genes")
        tidy(bot)
    axes[0, 0].set_ylabel("Permuted data sets")
    axes[1, 0].set_ylabel("Permuted data sets")
    h1, = axes[0, 0].plot([], [], color=BLUE, lw=1.2)
    h3 = plt.Rectangle((0, 0), 1, 1, color=LGRAY)
    fig.legend([h3, h1], ["permutation null (200 data sets)", "observed GA runs (10 seeds)"],
               frameon=False, fontsize=8, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "fig3_gate1_search.png")


def fig_groups():
    res = load("gate1_grp_union.pkl")
    if res is None:
        return
    Ks = sorted({r["K"] for r in res})
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    mo, mn, lo, hi, omin, omax = [], [], [], [], [], []
    for K in Ks:
        o = np.array([r["wci"] for r in res if r["K"] == K and r["kind"] == "obs"]); z = np.array([r["wci"] for r in res if r["K"] == K and r["kind"] == "null"])
        mo.append(o.mean()); mn.append(z.mean()); lo.append(np.quantile(z, .05)); hi.append(np.quantile(z, .95)); omin.append(o.min()); omax.append(o.max())
    ax.fill_between(Ks, lo, hi, color=LGRAY, alpha=0.7, lw=0, label="permutation null, 5-95%")
    ax.plot(Ks, mn, color=GRAY, lw=1.5)
    ax.fill_between(Ks, omin, omax, color=BLUE, alpha=0.25, lw=0)
    ax.plot(Ks, mo, color=BLUE, lw=2, marker="o", ms=4, label="observed (10 GA seeds)")
    ax.set_xlabel("Number of groups K")
    ax.set_ylabel("Weighted in-sample CI")
    ax.set_ylim(0.7, 1.0)
    tidy(ax)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.tight_layout()
    save(fig, "fig4_groups.png")


def fig_geneset():
    r = load("gate2_a.pkl")
    if r is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6), sharey=True)
    order = ["MT_all", "MT_matrisome(A)", "MT_mechano_core(D)", "MT_stem/EMT(C)"]
    nice = {"MT_all": "All MT genes", "MT_matrisome(A)": "Matrisome-type (block A)", "MT_mechano_core(D)": "Mechanotransduction core (D)", "MT_stem/EMT(C)": "Stemness / EMT (C)"}
    for ax, (key, col, title) in zip(axes, [("A_tcga", BLUE, "TCGA-GBM (PanCan Atlas), n = 154"), ("A_cgga", AQUA, "CGGA primary GBM, n = 218")]):
        rows = r.get(key, [])
        ms = sorted({x["model"] for x in rows}, key=lambda m: len(m))
        for i, s in enumerate(order):
            for j, m in enumerate([ms[0], ms[-1]]):
                x = [q for q in rows if q["set"] == s and q["model"] == m]
                if not x:
                    continue
                x = x[0]
                y = i + (-0.17 if j == 0 else 0.17)
                ax.plot([np.exp(x["beta"] - 1.96 * x["se"]), np.exp(x["beta"] + 1.96 * x["se"])], [y, y], color=col, lw=1.6, alpha=0.9)
                ax.plot(x["hr"], y, "o", ms=6, color=col if j == 1 else "white", mec=col, mew=1.6, zorder=3)
        ax.axvline(1, color=INK2, lw=0.9)
        ax.set_xscale("log")
        ax.set_xlim(0.72, 2.2)
        ax.set_xticks([0.75, 1, 1.5, 2])
        ax.set_xticklabels(["0.75", "1", "1.5", "2"])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("Hazard ratio per SD of the set score (95% CI)")
        ax.set_title(title, loc="left", fontsize=9)
        tidy(ax, "x")
        ax.set_ylim(len(order) - 0.4, -0.6)
        ax.plot([], [], "o", mfc="white", mec=col, color=col, label="age + sex" + (" + batch" if key == "A_cgga" else ""))
        ax.plot([], [], "o", color=col, label="fully adjusted (" + ("subtype, MGMT, IDH1" if key == "A_tcga" else "IDH, MGMT, RT, TMZ, batch") + ")")
        ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([nice[s] for s in order])
    fig.tight_layout()
    save(fig, "fig5_geneset.png")


def fig_cv_sets():
    r = load("gate2_b.pkl")
    if r is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2), sharey=True)
    for ax, key, title in zip(axes, ["B_tcga", "B_cgga"], ["TCGA-GBM", "CGGA primary GBM"]):
        x = r[key]
        lo, hi = min(x["mt"].min(), x["ctl"].min()), max(x["mt"].max(), x["ctl"].max())
        bins = np.linspace(lo - 0.005, hi + 0.005, 40)
        ax.hist(x["ctl"], bins=bins, color=LGRAY, alpha=0.95, label="expression-matched non-MT genes")
        ax.hist(x["mt"], bins=bins, color=BLUE, alpha=0.65, label="MT genes")
        ax.axvline(x["baseline"], color=INK2, lw=1.2, ls=":")
        ax.text(x["baseline"], ax.get_ylim()[1] * 0.97, " age + sex only", color=INK2, fontsize=7.5, va="top")
        ax.set_xlabel("Cross-validated C-index of a 100-gene ridge Cox model")
        ax.set_title("%s   (means %.3f vs %.3f; p = %.2g)" % (title, x["mt_mean"], x["ctl_mean"], x["mw_p"]), loc="left", fontsize=8.5)
        tidy(ax)
    axes[0].set_ylabel("Random 100-gene sets")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "fig6_random_sets_cv.png")


def fig_realcv():
    r = load("realcv.pkl")
    if r is None:
        return
    import pandas as pd
    df = pd.DataFrame(r["cv"])
    g = pd.DataFrame(r["ga"])
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 3.6), gridspec_kw={"width_ratios": [1.15, 1]})
    labs = [("base", "age + sex"), ("net_mt", "Coxnet, MT genes"), ("net_expr3000", "Coxnet, 3000 variable genes")]
    ypos, ylab = [], []
    y = 0
    for coh, col in (("tcga", BLUE), ("cgga", AQUA)):
        for k, lab in labs:
            v = df[(df.cohort == coh) & (df.model == k)].c
            a.plot([v.mean() - v.std(), v.mean() + v.std()], [y, y], color=col, lw=1.6)
            a.plot(v.mean(), y, "o", color=col, ms=6)
            ypos.append(y)
            ylab.append(("TCGA  " if coh == "tcga" else "CGGA  ") + lab)
            y += 1
        if coh == "tcga":
            for k, lab in (("ga_best", "nested GA, best solution"), ("ga_top10", "nested GA, consensus model")):
                a.plot([g[k].mean() - g[k].std(), g[k].mean() + g[k].std()], [y, y], color=BLUE, lw=1.6)
                a.plot(g[k].mean(), y, "o", color=BLUE, ms=6, mfc="white", mew=1.6)
                ypos.append(y)
                ylab.append("TCGA  " + lab)
                y += 1
        y += 0.6
    a.axvline(0.5, color=INK2, lw=0.9, ls=":")
    a.set_yticks(ypos)
    a.set_yticklabels(ylab, fontsize=8)
    a.invert_yaxis()
    a.set_xlabel("Held-out C-index (mean +/- SD over 15 folds)")
    tidy(a, "x")
    a.set_title("A  Within-cohort cross-validation", loc="left", fontsize=9)
    ypos, ylab, y = [], [], 0
    for c in r["cross"]:
        b.plot(c["c_net"], y, "o", color=BLUE if c["direction"].startswith("TCGA") else AQUA, ms=6.5)
        b.plot(c["c_agesex"], y, "|", color=INK2, ms=13, mew=2)
        ypos.append(y)
        ylab.append("%s, %s" % (c["direction"], "MT" if c["features"].startswith("MT") else "3000 var."))
        y += 1
    b.axvline(0.5, color=INK2, lw=0.9, ls=":")
    b.set_yticks(ypos)
    b.set_yticklabels(ylab, fontsize=8)
    b.invert_yaxis()
    b.set_xlabel("C-index in the other cohort")
    b.plot([], [], "o", color=BLUE, label="Coxnet (train cohort -> test cohort)")
    b.plot([], [], "|", color=INK2, ms=10, mew=2, label="age + sex only")
    b.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    tidy(b, "x")
    b.set_title("B  Cross-cohort transfer", loc="left", fontsize=9)
    fig.tight_layout()
    save(fig, "fig7_prediction.png")

def fig_sim():
    res = load("gate3_sim.pkl")
    if res is None:
        return
    sc = ["S0", "S1", "S2"]
    lab = {"S0": "S0 no signal", "S1": "S1 five drivers", "S2": "S2 three subgroups"}
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.4, 3.5), gridspec_kw={"width_ratios": [1.3, 1]})
    for i, s in enumerate(sc):
        rs = [r for r in res if r["scen"] == s]
        m = lambda k: np.nanmean([r[k] for r in rs if k in r])
        for off, (tr, te, col) in zip((-0.14, 0.14), (("ga_train", "ga_test", BLUE), ("net_train", "net_test", AQUA))):
            a.plot([i + off, i + off], [m(te), m(tr)], color=col, lw=2.2, alpha=0.85)
            a.plot(i + off, m(tr), "o", color=col, ms=6.5)
            a.plot(i + off, m(te), "o", color="white", mec=col, mew=1.8, ms=6.5)
        a.plot(i, m("oracle_test"), "_", color=INK, ms=26, mew=2)
    a.axhline(0.5, color=INK2, lw=0.8, ls=":")
    a.set_xticks(range(3))
    a.set_xticklabels([lab[s] for s in sc])
    a.set_ylabel("C-index")
    tidy(a)
    a.plot([], [], "o", color=BLUE, label="GA, in-sample (filled) -> new outcomes (open)")
    a.plot([], [], "o", color=AQUA, label="Coxnet, in-sample -> new outcomes")
    a.plot([], [], "_", color=INK, ms=14, mew=2, label="true model (upper bound)")
    a.legend(frameon=False, fontsize=7.2, loc="upper left", bbox_to_anchor=(0.0, 1.02))
    a.set_ylim(0.45, 1.05)
    a.set_title("A  Optimism of the selection step", loc="left", fontsize=9)
    for i, s in enumerate(sc):
        v = np.array([r["grp_wci"] for r in res if r["scen"] == s])
        b.plot(np.full(len(v), i) + np.random.default_rng(1).uniform(-0.12, 0.12, len(v)), v, "o", color=BLUE, ms=3.5, alpha=0.7)
        b.plot([i - 0.25, i + 0.25], [v.mean(), v.mean()], color=INK, lw=1.8)
    ari = np.array([r["grp_ari"] for r in res if "grp_ari" in r])
    b.set_xticks(range(3))
    b.set_xticklabels([lab[s] for s in sc], fontsize=8)
    b.set_ylabel("Weighted CI of the 3-group search")
    b.set_ylim(0.7, 1.0)
    tidy(b)
    b.text(0.97, 0.05, "S2: agreement of the groups with the\ntrue subgroups (ARI) = %.2f (mean)" % ari.mean(), transform=b.transAxes, ha="right", va="bottom", fontsize=7.5, color=INK2)
    b.set_title("B  Group search finds high CI everywhere", loc="left", fontsize=9)
    fig.tight_layout()
    save(fig, "fig8_simulation.png")


def fig_coassoc():
    r = load("coassoc.pkl")
    if r is None:
        return
    ks = [2, 3, 4, 5, 6]
    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    for kind, col, nm in (("null", GRAY, "permutation null (100)"), ("obs", BLUE, "observed (10)")):
        m = np.array([[x["sil%d" % k] for k in ks] for x in r[kind]])
        ax.plot(ks, m.mean(0), color=col, lw=2, marker="o", ms=4, label=nm)
        ax.fill_between(ks, np.quantile(m, .05, 0), np.quantile(m, .95, 0), color=col, alpha=0.2, lw=0)
    ax.set_xticks(ks)
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Silhouette of the co-association matrix")
    tidy(ax)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, "fig9_coassociation.png")


if __name__ == "__main__":
    for f in (fig_repro, fig_audit, fig_gate1, fig_groups, fig_geneset, fig_cv_sets, fig_realcv, fig_sim, fig_coassoc):
        try:
            f()
        except Exception as ex:
            print("FAILED", f.__name__, repr(ex))