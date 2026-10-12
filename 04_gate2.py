"""Gate 2: is the mechanotransduction (MT) gene set special? Gene-set level, expression-matched, adjusted, two cohorts.
Part A  competitive test: MT score vs expression-matched random gene sets (z-statistic of the score in Cox models)
Part B  predictive test: 100-gene ridge-Cox, cross-validated C-index,
        MT vs expression-matched non-MT vs age+sex only."""
import pickle
import sys
import types
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import cgga
import common
import dataset
import gset
from common import RESULTS

warnings.filterwarnings("ignore")
EXPR_THR = {"tcga": 10.0, "cgga": 1.0}     # median RSEM (TCGA normalised counts) / RSEM (CGGA) to call a gene expressed


def prep(name):
    if name == "tcga":
        d = dataset.load("pancan")
        blocks = d.gene_block
    else:
        d = cgga.load("primary_gbm")
        bl = common.mt_blocks()
        blocks = np.array([bl.get(g) or "" for g in d.genes])
    R = np.nan_to_num(d.rsem, nan=0.0)
    d.Zr = gset.zlog(R)
    d.med = np.median(R, axis=0)
    d.expressed = d.med >= EXPR_THR[name]
    d.block = blocks
    d.medlog = np.median(np.log2(R + 1.0), axis=0)
    return d


def models(d, name):
    """List of (label, row mask, covariate matrix). Rows are already time sorted."""
    n = d.n
    allr = np.ones(n, dtype=bool)
    base = np.column_stack([d.age, d.sex])
    out = [("age+sex", allr, base)]
    if name == "tcga":
        sub = d.extra["subtype"].values
        ok = pd.notna(sub)
        dummies = np.column_stack([(sub == s).astype(float) for s in ["Mesenchymal", "Neural", "Proneural", "G-CIMP"]])
        out.append(("age+sex+subtype", ok, np.column_stack([base, dummies])))
        ok2 = ok & d.extra["mgmt_meth"].notna().values & d.extra["idh1_mut"].notna().values
        out.append(("age+sex+subtype+MGMT+IDH1", ok2, np.column_stack([base, dummies, d.extra["mgmt_meth"].fillna(0).values, d.extra["idh1_mut"].fillna(0).values])))
    else:
        ex = d.extra
        batch = (ex["batch"].values == "325").astype(float)
        out.append(("age+sex+batch", allr, np.column_stack([base, batch])))
        ok = ex["idh_mut"].notna().values & ex["mgmt_meth"].notna().values & ex["radio"].notna().values & ex["chemo"].notna().values
        out.append(("age+sex+batch+IDH+MGMT+RT+TMZ", ok, np.column_stack([base, batch, ex["idh_mut"].fillna(0).values, ex["mgmt_meth"].fillna(0).values,
                                                                         ex["radio"].fillna(0).values, ex["chemo"].fillna(0).values])))
    return out


def stat(d, idx, mrow, C):
    sc = d.Zr[:, idx].mean(axis=1)
    b, se, z, p, ci = gset.cox_score(sc[mrow], C[mrow], d.time[mrow], d.event[mrow])
    return dict(beta=b, se=se, z=z, p=p, hr=float(np.exp(b)))


def part_a(name, d, n_sets=5000, seed=1):
    rng = np.random.default_rng(seed)
    strata = gset.strata_by_expression(d.medlog[d.expressed])
    st_full = np.full(len(d.genes), -1)
    st_full[np.where(d.expressed)[0]] = strata
    expr_idx = np.where(d.expressed)[0]
    mt = {"MT_all": d.expressed & (d.block != ""), "MT_matrisome(A)": d.expressed & (d.block == "A"),
          "MT_stem/EMT(C)": d.expressed & (d.block == "C"), "MT_mechano_core(D)": d.expressed & (d.block == "D")}
    pool = np.where(d.expressed & (d.block == ""))[0]
    res = []
    for sname, mask in mt.items():
        idx = np.where(mask)[0]
        if len(idx) < 5:
            continue
        sets = gset.matched_sets(rng, idx, pool, st_full, n_sets)
        for label, mrow, C in models(d, name):
            obs = stat(d, idx, mrow, C)
            rz = np.array([stat(d, s, mrow, C)["z"] for s in sets])
            rz = rz[np.isfinite(rz)]
            res.append(dict(cohort=name, set=sname, n_genes=len(idx), model=label, n=int(mrow.sum()), **obs,
                            p_comp=(1 + (np.abs(rz) >= abs(obs["z"])).sum()) / (1 + len(rz)),
                            rand_z_sd=float(rz.std()), rand_z_q=np.quantile(rz, [.025, .5, .975])))
            print(f"[{name}] {sname:20s} m={len(idx):4d} {label:32s} n={int(mrow.sum()):3d} HR/SD={obs['hr']:.3f} z={obs['z']:+.2f} p={obs['p']:.3f} | competitive p={res[-1]['p_comp']:.3f} (random z sd {rz.std():.2f})", flush=True)
    return res


def one_set_cv(d, idx, C, reps, folds, lam, seed):
    rng = np.random.default_rng(seed)
    Z = d.Zr[:, idx]
    X = np.column_stack([d.age, d.sex, (Z - Z.mean(0)) / np.where(Z.std(0) > 0, Z.std(0), 1)])
    return float(np.nanmean([gset.cv_cindex(X, d.time, d.event, lam, folds, rng) for _ in range(reps)]))


def part_b(name, d, m=100, n_sets=600, reps=2, folds=5, lam=0.5, seed=2, jobs=18):
    rng = np.random.default_rng(seed)
    strata = gset.strata_by_expression(d.medlog[d.expressed])
    st_full = np.full(len(d.genes), -1)
    st_full[np.where(d.expressed)[0]] = strata
    mt_idx = np.where(d.expressed & (d.block != ""))[0]
    pool = np.where(d.expressed & (d.block == ""))[0]
    mt_sets = [rng.choice(mt_idx, m, replace=False) for _ in range(n_sets)]
    ctl_sets = gset.matched_sets(rng, mt_idx, pool, st_full, n_sets, size=m)
    base_ci = float(np.nanmean([gset.cv_cindex(np.column_stack([d.age, d.sex]), d.time, d.event, lam, folds, np.random.default_rng(s)) for s in range(200)]))
    ctx = types.SimpleNamespace(Zr=d.Zr, age=d.age, sex=d.sex, time=d.time, event=d.event)
    run = lambda sets, off: Parallel(n_jobs=jobs)(delayed(one_set_cv)(ctx, s, None, reps, folds, lam, off + i) for i, s in enumerate(sets))
    a = np.array(run(mt_sets, 10_000)); b = np.array(run(ctl_sets, 20_000))
    from scipy.stats import mannwhitneyu
    u = mannwhitneyu(a, b, alternative="two-sided")
    print(f"[{name}] CV C-index, ridge-Cox on {m} random genes: MT {a.mean():.4f} (sd {a.std():.4f}) | matched non-MT {b.mean():.4f} (sd {b.std():.4f}) | age+sex only {base_ci:.4f} | Mann-Whitney p={u.pvalue:.3g}", flush=True)
    return dict(cohort=name, m=m, n_sets=n_sets, mt=a, ctl=b, baseline=base_ci, mw_p=float(u.pvalue), mt_mean=float(a.mean()), ctl_mean=float(b.mean()))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "ab"
    out = {}
    for name in ["tcga", "cgga"]:
        d = prep(name)
        print(f"== {name}: n={d.n}, events={int(d.event.sum())}, genes={len(d.genes)}, expressed={int(d.expressed.sum())}, MT expressed={int((d.expressed & (d.block != '')).sum())}", flush=True)
        if "a" in which:
            out[f"A_{name}"] = part_a(name, d)
        if "b" in which:
            out[f"B_{name}"] = part_b(name, d, jobs=int(sys.argv[2]) if len(sys.argv) > 2 else 18)
    pickle.dump(out, open(RESULTS / f"gate2_{which}.pkl", "wb"))
