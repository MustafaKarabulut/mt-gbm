"""Out-of-sample performance on real data: repeated 5-fold CV inside each cohort and cross-cohort transfer (TCGA <-> CGGA).
Models: age+sex Cox | Coxnet on MT (expressed) genes | Coxnet on top-3000 variable expressed genes | Coxnet on binary
eligible columns | nested GA (cols only, k=10; GA runs on the training folds only)."""
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import baselines
import cgga
import coxfast
import dataset
import ga
import gset
from common import RESULTS, mt_blocks

warnings.filterwarnings("ignore")


def base_risk(tr, te, age, sex, t, e):
    X = np.column_stack([age, sex]).astype(float)
    ok, ci, keep, b, se, p = coxfast.fit(X[tr], t[tr], e[tr], 1e-6, 0.0, 1.01, False, False)
    return X[te] @ np.where(np.isnan(b), 0.0, b)


def net_risk(tr, te, F, age, sex, t, e, seed):
    X = np.column_stack([age, sex, F])
    m = baselines.Coxnet(l1_ratio=0.5, seed=seed).fit(X[tr], t[tr], e[tr])
    return m.predict(X[te])


def cohort_tcga():
    d = dataset.load("pancan")
    Z = gset.zlog(np.nan_to_num(d.rsem, nan=0.0))
    med = np.median(np.nan_to_num(d.rsem, nan=0.0), axis=0)
    return d, Z, med >= 10.0, d.gene_block != ""


def cohort_cgga():
    d = cgga.load("primary_gbm")
    bl = mt_blocks()
    R = np.nan_to_num(d.rsem, nan=0.0)
    return d, gset.zlog(R), np.median(R, axis=0) >= 1.0, np.array([bool(bl.get(g)) for g in d.genes])


def feature_sets(Z, expressed, is_mt):
    mt = np.where(expressed & is_mt)[0]
    ex = np.where(expressed)[0]
    top = ex[np.argsort(-Z[:, ex].var(0), kind="stable")[:3000]]
    return {"net_mt": Z[:, mt], "net_expr3000": Z[:, top]}


def fold_task(name, model, rep, f, d, F, folds=5):
    n = d.n
    idx = np.random.default_rng(1000 + rep).permutation(n)
    te = np.sort(idx[f::folds])
    tr = np.setdiff1d(np.arange(n), te)
    if model == "base":
        r = base_risk(tr, te, d.age, d.sex, d.time, d.event)
    else:
        r = net_risk(tr, te, F[model], d.age, d.sex, d.time, d.event, rep)
    return dict(cohort=name, model=model, rep=rep, fold=f, c=baselines.cind(d.time[te], d.event[te], r))


def ga_fold(rep, f, arr, folds=5):
    n = len(arr["age"])
    idx = np.random.default_rng(1000 + rep).permutation(n)
    te = np.sort(idx[f::folds])
    tr = np.setdiff1d(np.arange(n), te)
    Btr = np.ascontiguousarray(arr["B"][tr])
    D = dict(B=Btr, age=arr["age"][tr], sex=arr["sex"][tr], t=arr["t"][tr], e=arr["e"][tr], colprev=Btr.mean(0))
    r = ga.run_cols(D, np.arange(Btr.shape[1]), 100 * rep + f, pop=300, gens=200)
    out = dict(cohort="tcga_binary", rep=rep, fold=f)
    for tag, cols in (("ga_best", r["best_cols"]), ("ga_top10", r["top_cols"])):
        (ok, ci, keep, beta, se, pv), uc, X, tt, ee = ga.single_eval(D, cols)
        Xte = np.column_stack([arr["age"][te], arr["sex"][te], arr["B"][te][:, uc]]).astype(float)
        out[tag] = baselines.cind(arr["t"][te], arr["e"][te], Xte @ np.where(np.isnan(beta), 0.0, beta))
        out[tag + "_train"] = float(ci)
    return out


def cross(dA, FA, dB, FB, genesA, genesB, name):
    """Train Coxnet on cohort A with the genes common to both cohorts, test on B."""
    common = sorted(set(genesA) & set(genesB))
    ia = {g: i for i, g in enumerate(genesA)}
    ib = {g: i for i, g in enumerate(genesB)}
    return common, ia, ib


if __name__ == "__main__":
    jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 18
    reps = 3
    t0 = time.time()
    res = []
    coh = {"tcga": cohort_tcga(), "cgga": cohort_cgga()}
    for name, (d, Z, expressed, is_mt) in coh.items():
        F = feature_sets(Z, expressed, is_mt)
        tasks = [(name, m, r, f, d, F) for m in ["base", "net_mt", "net_expr3000"] for r in range(reps) for f in range(5)]
        res += Parallel(n_jobs=jobs)(delayed(fold_task)(*t) for t in tasks)
        print(f"[{time.time()-t0:.0f}s] {name} within-cohort CV done", flush=True)
    du = dataset.load("union")
    arr = dict(B=du.B, age=du.age, sex=du.sex, t=du.time, e=du.event)
    ga.run_cols(dict(B=du.B[:, :100], age=du.age, sex=du.sex, t=du.time, e=du.event, colprev=du.B[:, :100].mean(0)), np.arange(100), 0, pop=20, gens=2)
    gres = Parallel(n_jobs=jobs)(delayed(ga_fold)(r, f, arr) for r in range(reps) for f in range(5))
    print(f"[{time.time()-t0:.0f}s] nested GA done", flush=True)
    # cross-cohort transfer with genes common to both cohorts (continuous within-cohort z)
    (dt, Zt, et, mt_t), (dc, Zc, ec, mt_c) = coh["tcga"], coh["cgga"]
    gt = {g: i for i, g in enumerate(dt.genes)}
    gc = {g: i for i, g in enumerate(dc.genes)}
    common = np.array(sorted(set(gt) & set(gc)))
    it, ic = np.array([gt[g] for g in common]), np.array([gc[g] for g in common])
    ex_both = et[it] & ec[ic]
    mt_both = mt_t[it] & mt_c[ic]
    cross_res = []
    for tag, mask in (("MT genes", ex_both & mt_both), ("top-3000 variable genes", ex_both)):
        sel = np.where(mask)[0]
        if tag.startswith("top"):
            v = Zt[:, it[sel]].var(0) + Zc[:, ic[sel]].var(0)
            sel = sel[np.argsort(-v, kind="stable")[:3000]]
        for a, b, Za, Zb, ja, jb, nm in ((dt, dc, Zt, Zc, it, ic, "TCGA->CGGA"), (dc, dt, Zc, Zt, ic, it, "CGGA->TCGA")):
            XA = np.column_stack([a.age, a.sex, Za[:, ja[sel]]]); XB = np.column_stack([b.age, b.sex, Zb[:, jb[sel]]])
            m = baselines.Coxnet(l1_ratio=0.5).fit(XA, a.time, a.event)
            base = coxfast.fit(np.column_stack([a.age, a.sex]).astype(float), a.time, a.event, 1e-6, 0.0, 1.01, False, False)[3]
            cross_res.append(dict(direction=nm, features=tag, n_features=len(sel), c_net=baselines.cind(b.time, b.event, m.predict(XB)),
                                  c_agesex=baselines.cind(b.time, b.event, np.column_stack([b.age, b.sex]) @ base), n_selected=len(m.selected())))
            print(cross_res[-1], flush=True)
    pickle.dump(dict(cv=res, ga=gres, cross=cross_res), open(RESULTS / "realcv.pkl", "wb"))
    import pandas as pd
    df = pd.DataFrame(res)
    print(df.groupby(["cohort", "model"]).c.agg(["mean", "std", "count"]).round(3))
    g = pd.DataFrame(gres)
    print("nested GA (TCGA binary): test C ga_best %.3f, ga_top10 %.3f | train C ga_best %.3f" % (g.ga_best.mean(), g.ga_top10.mean(), g.ga_best_train.mean()))
