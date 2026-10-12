"""Gate 1: do GA-selected gene / patient subsets reach their CI by chance?
Observed runs (several GA seeds) vs a permutation null in which the gene matrix rows are permuted jointly
(survival, age and sex stay together, genes are made independent of the outcome). Same GA settings in both."""
import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import coxcore
import dataset
import ga
from common import RESULTS


def heldout_ci(D, cols, rows):
    """C-index of the model fitted on `rows`, evaluated on the patients NOT in `rows`."""
    n = D["B"].shape[0]
    (ok, ci, keep, beta, se, pv), ucols, X, tt, ee = ga.single_eval(D, cols, rows)
    if not ok:
        return np.nan
    out = np.setdiff1d(np.arange(n), rows)
    Xo, to, eo = ga._build(D["B"], D["age"], D["sex"], D["t"], D["e"], out, len(out), ucols, len(ucols))
    risk = Xo @ np.where(np.isnan(beta), 0.0, beta)
    return float(coxcore.cindex(to, eo, risk))


def task(exp, kind, idx, arr, K=None):
    n = len(arr["age"])
    B = arr["B"]
    if kind == "null":
        B = B[np.random.default_rng(100000 + idx).permutation(n)]
    B = np.ascontiguousarray(B)
    D = dict(B=B, age=arr["age"], sex=arr["sex"], t=arr["t"], e=arr["e"], colprev=B.mean(0))
    allowed = arr["mt"] if exp.endswith("_mt") else arr["all"]
    seed = idx if kind == "obs" else 7000 + idx
    base = dict(exp=exp, kind=kind, idx=idx, seed=seed)
    if exp.startswith("grp"):
        r = ga.run_groups(D, allowed, seed, K=K, pop=300, gens=200)
        return dict(base, K=K, wci=r["wci"], labels=r["labels"], cols=r["cols"])
    use_rows = exp.startswith("rows")
    r = ga.run_cols(D, allowed, seed, pop=300, gens=300 if use_rows else 200, use_rows=use_rows)
    if not r["ok"]:
        return dict(base, ok=False)
    rec = dict(base, ok=True, best_ci=r["best_ci"], top_ci=r["top_ci"], n_pareto=r["n_pareto"], best_cols=r["best_cols"])
    if use_rows:
        rec.update(best_n=r["best_n"], heldout_ci=heldout_ci(D, r["best_cols"], r["best_rows"]))
        if kind == "obs":
            rec["best_rows"] = r["best_rows"]
    return rec


def summarize(exp, res):
    obs = [r for r in res if r["kind"] == "obs"]
    nul = [r for r in res if r["kind"] == "null"]
    key = "wci" if exp.startswith("grp") else "best_ci"
    o = np.array([r[key] for r in obs if r.get("ok", True)])
    z = np.array([r[key] for r in nul if r.get("ok", True)])
    line = f"{exp:10s} observed {key}: mean {o.mean():.3f} (min {o.min():.3f}, max {o.max():.3f}, n={len(o)}) | null: mean {z.mean():.3f}, q05 {np.quantile(z, .05):.3f}, q95 {np.quantile(z, .95):.3f}, max {z.max():.3f} (n={len(z)})"
    line += f" | empirical p (obs mean) = {(1 + (z >= o.mean()).sum()) / (1 + len(z)):.3f}"
    if exp.startswith("rows"):
        line += f" | subset size obs {np.mean([r['best_n'] for r in obs]):.0f} null {np.mean([r['best_n'] for r in nul]):.0f}"
        line += f" | held-out CI obs {np.nanmean([r['heldout_ci'] for r in obs]):.3f}"
    return line


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", default="cols_mt,cols_all,rows_mt,rows_all")
    ap.add_argument("--n_obs", type=int, default=10)
    ap.add_argument("--n_null", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=18)
    ap.add_argument("--ks", default="2,3,4,5,6,7,8,9")
    ap.add_argument("--cohort", default="union")
    a = ap.parse_args()
    d = dataset.load(a.cohort)
    print(dataset.summary(d), flush=True)
    arr = dict(B=d.B, age=d.age, sex=d.sex, t=d.time, e=d.event, mt=np.where(d.col_is_mt)[0], all=np.arange(d.B.shape[1]))
    ga.run_cols(dict(B=d.B, age=d.age, sex=d.sex, t=d.time, e=d.event, colprev=d.B.mean(0)), arr["mt"], 0, pop=20, gens=2)
    for exp in a.exps.split(","):
        t0 = time.time()
        if exp == "grp":
            jobs = [(f"grp_all", k, kind, i) for k in map(int, a.ks.split(",")) for kind, n in (("obs", a.n_obs), ("null", a.n_null)) for i in range(n)]
            res = Parallel(n_jobs=a.jobs, verbose=0)(delayed(task)(e, kind, i, arr, K=k) for e, k, kind, i in jobs)
            pickle.dump(res, open(RESULTS / f"gate1_{exp}_{a.cohort}.pkl", "wb"))
            for k in map(int, a.ks.split(",")):
                print(f"[{time.time()-t0:5.0f}s] K={k}", summarize("grp_all", [r for r in res if r["K"] == k]), flush=True)
            continue
        jobs = [(kind, i) for kind, n in (("obs", a.n_obs), ("null", a.n_null)) for i in range(n)]
        res = Parallel(n_jobs=a.jobs, verbose=0)(delayed(task)(exp, kind, i, arr) for kind, i in jobs)
        pickle.dump(res, open(RESULTS / f"gate1_{exp}_{a.cohort}.pkl", "wb"))
        print(f"[{time.time()-t0:5.0f}s]", summarize(exp, res), flush=True)
