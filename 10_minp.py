"""Single-column significance with multiplicity: univariate Cox (age, sex, one eligible binary column), observed vs permutation null."""
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import coxfast
import dataset
from common import RESULTS

warnings.filterwarnings("ignore")


def univ(B, age, sex, t, e, elig):
    z = np.full(len(elig), np.nan)
    hr = np.full(len(elig), np.nan)
    for i, c in enumerate(elig):
        X = np.column_stack([age, sex, B[:, c]]).astype(np.float64)
        ok, ci, keep, b, se, p = coxfast.fit(X, t, e, 1e-6, 0.0, 1.01, False, False)
        if ok:
            z[i], hr[i] = b[2] / se[2], np.exp(b[2])
    return z, hr


def bh(p):
    n = len(p)
    o = np.argsort(p)
    q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / (np.arange(n) + 1))[::-1])[::-1]
    return np.minimum(q, 1)


def job(arr, kind, i):
    B = arr["B"] if kind == "obs" else arr["B"][np.random.default_rng(777000 + i).permutation(len(arr["age"]))]
    z, hr = univ(np.ascontiguousarray(B), arr["age"], arr["sex"], arr["t"], arr["e"], arr["elig"])
    p = chi2.sf(z ** 2, 1)
    return dict(kind=kind, minp=float(np.nanmin(p)), n_q05=int(np.sum(bh(np.nan_to_num(p, nan=1.0)) < 0.05)), n_p05=int(np.sum(p < 0.05)), n_p001=int(np.sum(p < 0.001)), p=p if kind == "obs" else None, hr=hr if kind == "obs" else None)


if __name__ == "__main__":
    d = dataset.load("union")
    prev = d.B.mean(0)
    elig = np.where((prev >= 0.1127) & (prev <= 0.8873))[0]
    arr = dict(B=d.B, age=d.age, sex=d.sex, t=d.time, e=d.event, elig=elig)
    res = Parallel(n_jobs=4)([delayed(job)(arr, "obs", 0)] + [delayed(job)(arr, "null", i) for i in range(200)])
    pickle.dump(dict(res=res, elig=elig, colnames=d.colnames[elig]), open(RESULTS / "minp.pkl", "wb"))
    o, nul = res[0], [r for r in res if r["kind"] == "null"]
    print(f"eligible columns: {len(elig)}")
    print(f"observed: min p = {o['minp']:.2e}; columns with p<0.05: {o['n_p05']}; p<0.001: {o['n_p001']}; BH q<0.05: {o['n_q05']}")
    for k in ("minp", "n_p05", "n_p001", "n_q05"):
        v = np.array([r[k] for r in nul], float)
        print(f"null {k}: mean {v.mean():.4g}, q05 {np.quantile(v, .05):.4g}, q95 {np.quantile(v, .95):.4g}")
    print("empirical P(null min p <= observed min p) = %.3f" % ((1 + np.sum([r["minp"] <= o["minp"] for r in nul])) / 201))
    top = np.argsort(o["p"])[:12]
    for i in top:
        print(f"   {d.colnames[elig[i]]:22s} HR {o['hr'][i]:.2f}  p {o['p'][i]:.2e}  prevalence {prev[elig[i]]:.2f}")
