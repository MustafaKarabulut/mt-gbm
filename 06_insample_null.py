"""In-sample reference analyses (all-column multivariate Cox, univariate screening + top-X multivariate) on the
cohort and on permutation-null data (gene rows permuted jointly). Same filters and ridge penalty throughout."""
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import coxfast
import dataset
from common import RESULTS

warnings.filterwarnings("ignore")
XS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 350, 400, 450, 500]


def fit_cols(B, age, sex, t, e, cols):
    X = np.column_stack([age, sex, B[:, cols]]).astype(np.float64)
    ok, ci, keep, beta, se, p = coxfast.fit(X, t, e, 0.5, 0.1, 0.9, True, False)
    return (ci if ok else np.nan), keep[2:].sum(), beta, keep


def screen(B, age, sex, t, e, elig):
    """Univariate Cox (age, sex, one column) for every eligible column: returns CI and HR arrays."""
    ci = np.full(len(elig), np.nan)
    hr = np.full(len(elig), np.nan)
    for i, c in enumerate(elig):
        X = np.column_stack([age, sex, B[:, c]]).astype(np.float64)
        ok, c_i, keep, beta, se, p = coxfast.fit(X, t, e, 0.5, 0.1, 0.9, True, False)
        if ok and keep[2]:
            ci[i], hr[i] = c_i, np.exp(beta[2])
    return ci, hr


def analyse(B, age, sex, t, e, elig_all, elig_mt, with_curve):
    out = {}
    out["multi_mt_ci"], out["multi_mt_ncols"], _, _ = fit_cols(B, age, sex, t, e, elig_mt)
    out["multi_all_ci"], out["multi_all_ncols"], _, _ = fit_cols(B, age, sex, t, e, elig_all)
    ci_all, hr_all = screen(B, age, sex, t, e, elig_all)
    mt_pos = np.isin(elig_all, elig_mt)
    for name, sel in [("all", np.ones(len(elig_all), bool)), ("mt", mt_pos)]:
        idx = np.where(sel & np.isfinite(ci_all))[0]
        o = idx[np.argsort(-ci_all[idx], kind="stable")]
        out[f"top10_ci_{name}"] = fit_cols(B, age, sex, t, e, elig_all[o[:10]])[0]
        oh = idx[np.argsort(-np.abs(np.log(hr_all[idx])), kind="stable")]
        out[f"top10_hr_{name}"] = fit_cols(B, age, sex, t, e, elig_all[oh[:10]])[0]
        if with_curve and name == "all":
            out["curve"] = [fit_cols(B, age, sex, t, e, elig_all[o[:x]])[0] for x in XS]
    return out


def job(arr, kind, i, with_curve):
    B = arr["B"]
    if kind == "null":
        B = B[np.random.default_rng(555000 + i).permutation(len(arr["age"]))]
    B = np.ascontiguousarray(B)
    prev = B.mean(0)
    elig = np.where((prev >= 0.1127) & (prev <= 0.8873))[0]
    return dict(kind=kind, i=i, **analyse(B, arr["age"], arr["sex"], arr["t"], arr["e"], elig, elig[arr["is_mt"][elig]], with_curve))


if __name__ == "__main__":
    n_null = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 18
    d = dataset.load("union")
    arr = dict(B=d.B, age=d.age, sex=d.sex, t=d.time, e=d.event, is_mt=d.col_is_mt)
    t0 = time.time()
    res = Parallel(n_jobs=jobs)([delayed(job)(arr, "obs", 0, True)] + [delayed(job)(arr, "null", i, i < 20) for i in range(n_null)])
    pickle.dump(res, open(RESULTS / "insample_null.pkl", "wb"))
    obs, nul = res[0], [r for r in res if r["kind"] == "null"]
    print(f"done {time.time()-t0:.0f}s; eligible MT columns used in multivariate: {obs['multi_mt_ncols']}, all: {obs['multi_all_ncols']}")
    for k in ["multi_mt_ci", "multi_all_ci", "top10_ci_mt", "top10_hr_mt", "top10_ci_all", "top10_hr_all"]:
        z = np.array([r[k] for r in nul], float)
        print(f"{k:14s} observed {obs[k]:.3f} | null mean {np.nanmean(z):.3f} q05 {np.nanquantile(z,.05):.3f} q95 {np.nanquantile(z,.95):.3f} max {np.nanmax(z):.3f}")
    cur = np.array([r["curve"] for r in nul if "curve" in r], float)
    print("top-X (X, observed, null mean):", [(x, round(o, 3), round(m, 3)) for x, o, m in zip(XS, obs["curve"], np.nanmean(cur, 0))])
