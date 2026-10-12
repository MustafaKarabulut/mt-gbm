"""Step 2: validate the numba Cox/C-index engine against lifelines (lifelines)."""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.feature_selection import VarianceThreshold

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import coxfast
import dataset

warnings.filterwarnings("ignore")
d = dataset.load()
print(dataset.summary(d))
print(f"events {d.event.sum()}/{d.n} ({d.event.mean():.0%}),  median OS {np.median(d.time):.1f} mo,  "
      f"male {d.sex.mean():.0%},  mean age {d.age.mean():.1f}")
print("subtype counts:", d.extra["subtype"].value_counts(dropna=False).to_dict())

prev = d.B.mean(axis=0)
pool = np.where((prev >= 0.12) & (prev <= 0.9))[0]
print("binary columns:", d.B.shape[1], " with prevalence in [12%, 90%]:", len(pool))


def lifelines_fit(rows, cols):
    X = pd.DataFrame(np.column_stack([d.age[rows], d.sex[rows]] + [d.B[rows, c] for c in cols]).astype(float),
                     columns=["AGE", "SEX"] + [f"c{c}" for c in cols])
    X = X[X.columns[VarianceThreshold(0.1).fit(X).get_support(indices=True)]]
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    X = X.drop([c for c in upper.columns if any(upper[c] > 0.90)], axis=1)
    X["PatientId"] = d.pid[rows]
    X["OS_MONTHS"] = d.time[rows]
    X["OS_STATUS"] = d.event[rows]
    cph = CoxPHFitter(penalizer=0.5)
    cph.fit(X, duration_col="OS_MONTHS", event_col="OS_STATUS", cluster_col="PatientId")
    return cph


rng = np.random.default_rng(7)
rows_all = np.arange(d.n)
dc, dh, dp, nbad, kept_mismatch = [], [], [], 0, 0
t0 = time.time()
for trial in range(60):
    if trial < 20:
        rows = rows_all
    else:
        rows = np.sort(rng.choice(d.n, size=int(rng.integers(35, 120)), replace=False))
    cols = rng.choice(pool, size=10, replace=False)
    X = np.column_stack([d.age[rows], d.sex[rows]] + [d.B[rows, c] for c in cols]).astype(np.float64)
    ok, ci, keep, beta, se, p = coxfast.fit(X, d.time[rows], d.event[rows], 0.5, 0.1, 0.9, True, True)
    cph = lifelines_fit(rows, cols)
    names = ["AGE", "SEX"] + [f"c{c}" for c in cols]
    ll_keep = [n in cph.summary.index for n in names]
    if list(keep) != ll_keep:
        kept_mismatch += 1
        continue
    dc.append(abs(ci - cph.concordance_index_))
    s = cph.summary
    for j, n in enumerate(names):
        if keep[j]:
            dh.append(abs(np.exp(beta[j]) - s.loc[n, "exp(coef)"]) / s.loc[n, "exp(coef)"])
            dp.append(abs(p[j] - s.loc[n, "p"]))
print(f"\n60 random model fits ({time.time()-t0:.1f}s incl. lifelines); kept-column mismatches: {kept_mismatch}")
print(f"C-index   max |diff| = {np.max(dc):.2e}   mean = {np.mean(dc):.2e}")
print(f"HR        max rel diff = {np.max(dh):.2e}   mean = {np.mean(dh):.2e}")
print(f"p-value   max |diff| = {np.max(dp):.3f}   median = {np.median(dp):.4f}  (both robust/cluster-sandwich)")

# speed
X = np.column_stack([d.age, d.sex] + [d.B[:, c] for c in rng.choice(pool, 10, replace=False)]).astype(np.float64)
coxfast.fit(X, d.time, d.event, 0.5, 0.1, 0.9, True, True)
t0 = time.time()
for _ in range(2000):
    coxfast.fit(X, d.time, d.event, 0.5, 0.1, 0.9, True, True)
print(f"speed: {(time.time()-t0)/2000*1e3:.3f} ms per fit (n={d.n}, 12 covariates)")
