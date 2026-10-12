"""In-sample CI of 100 random columns, MT vs non-MT (1000 sets each), plus the number of columns the
variance filter keeps. Shows whether the small MT-vs-non-MT difference reflects survival information or just model size."""
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import coxfast
import dataset
from common import RESULTS

warnings.filterwarnings("ignore")
d = dataset.load("union")
prev = d.B.mean(0)
elig = (prev >= 0.1127) & (prev <= 0.8873)
rng = np.random.default_rng(11)
mt = np.where(d.col_is_mt)[0]
non = np.where(~d.col_is_mt)[0]
out = {}
for name, pool in (("MT", mt), ("nonMT", non)):
    ci, nk = [], []
    for _ in range(1000):
        cols = rng.choice(pool, 100, replace=False)
        X = np.column_stack([d.age, d.sex, d.B[:, cols]]).astype(np.float64)
        ok, c, keep, b, se, p = coxfast.fit(X, d.time, d.event, 0.5, 0.1, 0.9, True, False)
        ci.append(c if ok else np.nan)
        nk.append(int(keep[2:].sum()))
    out[name] = dict(ci=np.array(ci), nkeep=np.array(nk), n_elig_in_pool=int(elig[pool].sum()), pool=len(pool))
    print(f"{name}: pool {len(pool)} columns ({int(elig[pool].sum())} eligible = {100*elig[pool].mean():.1f}%); CI mean {np.nanmean(ci):.3f} sd {np.nanstd(ci):.3f} max {np.nanmax(ci):.3f} min {np.nanmin(ci):.3f}; kept columns mean {np.mean(nk):.2f}")
print("Mann-Whitney p (CI):", mannwhitneyu(out["MT"]["ci"], out["nonMT"]["ci"], nan_policy="omit").pvalue)
allci = np.concatenate([out["MT"]["ci"], out["nonMT"]["ci"]]); allk = np.concatenate([out["MT"]["nkeep"], out["nonMT"]["nkeep"]])
print("Spearman(CI, number of kept columns) over all 2000 sets: %.3f" % spearmanr(allci, allk, nan_policy="omit")[0])
for name in out:
    print(name, "mean CI at kept=0..8:", {k: round(float(np.nanmean(out[name]['ci'][out[name]['nkeep'] == k])), 3) for k in range(0, 9) if np.any(out[name]['nkeep'] == k)})
pickle.dump(out, open(RESULTS / "random_column_sets.pkl", "wb"))