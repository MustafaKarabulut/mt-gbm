"""Control for the CV comparison of random 100-gene sets: are MT sets better than random sets only because MT genes are co-expressed
(fewer independent noise dimensions)? Random sets are drawn with the same mean absolute pairwise correlation as MT sets."""
import importlib.util
import pickle
import sys
import types
import warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from common import RESULTS

warnings.filterwarnings("ignore")
spec = importlib.util.spec_from_file_location("g2", str(Path(__file__).resolve().parent / "04_gate2.py"))
g2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g2)


def mean_abs_corr(Z, idx):
    C = np.corrcoef(Z[:, idx].T)
    iu = np.triu_indices(len(idx), 1)
    return float(np.mean(np.abs(C[iu])))


def corr_matched_sets(rng, Z, pool, target, n_sets, m=100):
    Zp = (Z[:, pool] - Z[:, pool].mean(0)) / np.where(Z[:, pool].std(0) > 0, Z[:, pool].std(0), 1.0)
    best = None
    for T in (100, 150, 250, 400, 700, 1200, 2500, 5000, 8000, len(pool)):
        vals = []
        for s in rng.choice(len(pool), 20, replace=False):
            cor = Zp.T @ Zp[:, s] / len(Zp)
            cand = np.argsort(-cor)[:T]
            vals.append(mean_abs_corr(Z, pool[rng.choice(cand, m, replace=False)]))
        gap = abs(np.mean(vals) - target)
        if best is None or gap < best[0]:
            best = (gap, T, np.mean(vals))
    T = best[1]
    sets = []
    for s in rng.choice(len(pool), n_sets, replace=True):
        cor = Zp.T @ Zp[:, s] / len(Zp)
        cand = np.argsort(-cor)[:T]
        sets.append(pool[rng.choice(cand, m, replace=False)])
    return sets, T, best[2]


if __name__ == "__main__":
    out = {}
    for name in ("tcga", "cgga"):
        d = g2.prep(name)
        rng = np.random.default_rng(21)
        mt_idx = np.where(d.expressed & (d.block != ""))[0]
        pool = np.where(d.expressed & (d.block == ""))[0]
        mt_sets = [rng.choice(mt_idx, 100, replace=False) for _ in range(600)]
        mt_r = np.mean([mean_abs_corr(d.Zr, s) for s in mt_sets[:60]])
        ctl_sets, T, got = corr_matched_sets(rng, d.Zr, pool, mt_r, 600)
        ctx = types.SimpleNamespace(Zr=d.Zr, age=d.age, sex=d.sex, time=d.time, event=d.event)
        run = lambda sets, off: np.array(Parallel(n_jobs=18)(delayed(g2.one_set_cv)(ctx, s, None, 2, 5, 0.5, off + i) for i, s in enumerate(sets)))
        a, b = run(mt_sets, 10_000), run(ctl_sets, 30_000)
        u = mannwhitneyu(a, b, alternative="two-sided")
        out[name] = dict(mt=a, ctl=b, mt_corr=mt_r, ctl_corr=got, T=T, p=float(u.pvalue))
        print(f"[{name}] mean |r| in sets: MT {mt_r:.3f}, correlation-matched random {got:.3f} (candidate pool top-{T}); CV C-index MT {a.mean():.4f} vs correlation-matched {b.mean():.4f}; Mann-Whitney p = {u.pvalue:.2g}", flush=True)
    pickle.dump(out, open(RESULTS / "modules_cv.pkl", "wb"))