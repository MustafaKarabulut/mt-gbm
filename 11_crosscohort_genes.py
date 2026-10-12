"""Do gene-level prognostic effects replicate between TCGA and CGGA, and more so for MT genes than for expression-matched random genes?
Univariate Cox z-statistic of the continuous within-cohort z-score (age + sex adjusted) for every common expressed gene."""
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import cgga
import coxfast
import dataset
import gset
from common import RESULTS, mt_blocks

warnings.filterwarnings("ignore")


def gene_z(Z, age, sex, t, e):
    out = np.full(Z.shape[1], np.nan)
    base = np.column_stack([age, sex]).astype(float)
    for j in range(Z.shape[1]):
        X = np.column_stack([base, Z[:, j]])
        ok, ci, keep, b, se, p = coxfast.fit(X, t, e, 1e-6, 0.0, 1.01, False, False)
        if ok:
            out[j] = b[2] / se[2]
    return out


dt = dataset.load("pancan")
dc = cgga.load("primary_gbm")
Rt, Rc = np.nan_to_num(dt.rsem, nan=0.0), np.nan_to_num(dc.rsem, nan=0.0)
Zt, Zc = gset.zlog(Rt), gset.zlog(Rc)
common = sorted(set(dt.genes) & set(dc.genes))
it = {g: i for i, g in enumerate(dt.genes)}
ic = {g: i for i, g in enumerate(dc.genes)}
a = np.array([it[g] for g in common]); b = np.array([ic[g] for g in common])
expr = (np.median(Rt, axis=0)[a] >= 10) & (np.median(Rc, axis=0)[b] >= 1)
idx = np.where(expr)[0]
zt = gene_z(Zt[:, a[idx]], dt.age, dt.sex, dt.time, dt.event)
zc = gene_z(Zc[:, b[idx]], dc.age, dc.sex, dc.time, dc.event)
bl = mt_blocks()
blk = np.array([bl.get(common[i]) or "" for i in idx])
medlog = np.median(np.log2(Rt[:, a[idx]] + 1), axis=0)
strata = gset.strata_by_expression(medlog)
rng = np.random.default_rng(5)
pool = np.where(blk == "")[0]
res = {"n_genes": len(idx), "rho_all": spearmanr(zt, zc, nan_policy="omit")[0]}
for name, mask in (("MT_all", blk != ""), ("A", blk == "A"), ("C", blk == "C"), ("D", blk == "D")):
    mi = np.where(mask)[0]
    if len(mi) < 10:
        continue
    r_obs = spearmanr(zt[mi], zc[mi], nan_policy="omit")[0]
    sets = gset.matched_sets(rng, mi, pool, strata, 2000)
    r_null = np.array([spearmanr(zt[s], zc[s], nan_policy="omit")[0] for s in sets])
    res[name] = dict(n=len(mi), rho=r_obs, null_mean=float(r_null.mean()), null_q=np.quantile(r_null, [.025, .975]), p=(1 + (r_null >= r_obs).sum()) / 2001)
    print(f"{name:7s} n={len(mi):4d} rho(TCGA,CGGA gene z) = {r_obs:+.3f}; matched random sets: mean {r_null.mean():+.3f} (95% {np.quantile(r_null,.025):+.3f}..{np.quantile(r_null,.975):+.3f}); one-sided p = {res[name]['p']:.3f}")
print("all expressed genes common to both cohorts:", len(idx), " rho = %.3f" % res["rho_all"])
pickle.dump(res, open(RESULTS / "crosscohort_genes.pkl", "wb"))

# --- co-expression-aware (conservative) null: random "modules" = seed gene + its m most co-expressed non-MT genes ---
Zp = Zt[:, a[idx]][:, pool].astype(np.float64)
Zp = (Zp - Zp.mean(0)) / np.where(Zp.std(0) > 0, Zp.std(0), 1.0)
for name, mask in (("MT_all", blk != ""), ("A", blk == "A"), ("C", blk == "C"), ("D", blk == "D")):
    mi = np.where(mask)[0]
    if len(mi) < 10 or name not in res:
        continue
    m = len(mi)
    r_mod = []
    for s in rng.choice(len(pool), 1500, replace=False):
        cor = Zp.T @ Zp[:, s] / len(Zp)
        nn = np.argsort(-cor)[:m]
        r_mod.append(spearmanr(zt[pool[nn]], zc[pool[nn]], nan_policy="omit")[0])
    r_mod = np.array(r_mod)
    res[name]["mod_q"] = np.quantile(r_mod, [.025, .975])
    res[name]["mod_p"] = (1 + (r_mod >= res[name]["rho"]).sum()) / (1 + len(r_mod))
    print(f"{name:7s} module-like random sets: 95% {np.quantile(r_mod,.025):+.3f}..{np.quantile(r_mod,.975):+.3f}; one-sided p = {res[name]['mod_p']:.3f}")
pickle.dump(res, open(RESULTS / "crosscohort_genes.pkl", "wb"))