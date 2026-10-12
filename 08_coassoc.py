"""Report section 6.3-6.4 revisited: co-association matrix of GA groupings (K=2..8), Ward clustering into 3 groups,
and what those groups relate to (clinical variables) - observed data vs permutation-null data."""
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import chi2_contingency, kruskal
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import dataset
from common import RESULTS

warnings.filterwarnings("ignore")
KS = [2, 3, 4, 5, 6, 7, 8]


def coassoc(res, kind, idx):
    n = None
    M = None
    for K in KS:
        r = [x for x in res if x["K"] == K and x["kind"] == kind and x["idx"] == idx]
        if not r:
            return None
        lab = r[0]["labels"]
        n = len(lab)
        S = (lab[:, None] == lab[None, :]).astype(float)
        M = S if M is None else M + S
    return M


def chi_p(a, b):
    ok = np.array([x is not None and x == x for x in b])
    tab = np.array([[np.sum((a[ok] == i) & (np.array(b)[ok] == j)) for j in sorted(set(np.array(b)[ok]))] for i in sorted(set(a))])
    return chi2_contingency(tab)[1] if tab.shape[1] > 1 else np.nan


def analyse(M, d):
    out = {}
    for k in (2, 3, 4, 5, 6):
        lab = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(M)
        out[f"sil{k}"] = float(silhouette_score(M, lab))
        if k == 3:
            sub = d.extra["subtype"].values
            out["sizes"] = np.bincount(lab)
            out["p_sex"] = chi_p(lab, list(d.sex))
            out["p_subtype"] = chi_p(lab, [s if isinstance(s, str) else None for s in sub])
            out["p_age"] = float(kruskal(*[d.age[lab == g] for g in range(3)]).pvalue)
            out["p_time"] = float(kruskal(*[d.time[lab == g] for g in range(3)]).pvalue)
            out["p_event"] = chi_p(lab, list(d.event))
    return out


if __name__ == "__main__":
    d = dataset.load("union")
    res = pickle.load(open(RESULTS / "gate1_grp_union.pkl", "rb"))
    out = {"obs": [], "null": []}
    for kind, idxs in (("obs", range(10)), ("null", range(100))):
        for i in idxs:
            M = coassoc(res, kind, i)
            if M is not None:
                out[kind].append(analyse(M, d))
    pickle.dump(out, open(RESULTS / "coassoc.pkl", "wb"))
    for kind in ("obs", "null"):
        a = out[kind]
        print(kind, len(a), "replicates | silhouette k=3: %.3f (k=2 %.3f, k=4 %.3f, k=5 %.3f)" % tuple(np.mean([x[k] for x in a]) for k in ("sil3", "sil2", "sil4", "sil5")))
        for key in ("p_sex", "p_subtype", "p_age", "p_time", "p_event"):
            v = np.array([x[key] for x in a], float)
            print(f"   {key}: median p {np.nanmedian(v):.3g}; share of replicates with p<0.05: {np.nanmean(v < 0.05):.2f}")
