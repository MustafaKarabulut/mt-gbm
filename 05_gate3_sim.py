"""Gate 3: simulations with known ground truth on the real TCGA gene matrix (rows permuted -> no link to outcome).
S0 null | S1 five homogeneous driver columns (HR 2 / 0.5) | S2 three latent subgroups (80/50/30) with expression signatures
and subgroup-specific drivers (HR 3 / 1/3). Survival: exponential, median ~11 mo, ~20% censoring. The same data
generating process gives a fresh outcome vector for the same patients (test), so optimism = in-sample - test."""
import argparse
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import baselines
import dataset
import ga
from common import RESULTS

warnings.filterwarnings("ignore")
H0 = np.log(2) / 11.0


def simulate(arr, scen, seed):
    rng = np.random.default_rng(seed)
    n = len(arr["age"])
    B = arr["B"][rng.permutation(n)].copy()
    prev = B.mean(0)
    cand = np.where((prev >= 0.2) & (prev <= 0.6))[0]
    lp = 0.3 * (arr["age"] - arr["age"].mean()) / arr["age"].std()
    truth = dict(drivers=np.array([], dtype=int), labels=None)
    if scen == "S1":
        dr = rng.choice(cand, 5, replace=False)
        lp = lp + B[:, dr] @ (np.array([1, -1, 1, 1, -1]) * np.log(2.0))
        truth["drivers"] = dr
    elif scen == "S2":
        labels = rng.permutation(np.repeat([1, 2, 3], [80, 50, 30]))[:n]
        pick = rng.choice(cand, 39, replace=False)
        sig, drv = pick[:30].reshape(3, 10), pick[30:].reshape(3, 3)
        for k in range(3):
            for c in sig[k]:
                B[:, c] = (rng.random(n) < np.where(labels == k + 1, 0.85, 0.10)).astype(np.uint8)
            for j, c in enumerate(drv[k]):
                lp = lp + (np.log(3.0) if j < 2 else -np.log(3.0)) * B[:, c] * (labels == k + 1)
        truth = dict(drivers=drv.ravel(), labels=labels)
    haz = H0 * np.exp(lp - lp.mean())

    def outcome():
        T, Cn = rng.exponential(1.0 / haz), rng.exponential(1.0 / (0.25 * H0), n)
        return np.minimum(T, Cn), (T <= Cn).astype(np.int64)

    T, E = outcome()
    T2, E2 = outcome()
    o = np.argsort(T, kind="stable")
    return dict(B=np.ascontiguousarray(B[o]), age=arr["age"][o], sex=arr["sex"][o], t=T[o], e=E[o], t2=T2[o], e2=E2[o],
                lp=lp[o], labels=None if truth["labels"] is None else truth["labels"][o], drivers=truth["drivers"])


def run_dataset(arr, scen, seed, K=3):
    s = simulate(arr, scen, seed)
    B = s["B"]
    D = dict(B=B, age=s["age"], sex=s["sex"], t=s["t"], e=s["e"], colprev=B.mean(0))
    allc = np.arange(B.shape[1])
    out = dict(scen=scen, seed=seed, oracle_test=baselines.cind(s["t2"], s["e2"], s["lp"]),
               age_test=baselines.cind(s["t2"], s["e2"], s["age"]))
    r = ga.run_cols(D, allc, seed, pop=300, gens=200)
    if r["ok"]:
        (ok, ci, keep, beta, se, pv), uc, X, tt, ee = ga.single_eval(D, r["best_cols"])
        risk = X @ np.where(np.isnan(beta), 0.0, beta)
        out.update(ga_train=r["best_ci"], ga_test=baselines.cind(s["t2"], s["e2"], risk),
                   ga_hits=len(set(r["best_cols"].tolist()) & set(s["drivers"].tolist())))
        (ok2, ci2, *_), uc2, X2, _, _ = ga.single_eval(D, r["top_cols"])
        out.update(ga_top_train=r["top_ci"], ga_top_hits=len(set(r["top_cols"].tolist()) & set(s["drivers"].tolist())))
    g = ga.run_groups(D, allc, seed, K=K, pop=300, gens=200)
    out.update(grp_wci=g["wci"], grp_sizes=np.bincount(g["labels"], minlength=K + 1)[1:])
    if s["labels"] is not None:
        out["grp_ari"] = adjusted_rand_score(s["labels"], g["labels"])
        out["grp_hits"] = len(set(g["cols"].ravel().tolist()) & set(s["drivers"].tolist()))
    prev = B.mean(0)
    el = np.where((prev >= 0.1127) & (prev <= 0.8873))[0]
    Xn = np.column_stack([s["age"], s["sex"], B[:, el]]).astype(float)
    m = baselines.Coxnet(l1_ratio=0.5, seed=seed).fit(Xn, s["t"], s["e"])
    sel = el[m.selected()[m.selected() >= 2] - 2]
    out.update(net_cv=m.cv_c, net_train=baselines.cind(s["t"], s["e"], m.predict(Xn)), net_test=baselines.cind(s["t2"], s["e2"], m.predict(Xn)),
               net_nsel=len(m.selected()), net_hits=len(set(sel.tolist()) & set(s["drivers"].tolist())))
    return out


def fmt(rs, key):
    v = np.array([r[key] for r in rs if key in r], dtype=float)
    return f"{np.nanmean(v):.3f}" if len(v) else "  - "


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--jobs", type=int, default=18)
    ap.add_argument("--scens", default="S0,S1,S2")
    a = ap.parse_args()
    d = dataset.load("union")
    arr = dict(B=d.B, age=d.age, sex=d.sex)
    ga.run_cols(dict(B=d.B, age=d.age, sex=d.sex, t=d.time, e=d.event, colprev=d.B.mean(0)), np.arange(d.B.shape[1]), 0, pop=20, gens=2)
    ga.run_groups(dict(B=d.B, age=d.age, sex=d.sex, t=d.time, e=d.event, colprev=d.B.mean(0)), np.arange(d.B.shape[1]), 0, K=3, pop=20, gens=2)
    t0 = time.time()
    jobs = [(sc, 500 + i) for sc in a.scens.split(",") for i in range(a.n)]
    res = Parallel(n_jobs=a.jobs)(delayed(run_dataset)(arr, sc, sd) for sc, sd in jobs)
    pickle.dump(res, open(RESULTS / "gate3_sim.pkl", "wb"))
    print(f"done in {time.time()-t0:.0f}s")
    for sc in a.scens.split(","):
        rs = [r for r in res if r["scen"] == sc]
        print(f"{sc}: n={len(rs)} | GA cols: train CI {fmt(rs,'ga_train')} test CI {fmt(rs,'ga_test')} hits {fmt(rs,'ga_hits')} | "
              f"groups wCI {fmt(rs,'grp_wci')} ARI {fmt(rs,'grp_ari')} | Coxnet train {fmt(rs,'net_train')} test {fmt(rs,'net_test')} hits {fmt(rs,'net_hits')} (n sel {fmt(rs,'net_nsel')}) | oracle test {fmt(rs,'oracle_test')} age-only test {fmt(rs,'age_test')}")
