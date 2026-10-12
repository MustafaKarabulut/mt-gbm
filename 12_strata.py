"""Stratified analyses: stratifying by age or sex and re-running the 10-gene GA search inside each stratum.
Observed vs permutation null (gene rows permuted before stratifying)."""
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import dataset
import ga
from common import RESULTS

warnings.filterwarnings("ignore")
REPORT = {"Age <= 50": 0.88, "50 < Age <= 60": 0.85, "Age > 60": 0.79, "Male": 0.75, "Female": 0.83}


def strata(d):
    return {"Age <= 50": np.where(d.age <= 50)[0], "50 < Age <= 60": np.where((d.age > 50) & (d.age <= 60))[0], "Age > 60": np.where(d.age > 60)[0],
            "Male": np.where(d.sex == 1)[0], "Female": np.where(d.sex == 0)[0]}


def task(arr, name, rows, kind, i):
    B = arr["B"]
    if kind == "null":
        B = B[np.random.default_rng(31000 + i).permutation(len(arr["age"]))]
    Bs = np.ascontiguousarray(B[rows])
    D = dict(B=Bs, age=arr["age"][rows], sex=arr["sex"][rows], t=arr["t"][rows], e=arr["e"][rows], colprev=Bs.mean(0))
    r = ga.run_cols(D, np.arange(Bs.shape[1]), (i if kind == "obs" else 5000 + i), pop=300, gens=200)
    return dict(stratum=name, n=len(rows), kind=kind, i=i, best=r.get("best_ci", np.nan), top=r.get("top_ci", np.nan))


if __name__ == "__main__":
    jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    d = dataset.load("union")
    arr = dict(B=d.B, age=d.age, sex=d.sex, t=d.time, e=d.event)
    ga.run_cols(dict(B=d.B[:, :200], age=d.age, sex=d.sex, t=d.time, e=d.event, colprev=d.B[:, :200].mean(0)), np.arange(200), 0, pop=20, gens=2)
    st = strata(d)
    t0 = time.time()
    tasks = [(nm, rows, kind, i) for nm, rows in st.items() for kind, n in (("obs", 6), ("null", 60)) for i in range(n)]
    res = Parallel(n_jobs=jobs)(delayed(task)(arr, *t) for t in tasks)
    pickle.dump(res, open(RESULTS / "strata.pkl", "wb"))
    print(f"done {time.time()-t0:.0f}s")
    for nm in st:
        o = np.array([r["best"] for r in res if r["stratum"] == nm and r["kind"] == "obs"])
        z = np.array([r["best"] for r in res if r["stratum"] == nm and r["kind"] == "null"])
        print(f"{nm:15s} n={len(st[nm]):3d} report {REPORT[nm]:.2f} | observed best CI {o.mean():.3f} | null mean {z.mean():.3f} (5-95%: {np.quantile(z,.05):.3f}-{np.quantile(z,.95):.3f}) p={(1+(z>=o.mean()).sum())/(1+len(z)):.2f}")