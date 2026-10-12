"""Smoke test of the survival engine without any downloaded data: coxfast.fit must agree with lifelines on simulated data.
Run:  python tests/test_cox_engine.py   (or pytest)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import coxfast  # noqa: E402


def test_engine_matches_lifelines():
    rng = np.random.default_rng(7)
    n = 400
    X = rng.normal(size=(n, 3))
    t = rng.exponential(scale=np.exp(-(0.5 * X[:, 0] - 0.3 * X[:, 1])) * 20) + 0.01
    e = (rng.uniform(size=n) < 0.7).astype(np.int64)
    o = np.argsort(t, kind="stable")
    X, t, e = np.ascontiguousarray(X[o]), t[o], e[o]
    ok, ci, keep, beta, se, p = coxfast.fit(X, t, e, 1e-6, 0.0, 1.01, False, False)
    assert ok
    df = pd.DataFrame(X, columns=["a", "b", "c"])
    df["T"], df["E"] = t, e
    cph = CoxPHFitter().fit(df, "T", "E")
    assert np.allclose(beta, cph.params_.values, atol=1e-3)
    assert np.allclose(se, cph.standard_errors_.values, atol=1e-3)


if __name__ == "__main__":
    test_engine_matches_lifelines()
    print("engine test passed")
