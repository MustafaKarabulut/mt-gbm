"""Filter + fit layer on top of coxcore:
VarianceThreshold(0.1) -> |r| > 0.90 collinearity filter -> ridge Cox (penalizer 0.5) -> train C-index.
"""
import math

import numpy as np
from numba import njit

from coxcore import chol_inv, cindex, matvec, newton, robust_se

LAM = 0.5
VAR_THR = 0.1
CORR_THR = 0.90


@njit(cache=True)
def fit(X, t, e, lam, var_thr, corr_thr, do_filter, robust):
    """Returns (ok, ci, keep[q], beta_orig[q], se_orig[q], p[q]); dropped columns are NaN.

    robust=True : sandwich SE as lifelines with cluster_col=PatientId
    robust=False: model-based SE from the penalised information matrix
    """
    m, q = X.shape
    keep = np.ones(q, dtype=np.bool_)
    beta_o = np.full(q, np.nan)
    se_o = np.full(q, np.nan)
    pv = np.full(q, np.nan)
    if m < 4:
        return False, np.nan, keep, beta_o, se_o, pv
    nev = 0
    for i in range(m):
        nev += e[i]
    if nev < 2:
        return False, np.nan, keep, beta_o, se_o, pv
    mu = np.zeros(q)
    sd0 = np.zeros(q)
    var_keep = np.ones(q, dtype=np.bool_)
    for j in range(q):
        col = X[:, j]
        mu[j] = col.mean()
        v = np.mean((col - mu[j]) ** 2)
        sd0[j] = math.sqrt(v)
        if do_filter:
            var_keep[j] = v > var_thr
        else:
            var_keep[j] = v > 0.0
    drop_c = np.zeros(q, dtype=np.bool_)
    if do_filter:
        for j in range(q):
            if not var_keep[j]:
                continue
            for i in range(j):
                if not var_keep[i]:
                    continue
                c = np.mean((X[:, i] - mu[i]) * (X[:, j] - mu[j])) / (sd0[i] * sd0[j])
                if abs(c) > corr_thr:
                    drop_c[j] = True
                    break
    nk = 0
    for j in range(q):
        keep[j] = var_keep[j] and not drop_c[j]
        if keep[j]:
            nk += 1
    if nk == 0:
        return False, np.nan, keep, beta_o, se_o, pv
    Xk = np.empty((m, nk))
    sds = np.empty(nk)
    idx = np.empty(nk, dtype=np.int64)
    c = 0
    for j in range(q):
        if keep[j]:
            idx[c] = j
            col = X[:, j]
            s = math.sqrt(np.sum((col - mu[j]) ** 2) / (m - 1))   # pandas .std(): ddof = 1
            sds[c] = s
            Xk[:, c] = (col - mu[j]) / s
            c += 1
    beta, H, ok = newton(Xk, e, t, lam)
    if not ok:
        return False, np.nan, keep, beta_o, se_o, pv
    ci = cindex(t, e, matvec(Xk, beta))
    cov = chol_inv(H)
    rse = np.zeros(nk)
    if robust:
        rse = robust_se(Xk, e, beta, cov)
    for c in range(nk):
        j = idx[c]
        beta_o[j] = beta[c] / sds[c]
        se = rse[c] if robust else math.sqrt(cov[c, c])
        se_o[j] = se / sds[c]
        pv[j] = math.erfc(abs(beta[c]) / se / math.sqrt(2.0))
    return True, ci, keep, beta_o, se_o, pv
