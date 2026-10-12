"""Gene-set level tools: within-cohort z-scores, expression-matched random sets, Cox statistics, CV C-index."""
import numpy as np

import coxcore
import coxfast

LAM0 = 1e-6   # essentially unpenalised


def zlog(R):
    """z-score of log2(x+1) within the cohort, per gene (sd==0 -> 0)."""
    L = np.log2(np.nan_to_num(R, nan=0.0).astype(np.float64) + 1.0)
    sd = L.std(axis=0)
    z = (L - L.mean(axis=0)) / np.where(sd > 0, sd, 1.0)
    z[:, sd == 0] = 0.0
    return z


def design(d, covars):
    """Covariate matrix from names: age, sex, or a column of d.extra (one-hot for categorical handled by caller)."""
    cols = []
    for c in covars:
        cols.append(np.asarray(c, dtype=float))
    return np.column_stack(cols) if cols else np.empty((d.n, 0))


def cox_score(score, C, t, e):
    """Unpenalised Cox of [score_std, C...]. Returns (beta, se, z, p) for the score; rows must be time-sorted."""
    s = (score - score.mean()) / score.std()
    X = np.column_stack([s, C]) if C.shape[1] else s[:, None]
    ok, ci, keep, b, se, p = coxfast.fit(X, t, e, LAM0, 0.0, 1.01, False, False)
    if not ok:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    return b[0], se[0], b[0] / se[0], p[0], ci


def strata_by_expression(med_log, n_bins=10):
    """Quantile bin index (0..n_bins-1) of each gene by median log expression."""
    qs = np.quantile(med_log, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.searchsorted(qs, med_log)


def matched_sets(rng, mt_idx, pool_idx, strata, n_sets, size=None):
    """Random gene sets drawn from pool_idx whose stratum composition equals that of mt_idx (or a subsample of size)."""
    mt_idx = np.asarray(mt_idx)
    by = {s: pool_idx[strata[pool_idx] == s] for s in np.unique(strata[pool_idx])}
    need = np.bincount(strata[mt_idx], minlength=strata.max() + 1)
    if size is not None:
        need = np.round(need * size / need.sum()).astype(int)
    out = []
    for _ in range(n_sets):
        pick = [rng.choice(by[s], size=min(k, len(by[s])), replace=False) for s, k in enumerate(need) if k > 0 and s in by]
        out.append(np.concatenate(pick))
    return out


def cv_cindex(X, t, e, lam, folds, rng):
    """K-fold CV: ridge Cox on training folds, C-index of the linear predictor on each held-out fold (mean over folds)."""
    n = len(t)
    idx = rng.permutation(n)
    cis = []
    for f in range(folds):
        te = np.sort(idx[f::folds])
        tr = np.setdiff1d(np.arange(n), te)
        ok, ci, keep, b, se, p = coxfast.fit(X[tr], t[tr], e[tr], lam, 0.0, 1.01, False, False)
        if not ok:
            continue
        risk = X[te] @ np.where(np.isnan(b), 0.0, b)
        c = coxcore.cindex(t[te], e[te], risk)
        if c == c:
            cis.append(c)
    return float(np.mean(cis)) if cis else np.nan
