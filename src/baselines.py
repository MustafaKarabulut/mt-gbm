"""Standard baselines a reviewer will ask for: elastic-net Cox (Coxnet) with CV-chosen penalty, C-index helper."""
import warnings

import numpy as np
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

warnings.filterwarnings("ignore")


def cind(t, e, risk):
    """Harrell C (sksurv) of a risk score; larger risk = shorter survival."""
    ok = np.isfinite(risk)
    if e[ok].sum() < 2:
        return np.nan
    return float(concordance_index_censored(e[ok].astype(bool), t[ok], risk[ok])[0])


class Coxnet:
    """Elastic-net Cox; alpha chosen by K-fold CV C-index along a common path. Features are standardised."""

    def __init__(self, l1_ratio=0.5, folds=4, n_alphas=25, alpha_min_ratio=0.05, seed=0):
        self.l1, self.folds, self.na, self.amr, self.seed = l1_ratio, folds, n_alphas, alpha_min_ratio, seed

    def fit(self, X, t, e):
        self.mu, self.sd = X.mean(0), X.std(0)
        self.sd[self.sd == 0] = 1.0
        Xs = (X - self.mu) / self.sd
        y = Surv.from_arrays(event=e.astype(bool), time=t)
        path = CoxnetSurvivalAnalysis(l1_ratio=self.l1, n_alphas=self.na, alpha_min_ratio=self.amr,
                                      fit_baseline_model=False, max_iter=50000).fit(Xs, y)
        alphas = path.alphas_
        n = len(t)
        idx = np.random.default_rng(self.seed).permutation(n)
        score = np.zeros(len(alphas))
        cnt = np.zeros(len(alphas))
        for f in range(self.folds):
            te = idx[f::self.folds]
            tr = np.setdiff1d(np.arange(n), te)
            m = CoxnetSurvivalAnalysis(l1_ratio=self.l1, alphas=alphas, fit_baseline_model=False, max_iter=50000)
            try:
                m.fit(Xs[tr], y[tr])
            except Exception:
                continue
            for k, a in enumerate(alphas):
                lp = Xs[te] @ m.coef_[:, k]
                c = cind(t[te], e[te], lp) if np.any(lp != 0) else 0.5
                if c == c:
                    score[k] += c
                    cnt[k] += 1
        score = np.where(cnt > 0, score / np.maximum(cnt, 1), 0.5)
        best = int(np.argmax(score))
        self.alpha = alphas[best]
        self.coef = path.coef_[:, best].copy()
        self.cv_c = float(score[best])
        return self

    def predict(self, X):
        return ((X - self.mu) / self.sd) @ self.coef

    def selected(self):
        return np.nonzero(self.coef)[0]
