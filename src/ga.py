"""NSGA-II searches on top of the fast Cox engine.

ColRowProblem  : k columns (+ optionally a 0/1 row mask) -> objectives (-CI, -mean patient fraction, sum -|HR-1|),
                 constraints  p<=0.05 for every selected column (columns removed by the filters count as satisfied)
                 and, for row search, n_rows >= min_rows. The min_rows constraint is enforced explicitly
                 (pymoo treats G<=0 as feasible).
GroupProblem   : patient -> group labels (1..K) + k columns per group, objective  -sum_g CI_g * n_g / n.
All Cox fits go through coxfast.fit (penalizer 0.5, variance + collinearity filter, robust SE for p-values).
"""
import numpy as np
from numba import njit
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize

from coxfast import fit

LAM, VAR_THR, CORR_THR, P_THR = 0.5, 0.1, 0.90, 0.05


@njit(cache=True)
def _build(B, age, sex, t, e, rows, m, ucols, ku):
    X = np.empty((m, 2 + ku))
    tt = np.empty(m)
    ee = np.empty(m, dtype=np.int64)
    for i in range(m):
        r = rows[i]
        X[i, 0] = age[r]
        X[i, 1] = sex[r]
        for b in range(ku):
            X[i, 2 + b] = B[r, ucols[b]]
        tt[i] = t[r]
        ee[i] = e[r]
    return X, tt, ee


@njit(cache=True)
def eval_cols_rows(B, age, sex, t, e, colprev, Xpop, k, use_rows, min_rows, F, G):
    n = B.shape[0]
    pop = Xpop.shape[0]
    rows = np.empty(n, dtype=np.int64)
    for ind in range(pop):
        if use_rows:
            m = 0
            for r in range(n):
                if Xpop[ind, k + r] != 0:
                    rows[m] = r
                    m += 1
        else:
            m = n
            for r in range(n):
                rows[r] = r
        ucols = np.empty(k, dtype=np.int64)
        pos = np.empty(k, dtype=np.int64)
        ku = 0
        for a in range(k):
            f = -1
            for b in range(ku):
                if ucols[b] == Xpop[ind, a]:
                    f = b
                    break
            if f < 0:
                ucols[ku] = Xpop[ind, a]
                f = ku
                ku += 1
            pos[a] = f
        X, tt, ee = _build(B, age, sex, t, e, rows, m, ucols, ku)
        ok, ci, keep, beta, se, pv = fit(X, tt, ee, LAM, VAR_THR, CORR_THR, True, True)
        ncons = k + (1 if use_rows else 0)
        if not ok:
            F[ind, 0] = 0.0
            F[ind, 1] = 0.0
            F[ind, 2] = 0.0
            for c in range(ncons):
                G[ind, c] = 1.0
            continue
        pc = 0.0
        for a in range(k):
            pc += colprev[Xpop[ind, a]]
        hr = 0.0
        for j in range(2 + ku):
            if keep[j]:
                hr -= abs(np.exp(beta[j]) - 1.0)
        F[ind, 0] = -ci
        F[ind, 1] = -pc / k
        F[ind, 2] = hr
        for a in range(k):
            j = 2 + pos[a]
            if (not keep[j]) or pv[j] <= P_THR:
                G[ind, a] = -1.0
            else:
                G[ind, a] = pv[j]
        if use_rows:
            G[ind, k] = min_rows - m


@njit(cache=True)
def eval_groups(B, age, sex, t, e, Xpop, K, k, F):
    n = B.shape[0]
    pop = Xpop.shape[0]
    rows = np.empty(n, dtype=np.int64)
    for ind in range(pop):
        tot = 0.0
        for g in range(1, K + 1):
            m = 0
            for r in range(n):
                if Xpop[ind, r] == g:
                    rows[m] = r
                    m += 1
            if m < 4:
                continue
            ucols = np.empty(k, dtype=np.int64)
            ku = 0
            for a in range(k):
                c = Xpop[ind, n + (g - 1) * k + a]
                f = False
                for b in range(ku):
                    if ucols[b] == c:
                        f = True
                        break
                if not f:
                    ucols[ku] = c
                    ku += 1
            X, tt, ee = _build(B, age, sex, t, e, rows, m, ucols, ku)
            ok, ci, keep, beta, se, pv = fit(X, tt, ee, LAM, VAR_THR, CORR_THR, True, False)
            if ok and ci == ci:
                tot += ci * m
        F[ind, 0] = -tot / n


class ColRowProblem(Problem):
    def __init__(self, D, allowed, k=10, use_rows=False, min_rows=20):
        self.D, self.allowed, self.k, self.use_rows, self.min_rows = D, np.asarray(allowed, dtype=np.int64), k, use_rows, min_rows
        n = D["B"].shape[0]
        nvar = k + (n if use_rows else 0)
        xu = np.concatenate([np.full(k, len(self.allowed) - 1), np.ones(n if use_rows else 0)])
        super().__init__(n_var=nvar, n_obj=3, n_ieq_constr=k + (1 if use_rows else 0),
                         xl=np.zeros(nvar), xu=xu, vtype=int)

    def _evaluate(self, X, out, *args, **kwargs):
        Xi = np.rint(X).astype(np.int64)
        Xp = np.hstack([self.allowed[Xi[:, :self.k]], Xi[:, self.k:]])
        F = np.zeros((len(Xp), 3))
        G = np.zeros((len(Xp), self.n_ieq_constr))
        D = self.D
        eval_cols_rows(D["B"], D["age"], D["sex"], D["t"], D["e"], D["colprev"], Xp, self.k, self.use_rows,
                       self.min_rows, F, G)
        out["F"], out["G"] = F, G


class GroupProblem(Problem):
    def __init__(self, D, allowed, K=3, k=10):
        self.D, self.allowed, self.K, self.k = D, np.asarray(allowed, dtype=np.int64), K, k
        n = D["B"].shape[0]
        nvar = n + K * k
        xl = np.concatenate([np.ones(n), np.zeros(K * k)])
        xu = np.concatenate([np.full(n, K), np.full(K * k, len(self.allowed) - 1)])
        super().__init__(n_var=nvar, n_obj=1, n_ieq_constr=0, xl=xl, xu=xu, vtype=int)

    def _evaluate(self, X, out, *args, **kwargs):
        Xi = np.rint(X).astype(np.int64)
        n = self.D["B"].shape[0]
        Xp = np.hstack([Xi[:, :n], self.allowed[Xi[:, n:]]])
        F = np.zeros((len(Xp), 1))
        D = self.D
        eval_groups(D["B"], D["age"], D["sex"], D["t"], D["e"], Xp, self.K, self.k, F)
        out["F"] = F


def make_data(d, B=None):
    """Arrays for the numba kernels. B may be replaced (e.g. row-permuted) to build null data sets."""
    B = np.ascontiguousarray(d.B if B is None else B)
    return dict(B=B, age=d.age, sex=d.sex, t=d.time, e=d.event, colprev=B.mean(axis=0))


def _algo(pop, multi=True):
    kw = dict(pop_size=pop, sampling=IntegerRandomSampling(),
              crossover=SBX(prob=1.0, eta=10, vtype=float, repair=RoundingRepair()),
              mutation=PM(eta=10, vtype=float, repair=RoundingRepair()), eliminate_duplicates=True)
    return NSGA2(**kw) if multi else GA(**kw)


def single_eval(D, cols, rows=None, robust=True):
    """Full filtered/penalised Cox fit for one column set (and optional rows)."""
    n = D["B"].shape[0]
    rows = np.arange(n) if rows is None else np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    ucols = np.array(list(dict.fromkeys(cols.tolist())), dtype=np.int64)
    X, tt, ee = _build(D["B"], D["age"], D["sex"], D["t"], D["e"], rows, len(rows), ucols, len(ucols))
    return fit(X, tt, ee, LAM, VAR_THR, CORR_THR, True, robust), ucols, X, tt, ee


def run_cols(D, allowed, seed, pop=300, gens=200, k=10, use_rows=False, min_rows=20):
    prob = ColRowProblem(D, allowed, k=k, use_rows=use_rows, min_rows=min_rows)
    res = minimize(prob, _algo(pop), ("n_gen", gens), seed=seed, verbose=False)
    out = dict(seed=seed, ok=res.X is not None)
    if res.X is None:
        return out
    Xi = np.rint(res.X).astype(np.int64)
    cols = prob.allowed[Xi[:, :k]]
    ci = -res.F[:, 0]
    out.update(n_pareto=len(Xi), ci_all=ci, hr_sum=res.F[:, 2])
    b = int(np.argmax(ci))
    out["best_ci"], out["best_cols"] = float(ci[b]), cols[b]
    if use_rows:
        rows = np.nonzero(Xi[b, k:])[0]
        out["best_rows"] = rows
        out["best_n"] = len(rows)
        out["n_rows_all"] = Xi[:, k:].sum(1)
    # for target=cols: refit the k most frequent genes among the Pareto solutions
    vals, cnt = np.unique(cols.ravel(), return_counts=True)
    top = vals[np.argsort(-cnt, kind="stable")[:k]]
    out["top_cols"] = top
    (ok, ci_top, *_), _, _, _, _ = single_eval(D, top)
    out["top_ci"] = float(ci_top) if ok else np.nan
    return out


def run_groups(D, allowed, seed, K=3, pop=300, gens=200, k=10):
    prob = GroupProblem(D, allowed, K=K, k=k)
    res = minimize(prob, _algo(pop, multi=False), ("n_gen", gens), seed=seed, verbose=False)
    x = np.rint(res.X).astype(np.int64)
    n = D["B"].shape[0]
    return dict(seed=seed, K=K, wci=float(-res.F[0]), labels=x[:n], cols=prob.allowed[x[n:]].reshape(K, k))
