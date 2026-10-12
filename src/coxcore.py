"""Numba kernels: ridge-penalised Cox (Efron ties) and Harrell's C-index.

Mirrors lifelines CoxPHFitter(penalizer=0.5): pure L2 on standardised covariates,
penalty = n * lam / 2 * ||beta||^2. Rows must be sorted by time ascending.
"""
import math

import numpy as np
from numba import njit


@njit(cache=True)
def matvec(X, b):
    n, p = X.shape
    out = np.zeros(n)
    for i in range(n):
        s = 0.0
        for a in range(p):
            s += X[i, a] * b[a]
        out[i] = s
    return out


@njit(cache=True)
def chol_factor(A):
    """Lower Cholesky factor of a symmetric positive-definite matrix (no LAPACK needed)."""
    p = A.shape[0]
    L = np.zeros((p, p))
    for i in range(p):
        for j in range(i + 1):
            s = A[i, j]
            for k in range(j):
                s -= L[i, k] * L[j, k]
            if i == j:
                if s <= 0.0:
                    s = 1e-12
                L[i, i] = math.sqrt(s)
            else:
                L[i, j] = s / L[j, j]
    return L


@njit(cache=True)
def chol_solve(A, b):
    p = A.shape[0]
    L = chol_factor(A)
    y = np.zeros(p)
    for i in range(p):
        s = b[i]
        for k in range(i):
            s -= L[i, k] * y[k]
        y[i] = s / L[i, i]
    x = np.zeros(p)
    for i in range(p - 1, -1, -1):
        s = y[i]
        for k in range(i + 1, p):
            s -= L[k, i] * x[k]
        x[i] = s / L[i, i]
    return x


@njit(cache=True)
def chol_inv(A):
    p = A.shape[0]
    inv = np.zeros((p, p))
    e = np.zeros(p)
    for j in range(p):
        e[:] = 0.0
        e[j] = 1.0
        col = chol_solve(A, e)
        for i in range(p):
            inv[i, j] = col[i]
    return inv


@njit(cache=True)
def derivs(X, e, t, beta, lam):
    """Penalised Efron log-likelihood, gradient and information matrix."""
    n, p = X.shape
    eta = matvec(X, beta)
    mx = eta.max()
    w_all = np.exp(eta - mx)
    R0 = 0.0
    R1 = np.zeros(p)
    R2 = np.zeros((p, p))
    ll = 0.0
    g = np.zeros(p)
    H = np.zeros((p, p))
    i = n - 1
    while i >= 0:
        j = i
        while j - 1 >= 0 and t[j - 1] == t[i]:
            j -= 1
        E0 = 0.0
        E1 = np.zeros(p)
        E2 = np.zeros((p, p))
        d = 0
        sumx = np.zeros(p)
        sumeta = 0.0
        for k in range(j, i + 1):
            w = w_all[k]
            R0 += w
            for a in range(p):
                R1[a] += w * X[k, a]
                for b in range(a, p):
                    R2[a, b] += w * X[k, a] * X[k, b]
            if e[k] == 1:
                d += 1
                E0 += w
                sumeta += eta[k]
                for a in range(p):
                    E1[a] += w * X[k, a]
                    sumx[a] += X[k, a]
                    for b in range(a, p):
                        E2[a, b] += w * X[k, a] * X[k, b]
        if d > 0:
            ll += sumeta
            g += sumx
            for l in range(d):
                f = l / d
                D0 = R0 - f * E0
                ll -= math.log(D0) + mx
                v = (R1 - f * E1) / D0
                g -= v
                for a in range(p):
                    for b in range(a, p):
                        H[a, b] += (R2[a, b] - f * E2[a, b]) / D0 - v[a] * v[b]
        i = j - 1
    for a in range(p):
        for b in range(a):
            H[a, b] = H[b, a]
    ll -= n * lam * 0.5 * np.sum(beta * beta)
    g -= n * lam * beta
    for a in range(p):
        H[a, a] += n * lam
    return ll, g, H


@njit(cache=True)
def newton(X, e, t, lam):
    """Newton-Raphson with step halving. Returns (beta, info matrix, converged)."""
    n, p = X.shape
    beta = np.zeros(p)
    ll, g, H = derivs(X, e, t, beta, lam)
    ok = True
    for it in range(60):
        step = chol_solve(H, g)
        s = 1.0
        accepted = False
        nb = beta
        for _ in range(25):
            nb = beta + s * step
            ll2, g2, H2 = derivs(X, e, t, nb, lam)
            if ll2 >= ll - 1e-10 and np.isfinite(ll2):
                accepted = True
                break
            s *= 0.5
        if not accepted:
            ok = False
            break
        delta = np.max(np.abs(s * step))
        beta, ll, g, H = nb, ll2, g2, H2
        if delta < 1e-7:
            break
    return beta, H, ok


@njit(cache=True)
def robust_se(X, e, beta, V):
    """Sandwich SE (standardised scale) as lifelines computes it with cluster_col (one row per cluster):
    Breslow-style score residuals (no tie correction), dfbeta = score @ V, SE_j = sqrt(sum_i dfbeta_ij^2)."""
    n, p = X.shape
    eta = matvec(X, beta)
    phi = np.exp(eta - eta.max())
    S0 = np.zeros(n)
    S1 = np.zeros((n, p))
    r0 = 0.0
    r1 = np.zeros(p)
    for k in range(n - 1, -1, -1):
        r0 += phi[k]
        for a in range(p):
            r1[a] += phi[k] * X[k, a]
        S0[k] = r0
        S1[k, :] = r1
    A = 0.0
    B = np.zeros(p)
    ss = np.zeros(p)
    score = np.zeros(p)
    for i in range(n):
        if e[i] == 1:
            A += 1.0 / S0[i]
            for a in range(p):
                B[a] += S1[i, a] / (S0[i] * S0[i])
        for a in range(p):
            score[a] = -phi[i] * (A * X[i, a] - B[a])
            if e[i] == 1:
                score[a] += X[i, a] - S1[i, a] / S0[i]
        for j in range(p):
            s = 0.0
            for l in range(p):
                s += score[l] * V[l, j]
            ss[j] += s * s
    return np.sqrt(ss)


@njit(cache=True)
def cindex(t, e, risk):
    """Harrell's C with lifelines' pairing rules (risk: larger = shorter survival)."""
    n = len(t)
    num = 0.0
    den = 0.0
    for i in range(n):
        if e[i] == 0:
            continue
        for j in range(n):
            if t[j] > t[i] or (t[j] == t[i] and e[j] == 0):
                den += 1.0
                if risk[i] > risk[j]:
                    num += 1.0
                elif risk[i] == risk[j]:
                    num += 0.5
    if den == 0.0:
        return np.nan
    return num / den
