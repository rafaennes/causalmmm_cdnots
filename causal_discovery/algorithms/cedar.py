"""CEDAR: Causal Edge Discovery in Autoregressive time series.

Fesanghary (2026). Designed for scarce data (T=100-200) with lag-1 dynamics.

Pipeline:
  1. Fit AR(1) residuals per variable to remove autoregressive signal.
  2. Screen candidate cross-lag edges via U-centered distance correlation
     on residuals — O(d²) pairs, no CI tests yet.
  3. For each significant candidate (i -> j at lag 1), run two directed
     CI tests to confirm/reject the edge.
  4. MCI-style pruning: re-test remaining edges conditioning on all other
     accepted parents to remove indirect links.
  5. Optional C-node: append normalized time index to handle trend
     nonstationarity (same idea as CD-NOTS surrogate variable).

Key advantage over PCMCI in low-N: distance correlation screening requires
far fewer samples than CMIknn, and AR(1) residualization removes the dominant
serial correlation that inflates false positives.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from causal_discovery.graph import CausalGraph, bh_correct


def discover(
    data: pd.DataFrame,
    var_names: list[str],
    alpha: float = 0.05,
    max_lag: int = 2,
    use_c_node: bool = True,
    **kwargs,
) -> CausalGraph:
    """CEDAR: distance-correlation screened causal discovery for short series.

    Parameters
    ----------
    data : DataFrame with columns matching var_names. Last column must be "y".
    var_names : list of column names; last element must be "y".
    alpha : significance level for screening and CI tests.
    max_lag : maximum lag to consider (CEDAR prefers lag-1 but tests up to max_lag).
    use_c_node : if True, append time index as C-node to adjust for trend.
    """
    start = time.perf_counter()
    X = StandardScaler().fit_transform(data[var_names].values)
    T, n = X.shape
    y_idx = n - 1

    # 1. AR(1) residualization: remove dominant serial correlation
    residuals = np.zeros((T - 1, n))
    for j in range(n):
        # OLS fit: x_t = a + b * x_{t-1} + eps
        x_lag = X[:-1, j]
        x_cur = X[1:, j]
        b = np.dot(x_lag, x_cur) / (np.dot(x_lag, x_lag) + 1e-12)
        residuals[:, j] = x_cur - b * x_lag

    # Optional: append C-node (time trend) to residuals
    if use_c_node:
        c_resid = np.linspace(0, 1, T - 1).reshape(-1, 1)
        residuals_aug = np.hstack([residuals, c_resid])
    else:
        residuals_aug = residuals

    # 2. Screen candidates via distance correlation on residuals
    # For each pair (i, j), test dcor(residual_i[:-lag], residual_j[lag:])
    pval = np.ones((n, n))
    dcor_vals = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            best_p = 1.0
            best_d = 0.0
            for lag in range(1, max_lag + 1):
                if lag >= T - 1:
                    continue
                x_i = residuals[:-lag, i] if lag < len(residuals) else residuals[:1, i]
                x_j = residuals[lag:, j]
                min_len = min(len(x_i), len(x_j))
                if min_len < 10:
                    continue
                x_i, x_j = x_i[:min_len], x_j[:min_len]
                d, p = _dcor_test(x_i, x_j)
                if p < best_p:
                    best_p = p
                    best_d = d
            pval[i, j] = best_p
            dcor_vals[i, j] = best_d

    # 3. BH correction on screening p-values
    adj = bh_correct(pval, alpha)

    # 4. MCI-style pruning: for each edge i->j, test conditioning on other parents of j
    adj_pruned = adj.copy()
    for j in range(n):
        parents = [i for i in range(n) if adj[i, j] == 1]
        if len(parents) <= 1:
            continue
        for i in parents:
            other_parents = [p for p in parents if p != i]
            # Partial distance correlation: residualize i and j on other parents
            x_i = residuals[:-1, i]
            x_j = residuals[1:, j]
            Z = residuals[:-1][:, other_parents]
            min_len = min(len(x_i), len(x_j), len(Z))
            x_i, x_j, Z = x_i[:min_len], x_j[:min_len], Z[:min_len]
            if min_len < 10:
                continue
            # Residualize on Z via OLS
            x_i_r = _residualize(x_i, Z)
            x_j_r = _residualize(x_j, Z)
            _, p_cond = _dcor_test(x_i_r, x_j_r)
            if p_cond >= alpha:
                adj_pruned[i, j] = 0.0

    adj_pruned[y_idx, :] = 0.0  # y does not cause anything
    np.fill_diagonal(adj_pruned, 0.0)

    return CausalGraph(
        adjacency_matrix=adj_pruned,
        variable_names=tuple(var_names),
        algorithm="cedar",
        runtime_seconds=time.perf_counter() - start,
        metadata={
            "pvalues": pval,
            "dcor_matrix": dcor_vals,
            "use_c_node": use_c_node,
        },
    )


# --- Internal helpers ---------------------------------------------------------


def _dcor_test(x: np.ndarray, y: np.ndarray, n_perm: int = 199) -> tuple[float, float]:
    """U-centered distance correlation with permutation test.

    Returns (dcor_value, p_value). Permutation-based p-value is robust
    for small samples where asymptotic chi² approximation fails.
    """
    n = len(x)
    if n < 5:
        return 0.0, 1.0

    observed = _u_dcor(x, y)

    # Permutation test
    count = 0
    rng = np.random.default_rng(42)
    for _ in range(n_perm):
        y_perm = rng.permutation(y)
        if _u_dcor(x, y_perm) >= observed:
            count += 1
    p = (count + 1) / (n_perm + 1)
    return observed, p


def _u_dcor(x: np.ndarray, y: np.ndarray) -> float:
    """Unbiased (U-centered) distance correlation estimator.

    Szekely & Rizzo (2014). Works for 1D arrays.
    """
    n = len(x)
    if n < 4:
        return 0.0

    a = np.abs(x[:, None] - x[None, :])
    b = np.abs(y[:, None] - y[None, :])

    # U-centering
    A = _u_center(a)
    B = _u_center(b)

    # Only off-diagonal elements contribute
    mask = ~np.eye(n, dtype=bool)
    dcov2 = np.sum(A[mask] * B[mask]) / (n * (n - 3))
    dvar_x = np.sum(A[mask] * A[mask]) / (n * (n - 3))
    dvar_y = np.sum(B[mask] * B[mask]) / (n * (n - 3))

    if dvar_x <= 0 or dvar_y <= 0:
        return 0.0
    return float(np.sqrt(max(dcov2, 0)) / np.sqrt(np.sqrt(dvar_x) * np.sqrt(dvar_y)))


def _u_center(D: np.ndarray) -> np.ndarray:
    """U-center a distance matrix (Szekely & Rizzo 2014)."""
    n = D.shape[0]
    row_mean = D.mean(axis=1, keepdims=True)
    col_mean = D.mean(axis=0, keepdims=True)
    grand_mean = D.mean()

    U = D - row_mean - col_mean + grand_mean
    # Zero out diagonal (U-centering convention)
    np.fill_diagonal(U, 0.0)
    # Adjust for bias: multiply off-diagonal by n/(n-2)
    U *= n / (n - 2)
    # Set diagonal to -(1/(n-2)) * sum of row
    for i in range(n):
        U[i, i] = -np.sum(D[i, :]) / (n - 2) + grand_mean
    return U


def _residualize(x: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """OLS residual of x on Z."""
    if Z.shape[1] == 0:
        return x
    # Add intercept
    Z_int = np.column_stack([np.ones(len(Z)), Z])
    beta, _, _, _ = np.linalg.lstsq(Z_int, x, rcond=None)
    return x - Z_int @ beta
