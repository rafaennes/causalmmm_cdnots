"""Faithful CD-NOTS: Constraint-based Causal Discovery from Nonstationary Time Series.

Sadeghi, A., Gopal, A., & Fesanghary, M. (2024).
International Journal of Data Science and Analytics.

Key innovation over vanilla PCMCI: regime detection via PELT (ruptures).
The regime indicator is appended as a context variable. PCMCI then conditions
on regime, exploiting distributional shifts for causal identification instead
of treating nonstationarity as noise.

Pipeline:
  1. Detect regime boundaries (PELT, RBF kernel).
  2. Encode regime as integer context column appended to data.
  3. PCMCI with CMIknn on augmented data; regime can cause vars, not vice versa.
  4. Strip regime from output adjacency matrix.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from causal_discovery.graph import CausalGraph


def discover(
    data: pd.DataFrame,
    var_names: list[str],
    alpha: float = 0.05,
    max_lag: int = 2,
    n_regimes: int | None = None,
    **kwargs,
) -> CausalGraph:
    """CD-NOTS: regime-aware PCMCI with CMIknn.

    Parameters
    ----------
    data : DataFrame with columns matching var_names. Last column must be "y".
    var_names : list of column names; last element must be "y".
    alpha : significance level applied to BH-corrected q-values.
    max_lag : maximum temporal lag (tau_max). tau_min is always 1.
    n_regimes : number of regimes to detect. None = auto via BIC-like penalty.
    """
    start = time.perf_counter()
    X = StandardScaler().fit_transform(data[var_names].values)
    n = len(var_names)
    y_idx = n - 1

    # 1. Detect regimes
    regimes = _detect_regimes(X, n_bkps=n_regimes)
    n_reg = len(set(regimes))

    # 2. Augment data: append regime as integer context column
    X_aug = np.hstack([X, regimes.reshape(-1, 1).astype(float)])
    n_aug = n + 1
    regime_idx = n  # index of regime column in augmented matrix

    # 3. Build link assumptions: regime -> vars; nothing -> regime
    link_assumptions = _build_link_assumptions(n, regime_idx, max_lag)

    # 4. PCMCI with CMIknn on augmented data
    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI

    try:
        from tigramite.independence_tests.cmiknn import CMIknn
        ci_test = CMIknn(knn=5, null_fit=True, sig_samples=200)
    except (ImportError, AttributeError):
        warnings.warn("CMIknn unavailable. Falling back to ParCorr for CD-NOTS.")
        from tigramite.independence_tests.parcorr import ParCorr
        ci_test = ParCorr()

    dataframe = pp.DataFrame(X_aug, var_names=list(range(n_aug)))
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ci_test, verbosity=0)
    results = pcmci.run_pcmci(
        tau_max=max_lag,
        tau_min=1,
        pc_alpha=alpha,
        link_assumptions=link_assumptions,
    )
    p_matrix = results["p_matrix"]
    q_matrix = pcmci.get_corrected_pvalues(
        p_matrix=p_matrix, tau_min=1, tau_max=max_lag, fdr_method="fdr_bh"
    )

    # 5. Extract n×n submatrix (strip regime row/col)
    adj = np.zeros((n, n))
    pval = np.ones((n, n))
    qval = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            min_p = float(p_matrix[i, j, 1 : max_lag + 1].min())
            min_q = float(q_matrix[i, j, 1 : max_lag + 1].min())
            pval[i, j] = min_p
            qval[i, j] = min_q
            if min_q < alpha:
                adj[i, j] = 1.0

    adj[y_idx, :] = 0.0  # y does not cause anything

    return CausalGraph(
        adjacency_matrix=adj,
        variable_names=tuple(var_names),
        algorithm="cdnots",
        runtime_seconds=time.perf_counter() - start,
        metadata={
            "pvalues": pval,
            "qvalues": qval,
            "regime_boundaries": _get_boundaries(regimes),
            "n_regimes": n_reg,
        },
    )


# --- Internal helpers ---------------------------------------------------------


def _detect_regimes(X: np.ndarray, n_bkps: int | None = None) -> np.ndarray:
    """Detect regime changes via PELT with RBF cost (ruptures).

    RBF kernel detects both mean and variance shifts — appropriate for
    marketing data where campaign periods change spend dispersion, not just level.

    Returns integer array of length T where each value is a regime index.
    """
    import ruptures as rpt

    T = X.shape[0]
    algo = rpt.Pelt(model="rbf", min_size=10).fit(X)

    if n_bkps is not None:
        # n_bkps here is n_regimes; Binseg takes n_bkps = n_regimes - 1
        n_breakpoints = max(n_bkps - 1, 1)
        bkps = rpt.Binseg(model="rbf", min_size=10).fit(X).predict(n_bkps=n_breakpoints)
    else:
        # BIC-like automatic penalty: log(T) * log(N) balances fit vs complexity.
        pen = np.log(T) * np.log(max(X.shape[1], 2))
        bkps = algo.predict(pen=pen)

    # ruptures convention: bkps[-1] == T always (end-of-series sentinel).
    regimes = np.zeros(T, dtype=int)
    prev = 0
    for r, bp in enumerate(bkps[:-1]):
        regimes[prev:bp] = r
        prev = bp
    regimes[prev:] = len(bkps) - 1
    return regimes


def _get_boundaries(regimes: np.ndarray) -> list[tuple[int, int]]:
    """Return list of (start_idx, end_idx) inclusive for each regime."""
    T = len(regimes)
    boundaries: list[tuple[int, int]] = []
    start = 0
    for t in range(1, T):
        if regimes[t] != regimes[t - 1]:
            boundaries.append((start, t - 1))
            start = t
    boundaries.append((start, T - 1))
    return boundaries


def _build_link_assumptions(
    n_original: int, regime_idx: int, max_lag: int
) -> dict:
    """PCMCI link assumptions for CD-NOTS augmented graph.

    Allowed (tested):
      - Any variable including regime -> original variables (lagged)
    Forbidden (skipped):
      - Anything -> regime (regime is exogenous context)
    """
    n_aug = n_original + 1
    lags = [-tau for tau in range(1, max_lag + 1)]
    la: dict = {j: {} for j in range(n_aug)}

    for j in range(n_original):
        for i in range(n_aug):  # includes regime
            for lag in lags:
                la[j][(i, lag)] = "?->"

    # regime_idx: no incoming edges — leave la[regime_idx] = {} (empty)
    return la
