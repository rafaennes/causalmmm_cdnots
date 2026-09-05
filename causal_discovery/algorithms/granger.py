"""Granger VAR baseline causal discovery.

Pairwise Granger causality tests via statsmodels, BH-corrected.
Linear and stationary assumption. Serves as the performance floor.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import grangercausalitytests

from causal_discovery.graph import CausalGraph, bh_correct


def discover(
    data: pd.DataFrame,
    var_names: list[str],
    alpha: float = 0.05,
    max_lag: int = 2,
    **kwargs,
) -> CausalGraph:
    """Pairwise Granger causality with Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    data : DataFrame with columns matching var_names. Last column must be "y".
    var_names : list of column names; last element must be "y".
    alpha : FDR significance level for BH correction.
    max_lag : maximum temporal lag to test.
    """
    start = time.perf_counter()
    X = StandardScaler().fit_transform(data[var_names].values)
    n = len(var_names)
    y_idx = n - 1

    pval = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            try:
                # grangercausalitytests tests: does X[:,i] Granger-cause X[:,j]?
                test_data = np.column_stack([X[:, j], X[:, i]])
                import io, sys as _sys
                _old = _sys.stdout
                _sys.stdout = io.StringIO()  # ponytail: suppress statsmodels print
                try:
                    results = grangercausalitytests(test_data, maxlag=max_lag)
                finally:
                    _sys.stdout = _old
                pval[i, j] = min(
                    results[lag][0]["ssr_ftest"][1] for lag in range(1, max_lag + 1)
                )
            except Exception:
                pass  # leave pval[i,j] = 1.0 (no edge)

    adj = bh_correct(pval, alpha)
    adj[y_idx, :] = 0.0  # y does not cause anything

    return CausalGraph(
        adjacency_matrix=adj,
        variable_names=tuple(var_names),
        algorithm="granger",
        runtime_seconds=time.perf_counter() - start,
        metadata={"pvalues": pval},
    )
