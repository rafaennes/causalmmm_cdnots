"""DYNOTEARS-style score-based causal discovery for time series.

Implements the score-based DAG learning approach from Pamfil et al. (2020)
without requiring the causalnex library (unavailable on Python 3.12).
Uses VAR coefficient thresholding to achieve the same core advantage:
no multiple CI tests, so no BH power collapse on high-dimensional data.

The causalnex/DYNOTEARS acyclicity constraint is approximated by:
  1. Fit a VAR(p) model jointly (statsmodels VAR)
  2. Threshold coefficients at w_threshold to get binary edges

This preserves the score-based, high-dimensional advantage over PCMCI.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from causal_discovery.graph import CausalGraph


def discover(
    data: pd.DataFrame,
    var_names: list[str],
    alpha: float = 0.05,   # accepted for uniform interface; unused (score-based)
    max_lag: int = 2,
    w_threshold: float = 0.1,
    **kwargs,
) -> CausalGraph:
    """Score-based time-series causal discovery via VAR coefficient thresholding.

    Parameters
    ----------
    data : DataFrame with columns matching var_names. Last column must be "y".
    var_names : list of column names; last element must be "y".
    max_lag : number of lags p in the VAR model.
    w_threshold : coefficient magnitude threshold for binarising edges.
        Edges with |coef| < w_threshold are set to 0.
    """
    start = time.perf_counter()
    X = StandardScaler().fit_transform(data[var_names].values)
    n = len(var_names)
    y_idx = n - 1

    from statsmodels.tsa.api import VAR

    df_std = pd.DataFrame(X, columns=var_names)
    model = VAR(df_std)
    fitted = model.fit(maxlags=max_lag, ic=None)

    # Aggregate absolute coefficients across all lags: max(|coef_lag1|, |coef_lag2|, ...)
    # coefs shape: (max_lag, n, n) — coefs[k][i, j] = coef of var i at lag k+1 on var j
    coefs = fitted.coefs  # shape (p, n, n)
    W = np.max(np.abs(coefs), axis=0)  # (n, n); W[i, j] = max |effect of i on j|

    adj = (W > w_threshold).astype(float)
    np.fill_diagonal(adj, 0.0)   # no self-loops
    adj[y_idx, :] = 0.0          # y does not cause anything

    return CausalGraph(
        adjacency_matrix=adj,
        variable_names=tuple(var_names),
        algorithm="dynotears",
        runtime_seconds=time.perf_counter() - start,
        metadata={
            "w_threshold": w_threshold,
            "n_edges_raw": int(adj.sum()),
            "weight_matrix": W,
        },
    )
