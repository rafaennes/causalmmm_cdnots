"""RPCMCI: Joint regime learning + causal discovery (tigramite).

Saggioro, de Wiljes, Kretschmer & Runge (2020).

Unlike regime_pcmci.py which detects regimes first (PELT) then runs PCMCI,
RPCMCI jointly optimizes regime assignment and causal graph via alternating
MIP + PCMCI. This is more principled when the true regime structure depends
on the causal relationships themselves.

Requires: ortools (for MIP solver).
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
    num_regimes: int = 2,
    **kwargs,
) -> CausalGraph:
    """RPCMCI: joint regime + causal discovery.

    Parameters
    ----------
    data : DataFrame with columns matching var_names. Last column must be "y".
    var_names : list of column names; last element must be "y".
    alpha : significance level for PCMCI within RPCMCI.
    max_lag : maximum temporal lag.
    num_regimes : number of regimes to learn.
    """
    start = time.perf_counter()
    X = StandardScaler().fit_transform(data[var_names].values)
    T, n = X.shape
    y_idx = n - 1

    from tigramite import data_processing as pp
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.rpcmci import RPCMCI

    # ponytail: ParCorr only — RPCMCI runs PCMCI many times internally,
    # CMIknn would be prohibitively slow across annealing iterations
    dataframe = pp.DataFrame(X, var_names=list(range(n)))
    rpcmci = RPCMCI(dataframe=dataframe, cond_ind_test=ParCorr(), verbosity=0)

    # max_transitions: T//num_regimes is a reasonable default — allows
    # roughly one transition per regime-length window
    max_transitions = max(T // num_regimes, 2)

    result = rpcmci.run_rpcmci(
        num_regimes=num_regimes,
        max_transitions=max_transitions,
        tau_min=1,
        tau_max=max_lag,
        pc_alpha=alpha,
        alpha_level=alpha,
        n_jobs=1,  # ponytail: single-threaded to avoid nested parallelism issues
        num_iterations=10,
        max_anneal=3,  # ponytail: fewer annealings for speed; increase if quality matters
    )

    if result is None:
        # All annealings failed — return empty graph
        return CausalGraph(
            adjacency_matrix=np.zeros((n, n)),
            variable_names=tuple(var_names),
            algorithm="rpcmci",
            runtime_seconds=time.perf_counter() - start,
            metadata={"pvalues": np.ones((n, n)), "regimes": None,
                       "num_regimes": num_regimes, "error_free_annealings": 0},
        )

    regimes_raw = result["regimes"]
    causal_results = result["causal_results"]
    error_free = result["error_free_annealings"]

    # causal_results is the PCMCI results dict from the best annealing
    adj = np.zeros((n, n))
    pval = np.ones((n, n))

    p_matrix = causal_results.get("p_matrix", None) if isinstance(causal_results, dict) else None

    if p_matrix is not None:
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                min_p = float(p_matrix[i, j, 1 : max_lag + 1].min())
                pval[i, j] = min_p
                if min_p < alpha:
                    adj[i, j] = 1.0
    else:
        # Fallback: extract from graph string matrix
        graph_matrix = (causal_results.get("graph", None)
                        if isinstance(causal_results, dict) else None)
        if graph_matrix is not None:
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    for tau in range(1, max_lag + 1):
                        edge = graph_matrix[i, j, tau]
                        if isinstance(edge, str) and edge.endswith(">"):
                            adj[i, j] = 1.0
                            break

    adj[y_idx, :] = 0.0  # y does not cause anything

    # Decode regime assignments from one-hot
    if regimes_raw is not None and hasattr(regimes_raw, 'ndim') and regimes_raw.ndim == 2:
        regime_labels = np.argmax(regimes_raw, axis=0)
    else:
        regime_labels = regimes_raw

    return CausalGraph(
        adjacency_matrix=adj,
        variable_names=tuple(var_names),
        algorithm="rpcmci",
        runtime_seconds=time.perf_counter() - start,
        metadata={
            "pvalues": pval,
            "regimes": regime_labels,
            "num_regimes": num_regimes,
            "error_free_annealings": error_free,
        },
    )
