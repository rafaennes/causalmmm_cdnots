"""LPCMCI: PCMCI variant tolerant of latent confounders.

Outputs a Partial Ancestral Graph (PAG) rather than a DAG.
Converted to binary adjacency: edge present if graph mark ends with ">".
More conservative than PCMCI; better when unmeasured variables
(competitor spend, brand equity) confound observed channels.
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
    alpha: float = 0.05,
    max_lag: int = 2,
    **kwargs,
) -> CausalGraph:
    """LPCMCI with ParCorr CI test.

    Parameters
    ----------
    data : DataFrame with columns matching var_names. Last column must be "y".
    var_names : list of column names; last element must be "y".
    alpha : significance level for PC and MCI phases.
    max_lag : maximum temporal lag (tau_max). tau_min is always 1.
    """
    start = time.perf_counter()
    X = StandardScaler().fit_transform(data[var_names].values)
    n = len(var_names)
    y_idx = n - 1

    from tigramite import data_processing as pp
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.lpcmci import LPCMCI

    dataframe = pp.DataFrame(X, var_names=list(range(n)))
    lpcmci_obj = LPCMCI(dataframe=dataframe, cond_ind_test=ParCorr(), verbosity=0)

    results = lpcmci_obj.run_lpcmci(tau_max=max_lag, tau_min=1, pc_alpha=alpha)
    # graph_matrix[i, j, tau]: string edge mark of i(t-tau) w.r.t. j(t).
    # "-->" = directed i->j; "o->" = uncertain tail, arrow to j; "<->" = bidirected.
    # We treat any mark ending with ">" as i causes j (conservative inclusion).
    graph_matrix = results["graph"]  # shape (n, n, tau_max+1)

    adj = np.zeros((n, n))
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

    return CausalGraph(
        adjacency_matrix=adj,
        variable_names=tuple(var_names),
        algorithm="lpcmci",
        runtime_seconds=time.perf_counter() - start,
        metadata={"graph_matrix": graph_matrix},
    )
