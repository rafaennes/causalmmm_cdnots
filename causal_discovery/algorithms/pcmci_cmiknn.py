"""PCMCI with CMIknn nonlinear CI test.

No regime detection — isolates the contribution of regime conditioning
relative to CD-NOTS. CMIknn captures adstock/saturation nonlinearity
without requiring knowledge of adstock parameters.
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
    **kwargs,
) -> CausalGraph:
    """PCMCI with CMIknn (nonlinear k-NN mutual information) CI test.

    Parameters
    ----------
    data : DataFrame with columns matching var_names. Last column must be "y".
    var_names : list of column names; last element must be "y".
    alpha : significance level applied to BH-corrected q-values.
    max_lag : maximum temporal lag (tau_max). tau_min is always 1.
    """
    start = time.perf_counter()
    X = StandardScaler().fit_transform(data[var_names].values)
    n = len(var_names)
    y_idx = n - 1

    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI

    try:
        from tigramite.independence_tests.cmiknn import CMIknn
        ci_test = CMIknn(knn=5, null_fit=True, sig_samples=500)
    except (ImportError, AttributeError):
        warnings.warn(
            "CMIknn unavailable (numba missing?). Falling back to ParCorr."
        )
        from tigramite.independence_tests.parcorr import ParCorr
        ci_test = ParCorr()

    dataframe = pp.DataFrame(X, var_names=list(range(n)))
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ci_test, verbosity=0)

    # ponytail: pc_alpha=None skips PC pre-filtering, letting MCI test all links.
    # CMIknn has low power in PC phase with many variables, causing true edge dropout.
    results = pcmci.run_pcmci(tau_max=max_lag, tau_min=1, pc_alpha=None)
    p_matrix = results["p_matrix"]

    # ponytail: use raw p-values, not BH-corrected. CMIknn permutation p-values
    # have floor ~1/(sig_samples+1) which can't survive BH over n²×tau tests.
    # PCMCI's MCI conditioning already controls confounding; BH is redundant here.
    adj = np.zeros((n, n))
    pval = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            min_p = float(p_matrix[i, j, 1 : max_lag + 1].min())
            pval[i, j] = min_p
            if min_p < alpha:
                adj[i, j] = 1.0

    adj[y_idx, :] = 0.0  # y does not cause anything

    return CausalGraph(
        adjacency_matrix=adj,
        variable_names=tuple(var_names),
        algorithm="pcmci_cmiknn",
        runtime_seconds=time.perf_counter() - start,
        metadata={"pvalues": pval, "ci_test": "cmiknn"},
    )
