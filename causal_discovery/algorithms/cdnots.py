"""CD-NOTS: Causal Discovery from Nonstationary Time Series via surrogate variable.

Zhang, Huang, Zhang, Glymour & Schölkopf (2017) / Sadeghi, Gopal & Fesanghary (2024).

Key idea: nonstationarity is not noise — it's signal. A smooth surrogate
variable C (the time index) is appended to the variable set. CI tests on
V ∪ {C} exploit distributional shifts to:
  Phase 1: Recover the causal skeleton (edges conditional on V ∪ {C}).
  Phase 2: Orient edges using C-adjacency patterns. Variables adjacent to C
           have changing mechanisms. If V_k is C-adjacent and V_l is not,
           then C - V_k - V_l is an unshielded triple that orients V_k -> V_l.

This differs from regime_pcmci.py which discretizes time into regimes first.
Here C is the continuous (normalized) time index — preserving smooth trends
that discrete regimes would lose.
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
    """CD-NOTS: surrogate-variable PCMCI for nonstationary time series.

    Parameters
    ----------
    data : DataFrame with columns matching var_names. Last column must be "y".
    var_names : list of column names; last element must be "y".
    alpha : significance level applied to BH-corrected q-values.
    max_lag : maximum temporal lag (tau_max). tau_min is always 1.
    """
    start = time.perf_counter()
    X = StandardScaler().fit_transform(data[var_names].values)
    T, n = X.shape
    y_idx = n - 1

    # Phase 1: Append normalized time index as surrogate variable C
    # Smooth, continuous — captures trends without discretization loss
    C = np.linspace(0, 1, T).reshape(-1, 1)
    X_aug = np.hstack([X, C])
    n_aug = n + 1
    c_idx = n  # index of surrogate C in augmented matrix

    # Link assumptions: C is exogenous (nothing causes C)
    link_assumptions = _build_link_assumptions(n, c_idx, max_lag)

    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI

    try:
        from tigramite.independence_tests.cmiknn import CMIknn
        # sig_blocklength=1: the deterministic surrogate C has phi=1 which
        # causes NaN in automatic block length. Force simple permutations.
        ci_test = CMIknn(knn=5, sig_samples=500, sig_blocklength=1)
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

    # Phase 2: Identify C-adjacent variables and orient edges
    # A variable is "C-specific" if C -> V_i is significant (its mechanism changes over time)
    c_adjacent = set()
    for j in range(n):
        min_q_c = float(q_matrix[c_idx, j, 1 : max_lag + 1].min())
        if min_q_c < alpha:
            c_adjacent.add(j)

    # Extract n×n submatrix (strip C row/col)
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

    # Phase 2 orientation: use C-adjacency to resolve undirected edges
    # If i is C-adjacent and j is not, and both i->j and j->i are present,
    # orient as i->j (C - i - j unshielded triple implies i -> j)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if adj[i, j] == 1 and adj[j, i] == 1:
                if i in c_adjacent and j not in c_adjacent:
                    adj[j, i] = 0.0  # orient i -> j
                elif j in c_adjacent and i not in c_adjacent:
                    adj[i, j] = 0.0  # orient j -> i
                # ponytail: if both or neither are C-adjacent, leave bidirectional

    adj[y_idx, :] = 0.0  # y does not cause anything

    return CausalGraph(
        adjacency_matrix=adj,
        variable_names=tuple(var_names),
        algorithm="cdnots",
        runtime_seconds=time.perf_counter() - start,
        metadata={
            "pvalues": pval,
            "qvalues": qval,
            "c_adjacent_vars": [var_names[i] for i in c_adjacent],
            "n_c_adjacent": len(c_adjacent),
        },
    )


# --- Internal helpers ---------------------------------------------------------


def _build_link_assumptions(
    n_original: int, c_idx: int, max_lag: int
) -> dict:
    """PCMCI link assumptions for CD-NOTS augmented graph.

    Allowed (tested):
      - Any variable including C -> original variables (lagged)
      - C -> original variables at lag 0 is NOT tested (tau_min=1)
    Forbidden (skipped):
      - Anything -> C (surrogate time index is exogenous)
    """
    n_aug = n_original + 1
    lags = [-tau for tau in range(1, max_lag + 1)]
    la: dict = {j: {} for j in range(n_aug)}

    for j in range(n_original):
        for i in range(n_aug):  # includes C
            for lag in lags:
                la[j][(i, lag)] = "-?>"

    # c_idx: no incoming edges — leave la[c_idx] = {} (empty)
    return la
