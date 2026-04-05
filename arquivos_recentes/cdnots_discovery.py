# Copyright 2025
#
# Licensed under the Apache License, Version 2.0

"""CD-NOTS causal graph discovery for MMM benchmark data.

Discovers causal structure between marketing channels and target variable
using constraint-based causal discovery (CD-NOTS fallback to PC + Granger).
Returns adjacency matrix and channel classifications used by
cdnots_model_builder to adjust priors.

Usage:
    from mmm_param_recovery.benchmarking.cdnots_discovery import discover_graph
    
    graph = discover_graph(data_df, channel_columns)
    # graph.adjacency_matrix: [n_vars, n_vars]
    # graph.direct_channels: List[str]
    # graph.excluded_channels: List[str]
"""

import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class CausalGraph:
    """Immutable result of causal discovery on MMM data."""
    adjacency_matrix: np.ndarray
    edge_pvalues: np.ndarray
    variable_names: Tuple[str, ...]
    direct_channels: Tuple[str, ...]
    excluded_channels: Tuple[str, ...]
    mediated_channels: Tuple[str, ...]
    runtime_seconds: float
    ci_test_used: str
    n_edges: int


def discover_graph(
    data_df: pd.DataFrame,
    channel_columns: List[str],
    alpha: float = 0.05,
    max_lag: int = 2,
    console: Optional[Console] = None,
) -> CausalGraph:
    """Run causal discovery on benchmark MMM data.

    Parameters
    ----------
    data_df : pd.DataFrame
        Benchmark data with MultiIndex (date, geo) or flat index.
        Must contain channel columns and 'y'.
    channel_columns : List[str]
        Channel spend column names.
    alpha : float
        Significance level for CI tests.
    max_lag : int
        Maximum temporal lag to consider.
    console : Optional[Console]
        Rich console for output.

    Returns
    -------
    CausalGraph
        Discovery results with adjacency matrix and channel classifications.
    """
    if console is None:
        console = Console()

    start = time.perf_counter()

    discovery_vars = list(channel_columns) + ["y"]
    n_vars = len(discovery_vars)

    # Extract geos
    if isinstance(data_df.index, pd.MultiIndex):
        geos = data_df.index.get_level_values("geo").unique().tolist()
        n_time = data_df.index.get_level_values("date").nunique()
    else:
        geos = ["national"]
        n_time = len(data_df)

    # Auto-select CI test
    ci_test = "parcorr" if n_time < 200 else "rcot"

    console.print(f"\n  [bold]CD-NOTS Discovery[/bold]: {n_vars} vars, {len(geos)} geos, "
                   f"{n_time} timepoints, CI test: {ci_test}, alpha: {alpha}")

    # Run per-geo discovery
    per_geo_adj = {}
    per_geo_pval = {}

    for geo in geos:
        if isinstance(data_df.index, pd.MultiIndex):
            geo_data = data_df.xs(geo, level="geo")[discovery_vars].values.astype(float)
        else:
            geo_data = data_df[discovery_vars].values.astype(float)

        adj, pval = _discover_single_geo(geo_data, n_vars, alpha, max_lag, ci_test)
        per_geo_adj[geo] = adj
        per_geo_pval[geo] = pval

    # Consensus: majority vote (>= 50% of geos agree on edge)
    stacked = np.stack(list(per_geo_adj.values()), axis=0)
    stacked_pval = np.stack(list(per_geo_pval.values()), axis=0)
    agreement = stacked.mean(axis=0)
    consensus_adj = (agreement >= 0.5).astype(float)

    # Enforce arrow of time: y (last var) does not cause channels
    y_idx = n_vars - 1
    consensus_adj[y_idx, :] = 0

    # Average p-values where edges exist
    with np.errstate(divide="ignore", invalid="ignore"):
        pval_sum = np.where(stacked > 0, stacked_pval, 0).sum(axis=0)
        edge_count = np.maximum(stacked.sum(axis=0), 1)
        consensus_pval = np.where(consensus_adj > 0, pval_sum / edge_count, 1.0)

    # Classify channels
    direct = []
    excluded = []
    mediated = []

    for i, ch in enumerate(channel_columns):
        has_direct_to_y = consensus_adj[i, y_idx] > 0
        has_any_path = _has_path(consensus_adj, i, y_idx)

        if has_direct_to_y:
            direct.append(ch)
        elif has_any_path:
            mediated.append(ch)
        else:
            excluded.append(ch)

    n_edges = int(consensus_adj.sum())
    runtime = time.perf_counter() - start

    graph = CausalGraph(
        adjacency_matrix=consensus_adj,
        edge_pvalues=consensus_pval,
        variable_names=tuple(discovery_vars),
        direct_channels=tuple(direct),
        excluded_channels=tuple(excluded),
        mediated_channels=tuple(mediated),
        runtime_seconds=runtime,
        ci_test_used=ci_test,
        n_edges=n_edges,
    )

    _print_summary(graph, channel_columns, console)
    return graph


def _discover_single_geo(
    data: np.ndarray,
    n_vars: int,
    alpha: float,
    max_lag: int,
    ci_test: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run discovery on a single geo's data. Tries PC algorithm, falls back to Granger."""
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    try:
        return _pc_with_temporal_augmentation(data, n_vars, alpha, max_lag, ci_test)
    except ImportError:
        warnings.warn("causal-learn not available, falling back to Granger causality")
        return _granger_fallback(data, n_vars, alpha, max_lag)


def _pc_with_temporal_augmentation(
    data: np.ndarray,
    n_vars: int,
    alpha: float,
    max_lag: int,
    ci_test_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """PC algorithm with temporal augmentation (CD-NOTS-style)."""
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.cit import fisherz, kci

    T = data.shape[0]
    valid_start = max_lag

    # Build augmented data: contemporary + lagged + time node
    cols = []

    # Contemporary (t)
    contemporary = data[valid_start:]
    for i in range(n_vars):
        cols.append(contemporary[:, i])

    # Lagged (t-1, t-2, ...)
    for lag in range(1, max_lag + 1):
        lagged = data[valid_start - lag: T - lag]
        for i in range(n_vars):
            cols.append(lagged[:, i])

    # Time node (normalized)
    cols.append(np.linspace(0, 1, T - valid_start))

    augmented = np.column_stack(cols)

    ci_fn = fisherz if ci_test_name in ("parcorr", "fisherz") else kci

    cg = pc(augmented, alpha=alpha, indep_test=ci_fn, stable=True,
            uc_rule=0, uc_priority=2, verbose=False)

    full_graph = cg.G.graph

    # Extract contemporaneous adjacency + lagged contributions
    adj = np.zeros((n_vars, n_vars))
    pval = np.ones((n_vars, n_vars))

    for i in range(n_vars):
        for j in range(n_vars):
            if i == j:
                continue
            # Contemporaneous edge i -> j
            if full_graph[i, j] == -1 and full_graph[j, i] == 1:
                adj[i, j] = 1
                pval[i, j] = alpha / 2

            # Lagged edges: x_{t-k} -> y_t implies x -> y
            for lag in range(1, max_lag + 1):
                lag_i = i + n_vars * lag
                if lag_i < augmented.shape[1] - 1:  # exclude time node
                    if full_graph[lag_i, j] == -1 and full_graph[j, lag_i] == 1:
                        adj[i, j] = 1
                        pval[i, j] = min(pval[i, j], alpha / 2)

    return adj, pval


def _granger_fallback(
    data: np.ndarray,
    n_vars: int,
    alpha: float,
    max_lag: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pairwise Granger causality as minimal fallback."""
    from statsmodels.tsa.stattools import grangercausalitytests

    adj = np.zeros((n_vars, n_vars))
    pval = np.ones((n_vars, n_vars))

    for i in range(n_vars):
        for j in range(n_vars):
            if i == j:
                continue
            try:
                test_data = np.column_stack([data[:, j], data[:, i]])
                results = grangercausalitytests(test_data, maxlag=max_lag, verbose=False)
                min_p = min(results[lag][0]["ssr_ftest"][1] for lag in range(1, max_lag + 1))
                pval[i, j] = min_p
                if min_p < alpha:
                    adj[i, j] = 1
            except Exception:
                pass

    return adj, pval


def _has_path(adj: np.ndarray, source: int, target: int, max_depth: int = 3) -> bool:
    """BFS to check if a directed path exists from source to target."""
    visited = {source}
    queue = [source]
    depth = 0

    while queue and depth < max_depth:
        next_queue = []
        for node in queue:
            for child in range(adj.shape[1]):
                if adj[node, child] > 0 and child not in visited:
                    if child == target:
                        return True
                    visited.add(child)
                    next_queue.append(child)
        queue = next_queue
        depth += 1

    return False


def _print_summary(graph: CausalGraph, channel_columns: List[str], console: Console) -> None:
    """Print discovery results as Rich table."""
    table = Table(title="CD-NOTS Discovery Results")
    table.add_column("Channel", style="cyan")
    table.add_column("→ y", style="green")
    table.add_column("Category", style="magenta")

    y_idx = len(graph.variable_names) - 1

    for ch in channel_columns:
        ch_idx = graph.variable_names.index(ch)
        has_direct = "✓" if graph.adjacency_matrix[ch_idx, y_idx] > 0 else "✗"

        if ch in graph.direct_channels:
            cat = "direct"
        elif ch in graph.mediated_channels:
            cat = "mediated"
        else:
            cat = "excluded"

        table.add_row(ch, has_direct, cat)

    console.print(table)
    console.print(f"  Edges: {graph.n_edges} | Direct: {list(graph.direct_channels)} | "
                   f"Excluded: {list(graph.excluded_channels)} | Runtime: {graph.runtime_seconds:.1f}s")
