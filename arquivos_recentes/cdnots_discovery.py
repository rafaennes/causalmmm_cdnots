# Copyright 2025
#
# Licensed under the Apache License, Version 2.0

"""Causal graph discovery for MMM benchmark data.

VERSION 3: Replaces PC + temporal augmentation with PCMCI (tigramite) as
the primary discovery algorithm. PC is kept as a secondary fallback.

Why PCMCI over PC + temporal augmentation:
    - PC on the augmented matrix (vars × lags columns) explodes with
      dimensionality — false-positive rate grows with number of variables.
    - PCMCI conditions on parents of both endpoints (MCI test), which keeps
      conditioning sets small and dramatically reduces false positives.
    - PCMCI handles time series autocorrelation explicitly; PC does not.

Key features:
    - Primary: PCMCI via tigramite (ParCorr or CMIknn CI tests).
    - Fallback 1: PC + temporal augmentation via causal-learn.
    - Fallback 2: pairwise Granger causality via statsmodels.
    - control_columns: includes controls to detect endogeneity.
    - Geo sampling: max 5 geos (rest is redundant for discovery).
    - CausalGraph includes endogenous_channels, endogenous_r2, control info.

Usage:
    graph = discover_graph(
        data_df, channel_columns,
        control_columns=["c1", "c2"],
        alpha=0.05, max_lag=2, console=console,
        ci_test="auto",  # "auto" | "parcorr" | "kci"
    )
    # graph.endogenous_channels: channels confounded by a control
"""

import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class CausalGraph:
    """Immutable result of causal discovery on MMM data."""
    adjacency_matrix: np.ndarray
    edge_pvalues: np.ndarray   # raw MCI p-values — display/debug only
    edge_qvalues: np.ndarray   # BH-corrected q-values — used for prior calibration
    variable_names: Tuple[str, ...]
    direct_channels: Tuple[str, ...]
    excluded_channels: Tuple[str, ...]
    mediated_channels: Tuple[str, ...]
    endogenous_channels: Tuple[str, ...]   # NEW: channels confounded by controls
    runtime_seconds: float
    ci_test_used: str
    n_edges: int
    # Control info
    endogenous_r2: Dict[str, float] = field(default_factory=dict)  # NEW: R2 for endogenous channels
    control_names: Tuple[str, ...] = ()
    control_to_channel_edges: Tuple[Tuple[str, str], ...] = ()  # (control, channel) pairs


def _resolve_data_format(data_df: pd.DataFrame) -> pd.DataFrame:
    """Convert flat DataFrame with geo column to (date, geo) MultiIndex.

    prepare_dataset_for_modeling() returns a flat DataFrame where 'geo' and
    'time'/'date' are regular columns. discover_graph() expects a MultiIndex
    so it can split geos correctly. Without this, all geos are concatenated
    into a single 'national' pseudo-series, which breaks PCMCI and triggers
    KCI instead of parcorr.

    Normalises the time level name to 'date' for consistency with downstream
    code that calls get_level_values('date').
    """
    if isinstance(data_df.index, pd.MultiIndex):
        return data_df
    time_col = next((c for c in ("date", "time") if c in data_df.columns), None)
    if time_col is not None and "geo" in data_df.columns:
        df = data_df.set_index([time_col, "geo"])
        df.index.names = ["date", "geo"]
        return df
    return data_df


def discover_graph(
    data_df: pd.DataFrame,
    channel_columns: List[str],
    control_columns: Optional[List[str]] = None,
    alpha: float = 0.05,
    max_lag: int = 1,
    max_geos: int = 5,
    ci_test: str = "auto",
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
    control_columns : List[str], optional
        Control variable column names. If provided, included in the
        causal graph to detect endogeneity (control → channel confounding).
    alpha : float
        Significance level for CI tests.
    max_lag : int
        Maximum temporal lag to consider. Default 1 (faster).
    max_geos : int
        Maximum number of geos to use for discovery. Default 5.
    ci_test : str
        Teste de independência condicional. "auto" (default) escolhe
        kci (não-linear) se houver exatamente 1 geo efetivo, senão parcorr
        (linear Fisher-Z, muito mais rápido). Pode ser forçado para
        "parcorr" ou "kci" explicitamente.
    console : Optional[Console]
        Rich console for output.

    Returns
    -------
    CausalGraph
        Discovery results with channel classifications and endogeneity info.
    """
    if console is None:
        console = Console()
    if control_columns is None:
        control_columns = []

    data_df = _resolve_data_format(data_df)

    start = time.perf_counter()

    # Build discovery variable list: channels + controls + y
    # Controls go BEFORE y so that control→channel and control→y edges
    # are both discoverable
    discovery_vars = list(channel_columns) + list(control_columns) + ["y"]
    n_vars = len(discovery_vars)
    n_channels = len(channel_columns)
    n_controls = len(control_columns)
    y_idx = n_vars - 1

    # Index ranges
    channel_indices = list(range(n_channels))
    control_indices = list(range(n_channels, n_channels + n_controls))

    # Extract geos
    if isinstance(data_df.index, pd.MultiIndex):
        all_geos = data_df.index.get_level_values("geo").unique().tolist()
        n_time = data_df.index.get_level_values("date").nunique()
    else:
        all_geos = ["national"]
        n_time = len(data_df)

    # Sample geos if too many (discovery on 50 geos is redundant)
    if len(all_geos) > max_geos:
        np.random.seed(42)
        geos = list(np.random.choice(all_geos, max_geos, replace=False))
    else:
        geos = all_geos

    # ─── Seleção adaptativa do teste de independência condicional ──────────────
    # Política:
    #   • 1 geo  → kci (kernel-based, captura efeitos não-lineares de
    #              saturação/adstock; custo O(N³) aceitável pois N é pequeno).
    #   • N geos → parcorr (Fisher-Z linear, 100-1000× mais rápido; a
    #              replicação entre geos compensa em parte a perda de
    #              sensibilidade a não-linearidades).
    #
    # Como configurar:
    #   discover_graph(..., ci_test="auto")     # padrão, aplica regra acima
    #   discover_graph(..., ci_test="kci")      # força kci em qualquer cenário
    #   discover_graph(..., ci_test="parcorr")  # força parcorr em qualquer cenário
    # ────────────────────────────────────────────────────────────────────────────
    if ci_test == "auto":
        ci_test = "kci" if len(geos) == 1 else "parcorr"
    elif ci_test not in ("parcorr", "kci"):
        raise ValueError(
            f"ci_test deve ser 'auto', 'parcorr' ou 'kci' (recebido: {ci_test!r})"
        )

    console.print(
        f"\n  [bold]CD-NOTS Discovery v2[/bold]: {n_channels} channels + "
        f"{n_controls} controls + y = {n_vars} vars, "
        f"{len(geos)}/{len(all_geos)} geos, "
        f"{n_time} timepoints, CI: {ci_test}, alpha: {alpha}, max_lag: {max_lag}"
    )

    # Run per-geo discovery
    per_geo_adj = {}
    per_geo_pval = {}
    per_geo_qval = {}

    for geo in geos:
        if isinstance(data_df.index, pd.MultiIndex):
            geo_data = data_df.xs(geo, level="geo")[discovery_vars].values.astype(float)
        else:
            geo_data = data_df[discovery_vars].values.astype(float)

        adj, pval, qval = _discover_single_geo(
            geo_data, n_vars, alpha, max_lag, ci_test, n_channels, n_controls
        )
        per_geo_adj[geo] = adj
        per_geo_pval[geo] = pval
        per_geo_qval[geo] = qval

    # Consensus: majority vote on BH-corrected q-matrix (edge decisions)
    stacked = np.stack(list(per_geo_adj.values()), axis=0)
    stacked_pval = np.stack(list(per_geo_pval.values()), axis=0)
    stacked_qval = np.stack(list(per_geo_qval.values()), axis=0)
    agreement = stacked.mean(axis=0)
    consensus_adj = (agreement >= 0.5).astype(float)

    # Enforce constraints:
    # 1. y does not cause anything
    consensus_adj[y_idx, :] = 0
    # 2. channels do not cause controls (controls are exogenous by assumption)
    for ci in channel_indices:
        for cj in control_indices:
            consensus_adj[ci, cj] = 0

    # Average p-values and q-values where edges exist
    with np.errstate(divide="ignore", invalid="ignore"):
        pval_sum = np.where(stacked > 0, stacked_pval, 0).sum(axis=0)
        qval_sum = np.where(stacked > 0, stacked_qval, 0).sum(axis=0)
        edge_count = np.maximum(stacked.sum(axis=0), 1)
        consensus_pval = np.where(consensus_adj > 0, pval_sum / edge_count, 1.0)
        consensus_qval = np.where(consensus_adj > 0, qval_sum / edge_count, 1.0)

    # ============================================================
    # Classify channels
    # ============================================================
    direct = []
    excluded = []
    mediated = []
    endogenous = []
    graph_endogenous_r2 = {}
    control_to_channel = []

    for i, ch in enumerate(channel_columns):
        has_direct_to_y = consensus_adj[i, y_idx] > 0
        has_any_path = _has_path(consensus_adj, i, y_idx)

        if has_direct_to_y:
            direct.append(ch)
        elif has_any_path:
            mediated.append(ch)
        else:
            excluded.append(ch)

        # Check endogeneity: does any control cause this channel?
        for j, ctrl in enumerate(control_columns):
            ctrl_idx = n_channels + j
            if consensus_adj[ctrl_idx, i] > 0:
                endogenous.append(ch)
                control_to_channel.append((ctrl, ch))
                
                # Calculate R2 (data-driven penalty)
                try:
                    # Use full dataset for stable R2
                    c1 = data_df[ctrl].values
                    c2 = data_df[ch].values
                    r = np.corrcoef(c1, c2)[0, 1]
                    r2 = float(r**2)
                    graph_endogenous_r2[ch] = r2
                except Exception:
                    graph_endogenous_r2[ch] = 0.5  # safe fallback
                break  # one confounder is enough to flag

    n_edges = int(consensus_adj.sum())
    runtime = time.perf_counter() - start

    graph = CausalGraph(
        adjacency_matrix=consensus_adj,
        edge_pvalues=consensus_pval,
        edge_qvalues=consensus_qval,
        variable_names=tuple(discovery_vars),
        direct_channels=tuple(direct),
        excluded_channels=tuple(excluded),
        mediated_channels=tuple(mediated),
        endogenous_channels=tuple(endogenous),
        endogenous_r2=graph_endogenous_r2,
        runtime_seconds=runtime,
        ci_test_used=ci_test,
        n_edges=n_edges,
        control_names=tuple(control_columns),
        control_to_channel_edges=tuple(
            (ctrl, ch) for ctrl, ch in control_to_channel
        ),
    )

    _print_summary(graph, channel_columns, control_columns, console)
    return graph


def _discover_single_geo(
    data: np.ndarray,
    n_vars: int,
    alpha: float,
    max_lag: int,
    ci_test: str,
    n_channels: int = 0,
    n_controls: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run discovery on a single geo. PCMCI primary, PC fallback, Granger last resort.

    Returns (adj, pval, qval). For PC and Granger fallbacks qval=pval (no FDR
    correction available); this is documented and conservative.
    """
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    try:
        return _pcmci_discovery(data, n_vars, alpha, max_lag, ci_test,
                                n_channels, n_controls)
    except ImportError:
        warnings.warn(
            "tigramite not available, falling back to PC + temporal augmentation. "
            "Install with: pip install tigramite"
        )
        try:
            adj, pval = _pc_fallback(data, n_vars, alpha, max_lag, ci_test)
            return adj, pval, pval
        except ImportError:
            warnings.warn("causal-learn not available either, using Granger causality")
            adj, pval = _granger_fallback(data, n_vars, alpha, max_lag)
            return adj, pval, pval


def _pcmci_discovery(
    data: np.ndarray,
    n_vars: int,
    alpha: float,
    max_lag: int,
    ci_test_name: str,
    n_channels: int = 0,
    n_controls: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Primary discovery: PCMCI via tigramite with BH FDR correction.

    Lower false-positive rate than PC + temporal augmentation because:
    - MCI test conditions on parents of both endpoints → small conditioning
      sets regardless of variable count.
    - Handles time-series autocorrelation explicitly.
    - BH correction controls FDR across the ~n_vars² simultaneous CI tests.

    Parameters
    ----------
    data : np.ndarray
        Standardised (T, n_vars) array for one geo.
    n_vars : int
        Number of discovery variables (channels + controls + y).
    alpha : float
        Significance threshold applied to BH-corrected q-values.
    max_lag : int
        Maximum temporal lag (tau_max). tau_min is always 1 (lagged only).
    ci_test_name : str
        "parcorr" → ParCorr (linear Fisher-Z, fast).
        "kci"     → CMIknn (k-NN nonparametric, handles adstock/saturation).

    Returns
    -------
    adj : np.ndarray (n_vars, n_vars)
        adj[i, j] = 1 if q_ij < alpha (BH-corrected decision).
    pval : np.ndarray (n_vars, n_vars)
        Minimum raw MCI p-value across lags; 1.0 where no edge. For display.
    qval : np.ndarray (n_vars, n_vars)
        Minimum BH q-value across lags; 1.0 where no edge. Used for prior calibration.
    """
    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI

    from tigramite.independence_tests.parcorr import ParCorr

    if ci_test_name in ("parcorr", "fisherz"):
        cond_ind_test = ParCorr()
    else:
        # CMIknn: nonparametric k-NN CMI estimator — captures nonlinear
        # adstock/saturation effects without the O(N³) cost of kernel KCI.
        # Requires numba; falls back to ParCorr if numba is broken/missing.
        try:
            from tigramite.independence_tests.cmiknn import CMIknn
            cond_ind_test = CMIknn(knn=5, null_fit=True, sig_samples=200)
        except (ImportError, AttributeError):
            warnings.warn(
                "CMIknn unavailable (numba not importable). "
                "Falling back to ParCorr for kci mode."
            )
            cond_ind_test = ParCorr()

    # tigramite expects shape (T, N) — pass only the n_vars contemporary cols
    dataframe = pp.DataFrame(data[:, :n_vars], var_names=list(range(n_vars)))
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test, verbosity=0)

    # B: build targeted link_assumptions when structural info is available
    link_assumptions = None
    if n_channels > 0:
        link_assumptions = _build_mmm_link_assumptions(
            n_channels, n_controls, n_vars, max_lag
        )

    # A: max_conds_dim=4 caps k-NN dimensionality (curse of dimensionality);
    # link_assumptions restricts the PC and MCI phases to MMM-relevant edges.
    # tau_min=1: only lagged links — contemporaneous edges are ambiguous in MMM
    results = pcmci.run_pcmci(
        tau_max=max_lag,
        tau_min=1,
        pc_alpha=alpha,
        max_conds_dim=4,
        link_assumptions=link_assumptions,
    )

    # p_matrix[i, j, tau] = MCI p-value of X_i(t-tau) → X_j(t)
    # Shape: (n_vars, n_vars, tau_max+1); index 0 (contemporaneous) = 1.0
    p_matrix = results["p_matrix"]

    # BH correction across all (i, j, tau) tests — controls FDR, not FWER.
    # Reference: Benjamini & Hochberg (1995), JRSS-B 57(1):289-300.
    q_matrix = pcmci.get_corrected_pvalues(
        p_matrix=p_matrix,
        tau_min=1,
        tau_max=max_lag,
        fdr_method="fdr_bh",
    )

    adj = np.zeros((n_vars, n_vars))
    pval = np.ones((n_vars, n_vars))
    qval = np.ones((n_vars, n_vars))

    for i in range(n_vars):
        for j in range(n_vars):
            if i == j:
                continue
            min_p = float(p_matrix[i, j, 1 : max_lag + 1].min())
            min_q = float(q_matrix[i, j, 1 : max_lag + 1].min())
            pval[i, j] = min_p
            qval[i, j] = min_q
            if min_q < alpha:   # edge decision uses BH-corrected q, not raw p
                adj[i, j] = 1.0

    return adj, pval, qval


def _pc_fallback(
    data: np.ndarray,
    n_vars: int,
    alpha: float,
    max_lag: int,
    ci_test_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fallback discovery: PC algorithm with temporal augmentation (causal-learn).

    Used only when tigramite is not installed. Higher false-positive rate than
    PCMCI, especially for > 6 variables or short time series.
    """
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.cit import fisherz, kci

    T = data.shape[0]
    valid_start = max_lag

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

            # Lagged edges: x_{t-k} -> y_t
            for lag in range(1, max_lag + 1):
                lag_i = i + n_vars * lag
                if lag_i < augmented.shape[1] - 1:
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


def _build_mmm_link_assumptions(
    n_channels: int,
    n_controls: int,
    n_vars: int,
    max_lag: int,
) -> dict:
    """Restrict PCMCI to structurally possible edges in MMM.

    Allowed edges (tested):
      - channel/control → y      (direct / mediated detection)
      - channel_i → channel_j    (mediated path between channels)
      - channel_i → channel_i    (self-lag / autocorrelation conditioning)
      - control   → channel      (endogeneity detection)

    Forbidden edges (skipped, saves ~30% of CI tests):
      - y → anything             (y is the terminal outcome)
      - channel → control        (controls are exogenous by design)
      - control → control        (controls are assumed independent)
    """
    y_idx        = n_vars - 1
    channel_idxs = list(range(n_channels))
    control_idxs = list(range(n_channels, n_channels + n_controls))
    lags         = [-tau for tau in range(1, max_lag + 1)]

    la: dict = {j: {} for j in range(n_vars)}

    # anything → y
    for i in range(n_vars - 1):
        for lag in lags:
            la[y_idx][(i, lag)] = "?->"

    # channel/control → channel  (includes self-lags for autocorrelation)
    for j in channel_idxs:
        for i in channel_idxs + control_idxs:
            for lag in lags:
                la[j][(i, lag)] = "?->"

    # controls and y stay empty: no incoming edges

    return la


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


def _print_summary(
    graph: CausalGraph,
    channel_columns: List[str],
    control_columns: List[str],
    console: Console,
) -> None:
    """Print discovery results as Rich table."""
    table = Table(title="CD-NOTS Discovery Results (v2)")
    table.add_column("Channel", style="cyan")
    table.add_column("→ y", style="green")
    table.add_column("Category", style="magenta")
    table.add_column("Endogenous", style="red")

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

        endo = "⚠ YES" if ch in graph.endogenous_channels else ""

        table.add_row(ch, has_direct, cat, endo)

    console.print(table)

    console.print(
        f"  Edges: {graph.n_edges} | "
        f"Direct: {len(graph.direct_channels)} | "
        f"Mediated: {len(graph.mediated_channels)} | "
        f"Excluded: {len(graph.excluded_channels)} | "
        f"Endogenous: {len(graph.endogenous_channels)} | "
        f"Runtime: {graph.runtime_seconds:.1f}s"
    )

    if graph.control_to_channel_edges:
        console.print("\n  [bold red]Endogeneity detected:[/bold red]")
        for ctrl, ch in graph.control_to_channel_edges:
            console.print(f"    {ctrl} → {ch} (confounder: {ctrl} causes both {ch} and y)")