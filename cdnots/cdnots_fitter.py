# Copyright 2025
#
# Licensed under the Apache License, Version 2.0

"""Fitting functions for CD-NOTS informed PyMC-Marketing and Meridian models.

Follows the same interface as model_fitter.py:
    fit_* returns Tuple[Model, float, Dict[str, Optional[float]]]

Usage:
    from mmm_param_recovery.benchmarking.cdnots_fitter import (
        fit_pymc_with_graph,
        fit_meridian_with_graph,
    )

    pymc_model, runtime, ess = fit_pymc_with_graph(
        data_df, channel_columns, control_columns, graph,
        "nutpie", 4, 1000, 1000, 0.9, 42, console
    )
"""

import time
from typing import Dict, Optional, Tuple

import pandas as pd
from meridian.model import model
from pymc_marketing.mmm.multidimensional import MMM
from rich.console import Console

from . import diagnostics
from .cdnots_discovery import CausalGraph
from .cdnots_model_builder import (
    build_meridian_model_with_graph,
    build_pymc_model_with_graph,
)


def fit_pymc_with_graph(
    data_df: pd.DataFrame,
    channel_columns: list,
    control_columns: list,
    graph: CausalGraph,
    sampler: str,
    n_chains: int,
    n_draws: int,
    n_tune: int,
    target_accept: float,
    seed: int,
    console: Optional[Console] = None,
) -> Tuple[MMM, float, Dict[str, Optional[float]]]:
    """Fit PyMC-Marketing with CD-NOTS calibrated priors.

    Same interface as model_fitter.fit_pymc but uses graph-informed priors.
    Timing includes model building (with graph) + sampling.

    Parameters
    ----------
    data_df : pd.DataFrame
        Dataset
    channel_columns : list
        Channel column names
    control_columns : list
        Control column names
    graph : CausalGraph
        CD-NOTS discovery results
    sampler : str
        Sampler name ('pymc', 'blackjax', 'numpyro', 'nutpie')
    n_chains : int
        Number of chains
    n_draws : int
        Number of draws per chain
    n_tune : int
        Number of tuning samples
    target_accept : float
        Target acceptance probability
    seed : int
        Random seed
    console : Optional[Console]
        Rich console for output

    Returns
    -------
    Tuple[MMM, float, Dict]
        Fitted model, runtime in seconds, ESS statistics
    """
    if console is None:
        console = Console()

    console.print(
        f"  Fitting PyMC-Marketing + CD-NOTS with {sampler}, "
        f"{n_chains} chains, {n_draws} draws, {n_tune} tune steps"
    )
    console.print(
        f"    Graph: {len(graph.direct_channels)} direct, "
        f"{len(graph.mediated_channels)} mediated, "
        f"{len(graph.excluded_channels)} excluded"
    )

    kwargs = {}
    if sampler == "nutpie":
        kwargs = {"nuts_sampler_kwargs": {"backend": "jax", "gradient_backend": "jax"}}

    # Start timing BEFORE building model (same as baseline)
    start = time.perf_counter()

    pymc_model = build_pymc_model_with_graph(
        data_df, channel_columns, control_columns, graph
    )

    x = data_df.drop(columns=["y"])
    y = data_df["y"]

    pymc_model.fit(
        X=x,
        y=y,
        chains=n_chains,
        draws=n_draws,
        tune=n_tune,
        target_accept=target_accept,
        random_seed=seed,
        nuts_sampler=sampler,
        **kwargs,
    )

    pymc_model.sample_posterior_predictive(
        X=x, extend_idata=True, combined=True, random_seed=seed
    )

    runtime = time.perf_counter() - start
    ess = diagnostics.compute_ess(pymc_model.idata)

    console.print(
        f"  [green]✓[/green] PyMC + CD-NOTS - {sampler}: "
        f"{runtime:.1f}s, ESS min: {ess.get('min', 'N/A')}"
    )

    return pymc_model, runtime, ess


def fit_meridian_with_graph(
    data_df: pd.DataFrame,
    channel_columns: list,
    control_columns: list,
    graph: CausalGraph,
    n_chains: int,
    n_draws: int,
    n_tune: int,
    target_accept: float,
    seed: int,
    console: Optional[Console] = None,
) -> Tuple[model.Meridian, float, Dict[str, Optional[float]]]:
    """Fit Meridian with CD-NOTS calibrated priors.

    Same interface as model_fitter.fit_meridian but uses graph-informed priors.

    Parameters
    ----------
    data_df : pd.DataFrame
        Dataset
    channel_columns : list
        Channel column names
    control_columns : list
        Control column names
    graph : CausalGraph
        CD-NOTS discovery results
    n_chains : int
        Number of chains
    n_draws : int
        Number of draws per chain
    n_tune : int
        Number of tuning samples
    target_accept : float
        Target acceptance probability
    seed : int
        Random seed
    console : Optional[Console]
        Rich console for output

    Returns
    -------
    Tuple[model.Meridian, float, Dict]
        Fitted model, runtime in seconds, ESS statistics
    """
    if console is None:
        console = Console()

    console.print(
        f"  Fitting Meridian + CD-NOTS with "
        f"{n_chains} chains, {n_draws} draws, {n_tune} tune steps"
    )
    console.print(
        f"    Graph: {len(graph.direct_channels)} direct, "
        f"{len(graph.mediated_channels)} mediated, "
        f"{len(graph.excluded_channels)} excluded"
    )

    start = time.perf_counter()

    meridian_model = build_meridian_model_with_graph(
        data_df, channel_columns, control_columns, graph
    )

    meridian_model.sample_posterior(
        n_chains=n_chains,
        n_adapt=int(n_tune / 2),
        n_burnin=int(n_tune / 2),
        n_keep=n_draws,
        seed=(seed, seed),
        dual_averaging_kwargs={"target_accept_prob": target_accept},
    )

    runtime = time.perf_counter() - start
    ess = diagnostics.compute_ess(meridian_model.inference_data)

    console.print(
        f"  [green]✓[/green] Meridian + CD-NOTS: "
        f"{runtime:.1f}s, ESS min: {ess.get('min', 'N/A')}"
    )

    return meridian_model, runtime, ess
