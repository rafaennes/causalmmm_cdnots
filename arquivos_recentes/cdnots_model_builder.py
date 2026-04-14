# Copyright 2025
#
# Licensed under the Apache License, Version 2.0

"""Build PyMC-Marketing and Meridian models with CD-NOTS calibrated priors.

VERSION 2: Probabilistic prior adjustment ("humble priors").

Key change from v1: instead of fixed multipliers (0.1, 0.8, 1.2) that
caused prior-likelihood conflict and MCMC divergence (R-hat > 1.8),
this version uses a continuous adjustment function of CD-NOTS p-values.

Philosophy: the graph should INFORM the prior, not DICTATE it.

Adjustment formula:
    sigma_adjusted = sigma_base × (1 + damping × direction × confidence)

Where:
    confidence = clip(1 - p_value, 0, 1)
    direction  = +1 if channel has causal path to y (widen prior)
                 -1 if no causal path (shrink prior)
    damping    = max adjustment strength (default 0.5)

This ensures:
    - Strong evidence → meaningful but bounded adjustment
    - Weak evidence → nearly no adjustment (≈ baseline)
    - Excluded channels retain σ × 0.5 minimum (not 0.1) → no MCMC conflict
    - Model can always override prior if data disagrees
"""

from typing import List, Tuple

import numpy as np
import pandas as pd
import tensorflow_probability as tfp
from meridian import constants
from meridian.model import model, prior_distribution, spec
from pymc_marketing.mmm import GeometricAdstock, HillSaturationSigmoid
from pymc_marketing.mmm.multidimensional import MMM
from pymc_marketing.prior import Prior

from . import model_builder
from .cdnots_discovery import CausalGraph


# ============================================================
# Configuration
# ============================================================

# Max adjustment strength: 0=no effect, 0.5=moderate, 1.0=aggressive
DAMPING = 0.5

# Minimum multiplier floor (prevents prior-likelihood conflict)
FLOOR = 0.4


# ============================================================
# Probabilistic prior adjustment
# ============================================================

def _path_min_confidence_pvalue(
    adj: np.ndarray,
    pvals: np.ndarray,
    source: int,
    target: int,
    max_depth: int = 3,
) -> float:
    """Return the p-value of the weakest edge along the best path to target.

    A mediated channel's confidence is bounded by its flimsiest mediating
    edge. Among all paths source → ... → target (depth ≤ max_depth), the
    path's strength = max p-value on that path. We return the minimum of
    those path strengths (i.e., the best available path). Returns 1.0 if
    no path exists.
    """
    # BFS tracking best (lowest) path-max-pvalue to each node
    best = {source: 0.0}
    frontier = [source]
    for _ in range(max_depth):
        next_frontier = []
        for node in frontier:
            for child in range(adj.shape[1]):
                if adj[node, child] <= 0 or child == node:
                    continue
                path_max = max(best[node], float(pvals[node, child]))
                if path_max < best.get(child, np.inf):
                    best[child] = path_max
                    next_frontier.append(child)
        frontier = next_frontier
        if not frontier:
            break
    return best.get(target, 1.0)


def _compute_sigma_multipliers(
    channel_columns: List[str],
    graph: CausalGraph,
    damping: float = DAMPING,
    floor: float = FLOOR,
) -> np.ndarray:
    """Compute per-channel sigma multipliers as continuous function of p-values.

    Parameters
    ----------
    channel_columns : List[str]
        Channel column names (ordered).
    graph : CausalGraph
        Causal discovery results with p-values.
    damping : float
        Maximum adjustment factor.
    floor : float
        Minimum multiplier to prevent MCMC conflict.

    Returns
    -------
    np.ndarray
        Multipliers array of shape (n_channels,).
    """
    n_channels = len(channel_columns)
    multipliers = np.ones(n_channels)
    y_idx = len(graph.variable_names) - 1

    for i, ch in enumerate(channel_columns):
        if ch in graph.variable_names:
            ch_idx = graph.variable_names.index(ch)
            has_direct = graph.adjacency_matrix[ch_idx, y_idx] > 0
            if has_direct:
                p_value = graph.edge_pvalues[ch_idx, y_idx]
            elif ch in graph.mediated_channels:
                # Indirect path: use the weakest link's p-value along the
                # most confident path to y. Mediated confidence is bounded
                # by its flimsiest mediating edge.
                p_value = _path_min_confidence_pvalue(
                    graph.adjacency_matrix, graph.edge_pvalues, ch_idx, y_idx
                )
            else:
                p_value = 1.0
        else:
            p_value = 1.0

        confidence = np.clip(1.0 - p_value, 0.0, 1.0)
        has_path = (ch in graph.direct_channels) or (ch in graph.mediated_channels)

        if has_path:
            multipliers[i] = 1.0 + damping * confidence
        else:
            multipliers[i] = max(floor, 1.0 - damping * confidence)

        # Endogeneity penalty (Data-Driven): if a control confounds this channel,
        # shrink its prior proportionately to the variance explained by the control (R2).
        # We use Tolerance = 1 - R2 as the shrinkage factor.
        if hasattr(graph, 'endogenous_r2') and ch in graph.endogenous_r2:
            r2 = graph.endogenous_r2[ch]
            tolerance = max(0.1, 1.0 - r2)  # shrink by R2, min 10% tolerance
            multipliers[i] *= tolerance
            multipliers[i] = max(floor, multipliers[i])

    return multipliers


def _compute_adstock_params(
    channel_columns: List[str],
    graph: CausalGraph,
    damping: float = DAMPING,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-channel Beta(a, b) for adstock prior.

    Channels with stronger causal evidence get more flexible adstock.

    Returns
    -------
    tuple
        (alpha_a, alpha_b) arrays of shape (n_channels,)
    """
    n_channels = len(channel_columns)
    alpha_a = np.ones(n_channels)
    alpha_b = np.full(n_channels, 3.0)
    y_idx = len(graph.variable_names) - 1

    for i, ch in enumerate(channel_columns):
        if ch in graph.variable_names:
            ch_idx = graph.variable_names.index(ch)
            has_direct = graph.adjacency_matrix[ch_idx, y_idx] > 0
            if has_direct:
                p_value = graph.edge_pvalues[ch_idx, y_idx]
            elif ch in graph.mediated_channels:
                # Indirect path: use the weakest link's p-value along the
                # most confident path to y. Mediated confidence is bounded
                # by its flimsiest mediating edge.
                p_value = _path_min_confidence_pvalue(
                    graph.adjacency_matrix, graph.edge_pvalues, ch_idx, y_idx
                )
            else:
                p_value = 1.0
        else:
            p_value = 1.0

        confidence = np.clip(1.0 - p_value, 0.0, 1.0)
        has_path = (ch in graph.direct_channels) or (ch in graph.mediated_channels)

        if has_path:
            alpha_b[i] = 3.0 - damping * confidence * 2.0
        else:
            alpha_b[i] = 3.0 + damping * confidence * 1.0

        alpha_b[i] = np.clip(alpha_b[i], 1.0, 5.0)

    return alpha_a, alpha_b


# ============================================================
# PyMC-Marketing
# ============================================================

def build_pymc_model_with_graph(
    data_df: pd.DataFrame,
    channel_columns: List[str],
    control_columns: List[str],
    graph: CausalGraph,
    damping: float = DAMPING,
) -> MMM:
    """Build PyMC-Marketing model with probabilistic priors from causal graph."""
    prior_sigma = model_builder.calculate_prior_sigma(data_df, channel_columns)
    multipliers = _compute_sigma_multipliers(channel_columns, graph, damping=damping)
    adjusted_sigma = prior_sigma * multipliers[np.newaxis, :]

    _log_adjustments(channel_columns, multipliers, graph)

    saturation = HillSaturationSigmoid(
        priors={
            "sigma": Prior(
                "InverseGamma",
                mu=Prior("HalfNormal", sigma=adjusted_sigma.mean(axis=0), dims=("channel",)),
                sigma=Prior("HalfNormal", sigma=1.5),
                dims=("channel", "geo")
            ),
            "beta": Prior("HalfNormal", sigma=1.5, dims=("channel",)),
            "lam": Prior("HalfNormal", sigma=1.5, dims=("channel",)),
        },
    )

    _, alpha_b = _compute_adstock_params(channel_columns, graph, damping=damping)
    adstock = GeometricAdstock(
        l_max=8,
        priors={"alpha": Prior("Beta", alpha=1, beta=alpha_b.tolist(), dims=("channel",))},
    )

    mmm = MMM(
        date_column="time",
        target_column="y",
        channel_columns=channel_columns,
        control_columns=control_columns,
        dims=("geo",),
        scaling={
            "channel": {"method": "max", "dims": ()},
            "target": {"method": "max", "dims": ()},
        },
        saturation=saturation,
        adstock=adstock,
        yearly_seasonality=2,
    )

    x_train = data_df.drop(columns=["y"])
    y_train = data_df["y"]
    mmm.build_model(X=x_train, y=y_train)

    contribution_vars = [
        "channel_contribution",
        "intercept_contribution",
        "yearly_seasonality_contribution",
        "y",
    ]
    if control_columns:
        contribution_vars.insert(1, "control_contribution")
    mmm.add_original_scale_contribution_variable(var=contribution_vars)

    return mmm


# ============================================================
# Meridian
# ============================================================

def build_meridian_model_with_graph(
    data_df: pd.DataFrame,
    channel_columns: List[str],
    control_columns: List[str],
    graph: CausalGraph,
    damping: float = DAMPING,
) -> model.Meridian:
    """Build Meridian model with probabilistic priors from causal graph."""
    prior_sigma = model_builder.calculate_prior_sigma(data_df, channel_columns)
    multipliers = _compute_sigma_multipliers(channel_columns, graph, damping=damping)
    mean_sigma = prior_sigma.mean(axis=0) * multipliers

    _log_adjustments(channel_columns, multipliers, graph)

    built_data = model_builder.build_meridian_data(data_df, channel_columns, control_columns)
    build_media_channel_args = built_data.get_paid_media_channels_argument_builder()
    beta_m = build_media_channel_args(
        **{col: (0, float(mean_sigma[i])) for i, col in enumerate(channel_columns)}
    )
    beta_m_mu, beta_m_sigma = zip(*beta_m)

    alpha_a, alpha_b = _compute_adstock_params(channel_columns, graph, damping=damping)

    prior = prior_distribution.PriorDistribution(
        beta_m=tfp.distributions.LogNormal(
            beta_m_mu, beta_m_sigma, name=constants.BETA_M
        ),
        alpha_m=tfp.distributions.Beta(
            alpha_a.tolist(), alpha_b.tolist(), name=constants.ALPHA_M
        ),
    )

    model_spec = model_builder.build_meridian_model_spec(prior, len(built_data.time))
    return model.Meridian(input_data=built_data, model_spec=model_spec)


# ============================================================
# Logging
# ============================================================

def _log_adjustments(
    channel_columns: List[str],
    multipliers: np.ndarray,
    graph: CausalGraph,
) -> None:
    """Print adjustment summary for transparency and debugging."""
    y_idx = len(graph.variable_names) - 1

    print(f"\n  Prior adjustments (damping={DAMPING}, floor={FLOOR}):")
    for i, ch in enumerate(channel_columns):
        if ch in graph.variable_names:
            ch_idx = graph.variable_names.index(ch)
            p_val = graph.edge_pvalues[ch_idx, y_idx]
        else:
            p_val = 1.0

        category = (
            "direct" if ch in graph.direct_channels
            else "mediated" if ch in graph.mediated_channels
            else "excluded"
        )
        direction = "↑" if multipliers[i] > 1.0 else "↓" if multipliers[i] < 1.0 else "="
        print(
            f"    {ch:>30s}: σ×{multipliers[i]:.3f} {direction}  "
            f"(p={p_val:.4f}, {category})"
        )