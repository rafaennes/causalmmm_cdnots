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

# Minimum sigma ratio for the spike component of the continuous spike-and-slab.
# σ_adj = σ_base × (MIN_SIGMA_RATIO + (1 - MIN_SIGMA_RATIO) × PIP)
# Value 0.4 prevents prior-likelihood conflict (R-hat > 1.8 observed with values < 0.3).
# Derivation: docs/superpowers/specs/2026-05-13-cdnots-pipeline-fix-design.md
MIN_SIGMA_RATIO = 0.4
# DAMPING removed — new formula has no free parameters beyond MIN_SIGMA_RATIO.


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
) -> np.ndarray:
    """Compute per-channel sigma multipliers via Empirical Bayes / spike-and-slab.

    Formula (same for all channel categories):
        PIP  = 1 - q_value           (posterior inclusion probability, Storey 2002)
        mult = MIN_SIGMA_RATIO + (1 - MIN_SIGMA_RATIO) × PIP

    where q_value is the BH-corrected edge q-value from CausalGraph.edge_qvalues.
    For mediated channels the weakest-link path q-value is used.

    References
    ----------
    Storey (2002) JRSS-B 64(3):479-498 — q-value as P(H0|data).
    Efron (2010) Large-Scale Inference, Ch. 5 — PIP = 1 - q.
    Ishwaran & Rao (2005) Ann.Stat. 33(2):730-773 — continuous spike-and-slab.

    Returns
    -------
    np.ndarray
        Multipliers in [MIN_SIGMA_RATIO, 1.0] of shape (n_channels,).
    """
    n_channels = len(channel_columns)
    multipliers = np.ones(n_channels)
    y_idx = len(graph.variable_names) - 1

    for i, ch in enumerate(channel_columns):
        if ch in graph.variable_names:
            ch_idx = graph.variable_names.index(ch)
            has_direct = graph.adjacency_matrix[ch_idx, y_idx] > 0

            if has_direct:
                q_value = float(graph.edge_qvalues[ch_idx, y_idx])
            elif ch in graph.mediated_channels:
                # Mediated: weakest-link q along the most confident path to y.
                q_value = _path_min_confidence_pvalue(
                    graph.adjacency_matrix, graph.edge_qvalues, ch_idx, y_idx
                )
            else:
                # Excluded: high q → low PIP → multiplier near MIN_SIGMA_RATIO.
                q_value = float(graph.edge_qvalues[ch_idx, y_idx])
        else:
            q_value = 1.0

        # PIP = P(H1 | data) under BH empirical Bayes model
        pip = float(np.clip(1.0 - q_value, 0.0, 1.0))
        multipliers[i] = MIN_SIGMA_RATIO + (1.0 - MIN_SIGMA_RATIO) * pip

        # Endogeneity penalty: control → channel confounding shrinks prior
        # proportional to R² (variance explained by confounder).
        if hasattr(graph, 'endogenous_r2') and ch in graph.endogenous_r2:
            r2 = graph.endogenous_r2[ch]
            tolerance = max(MIN_SIGMA_RATIO, 1.0 - r2)
            multipliers[i] = max(MIN_SIGMA_RATIO, multipliers[i] * tolerance)

    return multipliers


def _compute_adstock_params(
    channel_columns: List[str],
    graph: CausalGraph,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-channel Beta(a, b) for adstock prior using PIP from q-values.

    Beta(1, alpha_b): lower alpha_b → more uniform (flexible decay);
                      higher alpha_b → concentrated near 0 (fast decay).
    BASE_B=3.0 is the neutral prior; MIN_B=1.0 is maximally flexible (Uniform).
    Channels with high PIP get more flexible adstock (lower alpha_b).

    Returns
    -------
    tuple
        (alpha_a, alpha_b) arrays of shape (n_channels,)
    """
    BASE_B, MIN_B = 3.0, 1.0
    n_channels = len(channel_columns)
    alpha_a = np.ones(n_channels)
    alpha_b = np.full(n_channels, BASE_B)
    y_idx = len(graph.variable_names) - 1

    for i, ch in enumerate(channel_columns):
        if ch in graph.variable_names:
            ch_idx = graph.variable_names.index(ch)
            has_direct = graph.adjacency_matrix[ch_idx, y_idx] > 0

            if has_direct:
                q_value = float(graph.edge_qvalues[ch_idx, y_idx])
            elif ch in graph.mediated_channels:
                q_value = _path_min_confidence_pvalue(
                    graph.adjacency_matrix, graph.edge_qvalues, ch_idx, y_idx
                )
            else:
                q_value = 1.0
        else:
            q_value = 1.0

        pip = float(np.clip(1.0 - q_value, 0.0, 1.0))
        alpha_b[i] = float(np.clip(BASE_B - (BASE_B - MIN_B) * pip, MIN_B, 5.0))

    return alpha_a, alpha_b


# ============================================================
# PyMC-Marketing
# ============================================================

def build_pymc_model_with_graph(
    data_df: pd.DataFrame,
    channel_columns: List[str],
    control_columns: List[str],
    graph: CausalGraph,
) -> MMM:
    """Build PyMC-Marketing model with Empirical Bayes priors from causal graph."""
    prior_sigma = model_builder.calculate_prior_sigma(data_df, channel_columns)
    multipliers = _compute_sigma_multipliers(channel_columns, graph)
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

    _, alpha_b = _compute_adstock_params(channel_columns, graph)
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
) -> model.Meridian:
    """Build Meridian model with Empirical Bayes priors from causal graph."""
    prior_sigma = model_builder.calculate_prior_sigma(data_df, channel_columns)
    multipliers = _compute_sigma_multipliers(channel_columns, graph)
    mean_sigma = prior_sigma.mean(axis=0) * multipliers

    _log_adjustments(channel_columns, multipliers, graph)

    built_data = model_builder.build_meridian_data(data_df, channel_columns, control_columns)
    build_media_channel_args = built_data.get_paid_media_channels_argument_builder()
    beta_m = build_media_channel_args(
        **{col: (0, float(mean_sigma[i])) for i, col in enumerate(channel_columns)}
    )
    beta_m_mu, beta_m_sigma = zip(*beta_m)

    alpha_a, alpha_b = _compute_adstock_params(channel_columns, graph)

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

    print(f"\n  Prior adjustments (MIN_SIGMA_RATIO={MIN_SIGMA_RATIO}, formula: σ×(0.4+0.6×PIP)):")
    for i, ch in enumerate(channel_columns):
        if ch in graph.variable_names:
            ch_idx = graph.variable_names.index(ch)
            q_val = float(graph.edge_qvalues[ch_idx, y_idx])
            p_val = float(graph.edge_pvalues[ch_idx, y_idx])
        else:
            q_val = p_val = 1.0

        pip = round(1.0 - q_val, 3)
        category = (
            "direct" if ch in graph.direct_channels
            else "mediated" if ch in graph.mediated_channels
            else "excluded"
        )
        print(
            f"    {ch:>30s}: σ×{multipliers[i]:.3f}  "
            f"(p={p_val:.4f}, q={q_val:.4f}, PIP={pip:.3f}, {category})"
        )