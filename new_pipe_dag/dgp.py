"""Parametric DGP for the T0.5 calibration ladder (spec v2 §3.5).

Each level isolates one factor. The fallback DGP in causal_discovery/compare.py
is a fixed "realistic" preset; this one exposes knobs for controlled ablation.

Reuses the same structural conventions: all channels → y, inter-channel edges
as (src_idx, tgt_idx) pairs, returns (data, var_names, true_adj, truth_var_names).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class LadderConfig:
    """One rung of the T0.5 calibration ladder."""

    name: str = "L0"
    n_channels: int = 4
    n_controls: int = 1
    n_periods: int = 1040
    inter_channel_edges: list = field(default_factory=lambda: [(0, 1)])
    noise_std: float = 0.001  # noise / signal ratio on y
    adstock_on: bool = False
    saturation_cy: bool = False       # saturation channel → y
    saturation_spill: bool = False    # saturation on spillover path
    seasonality: str = "off"          # "off", "S0" (hidden), "S1" (observed)
    phase_spread: str = "equispaced"  # "equispaced", "moderate", "concentrated"
    seed: int = 2025


# ponytail: pre-built ladder levels from spec T0.5
LADDER = {
    "L0": LadderConfig(name="L0", noise_std=0.001, n_periods=1040),
    "L1": LadderConfig(name="L1", noise_std=0.02, n_periods=1040),
    "L2": LadderConfig(name="L2", noise_std=0.02, n_periods=156),
    "L3": LadderConfig(name="L3", noise_std=0.02, n_periods=156, adstock_on=True),
    "L4": LadderConfig(name="L4", noise_std=0.02, n_periods=156, saturation_cy=True),
    "L5": LadderConfig(name="L5", noise_std=0.02, n_periods=156, saturation_spill=True),
    "L6": LadderConfig(name="L6", noise_std=0.02, n_periods=156, seasonality="S1"),
    "L7": LadderConfig(name="L7", noise_std=0.02, n_periods=156, seasonality="S0"),
    "L8": LadderConfig(
        name="L8", noise_std=0.02, n_periods=156,
        seasonality="S0", phase_spread="concentrated",
    ),
    "L9": LadderConfig(
        name="L9", noise_std=0.02, n_periods=156,
        adstock_on=True, saturation_cy=True, saturation_spill=True,
        seasonality="S0", phase_spread="concentrated",
    ),
}


def generate_ladder(
    cfg: LadderConfig,
    seed: int | None = None,
) -> tuple[pd.DataFrame, list[str], np.ndarray, list[str]]:
    """Generate synthetic data for one ladder rung.

    Returns (data_df, var_names, true_adj, truth_var_names).
    truth_var_names includes "season" when seasonality != "off".
    """
    s = seed if seed is not None else cfg.seed
    rng = np.random.default_rng(s)
    T = cfg.n_periods
    n_ch = cfg.n_channels
    n_co = cfg.n_controls
    edges = cfg.inter_channel_edges

    # --- Seasonal driver (latent or observed) ---
    season_active = cfg.seasonality in ("S0", "S1")
    if season_active:
        # Annual cycle, weekly resolution
        season_signal = np.cos(2 * np.pi * np.arange(T) / 52)
        # Phase-spread determines channel loading correlation
        if cfg.phase_spread == "equispaced":
            phases = np.linspace(0, 2 * np.pi, n_ch, endpoint=False)
        elif cfg.phase_spread == "moderate":
            phases = rng.normal(0.0, 1.0, n_ch)
        else:  # concentrated
            phases = rng.normal(0.0, 0.2, n_ch)
        ch_loadings = 0.3 * np.array([np.cos(p) for p in phases])
    else:
        season_signal = np.zeros(T)
        ch_loadings = np.zeros(n_ch)

    # --- Step 1: Channel innovations (independent) ---
    innovations = {}
    for i in range(n_ch):
        base = rng.uniform(800, 2000)
        noise = rng.normal(0, base * 0.12, T)
        seasonal_part = base * ch_loadings[i] * season_signal if season_active else 0.0
        innovations[i] = np.clip(base + seasonal_part + noise, 50, None)

    # --- Step 2: Inter-channel spillover ---
    spillover_coeffs = {}
    for src, tgt in edges:
        coeff = rng.uniform(0.15, 0.25)
        spillover_coeffs[(src, tgt)] = coeff
        src_signal = innovations[src].copy()

        # Optional saturation on spillover path (spec A5 / L5)
        if cfg.saturation_spill:
            K = np.median(src_signal)
            src_signal = K * np.log1p(src_signal / K)

        innovations[tgt][1:] += coeff * src_signal[:-1]
        innovations[tgt][2:] += (coeff * 0.4) * src_signal[:-2]

    # --- Step 3: Adstock (geometric carryover) ---
    spend = {}
    for i in range(n_ch):
        x = innovations[i].copy()
        if cfg.adstock_on:
            decay = rng.uniform(0.3, 0.5)
            for t in range(1, T):
                x[t] = x[t] + decay * x[t - 1]
        spend[i] = x

    # --- Step 4: Channel → y signal ---
    # Calibrate betas so total channel signal is ~3× noise level
    noise_level = 500 * cfg.noise_std / 0.02  # scale with noise_std
    if noise_level == 0:
        noise_level = 0.5  # floor for L0
    target_per_ch = noise_level * 3.0 / max(n_ch, 1)

    betas = []
    for i in range(n_ch):
        s_i = spend[i]
        # Optional saturation on channel → y (spec L4)
        if cfg.saturation_cy:
            K = np.median(s_i)
            s_i = K * np.log1p(s_i / K)
        std_i = s_i.std() + 1e-8
        betas.append(target_per_ch / std_i * rng.uniform(0.7, 1.0))

    base_sales = 10000.0
    y = np.full(T, base_sales)
    # Season → y
    if season_active:
        y += 0.05 * base_sales * season_signal
    # Channels → y at lag 1
    for i in range(n_ch):
        s_i = spend[i]
        if cfg.saturation_cy:
            K = np.median(s_i)
            s_i = K * np.log1p(s_i / K)
        y[1:] += betas[i] * s_i[:-1]
    # Controls
    controls = {}
    for j in range(n_co):
        c = rng.normal(0, 1, T)
        controls[j] = c
        y += rng.uniform(-30, 30) * c
    # Noise
    y += rng.normal(0, noise_level, T)

    # --- Build DataFrame ---
    data = {}
    ch_names = [f"x{i+1}" for i in range(n_ch)]
    co_names = [f"c{j+1}" for j in range(n_co)]
    for i in range(n_ch):
        data[ch_names[i]] = spend[i]
    for j in range(n_co):
        data[co_names[j]] = controls[j]
    # Season as observed variable (S1 only)
    if cfg.seasonality == "S1":
        data["season"] = season_signal
    data["y"] = y
    df = pd.DataFrame(data)
    var_names = list(df.columns)

    # --- Ground truth adjacency ---
    # Truth space: channels + [season] + y
    truth_names = list(ch_names)
    if season_active:
        truth_names.append("season")
    truth_names.append("y")
    n_truth = len(truth_names)
    idx = {n: i for i, n in enumerate(truth_names)}
    true_adj = np.zeros((n_truth, n_truth))

    # All channels → y
    for ch in ch_names:
        true_adj[idx[ch], idx["y"]] = 1.0
    # Inter-channel
    for src, tgt in edges:
        true_adj[idx[ch_names[src]], idx[ch_names[tgt]]] = 1.0
    # Season edges
    if season_active:
        true_adj[idx["season"], idx["y"]] = 1.0
        for i, ch in enumerate(ch_names):
            if abs(ch_loadings[i]) > 0.01:
                true_adj[idx["season"], idx[ch]] = 1.0

    return df, var_names, true_adj, truth_names
