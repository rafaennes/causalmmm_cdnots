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


# Beta calibration uses a fixed reference so that SNR actually varies with
# noise_std.  Total channel signal std ≈ sqrt(n_ch) * REFERENCE_NOISE * 3 / n_ch
# = 3 * REFERENCE_NOISE / sqrt(n_ch).  Per-channel SNR = 0.75 at noise_std=0.02.
REFERENCE_NOISE = 500.0

# Hill saturation: x^s / (k^s + x^s).  With slope=3 and kappa=p25,
# corr ≈ 0.994 vs raw spend (log1p was 0.999).  The narrow spend range
# ([~1300, ~2700], <2x dynamic range) limits how nonlinear any monotone
# transform can be.  This is inherent to the DGP's spend distribution.
HILL_SLOPE = 3
HILL_KAPPA_PERCENTILE = 25


def _hill(x: np.ndarray) -> np.ndarray:
    """Hill saturation: x^s / (kappa^s + x^s), kappa = p25(x), slope=3."""
    kappa = np.percentile(x, HILL_KAPPA_PERCENTILE) + 1e-8
    return x**HILL_SLOPE / (kappa**HILL_SLOPE + x**HILL_SLOPE)


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
    saturation_on: bool = False       # Hill saturation on both channel→y and spillover paths
    seasonality: str = "off"          # "off", "S0" (hidden), "S1" (observed)
    phase_spread: str = "equispaced"  # "equispaced", "moderate", "concentrated"
    seed: int = 2025


LADDER = {
    "L0": LadderConfig(name="L0", noise_std=0.001, n_periods=1040),
    "L1": LadderConfig(name="L1", noise_std=0.02, n_periods=1040),
    "L2": LadderConfig(name="L2", noise_std=0.02, n_periods=156),
    "L3": LadderConfig(name="L3", noise_std=0.02, n_periods=156, adstock_on=True),
    "L4": LadderConfig(name="L4", noise_std=0.02, n_periods=156, saturation_on=True),
    "L5": LadderConfig(name="L5", noise_std=0.02, n_periods=156, seasonality="S1"),
    "L6": LadderConfig(name="L6", noise_std=0.02, n_periods=156, seasonality="S0"),
    "L7": LadderConfig(
        name="L7", noise_std=0.02, n_periods=156,
        seasonality="S0", phase_spread="concentrated",
    ),
    "L8": LadderConfig(
        name="L8", noise_std=0.02, n_periods=156,
        adstock_on=True, saturation_on=True,
        seasonality="S0", phase_spread="concentrated",
    ),
}


def generate_ladder(
    cfg: LadderConfig,
    seed: int | None = None,
) -> tuple[pd.DataFrame, list[str], np.ndarray, list[str], dict[tuple[str, str], list[int]]]:
    """Generate synthetic data for one ladder rung.

    Returns (data_df, var_names, true_adj, truth_var_names, edge_lags).
    truth_var_names includes "season" when seasonality != "off".
    """
    s = seed if seed is not None else cfg.seed
    rng = np.random.default_rng(s)
    T = cfg.n_periods
    n_ch = cfg.n_channels
    n_co = cfg.n_controls
    edges = cfg.inter_channel_edges

    # --- Step 0: Seasonal driver (latent or observed) ---
    # Each channel gets a phase-shifted seasonal signal: cos(wt - phi_i).
    # The phase controls WHEN the channel peaks in the year.
    season_active = cfg.seasonality in ("S0", "S1")
    omega_t = 2 * np.pi * np.arange(T) / 52  # annual cycle, weekly resolution

    if season_active:
        if cfg.phase_spread == "equispaced":
            # [0, pi) avoids anticorrelated pairs: cos(wt-pi) = -cos(wt)
            phases = np.linspace(0, np.pi, n_ch, endpoint=False)
        elif cfg.phase_spread == "moderate":
            phases = rng.normal(0.0, 1.0, n_ch)
        else:  # concentrated — all channels peak near the same week
            phases = rng.normal(0.0, 0.2, n_ch)
        # Per-channel shifted seasonal signals
        season_signals = np.array([np.cos(omega_t - phi) for phi in phases])
        # Reference signal for season→y (unshifted)
        season_ref = np.cos(omega_t)
    else:
        phases = np.zeros(n_ch)
        season_signals = np.zeros((n_ch, T))
        season_ref = np.zeros(T)

    # --- Step 1: Channel innovations (independent of each other) ---
    innovations = {}
    for i in range(n_ch):
        base = rng.uniform(800, 2000)
        noise = rng.normal(0, base * 0.12, T)
        seasonal_part = base * 0.3 * season_signals[i] if season_active else 0.0
        innovations[i] = np.clip(base + seasonal_part + noise, 0, None)

    # --- Step 2: Inter-channel spillover (order-independent) ---
    # Freeze innovations before spillover so processing order doesn't matter.
    # Sort edges canonically so RNG draws are deterministic regardless of
    # the order the caller specifies them.
    pre_spillover = {k: v.copy() for k, v in innovations.items()}
    spillover_coeffs = {}
    for src, tgt in sorted(edges):
        coeff = rng.uniform(0.15, 0.25)
        spillover_coeffs[(src, tgt)] = coeff
        src_signal = pre_spillover[src]

        if cfg.saturation_on:
            src_signal = _hill(src_signal)
            # Scale back to original magnitude for additive spillover
            src_signal = src_signal * pre_spillover[src].std() / (src_signal.std() + 1e-8)

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
    # Betas calibrated at a FIXED reference noise level so that SNR actually
    # varies when noise_std changes (L0→L1 is a real noise experiment).
    target_per_ch = REFERENCE_NOISE * 3.0 / max(n_ch, 1)

    betas = []
    for i in range(n_ch):
        s_i = _hill(spend[i]) if cfg.saturation_on else spend[i]
        std_i = s_i.std() + 1e-8
        betas.append(target_per_ch / std_i * rng.uniform(0.7, 1.0))

    base_sales = 10000.0
    y = np.full(T, base_sales)
    # Season → y (uses unshifted reference signal)
    if season_active:
        y += 0.05 * base_sales * season_ref
    # Channels → y at lag 1
    for i in range(n_ch):
        s_i = _hill(spend[i]) if cfg.saturation_on else spend[i]
        y[1:] += betas[i] * s_i[:-1]
    # Controls (c1..cN) cause y but are excluded from the ground truth by
    # design: the discovery task is restricted to the media system
    # {channels, [season], y}. Controls act as nuisance variables that make
    # the problem harder but are not part of the evaluation target.
    controls = {}
    for j in range(n_co):
        c = rng.normal(0, 1, T)
        controls[j] = c
        y += rng.uniform(-30, 30) * c
    # Noise
    noise_level = 500 * cfg.noise_std / 0.02
    y += rng.normal(0, noise_level, T)

    # --- Build DataFrame ---
    ch_names = [f"x{i+1}" for i in range(n_ch)]
    co_names = [f"c{j+1}" for j in range(n_co)]
    data = {}
    for i in range(n_ch):
        data[ch_names[i]] = spend[i]
    for j in range(n_co):
        data[co_names[j]] = controls[j]
    if cfg.seasonality == "S1":
        data["season"] = season_ref
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
    # Inter-channel spillover
    for src, tgt in edges:
        true_adj[idx[ch_names[src]], idx[ch_names[tgt]]] = 1.0
    # Season edges: when active, season causes ALL channels (amplitude 0.3)
    # and y directly. No threshold needed — every channel gets seasonal loading.
    if season_active:
        true_adj[idx["season"], idx["y"]] = 1.0
        for ch in ch_names:
            true_adj[idx["season"], idx[ch]] = 1.0

    # --- Edge lags (for recall_ceiling) ---
    # Season edges are structurally contemporaneous (lag 0), but because
    # season(t-1) ≈ season(t) (autocorr ≈ 0.993), methods constrained to
    # tau_min=1 can detect them as lag-1 proxies.  The recall_ceiling
    # computation treats them as unrecoverable, which is conservative.
    edge_lags: dict[tuple[str, str], list[int]] = {}
    for ch in ch_names:
        edge_lags[(ch, "y")] = [1]
    for src, tgt in edges:
        edge_lags[(ch_names[src], ch_names[tgt])] = [1, 2]
    if season_active:
        edge_lags[("season", "y")] = [0]
        for ch in ch_names:
            edge_lags[("season", ch)] = [0]

    return df, var_names, true_adj, truth_names, edge_lags
