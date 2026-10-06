"""DGP verification tests.  Run: .venv/bin/python3.12 -m pytest new_pipe_dag/test_dgp.py -v"""
from __future__ import annotations

import numpy as np
import pytest

from new_pipe_dag.dgp import LadderConfig, generate_ladder


def test_phase_shifts_peak_timing():
    """Phase = pi/2 should peak around week 13 (quarter-year), not week 0."""
    T = 52
    cfg = LadderConfig(
        n_channels=1, n_controls=0, n_periods=T,
        inter_channel_edges=[], seasonality="S1", seed=42,
    )
    df, *_ = generate_ladder(cfg)
    season = df["season"].values
    # Unshifted cos peaks at t=0
    assert np.argmax(season[:52]) == 0

    # The channel's seasonal component should peak at ~week 13 when
    # equispaced phase for 1 channel is 0.  Test with 4 channels where
    # the second channel gets phase ≈ pi/4.
    cfg4 = LadderConfig(
        n_channels=4, n_controls=0, n_periods=104,
        inter_channel_edges=[], seasonality="S1",
        phase_spread="equispaced", seed=42,
    )
    df4, *_ = generate_ladder(cfg4)
    # Phases are linspace(0, pi, 4, endpoint=False) = [0, pi/4, pi/2, 3pi/4]
    # x2 (phase=pi/4) should peak ~6.5 weeks after x1 (phase=0)
    # Extract seasonal component by subtracting mean (base spend dominates)
    for col in ["x1", "x2", "x3", "x4"]:
        vals = df4[col].values
        # Seasonal oscillation rides on top of base; check that peaks differ
        assert vals.std() > 0, f"{col} has zero variance"

    # Verify phase ordering: peak of x1 should come before peak of x2
    # by looking at the first 52 weeks after detrending
    x1 = df4["x1"].values[:52] - df4["x1"].values[:52].mean()
    x2 = df4["x2"].values[:52] - df4["x2"].values[:52].mean()
    peak_x1 = np.argmax(x1)
    peak_x2 = np.argmax(x2)
    assert peak_x2 > peak_x1, f"x2 should peak after x1: x1@{peak_x1}, x2@{peak_x2}"


def test_spillover_order_independent():
    """Edge processing order must not affect results."""
    cfg_ab = LadderConfig(
        n_channels=3, n_controls=0, n_periods=156,
        inter_channel_edges=[(0, 1), (1, 2)], seed=2025,
    )
    cfg_ba = LadderConfig(
        n_channels=3, n_controls=0, n_periods=156,
        inter_channel_edges=[(1, 2), (0, 1)], seed=2025,
    )
    df_ab, *_ = generate_ladder(cfg_ab)
    df_ba, *_ = generate_ladder(cfg_ba)
    # x3 must be identical regardless of edge order
    np.testing.assert_allclose(
        df_ab["x3"].values, df_ba["x3"].values,
        atol=1e-10, err_msg="x3 differs between edge orders",
    )


def test_no_spillover_isolation():
    """With edges=[], no systematic lag-1 cross-channel correlation."""
    cfg = LadderConfig(
        n_channels=4, n_controls=0, n_periods=1040,
        inter_channel_edges=[], noise_std=0.001, seed=2025,
    )
    df, *_ = generate_ladder(cfg)
    channels = df[["x1", "x2", "x3", "x4"]].values
    # Check all lag-1 cross-correlations are small
    for i in range(4):
        for j in range(4):
            if i == j:
                continue
            r = np.corrcoef(channels[:-1, i], channels[1:, j])[0, 1]
            assert abs(r) < 0.15, f"Lag-1 corr(x{i+1}, x{j+1}) = {r:.3f}, expected < 0.15"


def test_seed_reproducibility():
    """Same seed → identical output; different seed → different."""
    cfg = LadderConfig(seed=123)
    df1, _, adj1, *_ = generate_ladder(cfg)
    df2, _, adj2, *_ = generate_ladder(cfg)
    np.testing.assert_array_equal(df1.values, df2.values)
    np.testing.assert_array_equal(adj1, adj2)

    df3, *_ = generate_ladder(cfg, seed=456)
    assert not np.allclose(df1["y"].values, df3["y"].values), "Different seeds should differ"


def test_truth_matches_dgp():
    """Every DGP mechanism has a corresponding edge in the truth graph."""
    cfg = LadderConfig(
        n_channels=4, n_controls=1, n_periods=156,
        inter_channel_edges=[(0, 1), (2, 3)],
        seasonality="S0", seed=2025,
    )
    _, _, true_adj, truth_names, edge_lags = generate_ladder(cfg)
    idx = {n: i for i, n in enumerate(truth_names)}

    # All channels → y
    for i in range(4):
        ch = f"x{i+1}"
        assert true_adj[idx[ch], idx["y"]] == 1.0, f"Missing {ch}→y"
        assert (ch, "y") in edge_lags

    # Spillover edges
    assert true_adj[idx["x1"], idx["x2"]] == 1.0, "Missing x1→x2"
    assert true_adj[idx["x3"], idx["x4"]] == 1.0, "Missing x3→x4"
    # No spurious spillover
    assert true_adj[idx["x1"], idx["x3"]] == 0.0, "Spurious x1→x3"

    # Season → all channels and y (S0 active)
    assert "season" in truth_names
    assert true_adj[idx["season"], idx["y"]] == 1.0, "Missing season→y"
    for i in range(4):
        ch = f"x{i+1}"
        assert true_adj[idx["season"], idx[ch]] == 1.0, f"Missing season→{ch}"

    # Controls NOT in truth (by design)
    assert "c1" not in truth_names


def test_hill_saturation_nonlinear():
    """Hill saturation should meaningfully change the signal (not a no-op)."""
    cfg_off = LadderConfig(n_periods=156, saturation_on=False, seed=2025)
    cfg_on = LadderConfig(n_periods=156, saturation_on=True, seed=2025)
    df_off, *_ = generate_ladder(cfg_off)
    df_on, *_ = generate_ladder(cfg_on)
    # y should differ meaningfully between saturated and unsaturated
    corr = np.corrcoef(df_off["y"].values, df_on["y"].values)[0, 1]
    assert corr < 0.999, f"Saturation had no effect: corr={corr:.6f}"


def test_equispaced_no_anticorrelation():
    """Equispaced phases over [0,pi) should not produce |corr| ≈ 1.0."""
    cfg = LadderConfig(
        n_channels=4, n_controls=0, n_periods=520,
        inter_channel_edges=[], seasonality="S1",
        phase_spread="equispaced", seed=2025,
    )
    df, *_ = generate_ladder(cfg)
    channels = df[["x1", "x2", "x3", "x4"]].values
    corr = np.corrcoef(channels, rowvar=False)
    np.fill_diagonal(corr, 0)
    max_abs = np.abs(corr).max()
    assert max_abs < 0.95, f"max |corr| = {max_abs:.3f}, phases still anticorrelated"
