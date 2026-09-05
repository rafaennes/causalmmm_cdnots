"""Stationarize causal presets and run causal discovery algorithms.

Runs algorithms that assume stationarity on pre-differenced data:
  - granger:      linear, pairwise (baseline)
  - pcmci_cmiknn: nonlinear via CMIknn
  - cedar:        nonlinear via distance correlation
  - dynotears:    linear VAR thresholding

Usage:
    .venv/bin/python3.12 -m causal_discovery.stationary.run_stationary_granger
    .venv/bin/python3.12 -m causal_discovery.stationary.run_stationary_granger --presets causal_business causal_large
    .venv/bin/python3.12 -m causal_discovery.stationary.run_stationary_granger --algorithms granger pcmci_cmiknn
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from causal_discovery.compare import (
    _load_preset, _PRESET_SPECS, _plot_adjacency_comparison,
)
from causal_discovery.algorithms import REGISTRY
from causal_discovery.graph import CausalGraph

# Algorithms that assume stationarity — pre-differencing helps these
STATIONARY_ALGORITHMS = ["granger", "pcmci_cmiknn", "cedar", "dynotears"]


OUTPUT_DIR = Path("causal_discovery/results/stationary")
ADF_ALPHA = 0.05
MAX_DIFF = 2  # at most second differencing


def stationarize(series: pd.Series, alpha: float = ADF_ALPHA) -> tuple[pd.Series, int]:
    """Difference a series until ADF rejects the unit root (up to MAX_DIFF)."""
    s = series.dropna()
    for d in range(MAX_DIFF + 1):
        pval = adfuller(s, autolag="AIC")[1]
        if pval < alpha:
            return s, d
        s = s.diff().dropna()
    return s, MAX_DIFF


def verify_stationarity(
    data: pd.DataFrame, var_names: list[str], alpha: float = ADF_ALPHA
) -> dict[str, float]:
    """Run ADF on each column, return {col: p-value}. Warns if any still non-stationary."""
    adf_pvals = {}
    for col in var_names:
        pval = adfuller(data[col].dropna(), autolag="AIC")[1]
        adf_pvals[col] = pval
        if pval >= alpha:
            print(f"  WARNING: {col} still non-stationary after differencing (ADF p={pval:.4f})")
    return adf_pvals


def stationarize_df(
    data: pd.DataFrame, var_names: list[str], alpha: float = ADF_ALPHA
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Stationarize each variable, align lengths, return cleaned DataFrame + diff orders."""
    diff_orders: dict[str, int] = {}
    stationary_cols: dict[str, pd.Series] = {}
    min_len = len(data)

    for col in var_names:
        s, d = stationarize(data[col], alpha)
        diff_orders[col] = d
        stationary_cols[col] = s.reset_index(drop=True)
        min_len = min(min_len, len(s))

    # ponytail: align all series to shortest length after differencing
    df_out = pd.DataFrame({c: stationary_cols[c].iloc[:min_len] for c in var_names})
    return df_out, diff_orders


def run_one_preset(
    preset_name: str,
    algorithms: list[str],
    alpha: float = 0.05,
    max_lag: int = 2,
) -> dict:
    """Load preset, stationarize, run algorithms, return results dict."""
    data_df, var_names, true_adj, truth_var_names = _load_preset(preset_name)

    stat_df, diff_orders = stationarize_df(data_df, var_names)
    print(f"  Diff orders: {diff_orders}")

    adf_pvals = verify_stationarity(stat_df, var_names)
    n_fail = sum(1 for p in adf_pvals.values() if p >= alpha)
    print(f"  ADF check: {len(var_names) - n_fail}/{len(var_names)} stationary")

    graphs: dict[str, CausalGraph] = {}
    metrics_all: dict[str, dict] = {}
    eval_vars = list(truth_var_names)

    for alg_name in algorithms:
        fn = REGISTRY[alg_name]
        print(f"  Running {alg_name}...", end=" ", flush=True)
        try:
            g = fn(stat_df, var_names, alpha=alpha, max_lag=max_lag)
            graphs[alg_name] = g
            m = g.evaluate(true_adj, var_subset=eval_vars)
            m["runtime_seconds"] = g.runtime_seconds
            metrics_all[alg_name] = m
            print(f"done ({g.runtime_seconds:.1f}s)")
        except Exception as exc:
            print(f"FAILED: {exc}")

    return {
        "metrics": metrics_all,
        "graphs": graphs,
        "diff_orders": diff_orders,
        "adf_pvals": adf_pvals,
        "true_adj": true_adj,
        "truth_var_names": truth_var_names,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stationarized causal discovery on presets")
    parser.add_argument(
        "--presets", nargs="*", default=list(_PRESET_SPECS.keys()),
        help="Preset names to run (default: all)",
    )
    parser.add_argument(
        "--algorithms", nargs="*", default=STATIONARY_ALGORITHMS,
        help=f"Algorithms to run (default: {STATIONARY_ALGORITHMS})",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--max-lag", type=int, default=2)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []

    for preset in args.presets:
        print(f"\n{'='*60}")
        print(f"Preset: {preset}")
        print(f"{'='*60}")

        result = run_one_preset(preset, args.algorithms, alpha=args.alpha, max_lag=args.max_lag)

        preset_dir = OUTPUT_DIR / preset
        preset_dir.mkdir(parents=True, exist_ok=True)

        # Save per-algorithm adjacency matrices
        for alg_name, g in result["graphs"].items():
            adj_df = pd.DataFrame(
                g.adjacency_matrix, index=g.variable_names, columns=g.variable_names,
            )
            adj_df.to_csv(preset_dir / f"adjacency_matrix_{alg_name}.csv")

        # Save ground truth + stationarity info
        pd.DataFrame(
            result["true_adj"],
            index=result["truth_var_names"], columns=result["truth_var_names"],
        ).to_csv(preset_dir / "true_adjacency_matrix.csv")

        pd.Series(result["diff_orders"]).to_csv(preset_dir / "diff_orders.csv", header=["order"])
        pd.Series(result["adf_pvals"]).to_csv(preset_dir / "adf_pvals_post.csv", header=["adf_pval"])

        # Plot all algorithms side by side
        _plot_adjacency_comparison(
            result["graphs"], result["true_adj"], result["truth_var_names"], preset_dir,
        )

        # Collect metrics
        for alg_name, m in result["metrics"].items():
            row = {"preset": preset, "algorithm": alg_name, **m}
            all_rows.append(row)
            print(f"  {alg_name}: P={m['precision']:.3f}  R={m['recall']:.3f}  "
                  f"F1={m['f1']:.3f}  SHD={m['shd']}  runtime={m['runtime_seconds']:.2f}s")

    # Save combined metrics
    metrics_df = pd.DataFrame(all_rows)
    metrics_df.to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
