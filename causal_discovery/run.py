#!/usr/bin/env python
"""CLI entry point for causal discovery algorithm comparison.

Usage examples:
    # Run all algorithms on causal_business preset
    python -m causal_discovery.run --preset causal_business

    # Run selected algorithms on causal_large with custom significance level
    python -m causal_discovery.run --preset causal_large \\
        --algorithms cdnots dynotears --alpha 0.10 --max_lag 2

    # Write results to a custom directory
    python -m causal_discovery.run --preset causal_business \\
        --output_dir /tmp/cd_results
"""
from __future__ import annotations

import argparse
from pathlib import Path

from causal_discovery.algorithms import REGISTRY
from causal_discovery.compare import run_comparison


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare causal discovery algorithms on MMM synthetic data"
    )
    parser.add_argument(
        "--preset",
        default="causal_business",
        help="Dataset preset (e.g. small_business, causal_business, causal_large)",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=None,
        choices=list(REGISTRY.keys()),
        help="Algorithms to run (default: all)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for CI-based algorithms (default: 0.05)",
    )
    parser.add_argument(
        "--max_lag",
        type=int,
        default=2,
        help="Maximum temporal lag (default: 2)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("causal_discovery/results"),
        help="Root directory for output plots and CSV (default: causal_discovery/results)",
    )
    args = parser.parse_args()

    results = run_comparison(
        preset_name=args.preset,
        algorithms=args.algorithms,
        alpha=args.alpha,
        max_lag=args.max_lag,
        output_dir=args.output_dir,
    )
    print("\n" + results.to_string())


if __name__ == "__main__":
    main()
