"""CLI: python -m new_pipe_dag --ladder L0 L1 L2 --algorithms granger pcmci_cmiknn"""
from __future__ import annotations

import argparse
from pathlib import Path

from new_pipe_dag.runner import run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Causal discovery benchmark — spec v2")
    parser.add_argument("--presets", nargs="+", default=None,
                        help="Preset names (causal_business, causal_large)")
    parser.add_argument("--ladder", nargs="+", default=None,
                        help="T0.5 ladder levels (L0-L9, or 'all')")
    parser.add_argument("--algorithms", nargs="+", default=None,
                        help="Algorithm names (default: all registered)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2025])
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--max-lag", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("new_pipe_dag/results"))
    parser.add_argument("--spec-version", choices=["v1_legacy", "v2"], default="v1_legacy",
                        help="DGP version (v1_legacy=original bugs, v2=fixed)")
    args = parser.parse_args()

    ladder = args.ladder
    if ladder and "all" in ladder:
        ladder = [f"L{i}" for i in range(10)]

    df = run_benchmark(
        presets=args.presets,
        ladder_levels=ladder,
        algorithms=args.algorithms,
        seeds=args.seeds,
        alpha=args.alpha,
        max_lag=args.max_lag,
        output_dir=args.output_dir,
        spec_version=args.spec_version,
    )
    if not df.empty:
        print(f"\n{'='*60}")
        print("Summary by level × algorithm × stratum:")
        summary = df.groupby(["preset", "algorithm", "stratum"])[
            ["precision", "recall", "f1", "shd"]
        ].mean().round(3)
        print(summary.to_string())


if __name__ == "__main__":
    main()
