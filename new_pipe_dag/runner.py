"""Benchmark runner — spec v2 §9 schema.

One row per seed × algorithm × stratum → results/runs.csv.
Supports both causal_discovery presets and T0.5 ladder configs.
"""
from __future__ import annotations

import re
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from new_pipe_dag.evaluation import (
    collinearity_diagnostics,
    evaluate_stratified,
    varsortability,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_preset(preset_name: str, seed: int = 2025,
                spec_version: str = "v1_legacy"):
    """Load data + ground truth from a causal preset.

    Falls back to causal_discovery's built-in synthetic DGP.
    Returns (data_df, var_names, channel_names, control_names,
             ghost_names, true_adj, truth_var_names).
    """
    repo_root = Path(__file__).parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    try:
        from cdnots.presets import get_preset_config
        from cdnots.core import generate_mmm_dataset

        config = get_preset_config(preset_name, seed=seed)
        config.spec_version = spec_version
        result = generate_mmm_dataset(config)
        data = result["data"]

        channel_cols = sorted(
            c for c in data.columns if re.match(r"^x\d+", c) and "contribution" not in c
        )
        control_cols = sorted(c for c in data.columns if re.match(r"^c\d+$", c))
        if "geo" in data.columns:
            first_geo = data["geo"].iloc[0]
            data = data[data["geo"] == first_geo].copy()

        var_names = channel_cols + control_cols + ["y"]
        data = data[var_names].reset_index(drop=True)
        true_graph = result["ground_truth"]["causal_graph"]
        true_adj = true_graph["adjacency_matrix"].astype(float)
        truth_var_names = list(true_graph["variable_names"])

        ghost_names = []
        for i, ch in enumerate(config.channels):
            col = f"x{i+1}_{ch.name}" if ch.name else f"x{i+1}"
            if ch.base_effectiveness == 0.0 and col in channel_cols:
                ghost_names.append(col)

        # Build edge_lags from CausalEdgeConfig when available
        edge_lags: dict[tuple[str, str], list[int]] = {}
        for ch_col in channel_cols:
            edge_lags[(ch_col, "y")] = [1]  # default: all channels → y at lag 1
        if hasattr(config, "causal_edges") and config.causal_edges:
            for ce in config.causal_edges:
                src_col = next((c for c in channel_cols if ce.source_channel in c), None)
                tgt_col = next((c for c in channel_cols if ce.target_channel in c), None)
                if src_col and tgt_col:
                    edge_lags[(src_col, tgt_col)] = [ce.lag]

    except (ImportError, ModuleNotFoundError):
        from causal_discovery.compare import _load_preset
        data, var_names, true_adj, truth_var_names = _load_preset(preset_name)
        truth_var_names = list(truth_var_names)
        channel_cols = [c for c in var_names if c.startswith("x")]
        control_cols = [c for c in var_names if c.startswith("c") and c != "y"]
        ghost_names = []
        edge_lags = {}  # unknown lags in fallback

    return (data, var_names, channel_cols, control_cols,
            ghost_names, true_adj, truth_var_names, edge_lags)


def load_ladder(level_name: str, seed: int = 2025):
    """Load data from T0.5 calibration ladder.

    Returns same shape as load_preset, plus edge_lags dict.
    """
    from new_pipe_dag.dgp import LADDER, generate_ladder

    cfg = LADDER[level_name]
    data, var_names, true_adj, truth_var_names, edge_lags = generate_ladder(cfg, seed=seed)
    channel_cols = [c for c in var_names if c.startswith("x")]
    control_cols = [c for c in var_names if c.startswith("c") and c != "y" and c != "season"]
    ghost_names = []
    return (data, var_names, channel_cols, control_cols,
            ghost_names, true_adj, truth_var_names, edge_lags)


# ---------------------------------------------------------------------------
# Recall ceiling (spec T0.1)
# ---------------------------------------------------------------------------

def recall_ceiling(true_adj: np.ndarray, truth_var_names: list[str],
                   max_lag: int, tau_min: int = 1,
                   edge_lags: dict[tuple[str, str], list[int]] | None = None,
                   ) -> dict[str, float]:
    """Analytic recall ceiling given max_lag and tau_min.

    An edge is detectable if any of its lags fall in [tau_min, max_lag].
    When edge_lags is empty/None, assumes all edges are detectable (conservative).
    """
    total = int(true_adj.sum())
    if not edge_lags or total == 0:
        return {"recall_ceiling": 1.0, "detectable_edges": total,
                "total_true_edges": total}

    detectable = 0
    n = true_adj.shape[0]
    for i in range(n):
        for j in range(n):
            if true_adj[i, j] == 0:
                continue
            src, tgt = truth_var_names[i], truth_var_names[j]
            lags = edge_lags.get((src, tgt), [1])
            if any(tau_min <= lag <= max_lag for lag in lags):
                detectable += 1
    ceiling = detectable / total if total > 0 else 1.0
    return {"recall_ceiling": ceiling, "detectable_edges": detectable,
            "total_true_edges": total}


# ---------------------------------------------------------------------------
# Edge signal strength (spec T0.7)
# ---------------------------------------------------------------------------

def edge_r2(data: pd.DataFrame, true_adj: np.ndarray,
            truth_var_names: list[str]) -> float:
    """Mean partial R² gain of true edges — proxy for signal strength."""
    import statsmodels.api as sm

    n = true_adj.shape[0]
    gains = []
    cols = [c for c in truth_var_names if c in data.columns]
    for i in range(n):
        for j in range(n):
            if true_adj[i, j] == 0 or i == j:
                continue
            src, tgt = truth_var_names[i], truth_var_names[j]
            if src not in data.columns or tgt not in data.columns:
                continue
            # Lag-1 regression
            df = data[[src, tgt]].copy()
            df["src_lag1"] = df[src].shift(1)
            df = df.dropna()
            if len(df) < 10:
                continue
            try:
                X0 = sm.add_constant(np.ones(len(df)))
                X1 = sm.add_constant(df[["src_lag1"]])
                r2_0 = sm.OLS(df[tgt], X0).fit().rsquared
                r2_1 = sm.OLS(df[tgt], X1).fit().rsquared
                gains.append(r2_1 - r2_0)
            except Exception:
                pass
    return float(np.mean(gains)) if gains else 0.0


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_benchmark(
    presets: list[str] | None = None,
    ladder_levels: list[str] | None = None,
    algorithms: list[str] | None = None,
    seeds: list[int] | None = None,
    seed_algos: list[int] | None = None,
    alpha: float = 0.05,
    max_lag: int = 3,
    output_dir: Path = Path("new_pipe_dag/results"),
    spec_version: str = "v1_legacy",
    use_background_knowledge: bool = False,
) -> pd.DataFrame:
    """Run benchmark grid → results/runs.csv (spec §9 schema)."""
    from causal_discovery.algorithms import REGISTRY

    if algorithms is None:
        algorithms = list(REGISTRY.keys())
    if seeds is None:
        seeds = [2025]
    if seed_algos is None:
        seed_algos = seeds  # default: same as data seeds

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build task list: (label, loader_fn, loader_kwargs)
    tasks = []
    for p in (presets or []):
        tasks.append(("preset", p, load_preset, {"spec_version": spec_version}))
    for lv in (ladder_levels or []):
        tasks.append(("ladder", lv, load_ladder, {}))

    if not tasks:
        print("Nothing to run. Pass --presets or --ladder.")
        return pd.DataFrame()

    rows: list[dict] = []

    for task_type, task_name, loader, loader_kw in tasks:
        for seed in seeds:
            print(f"\n{'='*60}")
            print(f"{task_type}: {task_name}  seed: {seed}  spec={spec_version}")
            print(f"{'='*60}")

            try:
                (data, var_names, ch_cols, ctrl_cols,
                 ghost_names, true_adj, truth_vars, edge_lags) = loader(task_name, seed, **loader_kw)
            except Exception as exc:
                print(f"  SKIP {task_name} seed={seed}: {exc}")
                continue

            # ponytail: persist generated data + ground truth for reproducibility
            data_dir = output_dir / "data" / f"{task_name}_seed{seed}_{spec_version}"
            data_dir.mkdir(parents=True, exist_ok=True)
            data.to_csv(data_dir / "data.csv", index=False)
            np.savetxt(data_dir / "true_adj.csv", true_adj, delimiter=",", fmt="%.0f")
            pd.Series(truth_vars).to_csv(data_dir / "variable_names.csv", index=False, header=False)
            pd.Series(ghost_names).to_csv(data_dir / "ghost_names.csv", index=False, header=False)

            # --- Diagnostics ---
            ch_data = data[[c for c in ch_cols if c in data.columns]].values
            col_diag = collinearity_diagnostics(ch_data, ch_cols)

            # ponytail: varsortability needs columns matching truth_vars
            truth_cols_in_data = [v for v in truth_vars if v in data.columns]
            if len(truth_cols_in_data) == len(truth_vars):
                vs_data = data[truth_cols_in_data].values
                vs_raw = varsortability(true_adj, vs_data)
                vs_z = varsortability(true_adj, StandardScaler().fit_transform(vs_data))
            else:
                vs_raw = vs_z = float("nan")

            ceil = recall_ceiling(true_adj, truth_vars, max_lag, edge_lags=edge_lags)
            mean_r2 = edge_r2(data, true_adj, truth_vars)

            # pct_zero_spend (spec A0.3 diagnostic)
            ch_vals = data[[c for c in ch_cols if c in data.columns]].values
            pct_zero = float((ch_vals == 0).mean()) if ch_vals.size else 0.0

            # Background knowledge: domain constraints to skip impossible links.
            # Reduces test count (especially on large variable sets) without
            # affecting recall on plausible edges.
            forbidden_links = _build_forbidden_links(
                var_names, ch_cols, ctrl_cols, use_background_knowledge
            )

            print(f"  channels={len(ch_cols)} controls={len(ctrl_cols)} "
                  f"ghosts={len(ghost_names)} T={len(data)} "
                  f"forbidden={len(forbidden_links)}")
            print(f"  max|r|={col_diag['max_abs_corr']:.3f}  "
                  f"varsort_z={vs_z:.3f}  mean_r2={mean_r2:.4f}  "
                  f"pct_zero={pct_zero:.3f}")

            for seed_algo in seed_algos:
              for alg_name in algorithms:
                if alg_name not in REGISTRY:
                    print(f"  SKIP unknown algorithm: {alg_name}")
                    continue

                fn = REGISTRY[alg_name]
                print(f"  Running {alg_name} (seed_algo={seed_algo})...", end=" ", flush=True)
                run_id = str(uuid.uuid4())[:8]

                try:
                    t0 = time.perf_counter()
                    graph = fn(data, var_names, alpha=alpha, max_lag=max_lag,
                              forbidden_links=forbidden_links)
                    runtime = time.perf_counter() - t0
                    print(f"done ({runtime:.1f}s)")
                except Exception as exc:
                    print(f"FAILED: {exc}")
                    continue

                # ponytail: evaluate on the intersection of truth and learned vars.
                # Hidden nodes (e.g. "season" in S0/S2) are in truth but not learned.
                learned_names = list(graph.variable_names)
                observable_truth = [v for v in truth_vars if v in learned_names]
                learned_sub = _extract_submatrix(
                    graph.adjacency_matrix, learned_names, observable_truth
                )
                true_sub = _extract_submatrix(true_adj, truth_vars, observable_truth)

                strata = evaluate_stratified(
                    learned=learned_sub,
                    true=true_sub,
                    variable_names=observable_truth,
                    channel_names=[v for v in observable_truth if v not in ("y", "season")],
                    ghost_names=[g for g in ghost_names if g in observable_truth],
                    target="y",
                )

                for stratum, metrics in strata.items():
                    rows.append({
                        # --- identification (§9) ---
                        "run_id": run_id,
                        "timestamp": datetime.now().isoformat(),
                        "phase": "B" if task_type == "ladder" else "exploratory",
                        # --- data ---
                        "preset": task_name,
                        "task_type": task_type,
                        "spec_version": "v2_ladder" if task_type == "ladder" else spec_version,
                        "seed_data": seed,
                        "seed_algo": seed_algo,
                        # --- algorithm ---
                        "algorithm": alg_name,
                        "ci_test": graph.metadata.get("ci_test", alg_name),
                        "alpha": alpha,
                        "max_lag": max_lag,
                        "tau_min": 1,  # ponytail: all current algorithms use tau_min=1
                        # --- stratum + metrics ---
                        "stratum": stratum,
                        **metrics,
                        # --- diagnostics ---
                        "recall_ceiling": ceil["recall_ceiling"],
                        "max_abs_corr_channels": col_diag["max_abs_corr"],
                        "median_abs_corr_channels": col_diag["median_abs_corr"],
                        "condition_number": col_diag["condition_number"],
                        "varsortability_raw": vs_raw,
                        "varsortability_z": vs_z,
                        "pct_zero_spend": pct_zero,
                        "mean_r2_partial_edges": mean_r2,
                        # --- dimensions ---
                        "n_channels": len(ch_cols),
                        "n_controls": len(ctrl_cols),
                        "n_ghosts": len(ghost_names),
                        "n_periods": len(data),
                        "n_true_edges": int(true_adj.sum()),
                        "runtime_seconds": runtime,
                    })

                ov = strata.get("overall", {})
                print(f"    P={ov.get('precision', 0):.3f}  "
                      f"R={ov.get('recall', 0):.3f}  "
                      f"F1={ov.get('f1', 0):.3f}  "
                      f"SHD={ov.get('shd', '?')}")

    df = pd.DataFrame(rows)
    if not df.empty:
        csv_path = output_dir / "runs.csv"
        write_header = not csv_path.exists()
        df.to_csv(csv_path, mode="a", index=False, header=write_header)
        print(f"\nResults: {csv_path} ({len(df)} new rows, append={'existing' if not write_header else 'new file'})")
    return df


def _build_forbidden_links(
    var_names: list[str],
    channel_cols: list[str],
    control_cols: list[str],
    use_background_knowledge: bool,
) -> set[tuple[str, str]]:
    """Build forbidden links from MMM domain knowledge.

    Encodes impossible causal directions to skip testing them entirely,
    reducing runtime on large variable sets without affecting recall on
    plausible edges.

    Constraints:
      - y → anything: sales don't cause ad spend or controls
      - controls → channels: exogenous controls don't cause channel spend
      - channels → controls: media spend doesn't cause exogenous controls
      - y → controls: sales don't cause controls (redundant with y→anything, explicit for clarity)
    """
    if not use_background_knowledge:
        return set()

    forbidden: set[tuple[str, str]] = set()
    # y cannot cause anything
    for v in var_names:
        if v != "y":
            forbidden.add(("y", v))
    # Controls cannot cause channels (exogenous by design)
    for ctrl in control_cols:
        for ch in channel_cols:
            forbidden.add((ctrl, ch))
    # Channels cannot cause controls (media spend doesn't drive exogenous vars)
    for ch in channel_cols:
        for ctrl in control_cols:
            forbidden.add((ch, ctrl))
    return forbidden


def _extract_submatrix(
    adj: np.ndarray, all_names: list[str], sub_names: list[str]
) -> np.ndarray:
    """Extract submatrix matching sub_names from a larger adjacency matrix."""
    indices = [all_names.index(v) for v in sub_names if v in all_names]
    return adj[np.ix_(indices, indices)]
