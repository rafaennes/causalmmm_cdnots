"""Comparison pipeline: run all algorithms, evaluate, plot.

Entry points:
  run_comparison(preset_name, ...) -> pd.DataFrame
  _load_preset(preset_name) -> (data_df, var_names, true_adj, truth_var_names)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for background execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console

from causal_discovery.algorithms import REGISTRY
from causal_discovery.graph import CausalGraph


def run_comparison(
    preset_name: str,
    algorithms: list[str] | None = None,
    alpha: float = 0.05,
    max_lag: int = 2,
    output_dir: Path = Path("causal_discovery/results"),
    console: Console | None = None,
) -> pd.DataFrame:
    """Run causal discovery algorithms on a preset and produce comparison outputs.

    Parameters
    ----------
    preset_name : preset name passed to get_preset_config() (e.g. "causal_business").
    algorithms : list of algorithm names to run, or None for all in REGISTRY.
    alpha : significance level for CI-based algorithms.
    max_lag : maximum temporal lag.
    output_dir : root output directory. Results written to output_dir/preset_name/.
    console : rich Console for progress output. Created if None.

    Returns
    -------
    DataFrame with rows=algorithms, cols=structural metrics + runtime_seconds.
    """
    if console is None:
        console = Console()

    out = Path(output_dir) / preset_name
    out.mkdir(parents=True, exist_ok=True)

    alg_fns = {
        k: v for k, v in REGISTRY.items()
        if algorithms is None or k in algorithms
    }

    data_df, var_names, true_adj, truth_var_names = _load_preset(preset_name)

    graphs: dict[str, CausalGraph] = {}
    for name, fn in alg_fns.items():
        console.print(f"  Running [bold]{name}[/bold]...")
        try:
            graphs[name] = fn(data_df, var_names, alpha=alpha, max_lag=max_lag)
            console.print(f"    done in {graphs[name].runtime_seconds:.1f}s")
        except Exception as exc:
            console.print(f"  [red]{name} failed: {exc}[/red]")

    # Evaluate: extract channels+y submatrix to match true_adj (no controls)
    eval_vars = list(truth_var_names)
    metrics: dict[str, dict] = {}
    for name, g in graphs.items():
        m = g.evaluate(true_adj, var_subset=eval_vars)
        m["runtime_seconds"] = g.runtime_seconds
        metrics[name] = m

    _plot_adjacency_comparison(graphs, true_adj, truth_var_names, out)
    _plot_metrics_bar(metrics, out)
    _plot_runtime_vs_f1(metrics, out)

    df = pd.DataFrame(metrics).T
    df.to_csv(out / "metrics.csv")
    console.print(f"\n  Results saved to [bold]{out}[/bold]")
    return df


def _load_preset(
    preset_name: str,
) -> tuple[pd.DataFrame, list[str], np.ndarray, tuple[str, ...]]:
    """Load synthetic dataset from cdnots preset.

    Tries cdnots.core first (requires pixi env); falls back to a built-in
    minimal synthetic generator if the cdnots submodules are unavailable.

    Returns
    -------
    data_df : single-geo DataFrame with channel, control and y columns.
    var_names : channel_cols + control_cols + ["y"].
    true_adj : (n_channels+1, n_channels+1) ground truth adjacency matrix.
    truth_var_names : variable names matching true_adj rows/cols.
    """
    repo_root = Path(__file__).parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    try:
        from cdnots.core import generate_mmm_dataset
        from cdnots.presets import get_preset_config

        config = get_preset_config(preset_name)
        result = generate_mmm_dataset(config)
        data = result["data"]

        channel_cols = sorted(
            c for c in data.columns if re.match(r"^x\d+", c) and "contribution" not in c
        )
        control_cols = sorted(
            c for c in data.columns if re.match(r"^c\d+$", c)
        )

        # Single-geo: use first geo to match experiment setup
        if "geo" in data.columns:
            first_geo = data["geo"].iloc[0]
            data = data[data["geo"] == first_geo].copy()

        var_names = channel_cols + control_cols + ["y"]
        data = data[var_names].reset_index(drop=True)

        true_graph = result["ground_truth"]["causal_graph"]
        true_adj = true_graph["adjacency_matrix"].astype(float)
        truth_var_names = tuple(true_graph["variable_names"])

        return data, var_names, true_adj, truth_var_names

    except (ImportError, ModuleNotFoundError):
        return _synthetic_preset(preset_name)


# Preset specs: (n_channels, n_controls, T, causal_edges)
# causal_edges: list of (src_idx, tgt_idx) inter-channel edges
_PRESET_SPECS: dict[str, tuple[int, int, int, list]] = {
    "small_business":    (4,  1, 104, []),
    "medium_business":   (6,  2, 156, []),
    "large_business":    (8,  3, 208, []),
    "causal_business":   (4,  1, 104, [(0, 1)]),
    "causal_large":      (10, 3, 260, [(0, 1), (2, 3), (4, 5)]),
    "growing_business":  (4,  1, 104, []),
    "basic":             (3,  0, 52,  []),
    "seasonal":          (4,  1, 104, []),
    "multi_region":      (4,  1, 104, []),
}


def _synthetic_preset(
    preset_name: str,
) -> tuple[pd.DataFrame, list[str], np.ndarray, tuple[str, ...]]:
    """Synthetic MMM preset with realistic marketing data characteristics.

    DGP mirrors real marketing data:
      1. Weekly spend innovations with per-channel regime shifts (budget changes)
      2. Inter-channel causal spillover at lag 1 + lag 2 (e.g., TV → Search)
      3. Geometric adstock (carryover decay) per channel
      4. Diminishing returns via log(1+x) saturation
      5. Sales (y) driven by saturated adstocked spend at lag 1 + lag 2
      6. Quarterly seasonality in y
      7. Controls (macro factors) with weak effect

    Causal structure is injected at the innovation level (before adstock)
    so the signal propagates through the full marketing pipeline. Each
    channel has independent regime timing to avoid shared-trend confounding.
    Coefficients are calibrated for detectable SNR at T=104.
    """
    if preset_name not in _PRESET_SPECS:
        n_ch, n_co, T, edges = 4, 1, 104, []
    else:
        n_ch, n_co, T, edges = _PRESET_SPECS[preset_name]

    rng = np.random.default_rng(2025)

    # === Step 1: Generate independent weekly spend innovations ===
    # Each channel has its own base level, noise, and regime shift timing
    innovations = {}
    for i in range(n_ch):
        base_spend = rng.uniform(500, 2000)         # weekly base spend ($)
        noise_std = base_spend * 0.15                # 15% week-to-week variation
        shift_t = rng.integers(T // 4, 3 * T // 4)  # channel-specific breakpoint
        shift_dir = rng.choice([-1, 1])
        shift_pct = rng.uniform(0.3, 0.6)           # 30-60% budget change

        innov = np.empty(T)
        for t in range(T):
            mu = base_spend * (1 + shift_dir * shift_pct) if t >= shift_t else base_spend
            std = noise_std * rng.uniform(1.2, 1.5) if t >= shift_t else noise_std
            innov[t] = rng.normal(mu, std)
        innov = np.clip(innov, 50, None)  # floor: min $50/week
        innovations[i] = innov

    # === Step 2: Inter-channel causal effects at lag 1 + lag 2 ===
    # Applied to innovations (budget decisions), before adstock.
    # "TV spend this week causes search spend to increase next week"
    # Coefficients are proportional: a fraction of src spend spills into tgt.
    for src, tgt in edges:
        spillover_lag1 = rng.uniform(0.15, 0.25)  # 15-25% of src spills at lag 1
        spillover_lag2 = rng.uniform(0.05, 0.12)   # 5-12% at lag 2
        innovations[tgt][1:] += spillover_lag1 * innovations[src][:-1]
        innovations[tgt][2:] += spillover_lag2 * innovations[src][:-2]

    # === Step 3: Geometric adstock (carryover) ===
    spend = {}
    adstock_decays = {}
    for i in range(n_ch):
        decay = rng.uniform(0.3, 0.6)  # typical weekly decay rates
        adstock_decays[i] = decay
        x = innovations[i].copy()
        for t in range(1, T):
            x[t] = x[t] + decay * x[t - 1]
        spend[f"x{i+1}"] = x

    # === Step 4: Saturation (diminishing returns) ===
    # log(1 + x/K) where K = median spend — standard MMM transform.
    # This compresses the signal, so we calibrate channel→y coefficients
    # to compensate.
    saturated = {}
    for i in range(n_ch):
        x = spend[f"x{i+1}"]
        K = np.median(x)
        saturated[i] = np.log1p(x / K)

    # === Step 5: Controls ===
    controls = {}
    for j in range(n_co):
        controls[f"c{j+1}"] = rng.normal(0, 1, T)

    # === Step 6: Generate y (sales) ===
    # y(t) = intercept + seasonality + sum_i[beta1_i * sat_i(t-1) + beta2_i * sat_i(t-2)]
    #         + control_effects + noise
    #
    # Calibration: compute the std of the saturated signals to set betas
    # so that the total channel contribution has std ≈ 2-3x the noise std.
    sat_stds = [saturated[i].std() for i in range(n_ch)]
    noise_std_y = 500  # weekly sales noise ($)
    # Target: each channel contributes ~(noise_std_y * 2.0) / n_ch in std.
    # Total channel signal ~ 2× noise — strong enough to survive log
    # saturation compression and remain detectable at T=104 with
    # nonlinear CI tests. Per-channel SNR ≈ 0.5 for 4 channels.
    target_per_ch = noise_std_y * 2.0 / max(n_ch, 1)

    beta_lag1 = np.array([target_per_ch / (s + 1e-8) * rng.uniform(0.7, 1.0)
                          for s in sat_stds])
    beta_lag2 = np.array([target_per_ch / (s + 1e-8) * rng.uniform(0.25, 0.45)
                          for s in sat_stds])

    base_sales = rng.uniform(8000, 15000)
    # Quarterly seasonality
    season = 0.03 * base_sales * np.sin(2 * np.pi * np.arange(T) / 13)

    y = np.full(T, base_sales) + season + rng.normal(0, noise_std_y, T)
    for i in range(n_ch):
        s = saturated[i]
        y[1:] += beta_lag1[i] * s[:-1]
        y[2:] += beta_lag2[i] * s[:-2]
    for j in range(n_co):
        y += rng.uniform(-50, 50) * controls[f"c{j+1}"]

    # === Build DataFrame ===
    data = pd.DataFrame({**spend, **controls, "y": y})
    channel_cols = [f"x{i+1}" for i in range(n_ch)]
    control_cols = [f"c{j+1}" for j in range(n_co)]
    var_names = channel_cols + control_cols + ["y"]

    # === Ground truth adjacency ===
    truth_var_names = tuple(channel_cols + ["y"])
    n_truth = len(truth_var_names)
    true_adj = np.zeros((n_truth, n_truth))
    y_truth_idx = n_truth - 1
    for i in range(n_ch):
        true_adj[i, y_truth_idx] = 1.0
    for src, tgt in edges:
        true_adj[src, tgt] = 1.0

    return data[var_names], var_names, true_adj, truth_var_names


# --- Plotting helpers ---------------------------------------------------------


def _get_sub_adj(
    graph: CausalGraph, truth_var_names: tuple[str, ...]
) -> np.ndarray:
    """Extract submatrix of graph.adjacency_matrix matching truth_var_names."""
    var_list = list(graph.variable_names)
    indices = [var_list.index(v) for v in truth_var_names]
    return graph.adjacency_matrix[np.ix_(indices, indices)]


def _make_rgba(learned: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Build RGBA image: TP=green, FP=red, FN=blue, TN=white, diagonal=light gray."""
    n = true.shape[0]
    rgba = np.ones((n, n, 4))
    for i in range(n):
        for j in range(n):
            if i == j:
                rgba[i, j] = [0.85, 0.85, 0.85, 1.0]
            elif true[i, j] == 1 and learned[i, j] == 1:
                rgba[i, j] = [0.0, 0.7, 0.0, 1.0]   # TP: green
            elif true[i, j] == 0 and learned[i, j] == 1:
                rgba[i, j] = [0.8, 0.0, 0.0, 1.0]   # FP: red
            elif true[i, j] == 1 and learned[i, j] == 0:
                rgba[i, j] = [0.0, 0.0, 0.8, 1.0]   # FN: blue
            # TN stays white (default ones)
    return rgba


def _label_axes(ax: plt.Axes, var_names: tuple[str, ...]) -> None:
    n = len(var_names)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(var_names, rotation=45, ha="right", fontsize=6)
    ax.set_yticklabels(var_names, fontsize=6)


def _plot_adjacency_comparison(
    graphs: dict[str, CausalGraph],
    true_adj: np.ndarray,
    truth_var_names: tuple[str, ...],
    out: Path,
) -> None:
    """One heatmap per algorithm + ground truth. TP=green, FP=red, FN=blue."""
    n_plots = len(graphs) + 1
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4.5))
    if n_plots == 1:
        axes = [axes]

    # Ground truth panel
    gt_rgba = _make_rgba(true_adj, true_adj)
    axes[0].imshow(gt_rgba, aspect="auto")
    axes[0].set_title("Ground Truth", fontsize=9, fontweight="bold")
    _label_axes(axes[0], truth_var_names)

    for ax, (name, g) in zip(axes[1:], graphs.items()):
        sub = _get_sub_adj(g, truth_var_names)
        rgba = _make_rgba(sub, true_adj)
        ax.imshow(rgba, aspect="auto")
        ax.set_title(name, fontsize=9)
        _label_axes(ax, truth_var_names)

    import matplotlib.patches as mpatches
    legend_patches = [
        mpatches.Patch(color=(0, 0.7, 0), label="TP"),
        mpatches.Patch(color=(0.8, 0, 0), label="FP"),
        mpatches.Patch(color=(0, 0, 0.8), label="FN"),
        mpatches.Patch(color=(1, 1, 1), label="TN"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.05))
    plt.tight_layout()
    plt.savefig(out / "adjacency_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_metrics_bar(metrics: dict[str, dict], out: Path) -> None:
    """Grouped bar chart: precision / recall / F1 / FDR per algorithm."""
    metric_keys = ["precision", "recall", "f1", "fdr"]
    alg_names = list(metrics.keys())
    x = np.arange(len(alg_names))
    width = 0.18

    fig, ax = plt.subplots(figsize=(max(8, len(alg_names) * 2), 5))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    for i, (key, color) in enumerate(zip(metric_keys, colors)):
        vals = [metrics[a].get(key, 0.0) for a in alg_names]
        ax.bar(x + i * width, vals, width, label=key, color=color)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(alg_names, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Structural Recovery Metrics by Algorithm")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "metrics_bar.png", dpi=150)
    plt.close()


def _plot_runtime_vs_f1(metrics: dict[str, dict], out: Path) -> None:
    """Efficiency frontier: F1 vs runtime (log scale)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, m in metrics.items():
        rt = m.get("runtime_seconds", 0)
        f1 = m.get("f1", 0)
        ax.scatter(max(rt, 0.01), f1, s=120, zorder=5)
        ax.annotate(
            name,
            (max(rt, 0.01), f1),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=9,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Runtime (seconds, log scale)")
    ax.set_ylabel("F1 Score")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Efficiency Frontier: F1 vs Runtime")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "runtime_vs_f1.png", dpi=150)
    plt.close()
