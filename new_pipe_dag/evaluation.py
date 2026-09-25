"""Stratified causal structure evaluation (spec v2 T0.2) + varsortability (L2)."""
from __future__ import annotations

import numpy as np


def _confusion(true_flat: np.ndarray, learned_flat: np.ndarray) -> dict[str, float]:
    tp = int(((true_flat == 1) & (learned_flat == 1)).sum())
    fp = int(((true_flat == 0) & (learned_flat == 1)).sum())
    fn = int(((true_flat == 1) & (learned_flat == 0)).sum())
    tn = int(((true_flat == 0) & (learned_flat == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if (tp + fp) and (tp + fn) and (precision + recall) > 0 else float("nan"))
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "fdr": fp / (tp + fp) if (tp + fp) else float("nan"),
        "shd": int((true_flat != learned_flat).sum()),
    }


def evaluate_stratified(
    learned: np.ndarray,
    true: np.ndarray,
    variable_names: list[str],
    channel_names: list[str],
    control_names: list[str] = (),
    ghost_names: list[str] = (),
    target: str = "y",
) -> dict[str, dict[str, float]]:
    """Evaluate learned vs true adjacency, stratified by edge type.

    Returns dict of stratum_name -> metrics dict.
    """
    idx = {n: i for i, n in enumerate(variable_names)}
    y = idx[target]
    ch = [idx[c] for c in channel_names if c in idx]
    ct = [idx[c] for c in control_names if c in idx]
    gh = {idx[g] for g in ghost_names if g in idx}
    n = len(variable_names)

    strata = {
        "overall": [(i, j) for i in range(n) for j in range(n) if i != j],
        "channel_to_y": [(i, y) for i in ch if i not in gh],
        "ghost_to_y": [(i, y) for i in gh],
        "channel_to_channel": [(i, j) for i in ch for j in ch if i != j],
    }
    if ct:
        strata["control_to_channel"] = [(i, j) for i in ct for j in ch]

    out = {}
    for name, pairs in strata.items():
        if not pairs:
            continue
        t = np.array([true[i, j] for i, j in pairs])
        l = np.array([learned[i, j] for i, j in pairs])
        out[name] = _confusion(t, l)
    return out


def varsortability(adj: np.ndarray, data: np.ndarray) -> float:
    """Fraction of directed edges where var(source) < var(target).

    Reisach, Seiler & Weichwald (2021). 0.5 = chance, 1.0 = perfectly sortable.
    """
    variances = np.var(data, axis=0)
    n = adj.shape[0]
    total = 0
    correct = 0
    for i in range(n):
        for j in range(n):
            if adj[i, j] > 0:
                total += 1
                if variances[i] < variances[j]:
                    correct += 1
    return correct / total if total else float("nan")


def collinearity_diagnostics(data: np.ndarray, names: list[str]) -> dict[str, float]:
    """Max |r|, median |r|, condition number, VIF over channel columns."""
    nan3 = {"max_abs_corr": float("nan"), "median_abs_corr": float("nan"),
            "condition_number": float("nan"), "mean_vif": float("nan"),
            "max_vif": float("nan")}
    if data.shape[1] < 2:
        return nan3
    corr = np.corrcoef(data, rowvar=False)
    np.fill_diagonal(corr, 0.0)
    abs_corr = np.abs(corr)
    upper = abs_corr[np.triu_indices_from(abs_corr, k=1)]
    # ponytail: condition number on standardized data, not raw
    std = data.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    X = (data - data.mean(axis=0)) / std
    cond = float(np.linalg.cond(X))
    # ponytail: VIF via OLS R², no statsmodels import needed
    # VIF_j = 1/(1 - R²_j) where R²_j is from regressing col j on all others
    vifs = _compute_vif(X)
    return {
        "max_abs_corr": float(upper.max()) if len(upper) else float("nan"),
        "median_abs_corr": float(np.median(upper)) if len(upper) else float("nan"),
        "condition_number": cond,
        "mean_vif": float(np.mean(vifs)),
        "max_vif": float(np.max(vifs)),
    }


def _compute_vif(X: np.ndarray) -> np.ndarray:
    """VIF for each column of standardized X via least-squares R²."""
    n, p = X.shape
    vifs = np.empty(p)
    for j in range(p):
        others = np.delete(X, j, axis=1)
        y = X[:, j]
        # ponytail: lstsq is stdlib-numpy, no statsmodels needed
        coef, residuals, _, _ = np.linalg.lstsq(others, y, rcond=None)
        ss_res = np.sum((y - others @ coef) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        vifs[j] = 1.0 / (1.0 - r2) if r2 < 1.0 else float("inf")
    return vifs
