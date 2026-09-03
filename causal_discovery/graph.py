from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from statsmodels.stats.multitest import multipletests


@dataclass
class CausalGraph:
    """Result of causal discovery on time-series data.

    adjacency_matrix: (n_vars, n_vars) binary float. adj[i, j] = 1 means i -> j.
    variable_names:   length n_vars; last element is always "y".
    algorithm:        one of "granger", "pcmci_cmiknn", "lpcmci", "dynotears", "cdnots".
    runtime_seconds:  wall-clock time of the discover() call.
    metadata:         algorithm-specific diagnostics (p-values, weight matrix, etc.).
    """

    adjacency_matrix: np.ndarray
    variable_names: tuple[str, ...]
    algorithm: str
    runtime_seconds: float
    metadata: dict = field(default_factory=dict)

    def evaluate(
        self,
        true_adj: np.ndarray,
        var_subset: list[str] | None = None,
    ) -> dict[str, float]:
        """Compare learned graph against ground truth adjacency matrix.

        Parameters
        ----------
        true_adj : (m, m) binary array — ground truth edges.
        var_subset : if given, extract that submatrix from self.adjacency_matrix
            before comparing. Use when true_adj covers only channels+y but the
            learned graph covers channels+controls+y.

        Returns
        -------
        dict with keys: precision, recall, f1, fdr, shd, tp, fp, fn, tn.
        """
        if var_subset is not None:
            var_list = list(self.variable_names)
            indices = [var_list.index(v) for v in var_subset]
            learned = self.adjacency_matrix[np.ix_(indices, indices)]
        else:
            learned = self.adjacency_matrix

        n = true_adj.shape[0]
        mask = ~np.eye(n, dtype=bool)
        true_flat = true_adj[mask].astype(float)
        learned_flat = learned[mask].astype(float)

        tp = int(np.sum((true_flat == 1) & (learned_flat == 1)))
        fp = int(np.sum((true_flat == 0) & (learned_flat == 1)))
        fn = int(np.sum((true_flat == 1) & (learned_flat == 0)))
        tn = int(np.sum((true_flat == 0) & (learned_flat == 0)))

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        fdr = fp / (tp + fp + 1e-8)
        shd = int(np.sum(true_flat != learned_flat))

        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "fdr": float(fdr),
            "shd": shd,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }


def bh_correct(pval_matrix: np.ndarray, alpha: float) -> np.ndarray:
    """Apply Benjamini-Hochberg FDR correction to an off-diagonal p-value matrix.

    Parameters
    ----------
    pval_matrix : (n, n) array. Diagonal values are ignored.
    alpha : FDR significance level.

    Returns
    -------
    (n, n) binary float adjacency matrix. Diagonal is always 0.
    """
    n = pval_matrix.shape[0]
    mask = ~np.eye(n, dtype=bool)
    flat_p = pval_matrix[mask]
    reject, _, _, _ = multipletests(flat_p, alpha=alpha, method="fdr_bh")
    adj = np.zeros((n, n))
    adj[mask] = reject.astype(float)
    return adj
