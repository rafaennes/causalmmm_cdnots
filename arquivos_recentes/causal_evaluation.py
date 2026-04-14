# Copyright 2025
#
# Licensed under the Apache License, Version 2.0

"""Evaluation metrics for causal structure discovery.

Extracted from causalmmm.metrics.evaluation for standalone use
in the CD-NOTS benchmark pipeline (no TensorFlow dependency).
"""

import numpy as np
from typing import Dict


def evaluate_causal_structure(
    learned_graph: np.ndarray,
    true_graph: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate causal structure discovery.

    Metrics:
    - Precision, Recall, F1
    - Structural Hamming Distance (SHD)
    - False Discovery Rate (FDR)

    Args:
        learned_graph: [n, n] - Learned adjacency matrix
        true_graph: [n, n] - Ground truth adjacency matrix

    Returns:
        Dictionary of metrics
    """
    n = true_graph.shape[0]
    mask = ~np.eye(n, dtype=bool)

    true_flat = true_graph[mask].flatten()
    learned_flat = learned_graph[mask].flatten()

    # Confusion matrix
    tp = np.sum((true_flat == 1) & (learned_flat == 1))
    fp = np.sum((true_flat == 0) & (learned_flat == 1))
    fn = np.sum((true_flat == 1) & (learned_flat == 0))
    tn = np.sum((true_flat == 0) & (learned_flat == 0))

    # Metrics
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    fdr = fp / (tp + fp + 1e-8)
    shd = np.sum(true_flat != learned_flat)

    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'fdr': float(fdr),
        'shd': int(shd),
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn)
    }
