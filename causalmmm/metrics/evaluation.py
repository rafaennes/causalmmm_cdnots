"""
Evaluation metrics for CausalMMM.
"""

import numpy as np
import tensorflow as tf
from typing import Dict, Optional, Tuple, Callable


def mae(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Mean Absolute Error."""
    return tf.reduce_mean(tf.abs(y_pred - y_true))


def mape(y_true: tf.Tensor, y_pred: tf.Tensor, eps: float = 1e-8) -> tf.Tensor:
    """Mean Absolute Percentage Error."""
    return 100.0 * tf.reduce_mean(tf.abs((y_pred - y_true) / (y_true + eps)))


def smape(y_true: tf.Tensor, y_pred: tf.Tensor, eps: float = 1e-8) -> tf.Tensor:
    """Symmetric Mean Absolute Percentage Error."""
    num = tf.abs(y_pred - y_true)
    den = tf.abs(y_true) + tf.abs(y_pred) + eps
    return 200.0 * tf.reduce_mean(num / den)


def rmse(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Root Mean Squared Error."""
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))


def evaluate_forecast(
    model,
    X: np.ndarray,
    y: np.ndarray,
    context: Optional[np.ndarray] = None,
    time_idx: Optional[np.ndarray] = None,
    metrics: Tuple[Callable, ...] = (mae, mape, smape, rmse)
) -> Dict[str, float]:
    """
    Evaluate forecast performance.
    
    Args:
        model: Trained CausalMMM model
        X, y, context, time_idx: Test data
        metrics: Tuple of metric functions
        
    Returns:
        Dictionary of metric values
    """
    predictions = model.predict(X, context, time_idx)
    y_true = tf.convert_to_tensor(y[:, :, 0], dtype=tf.float32)
    y_pred = tf.convert_to_tensor(predictions, dtype=tf.float32)
    
    results = {}
    for metric_fn in metrics:
        results[metric_fn.__name__] = float(metric_fn(y_true, y_pred).numpy())
    
    return results


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


def compute_attribution(
    model,
    X: np.ndarray,
    y: np.ndarray,
    context: Optional[np.ndarray] = None,
    time_idx: Optional[np.ndarray] = None,
    method: str = 'shapley'
) -> Dict[str, float]:
    """
    Compute channel attribution/contribution.
    
    Args:
        model: Trained model
        X, y, context, time_idx: Data
        method: Attribution method ('shapley', 'ablation', 'gradient')
        
    Returns:
        Dictionary of channel contributions
    """
    n_channels = X.shape[2]
    baseline_pred = model.predict(X, context, time_idx)
    baseline_mean = baseline_pred.mean()
    
    if method == 'ablation':
        # Ablation: Set each channel to zero and measure impact
        contributions = {}
        for ch_idx in range(n_channels):
            X_ablated = X.copy()
            X_ablated[:, :, ch_idx] = 0
            pred_ablated = model.predict(X_ablated, context, time_idx)
            impact = baseline_mean - pred_ablated.mean()
            contributions[f'channel_{ch_idx}'] = float(impact)
    
    elif method == 'gradient':
        # Gradient-based attribution
        X_tf = tf.convert_to_tensor(X, dtype=tf.float32)
        y_tf = tf.zeros_like(tf.convert_to_tensor(y, dtype=tf.float32))
        context_tf = tf.convert_to_tensor(context, dtype=tf.float32) if context is not None else None
        time_tf = tf.convert_to_tensor(time_idx, dtype=tf.int32) if time_idx is not None else None
        
        with tf.GradientTape() as tape:
            tape.watch(X_tf)
            mu, _, _, _ = model((X_tf, y_tf, context_tf, time_tf), training=False)
            target_pred = mu[:, :, -1]
            output = tf.reduce_mean(target_pred)
        
        gradients = tape.gradient(output, X_tf)
        contributions = {}
        for ch_idx in range(n_channels):
            grad_importance = tf.reduce_mean(tf.abs(gradients[:, :, ch_idx]))
            contributions[f'channel_{ch_idx}'] = float(grad_importance.numpy())
    
    else:  # shapley
        # Simplified Shapley values (approximate)
        contributions = {}
        for ch_idx in range(n_channels):
            shapley_value = 0.0
            n_samples = 10  # Number of coalition samples
            
            for _ in range(n_samples):
                # Random coalition
                coalition = np.random.rand(n_channels) > 0.5
                
                # With channel
                X_with = X.copy()
                for j in range(n_channels):
                    if not coalition[j] and j != ch_idx:
                        X_with[:, :, j] = 0
                pred_with = model.predict(X_with, context, time_idx).mean()
                
                # Without channel
                X_without = X_with.copy()
                X_without[:, :, ch_idx] = 0
                pred_without = model.predict(X_without, context, time_idx).mean()
                
                shapley_value += (pred_with - pred_without)
            
            contributions[f'channel_{ch_idx}'] = float(shapley_value / n_samples)
    
    return contributions


def compute_roas(
    model,
    X: np.ndarray,
    y: np.ndarray,
    spend: np.ndarray,
    context: Optional[np.ndarray] = None,
    time_idx: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute Return on Ad Spend (ROAS) per channel.
    
    Args:
        model: Trained model
        X: Channel data
        y: Target data
        spend: Actual spend per channel [B, T, d]
        context, time_idx: Additional data
        
    Returns:
        ROAS per channel
    """
    n_channels = X.shape[2]
    baseline_pred = model.predict(X, context, time_idx)
    
    roas = {}
    for ch_idx in range(n_channels):
        # Ablate channel
        X_ablated = X.copy()
        X_ablated[:, :, ch_idx] = 0
        pred_ablated = model.predict(X_ablated, context, time_idx)
        
        # Incremental sales
        incremental_sales = (baseline_pred - pred_ablated).sum()
        
        # Total spend on channel
        total_spend = spend[:, :, ch_idx].sum()
        
        # ROAS
        roas[f'channel_{ch_idx}'] = float(incremental_sales / (total_spend + 1e-8))
    
    return roas