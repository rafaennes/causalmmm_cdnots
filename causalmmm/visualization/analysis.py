"""
Analysis and dashboard visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional


def plot_channel_contribution(
    contributions: Dict[str, float],
    channel_names: Optional[List[str]] = None,
    figsize: tuple = (10, 6)
):
    """
    Plot channel contributions as bar chart.
    """
    channels = list(contributions.keys())
    values = list(contributions.values())
    
    if channel_names:
        labels = channel_names
    else:
        labels = channels
    
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.viridis(np.linspace(0, 1, len(channels)))
    
    bars = ax.barh(labels, values, color=colors, alpha=0.8)
    ax.set_xlabel('Contribution', fontsize=12)
    ax.set_title('Channel Contribution to Sales', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for bar, val in zip(bars, values):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
                f'{val:.2f}',
                ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.show()


def plot_saturation_curves(
    model,
    channel_idx: int,
    channel_name: str,
    spend_range: tuple = (0, 100),
    n_points: int = 100,
    figsize: tuple = (10, 6)
):
    """
    Plot saturation curve for a specific channel.
    """
    from causalmmm.models.decoder import SaturationModule
    
    # Extract saturation module
    saturation = model.decoder.saturation
    
    # Generate spend values
    spend_values = np.linspace(spend_range[0], spend_range[1], n_points)
    
    # Compute saturated response
    import tensorflow as tf
    spend_tf = tf.constant(spend_values.reshape(-1, 1), dtype=tf.float32)
    response_tf = saturation(spend_tf, context=None)
    response_values = response_tf.numpy().flatten()
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(spend_values, response_values, linewidth=2, color='steelblue')
    ax.fill_between(spend_values, 0, response_values, alpha=0.3, color='steelblue')
    
    ax.set_xlabel(f'{channel_name} Spend', fontsize=12)
    ax.set_ylabel('Response', fontsize=12)
    ax.set_title(f'Saturation Curve: {channel_name}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Mark diminishing returns point (inflection)
    if len(response_values) > 2:
        # Second derivative to find inflection
        d2 = np.diff(np.diff(response_values))
        if len(d2) > 0:
            inflection_idx = np.argmin(np.abs(d2))
            ax.axvline(spend_values[inflection_idx], color='red', 
                      linestyle='--', alpha=0.5, label='Inflection Point')
            ax.legend()
    
    plt.tight_layout()
    plt.show()


def plot_response_curves(
    model,
    X: np.ndarray,
    y: np.ndarray,
    channel_names: List[str],
    context: Optional[np.ndarray] = None,
    time_idx: Optional[np.ndarray] = None,
    figsize: tuple = (14, 10)
):
    """
    Plot response curves for all channels.
    """
    n_channels = X.shape[2]
    n_cols = 2
    n_rows = (n_channels + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    baseline_pred = model.predict(X, context, time_idx).mean()
    
    for ch_idx in range(n_channels):
        ax = axes[ch_idx]
        
        # Test different spend levels
        multipliers = np.linspace(0.5, 2.0, 20)
        responses = []
        
        for mult in multipliers:
            X_test = X.copy()
            X_test[:, :, ch_idx] *= mult
            pred = model.predict(X_test, context, time_idx).mean()
            responses.append(pred - baseline_pred)
        
        # Plot
        ax.plot(multipliers, responses, linewidth=2, marker='o', markersize=4)
        ax.axhline(0, color='black', linestyle='--', alpha=0.3)
        ax.axvline(1.0, color='red', linestyle='--', alpha=0.3)
        ax.set_xlabel('Spend Multiplier', fontsize=10)
        ax.set_ylabel('Incremental Sales', fontsize=10)
        ax.set_title(f'{channel_names[ch_idx]}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Hide extra subplots
    for idx in range(n_channels, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Channel Response Curves', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


def plot_attribution_waterfall(
    attributions: Dict[str, float],
    channel_names: Optional[List[str]] = None,
    baseline: float = 0,
    figsize: tuple = (12, 6)
):
    """
    Waterfall chart showing cumulative attribution.
    """
    channels = list(attributions.keys())
    values = list(attributions.values())
    
    if channel_names:
        labels = channel_names
    else:
        labels = [f'Channel {i}' for i in range(len(channels))]
    
    # Sort by contribution
    sorted_indices = np.argsort(values)[::-1]
    labels = [labels[i] for i in sorted_indices]
    values = [values[i] for i in sorted_indices]
    
    # Compute cumulative
    cumulative = np.cumsum([baseline] + values)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot bars
    colors = ['green' if v > 0 else 'red' for v in values]
    
    for i, (label, value, cum) in enumerate(zip(labels, values, cumulative[:-1])):
        ax.bar(i, value, bottom=cum, color=colors[i], alpha=0.7, edgecolor='black')
        
        # Connector lines
        if i < len(labels) - 1:
            ax.plot([i, i+1], [cumulative[i+1], cumulative[i+1]], 
                   'k--', alpha=0.3, linewidth=1)
        
        # Value labels
        ax.text(i, cum + value/2, f'{value:.1f}',
               ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Total bar
    ax.bar(len(labels), cumulative[-1], bottom=0, 
           color='blue', alpha=0.5, edgecolor='black', label='Total')
    ax.text(len(labels), cumulative[-1]/2, f'{cumulative[-1]:.1f}',
           ha='center', va='center', fontsize=10, fontweight='bold')
    
    ax.set_xticks(range(len(labels) + 1))
    ax.set_xticklabels(labels + ['Total'], rotation=45, ha='right')
    ax.set_ylabel('Contribution', fontsize=12)
    ax.set_title('Channel Attribution Waterfall', fontsize=14, fontweight='bold')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
