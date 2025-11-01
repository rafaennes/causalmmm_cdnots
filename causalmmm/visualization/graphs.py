"""
Visualization tools for causal graphs
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def plot_causal_graph(
    adjacency_matrix,
    channel_names=None,
    target_name='target',
    threshold=0.0,
    save_path=None,
    figsize=(10, 8),
    node_size=3000,
    font_size=12,
    **kwargs
):
    """
    Plot a causal graph from an adjacency matrix.

    Parameters
    ----------
    adjacency_matrix : np.ndarray
        Adjacency matrix of shape (n_nodes, n_nodes) where entry (i, j)
        represents the edge weight from node i to node j.
    channel_names : list of str, optional
        Names of the channel nodes (excluding target).
        If None, uses ['X0', 'X1', ...].
    target_name : str, default='target'
        Name of the target node.
    threshold : float, default=0.0
        Minimum edge weight to display.
    save_path : str, optional
        Path to save the figure.
    figsize : tuple, default=(10, 8)
        Figure size.
    node_size : int, default=3000
        Size of nodes.
    font_size : int, default=12
        Font size for labels.
    **kwargs
        Additional arguments passed to nx.draw_networkx.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object.
    """
    # Prepare node names
    n_nodes = adjacency_matrix.shape[0]
    n_channels = n_nodes - 1  # Assuming last node is target

    if channel_names is None:
        channel_names = [f'X{i}' for i in range(n_channels)]

    node_names = channel_names + [target_name]

    # Create directed graph
    G = nx.DiGraph()

    # Add nodes
    G.add_nodes_from(node_names)

    # Add edges above threshold
    edges = []
    edge_labels = {}
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j and adjacency_matrix[i, j] > threshold:
                weight = adjacency_matrix[i, j]
                edges.append((node_names[i], node_names[j], weight))
                edge_labels[(node_names[i], node_names[j])] = f'{weight:.2f}'

    G.add_weighted_edges_from(edges)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Use spring layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # Draw nodes
    # Different colors for channels vs target
    channel_nodes = channel_names
    target_nodes = [target_name]

    nx.draw_networkx_nodes(
        G, pos,
        nodelist=channel_nodes,
        node_color='lightblue',
        node_size=node_size,
        ax=ax
    )

    nx.draw_networkx_nodes(
        G, pos,
        nodelist=target_nodes,
        node_color='lightcoral',
        node_size=node_size,
        ax=ax
    )

    # Draw edges with varying width based on weight
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    if weights:
        max_weight = max(weights)
        edge_widths = [3 * w / max_weight for w in weights]
    else:
        edge_widths = []

    nx.draw_networkx_edges(
        G, pos,
        width=edge_widths,
        alpha=0.6,
        edge_color='gray',
        arrows=True,
        arrowsize=20,
        arrowstyle='->',
        connectionstyle='arc3,rad=0.1',
        ax=ax
    )

    # Draw labels
    nx.draw_networkx_labels(
        G, pos,
        font_size=font_size,
        font_weight='bold',
        ax=ax
    )

    # Draw edge labels
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_size=font_size - 2,
        ax=ax
    )

    ax.set_title('Causal Graph', fontsize=16, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()

    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax


def plot_graph_comparison(
    true_graph,
    learned_graph,
    channel_names=None,
    target_name='target',
    threshold=0.0,
    save_path=None,
    figsize=(16, 6)
):
    """
    Plot true vs learned causal graphs side by side.

    Parameters
    ----------
    true_graph : np.ndarray
        True adjacency matrix.
    learned_graph : np.ndarray
        Learned adjacency matrix.
    channel_names : list of str, optional
        Names of the channel nodes.
    target_name : str, default='target'
        Name of the target node.
    threshold : float, default=0.0
        Minimum edge weight to display.
    save_path : str, optional
        Path to save the figure.
    figsize : tuple, default=(16, 6)
        Figure size.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot true graph
    plot_causal_graph(
        true_graph,
        channel_names=channel_names,
        target_name=target_name,
        threshold=threshold,
        save_path=None
    )
    ax1.set_title('True Causal Graph', fontsize=14, fontweight='bold')

    # Plot learned graph
    plot_causal_graph(
        learned_graph,
        channel_names=channel_names,
        target_name=target_name,
        threshold=threshold,
        save_path=None
    )
    ax2.set_title('Learned Causal Graph', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_temporal_effects(
    time_idx,
    effects,
    channel_names=None,
    title='Temporal Channel Effects',
    save_path=None,
    figsize=(12, 6)
):
    """
    Plot temporal effects of channels over time.

    Parameters
    ----------
    time_idx : array-like
        Time indices.
    effects : np.ndarray
        Effect values of shape (n_timesteps, n_channels).
    channel_names : list of str, optional
        Names of channels.
    title : str
        Plot title.
    save_path : str, optional
        Path to save the figure.
    figsize : tuple, default=(12, 6)
        Figure size.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    """
    n_channels = effects.shape[1]

    if channel_names is None:
        channel_names = [f'Channel {i}' for i in range(n_channels)]

    fig, ax = plt.subplots(figsize=figsize)

    for i, name in enumerate(channel_names):
        ax.plot(time_idx, effects[:, i], label=name, marker='o', markersize=3)

    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Effect', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig
