"""
Implements the three-stage encoder from CausalMMM paper:
1. Pairwise Embedding: Local relationship modeling
2. Relational Interaction: Global aggregation
3. Gumbel Softmax Sampling: Differentiable graph sampling

Reference: Gong et al. (2024), Figure 2
"""

import tensorflow as tf
from tensorflow.keras import layers as KL
import numpy as np
from typing import Optional, Tuple


class TimeEncoder(KL.Layer):
    """Encodes time indices for temporal modeling."""
    
    def __init__(self, encoding: str = 'cyclical', hidden_dim: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.encoding = encoding
        self.hidden_dim = hidden_dim
        
        if encoding == 'learned':
            self.time_mlp = tf.keras.Sequential([
                KL.Dense(hidden_dim, activation='relu'),
                KL.Dense(hidden_dim)
            ])
        elif encoding == 'cyclical':
            # Annual and weekly cycles
            self.freq_year = 2 * np.pi / 365.25
            self.freq_week = 2 * np.pi / 7
    
    def call(self, time_idx: tf.Tensor) -> tf.Tensor:
        """
        Args:
            time_idx: [B, T] - Time indices
        Returns:
            time_emb: [B, T, H] - Time embeddings
        """
        time_float = tf.cast(time_idx, tf.float32)
        
        if self.encoding == 'linear':
            time_norm = time_float / (tf.reduce_max(time_float) + 1e-8)
            time_emb = tf.expand_dims(time_norm, -1)
            time_emb = tf.tile(time_emb, [1, 1, self.hidden_dim])
            
        elif self.encoding == 'cyclical':
            sin_year = tf.sin(self.freq_year * time_float)
            cos_year = tf.cos(self.freq_year * time_float)
            sin_week = tf.sin(self.freq_week * time_float)
            cos_week = tf.cos(self.freq_week * time_float)
            time_features = tf.stack([sin_year, cos_year, sin_week, cos_week], axis=-1)
            # Expand to hidden_dim
            time_emb = tf.tile(time_features, [1, 1, self.hidden_dim // 4])
            
        elif self.encoding == 'learned':
            time_norm = time_float / (tf.reduce_max(time_float) + 1e-8)
            time_emb = self.time_mlp(tf.expand_dims(time_norm, -1))
        
        return time_emb


class PairwiseEmbedding(KL.Layer):
    """
    Stage 1: Pairwise Embedding
    
    Creates initial edge representations by concatenating node features.
    Models local relationships between channel pairs.
    
    Reference: Paper Section 3.2.1
    """
    
    def __init__(self, hidden_dim: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        
        # Per-variable temporal encoder (RNN)
        self.temporal_encoder = KL.GRU(hidden_dim, return_sequences=False)
        
        # Pairwise MLP
        self.pairwise_mlp = tf.keras.Sequential([
            KL.Dense(hidden_dim, activation='relu'),
            KL.Dropout(0.1),
            KL.Dense(hidden_dim)
        ])
    
    def call(self, X: tf.Tensor, y: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Args:
            X: [B, T, d] - Channel time series
            y: [B, T, 1] - Target time series
            
        Returns:
            h_pair: [B, n, n, H] - Pairwise embeddings
        """
        B = tf.shape(X)[0]
        T = tf.shape(X)[1]
        d = tf.shape(X)[2]
        n = d + 1  # channels + target
        
        # Concatenate all variables
        XY = tf.concat([X, y], axis=-1)  # [B, T, n]
        
        # Per-variable temporal encoding
        # Reshape to [B*n, T, 1] for independent RNN processing
        XY_reshaped = tf.reshape(tf.transpose(XY, [0, 2, 1]), [B * n, T, 1])
        h_node = self.temporal_encoder(XY_reshaped, training=training)  # [B*n, H]
        h_node = tf.reshape(h_node, [B, n, self.hidden_dim])  # [B, n, H]
        
        # Create pairwise features: [hi, hj] for all (i, j) pairs
        hi = tf.expand_dims(h_node, 2)  # [B, n, 1, H]
        hj = tf.expand_dims(h_node, 1)  # [B, 1, n, H]
        hi_tiled = tf.tile(hi, [1, 1, n, 1])  # [B, n, n, H]
        hj_tiled = tf.tile(hj, [1, n, 1, 1])  # [B, n, n, H]
        
        h_pair_cat = tf.concat([hi_tiled, hj_tiled], axis=-1)  # [B, n, n, 2H]
        h_pair = self.pairwise_mlp(h_pair_cat, training=training)  # [B, n, n, H]
        
        return h_pair


class RelationalInteraction(KL.Layer):
    """
    Stage 2: Relational Interaction
    
    Aggregates information across all edges to capture global structure.
    Refines edge embeddings with global context.
    
    Reference: Paper Section 3.2.2
    """
    
    def __init__(self, hidden_dim: int = 64, aggregation: str = 'mean', **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.aggregation = aggregation
        
        # Node update MLP (aggregates incoming edges)
        self.node_mlp = tf.keras.Sequential([
            KL.Dense(hidden_dim, activation='relu'),
            KL.Dropout(0.1),
            KL.Dense(hidden_dim)
        ])
        
        # Edge refinement MLP
        self.edge_refine_mlp = tf.keras.Sequential([
            KL.Dense(hidden_dim, activation='relu'),
            KL.Dropout(0.1),
            KL.Dense(hidden_dim)
        ])
        
        if aggregation == 'attention':
            self.attention = KL.MultiHeadAttention(
                num_heads=4,
                key_dim=hidden_dim // 4
            )
    
    def call(self, h_pair: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Args:
            h_pair: [B, n, n, H] - Pairwise embeddings from Stage 1
            
        Returns:
            h_pair_refined: [B, n, n, H] - Refined pairwise embeddings
        """
        # Aggregate incoming edges for each node
        if self.aggregation == 'mean':
            h_node = tf.reduce_mean(h_pair, axis=1)  # [B, n, H]
        elif self.aggregation == 'sum':
            h_node = tf.reduce_sum(h_pair, axis=1)  # [B, n, H]
        elif self.aggregation == 'attention':
            B, n, _, H = tf.shape(h_pair)[0], tf.shape(h_pair)[1], tf.shape(h_pair)[2], tf.shape(h_pair)[3]
            h_pair_flat = tf.reshape(h_pair, [B, n * n, H])
            h_node = self.attention(h_pair_flat, h_pair_flat, training=training)
            h_node = tf.reshape(h_node, [B, n, n, H])
            h_node = tf.reduce_mean(h_node, axis=1)
        
        # Update node representations
        h_node_updated = self.node_mlp(h_node, training=training)  # [B, n, H]
        
        # Refine edges with global context
        hi_global = tf.expand_dims(h_node_updated, 2)  # [B, n, 1, H]
        hj_global = tf.expand_dims(h_node_updated, 1)  # [B, 1, n, H]
        n = tf.shape(h_node)[1]
        hi_tiled = tf.tile(hi_global, [1, 1, n, 1])
        hj_tiled = tf.tile(hj_global, [1, n, 1, 1])
        
        h_pair_global = tf.concat([hi_tiled, hj_tiled, h_pair], axis=-1)  # [B, n, n, 3H]
        h_pair_refined = self.edge_refine_mlp(h_pair_global, training=training)  # [B, n, n, H]
        
        return h_pair_refined


def gumbel_softmax(logits: tf.Tensor, temperature: float = 1.0, hard: bool = False) -> tf.Tensor:
    """
    Gumbel-Softmax sampling for differentiable discrete variables.
    
    Args:
        logits: [B, n, n, K] - Logits for K classes
        temperature: Temperature parameter (lower = more discrete)
        hard: If True, use straight-through estimator
        
    Returns:
        y: [B, n, n, K] - Sampled probabilities
    """
    def sample_gumbel(shape, eps=1e-20):
        u = tf.random.uniform(shape, minval=0.0, maxval=1.0)
        return -tf.math.log(-tf.math.log(u + eps) + eps)
    
    gumbel_noise = sample_gumbel(tf.shape(logits))
    y = tf.nn.softmax((logits + gumbel_noise) / temperature, axis=-1)
    
    if hard:
        k = tf.shape(logits)[-1]
        y_hard = tf.one_hot(tf.argmax(y, axis=-1), depth=k)
        y = tf.stop_gradient(y_hard - y) + y
    
    return y


class CausalRelationalEncoder(tf.keras.Model):
    """
    Complete Causal Relational Encoder
    
    Implements full three-stage architecture:
    1. Pairwise Embedding
    2. Relational Interaction  
    3. Gumbel Softmax Sampling
    
    Reference: Paper Section 3.2, Figure 2(a)
    """
    
    def __init__(self, config, context_dim: int = 0, prior_graph: Optional[np.ndarray] = None):
        super().__init__()
        self.config = config
        self.enc_config = config.encoder
        self.n_variables = config.n_variables
        
        # Time encoding (optional)
        if self.enc_config.use_time_node:
            self.time_encoder = TimeEncoder(
                self.enc_config.time_encoding,
                self.enc_config.hidden_dim
            )
        
        # Stage 1: Pairwise Embedding
        self.pairwise_embedding = PairwiseEmbedding(self.enc_config.hidden_dim)
        
        # Stage 2: Relational Interaction (multiple layers)
        self.interaction_layers = [
            RelationalInteraction(self.enc_config.hidden_dim, self.enc_config.aggregation)
            for _ in range(self.enc_config.n_layers)
        ]
        
        # Stage 3: Edge logits head
        self.edge_logit_head = tf.keras.Sequential([
            KL.Dense(self.enc_config.hidden_dim, activation='relu'),
            KL.Dropout(self.enc_config.dropout),
            KL.Dense(2)  # [no-edge, edge]
        ])
        
        # Context integration (optional)
        if context_dim > 0:
            self.context_mlp = tf.keras.Sequential([
                KL.Dense(self.enc_config.hidden_dim, activation='relu'),
                KL.Dense(self.enc_config.hidden_dim)
            ])
        
        self.prior_graph = tf.constant(prior_graph, dtype=tf.float32) if prior_graph is not None else None
    
    def call(self, X: tf.Tensor, y: tf.Tensor, context: Optional[tf.Tensor] = None,
             time_idx: Optional[tf.Tensor] = None, training: bool = False) -> tf.Tensor:
        """
        Args:
            X: [B, T, d] - Channel data
            y: [B, T, 1] - Target data
            context: [B, c] - Context features (optional)
            time_idx: [B, T] - Time indices (optional)
            
        Returns:
            logits: [B, n, n, 2] - Edge logits [no-edge, edge]
        """
        B = tf.shape(X)[0]
        n = self.n_variables
        
        # Stage 1: Pairwise Embedding
        h_pair = self.pairwise_embedding(X, y, training=training)  # [B, n, n, H]
        
        # Add time influence if enabled
        if self.enc_config.use_time_node and time_idx is not None:
            time_emb = self.time_encoder(time_idx)  # [B, T, H]
            time_emb_agg = tf.reduce_mean(time_emb, axis=1)  # [B, H]
            time_emb_agg = tf.reshape(time_emb_agg, [B, 1, 1, -1])
            h_pair = h_pair + time_emb_agg
        
        # Add context influence if provided
        if context is not None:
            context_emb = self.context_mlp(context, training=training)  # [B, H]
            context_emb = tf.reshape(context_emb, [B, 1, 1, -1])
            h_pair = h_pair + context_emb
        
        # Stage 2: Relational Interaction (multi-layer)
        for interaction_layer in self.interaction_layers:
            h_pair = interaction_layer(h_pair, training=training)
        
        # Stage 3: Edge logits
        logits = self.edge_logit_head(h_pair, training=training)  # [B, n, n, 2]
        
        # Mask self-loops
        eye = tf.eye(n, batch_shape=[B])
        mask = 1.0 - tf.expand_dims(eye, -1)
        logits = logits * mask + (-1e9) * (1 - mask)
        
        return logits
    
    def sample_graph(self, logits: tf.Tensor, temperature: float, hard: bool = False) -> tf.Tensor:
        """
        Sample causal graph using Gumbel-Softmax.
        
        Args:
            logits: [B, n, n, 2] - Edge logits
            temperature: Sampling temperature
            hard: Use hard sampling
            
        Returns:
            z: [B, n, n] - Sampled adjacency matrix (soft or hard)
        """
        # Gumbel-Softmax sampling
        z_soft = gumbel_softmax(logits, temperature, hard)  # [B, n, n, 2]
        z = z_soft[..., 1]  # Take "edge" class
        
        # Remove self-loops
        B = tf.shape(z)[0]
        n = tf.shape(z)[1]
        eye = tf.eye(n, batch_shape=[B])
        z = z * (1.0 - eye)
        
        return z