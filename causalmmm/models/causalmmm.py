"""
Main CausalMMM Model
====================

Variational Graph Autoencoder for causal structure learning in MMM.

Reference: Gong et al. (2024)
"""

import tensorflow as tf
import numpy as np
from typing import Optional, Tuple, Dict
import math


class CausalMMM(tf.keras.Model):
    """
    CausalMMM: Learning Causal Structure for Marketing Mix Modeling
    
    Full VAE framework with:
    - Encoder: Infers causal graph structure
    - Decoder: Forecasts with discovered graph
    - ELBO loss: Reconstruction + KL regularization
    
    Example:
        >>> config = CausalMMMConfig(n_channels=5)
        >>> model = CausalMMM(config)
        >>> model.fit(X_train, y_train, epochs=50)
        >>> graph = model.get_causal_graph()
        >>> predictions = model.predict(X_test)
    """
    
    def __init__(self, config, context_dim: int = 0, prior_graph: Optional[np.ndarray] = None):
        super().__init__()
        self.config = config
        self.context_dim = context_dim
        
        # Encoder and Decoder
        from causalmmm.models.encoder import CausalRelationalEncoder
        from causalmmm.models.decoder import MarketingResponseDecoder
        
        self.encoder = CausalRelationalEncoder(config, context_dim, prior_graph)
        self.decoder = MarketingResponseDecoder(config, context_dim)
        
        # Temperature for Gumbel-Softmax (annealed during training)
        self.temperature = tf.Variable(
            config.temperature,
            trainable=False,
            dtype=tf.float32,
            name='temperature'
        )
        
        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(config.learning_rate)
        
        # Metrics tracking
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.nll_tracker = tf.keras.metrics.Mean(name='nll')
        self.kl_tracker = tf.keras.metrics.Mean(name='kl')
        self.dag_tracker = tf.keras.metrics.Mean(name='dag')
    
    def call(self, inputs, training: bool = False):
        """
        Forward pass.
        
        Args:
            inputs: Tuple of (X, y, context, time_idx)
            training: Training mode
            
        Returns:
            mu: Predictions
            sigma: Uncertainty
            edge_logits: Causal graph logits
            z_graph: Sampled adjacency matrix
        """
        X, y, context, time_idx = inputs
        
        # Encode: Get causal structure
        edge_logits = self.encoder(X, y, context, time_idx, training=training)
        
        # Sample graph
        z_graph = self.encoder.sample_graph(
            edge_logits,
            self.temperature,
            hard=self.config.hard_gumbel
        )
        
        # Decode: Forecast with graph
        mu, sigma = self.decoder(X, y, z_graph, context, training=training)
        
        return mu, sigma, edge_logits, z_graph
    
    def compute_loss(self, X, y, context, time_idx, training: bool = True):
        """
        Compute ELBO loss (if VAE mode) or direct loss.
        
        Loss = NLL + λ_KL·KL + λ_DAG·h(A) + λ_temporal·R_t + λ_saturation·R_s
        """
        # Forward pass
        mu, sigma, edge_logits, z_graph = self(
            (X, y, context, time_idx),
            training=training
        )
        
        # 1. Negative Log-Likelihood (Reconstruction)
        # Now mu has shape [B, T-1, 1] - only target predictions
        target = y[:, 1:, :]  # [B, T-1, 1] - only target values

        if self.config.vae_mode:
            # Gaussian likelihood: log p(y|X, z)
            resid = target - mu
            var = sigma ** 2
            const = 0.5 * tf.math.log(2 * math.pi)
            nll = tf.reduce_mean(
                0.5 * tf.reduce_sum(
                    (resid ** 2) / var + 2 * (tf.math.log(sigma) + const),
                    axis=[1, 2]
                )
            )
        else:
            # Simple MSE
            nll = tf.reduce_mean(tf.square(target - mu))
        
        # 2. KL Divergence: KL(q(z|X,y) || p(z))
        edge_probs = tf.nn.sigmoid(edge_logits[..., 1])
        prior = tf.constant(self.config.prior_pi, dtype=tf.float32)
        
        eps = 1e-8
        kl_per_edge = (
            edge_probs * (tf.math.log(edge_probs + eps) - tf.math.log(prior + eps)) +
            (1 - edge_probs) * (tf.math.log(1 - edge_probs + eps) - tf.math.log(1 - prior + eps))
        )
        
        # Mask self-loops
        B = tf.shape(edge_probs)[0]
        n = tf.shape(edge_probs)[1]
        eye = tf.eye(n, batch_shape=[B])
        kl_per_edge = kl_per_edge * (1.0 - eye)
        kl = tf.reduce_mean(tf.reduce_sum(kl_per_edge, axis=[1, 2]))
        
        # 3. DAG Constraint: h(A) = tr(exp(A)) - d
        def compute_dag_penalty(A_i):
            exp_A = tf.linalg.expm(A_i)
            h = tf.linalg.trace(exp_A) - tf.cast(n, tf.float32)
            return h

        A = edge_probs * (1.0 - eye)
        h_all = tf.map_fn(compute_dag_penalty, A, dtype=tf.float32)
        dag_penalty = tf.reduce_mean(h_all)

        # Clip DAG penalty to prevent numerical explosion
        dag_penalty = tf.clip_by_value(dag_penalty, -100.0, 100.0)
        
        # 4. Temporal Regularization (optional)
        # Encourage temporal consistency
        temporal_reg = 0.0
        if self.config.lambda_temporal > 0:
            # Penalize large changes in predictions
            mu_diff = mu[:, 1:, :] - mu[:, :-1, :]
            temporal_reg = tf.reduce_mean(tf.square(mu_diff))
        
        # 5. Saturation Regularization (optional)
        # Ensure saturation curves are reasonable
        saturation_reg = 0.0
        if self.config.lambda_saturation > 0:
            # Penalize predictions outside reasonable range
            saturation_reg = tf.reduce_mean(tf.nn.relu(mu - 10.0) + tf.nn.relu(-mu - 10.0))
        
        # Total loss
        loss = (nll + 
                self.config.lambda_kl * kl +
                self.config.lambda_dag * dag_penalty +
                self.config.lambda_temporal * temporal_reg +
                self.config.lambda_saturation * saturation_reg)
        
        return loss, nll, kl, dag_penalty
    
    @tf.function
    def train_step(self, X, y, context, time_idx):
        """Single training step."""
        with tf.GradientTape() as tape:
            loss, nll, kl, dag = self.compute_loss(X, y, context, time_idx, training=True)
        
        # Backpropagation
        grads = tape.gradient(loss, self.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, self.config.gradient_clip)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        # Update temperature (annealing)
        new_temp = tf.maximum(
            self.config.temperature_min,
            self.temperature * self.config.temperature_decay
        )
        self.temperature.assign(new_temp)
        
        # Update metrics
        self.loss_tracker.update_state(loss)
        self.nll_tracker.update_state(nll)
        self.kl_tracker.update_state(kl)
        self.dag_tracker.update_state(dag)
        
        return {
            'loss': loss,
            'nll': nll,
            'kl': kl,
            'dag': dag,
            'temperature': self.temperature
        }
    
    def fit(self, X, y, context=None, time_idx=None, epochs=50, verbose=1):
        """
        Train the model.

        Args:
            X: [B, T, d] - Channel data
            y: [B, T, 1] - Target data
            context: [B, c] - Context features
            time_idx: [B, T] - Time indices
            epochs: Number of training epochs
            verbose: Verbosity level

        Returns:
            history: Dict with training metrics per epoch
        """
        # Convert to tensors
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        y = tf.convert_to_tensor(y, dtype=tf.float32)
        context = tf.convert_to_tensor(context, dtype=tf.float32) if context is not None else None
        time_idx = tf.convert_to_tensor(time_idx, dtype=tf.int32) if time_idx is not None else None

        # History tracking
        history = {
            'loss': [],
            'nll': [],
            'kl': [],
            'dag': [],
            'temperature': []
        }

        for epoch in range(1, epochs + 1):
            # Reset metrics
            self.loss_tracker.reset_state()
            self.nll_tracker.reset_state()
            self.kl_tracker.reset_state()
            self.dag_tracker.reset_state()

            # Train step
            metrics = self.train_step(X, y, context, time_idx)

            # Track history
            history['loss'].append(float(metrics['loss']))
            history['nll'].append(float(metrics['nll']))
            history['kl'].append(float(metrics['kl']))
            history['dag'].append(float(metrics['dag']))
            history['temperature'].append(float(metrics['temperature']))

            # Print progress
            if verbose and epoch % max(1, epochs // 20) == 0:
                print(f"Epoch {epoch:03d}/{epochs} | "
                      f"Loss: {metrics['loss']:.4f} | "
                      f"NLL: {metrics['nll']:.4f} | "
                      f"KL: {metrics['kl']:.4f} | "
                      f"DAG: {metrics['dag']:.4f} | "
                      f"Temp: {metrics['temperature']:.3f}")

        return history
    
    def predict(self, X, context=None, time_idx=None):
        """
        Make predictions.
        
        Args:
            X: [B, T, d] - Channel data
            context: [B, c] - Context
            time_idx: [B, T] - Time indices
            
        Returns:
            predictions: [B, T-1] - Target predictions
        """
        # Dummy y for encoding
        y = tf.zeros([tf.shape(X)[0], tf.shape(X)[1], 1], dtype=tf.float32)
        
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        context = tf.convert_to_tensor(context, dtype=tf.float32) if context is not None else None
        time_idx = tf.convert_to_tensor(time_idx, dtype=tf.int32) if time_idx is not None else None
        
        mu, _, _, _ = self((X, y, context, time_idx), training=False)
        return mu[:, :, 0].numpy()  # Return target predictions [B, T-1]
    
    def get_causal_graph(self, X=None, y=None, context=None, time_idx=None, threshold=0.3):
        """
        Extract learned causal graph.
        
        Args:
            X, y, context, time_idx: Data for encoding (optional, uses training data)
            threshold: Probability threshold for edges
            
        Returns:
            graph: [n, n] - Binary adjacency matrix
        """
        if X is None:
            # Use saved training data
            raise NotImplementedError("Must provide data or store during training")
        
        y_dummy = tf.zeros([tf.shape(X)[0], tf.shape(X)[1], 1], dtype=tf.float32) if y is None else y
        
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        y_dummy = tf.convert_to_tensor(y_dummy, dtype=tf.float32)
        context = tf.convert_to_tensor(context, dtype=tf.float32) if context is not None else None
        time_idx = tf.convert_to_tensor(time_idx, dtype=tf.int32) if time_idx is not None else None
        
        edge_logits = self.encoder(X, y_dummy, context, time_idx, training=False)
        edge_probs = tf.nn.sigmoid(edge_logits[..., 1])
        
        # Average over batch and threshold
        edge_probs_mean = tf.reduce_mean(edge_probs, axis=0)
        graph = (edge_probs_mean > threshold).numpy().astype(int)
        
        return graph