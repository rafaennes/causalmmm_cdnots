"""
Marketing Response Decoder
===========================

Implements temporal forecasting with:
1. Carryover effects (adstock)
2. Saturation curves (S-curve, Hill equation)
3. Graph-guided message passing

Reference: Gong et al. (2024), Section 3.3, Figure 2(b)
"""

import tensorflow as tf
from tensorflow.keras import layers as KL
import numpy as np
from typing import Optional, Tuple, Literal


class CarryoverModule(KL.Layer):
    """
    Models carryover/adstock effects.
    
    Implements different decay patterns:
    - Geometric: Simple exponential decay
    - Weibull: Flexible decay shape
    """
    
    def __init__(self, carryover_fn: str = 'geometric', max_lag: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.carryover_fn = carryover_fn
        self.max_lag = max_lag
        
        if carryover_fn == 'geometric':
            # Learnable decay rate per channel
            self.decay_logit = self.add_weight(
                name='decay_logit',
                shape=(1,),
                initializer='zeros',
                trainable=True
            )
        elif carryover_fn == 'weibull':
            # Weibull parameters: shape (k) and scale (λ)
            self.weibull_k = self.add_weight(
                name='weibull_k',
                shape=(1,),
                initializer=tf.constant_initializer(2.0),
                trainable=True
            )
            self.weibull_lambda = self.add_weight(
                name='weibull_lambda',
                shape=(1,),
                initializer=tf.constant_initializer(1.0),
                trainable=True
            )
    
    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Apply carryover transformation.
        
        Args:
            x: [B, T, d] - Input time series
            
        Returns:
            x_carryover: [B, T, d] - With carryover effects
        """
        if self.carryover_fn == 'none':
            return x
        
        B, T, d = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        
        if self.carryover_fn == 'geometric':
            # Geometric adstock: x_t = x_t + α*x_{t-1} + α²*x_{t-2} + ...
            decay = tf.nn.sigmoid(self.decay_logit)
            
            # Convolve with decay weights
            weights = tf.pow(decay, tf.range(self.max_lag, dtype=tf.float32))
            weights = weights / tf.reduce_sum(weights)  # Normalize
            
            # Apply convolution per channel
            x_carryover = tf.nn.conv1d(
                x, 
                tf.reshape(weights, [self.max_lag, 1, 1]),
                stride=1,
                padding='SAME'
            )
            
        elif self.carryover_fn == 'weibull':
            # Weibull adstock
            k = tf.nn.softplus(self.weibull_k) + 0.001
            lam = tf.nn.softplus(self.weibull_lambda) + 0.001
            
            lags = tf.range(1, self.max_lag + 1, dtype=tf.float32)
            # Weibull PDF: (k/λ) * (t/λ)^(k-1) * exp(-(t/λ)^k)
            weights = (k / lam) * tf.pow(lags / lam, k - 1) * tf.exp(-tf.pow(lags / lam, k))
            weights = weights / tf.reduce_sum(weights)
            
            x_carryover = tf.nn.conv1d(
                x,
                tf.reshape(weights, [self.max_lag, 1, 1]),
                stride=1,
                padding='SAME'
            )
        
        return x_carryover


class SaturationModule(KL.Layer):
    """
    Models saturation/diminishing returns.
    
    Implements:
    - S-curve: f(x) = α·x / (α·x + γ)
    - Hill equation: f(x) = x^n / (k^n + x^n)
    """
    
    def __init__(self, saturation_fn: str = 'scurve', context_dim: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.saturation_fn = saturation_fn
        self.use_context = context_dim > 0
        
        if saturation_fn != 'none':
            if self.use_context:
                # Learn α, γ as functions of context
                self.alpha_net = tf.keras.Sequential([
                    KL.Dense(32, activation='relu'),
                    KL.Dense(1)
                ])
                self.gamma_net = tf.keras.Sequential([
                    KL.Dense(32, activation='relu'),
                    KL.Dense(1)
                ])
            else:
                # Fixed parameters
                self.alpha = self.add_weight(
                    name='alpha',
                    shape=(1,),
                    initializer='ones',
                    trainable=True
                )
                self.gamma = self.add_weight(
                    name='gamma',
                    shape=(1,),
                    initializer='ones',
                    trainable=True
                )
            
            if saturation_fn == 'hill':
                self.hill_n = self.add_weight(
                    name='hill_n',
                    shape=(1,),
                    initializer=tf.constant_initializer(2.0),
                    trainable=True
                )
    
    def call(self, x: tf.Tensor, context: Optional[tf.Tensor] = None) -> tf.Tensor:
        """
        Apply saturation transformation.
        
        Args:
            x: [...] - Input values
            context: [B, c] - Context features (optional)
            
        Returns:
            x_saturated: [...] - Saturated values
        """
        if self.saturation_fn == 'none':
            return x
        
        # Get saturation parameters
        if self.use_context and context is not None:
            alpha = tf.nn.softplus(self.alpha_net(context)) + 0.001
            gamma = tf.nn.softplus(self.gamma_net(context)) + 0.001
        else:
            alpha = tf.nn.softplus(self.alpha) + 0.001
            gamma = tf.nn.softplus(self.gamma) + 0.001
        
        if self.saturation_fn == 'scurve':
            # S-curve: f(x) = (α·x) / (α·x + γ)
            numerator = alpha * x
            denominator = alpha * x + gamma
            x_saturated = numerator / (denominator + 1e-8)
            
        elif self.saturation_fn == 'hill':
            # Hill equation: f(x) = x^n / (k^n + x^n)
            n = tf.nn.softplus(self.hill_n) + 1.0
            k = gamma  # Use gamma as half-saturation point
            x_n = tf.pow(tf.maximum(x, 0.0) + 1e-8, n)
            k_n = tf.pow(k, n)
            x_saturated = x_n / (k_n + x_n)
        
        return x_saturated


class MarketingResponseDecoder(tf.keras.Model):
    """
    Complete Marketing Response Decoder.
    
    Architecture:
    1. Graph-guided message passing
    2. Temporal modeling (RNN)
    3. Carryover effects
    4. Saturation transformation
    
    Reference: Paper Section 3.3, Figure 3
    """
    
    def __init__(self, config, context_dim: int = 0):
        super().__init__()
        self.config = config
        self.dec_config = config.decoder
        self.n_variables = config.n_variables
        
        # Edge message MLP
        self.edge_mlp = tf.keras.Sequential([
            KL.Dense(self.dec_config.hidden_dim, activation='relu'),
            KL.Dropout(self.dec_config.dropout),
            KL.Dense(self.dec_config.hidden_dim)
        ])
        
        # RNN for temporal modeling
        if self.dec_config.rnn_type == 'gru':
            self.rnn = KL.GRU(
                self.dec_config.hidden_dim,
                return_state=True,
                return_sequences=True
            )
        else:  # lstm
            self.rnn = KL.LSTM(
                self.dec_config.hidden_dim,
                return_state=True,
                return_sequences=True
            )
        
        # Prediction head
        self.pred_head = tf.keras.Sequential([
            KL.Dense(self.dec_config.hidden_dim, activation='relu'),
            KL.Dropout(self.dec_config.dropout),
            KL.Dense(1)
        ])
        
        # Carryover module
        self.carryover = CarryoverModule(self.dec_config.carryover_fn)
        
        # Saturation module (applied only to target)
        self.saturation = SaturationModule(self.dec_config.saturation_fn, context_dim)
        
        # Learnable output variance
        if self.dec_config.learn_sigma:
            self.log_sigma = tf.Variable(
                np.log(self.dec_config.sigma_init),
                trainable=True,
                dtype=tf.float32
            )
        else:
            self.log_sigma = tf.constant(np.log(self.dec_config.sigma_init), dtype=tf.float32)
    
    def call(self, X: tf.Tensor, y: tf.Tensor, z_graph: tf.Tensor,
             context: Optional[tf.Tensor] = None, training: bool = False) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Decode using causal graph.
        
        Args:
            X: [B, T, d] - Channel data
            y: [B, T, 1] - Target data
            z_graph: [B, n, n] - Causal adjacency matrix
            context: [B, c] - Context features
            
        Returns:
            mu: [B, T-1, n] - Predictions
            sigma: scalar - Output std
        """
        B = tf.shape(X)[0]
        T = tf.shape(X)[1]
        n = self.n_variables
        
        # Apply carryover to channels
        X_carryover = self.carryover(X)
        
        # Concatenate all variables
        XY = tf.concat([X_carryover, y], axis=-1)  # [B, T, n]
        
        # Temporal loop with tf.while_loop
        mu_array = tf.TensorArray(dtype=tf.float32, size=T-1)
        
        if self.dec_config.rnn_type == 'gru':
            h_state = tf.zeros([B, n, self.dec_config.hidden_dim])
            initial_state = (0, mu_array, h_state, XY)
        else:  # lstm
            h_state = tf.zeros([B, n, self.dec_config.hidden_dim])
            c_state = tf.zeros([B, n, self.dec_config.hidden_dim])
            initial_state = (0, mu_array, h_state, c_state, XY)
        
        def body_gru(t, mu_array, h_state, XY):
            xt = XY[:, t, :]  # [B, n]
            
            # Create pairwise features
            xi = tf.expand_dims(xt, 1)  # [B, 1, n]
            xj = tf.expand_dims(xt, 2)  # [B, n, 1]
            xi_tiled = tf.tile(xi, [1, n, 1])  # [B, n, n]
            xj_tiled = tf.tile(xj, [1, 1, n])  # [B, n, n]
            pair = tf.stack([xi_tiled, xj_tiled], axis=-1)  # [B, n, n, 2]
            
            # Edge messages weighted by graph
            msg_ij = self.edge_mlp(pair, training=training)  # [B, n, n, H]
            z_expand = tf.expand_dims(z_graph, -1)  # [B, n, n, 1]
            msg_j = tf.reduce_sum(z_expand * msg_ij, axis=1)  # [B, n, H]
            
            # RNN update (per node)
            msg_j_seq = tf.expand_dims(msg_j, 1)  # [B, 1, n, H]
            msg_j_seq = tf.reshape(msg_j_seq, [B * n, 1, self.dec_config.hidden_dim])
            h_state_flat = tf.reshape(h_state, [B * n, self.dec_config.hidden_dim])
            
            out_seq, h_new = self.rnn(msg_j_seq, initial_state=h_state_flat, training=training)
            h_state_new = tf.reshape(h_new, [B, n, self.dec_config.hidden_dim])
            
            # Predict next step
            pred_in = tf.reshape(out_seq[:, -1, :], [B, n, self.dec_config.hidden_dim])
            mu_t1 = self.pred_head(pred_in, training=training)  # [B, n, 1]
            mu_t1 = tf.squeeze(mu_t1, -1)  # [B, n]
            
            # Apply saturation only to target (last variable)
            mu_channels = mu_t1[:, :-1]
            mu_target = mu_t1[:, -1:]
            mu_target_sat = self.saturation(mu_target, context)
            mu_t1_final = tf.concat([mu_channels, mu_target_sat], axis=-1)
            
            mu_array_new = mu_array.write(t, mu_t1_final)
            return t + 1, mu_array_new, h_state_new, XY
        
        def body_lstm(t, mu_array, h_state, c_state, XY):
            # Similar to GRU but with cell state
            xt = XY[:, t, :]
            xi = tf.expand_dims(xt, 1)
            xj = tf.expand_dims(xt, 2)
            xi_tiled = tf.tile(xi, [1, n, 1])
            xj_tiled = tf.tile(xj, [1, 1, n])
            pair = tf.stack([xi_tiled, xj_tiled], axis=-1)
            
            msg_ij = self.edge_mlp(pair, training=training)
            z_expand = tf.expand_dims(z_graph, -1)
            msg_j = tf.reduce_sum(z_expand * msg_ij, axis=1)
            
            msg_j_seq = tf.reshape(tf.expand_dims(msg_j, 1), [B * n, 1, self.dec_config.hidden_dim])
            h_state_flat = tf.reshape(h_state, [B * n, self.dec_config.hidden_dim])
            c_state_flat = tf.reshape(c_state, [B * n, self.dec_config.hidden_dim])
            
            out_seq, h_new, c_new = self.rnn(
                msg_j_seq,
                initial_state=[h_state_flat, c_state_flat],
                training=training
            )
            h_state_new = tf.reshape(h_new, [B, n, self.dec_config.hidden_dim])
            c_state_new = tf.reshape(c_new, [B, n, self.dec_config.hidden_dim])
            
            pred_in = tf.reshape(out_seq[:, -1, :], [B, n, self.dec_config.hidden_dim])
            mu_t1 = tf.squeeze(self.pred_head(pred_in, training=training), -1)
            
            mu_channels = mu_t1[:, :-1]
            mu_target = mu_t1[:, -1:]
            mu_target_sat = self.saturation(mu_target, context)
            mu_t1_final = tf.concat([mu_channels, mu_target_sat], axis=-1)
            
            mu_array_new = mu_array.write(t, mu_t1_final)
            return t + 1, mu_array_new, h_state_new, c_state_new, XY
        
        # Run temporal loop
        if self.dec_config.rnn_type == 'gru':
            _, mu_array_final, _, _ = tf.while_loop(
                cond=lambda t, *_: t < T - 1,
                body=body_gru,
                loop_vars=initial_state,
                parallel_iterations=1
            )
        else:
            _, mu_array_final, _, _, _ = tf.while_loop(
                cond=lambda t, *_: t < T - 1,
                body=body_lstm,
                loop_vars=initial_state,
                parallel_iterations=1
            )
        
        mu_all = mu_array_final.stack()  # [T-1, B, n]
        mu_all = tf.transpose(mu_all, [1, 0, 2])  # [B, T-1, n]
        sigma = tf.exp(self.log_sigma)
        
        return mu_all, sigma
