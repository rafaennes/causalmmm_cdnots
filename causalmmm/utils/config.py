"""Configuration classes for CausalMMM models."""

from dataclasses import dataclass, field
from typing import Optional, List, Literal


@dataclass
class EncoderConfig:
    """Configuration for Causal Relational Encoder.
    
    Attributes:
        hidden_dim: Hidden dimension for embeddings
        n_layers: Number of interaction layers
        aggregation: Type of aggregation ('mean', 'sum', 'attention')
        use_time_node: Whether to include time node (CD-NOTS style)
        time_encoding: Time encoding method ('linear', 'cyclical', 'learned')
    """
    hidden_dim: int = 64
    n_layers: int = 2
    aggregation: Literal['mean', 'sum', 'attention'] = 'mean'
    use_time_node: bool = False
    time_encoding: Literal['linear', 'cyclical', 'learned'] = 'cyclical'
    dropout: float = 0.1


@dataclass
class DecoderConfig:
    """Configuration for Marketing Response Decoder.
    
    Attributes:
        hidden_dim: Hidden dimension for GRU/LSTM
        rnn_type: Type of RNN ('gru', 'lstm')
        saturation_fn: Saturation function ('scurve', 'hill', 'none')
        carryover_fn: Carryover function ('geometric', 'weibull', 'none')
        learn_sigma: Whether to learn output variance
    """
    hidden_dim: int = 64
    rnn_type: Literal['gru', 'lstm'] = 'gru'
    saturation_fn: Literal['scurve', 'hill', 'none'] = 'scurve'
    carryover_fn: Literal['geometric', 'weibull', 'none'] = 'geometric'
    learn_sigma: bool = True
    sigma_init: float = 0.1
    dropout: float = 0.1


@dataclass
class CausalMMMConfig:
    """Main configuration for CausalMMM model.
    
    This follows the architecture from Gong et al. (2024) paper.
    
    Attributes:
        n_channels: Number of marketing channels
        encoder: Encoder configuration
        decoder: Decoder configuration
        vae_mode: Use full VAE framework (ELBO loss)
        temperature: Initial Gumbel-Softmax temperature
        temperature_min: Minimum temperature
        temperature_decay: Temperature decay rate
        hard_gumbel: Use hard Gumbel-Softmax
        prior_pi: Prior edge probability (sparsity)
        lambda_kl: Weight for KL divergence loss
        lambda_dag: Weight for DAG constraint
        lambda_temporal: Weight for temporal regularization
        lambda_saturation: Weight for saturation regularization
    """
    n_channels: int
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    
    # VAE settings
    vae_mode: bool = True
    
    # Gumbel-Softmax
    temperature: float = 1.0
    temperature_min: float = 0.5
    temperature_decay: float = 0.9995
    hard_gumbel: bool = False
    
    # Regularization
    prior_pi: float = 0.1
    lambda_kl: float = 1.0
    lambda_dag: float = 1.0
    lambda_temporal: float = 0.1
    lambda_saturation: float = 0.1
    
    # Optimization
    learning_rate: float = 1e-3
    gradient_clip: float = 1.0
    
    @property
    def n_variables(self) -> int:
        """Total number of variables (channels + target)."""
        return self.n_channels + 1


@dataclass
class HybridConfig(CausalMMMConfig):
    """Configuration for Hybrid model with CD-NOTS.
    
    Extends CausalMMMConfig with CD-NOTS specific settings.
    
    Attributes:
        use_cdnots_prior: Whether to use CD-NOTS for prior discovery
        cdnots_alpha: Significance level for CD-NOTS tests
        cdnots_indep_test: Independence test type
        lambda_prior: Weight for prior structure loss
        detect_regimes: Enable regime change detection
        regime_window: Window size for regime detection
    """
    use_cdnots_prior: bool = True
    cdnots_alpha: float = 0.05
    cdnots_indep_test: Literal['fisherz', 'kci', 'chisq', 'gsq'] = 'fisherz'
    lambda_prior: float = 10.0
    detect_regimes: bool = True
    regime_window: int = 10
