"""
CausalMMM: Learning Causal Structure for Marketing Mix Modeling
================================================================

A TensorFlow implementation of CausalMMM with extensions for non-stationary data.

Features:
- Automatic causal structure discovery using Graph VAE
- Temporal modeling with GRU/LSTM
- Saturation curves (S-curve, Hill equation)
- Carryover/adstock effects
- CD-NOTS integration for non-stationary data
- Regime change detection
- Comprehensive visualization tools

References:
    Gong, C., et al. (2024). CausalMMM: Learning Causal Structure for 
    Marketing Mix Modeling. WSDM '24. https://arxiv.org/abs/2406.16728

Example:
    >>> from causalmmm import CausalMMM, CausalMMMLConfig
    >>> from causalmmm.preprocessing import PanelDataLoader
    >>> 
    >>> # Load data
    >>> loader = PanelDataLoader()
    >>> X, y, context = loader.load_from_dataframe(df)
    >>> 
    >>> # Configure and train
    >>> config = CausalMMM.Config(n_channels=5, hidden_dim=64)
    >>> model = CausalMMM(config)
    >>> model.fit(X, y, context, epochs=50)
    >>> 
    >>> # Get causal graph
    >>> graph = model.get_causal_graph(threshold=0.3)
    >>> 
    >>> # Predict
    >>> predictions = model.predict(X_test, context_test)
"""

__version__ = "0.1.0"
__author__ = "CausalMMM Contributors"
__license__ = "MIT"

# Core models
from causalmmm.models.causalmmm import CausalMMM
from causalmmm.models.hybrid import HybridCausalMMM

# Configuration
from causalmmm.utils.config import (
    CausalMMMConfig,
    HybridConfig,
    EncoderConfig,
    DecoderConfig
)

# Preprocessing
from causalmmm.preprocessing.transformers import (
    PanelDataLoader,
    GroupStandardizer,
    AdstockTransformer
)

# Discovery
from causalmmm.discovery.cdnots import CDNOTSDiscovery
from causalmmm.discovery.regime_detection import RegimeDetector

# Metrics
from causalmmm.metrics.evaluation import (
    evaluate_forecast,
    evaluate_causal_structure,
    compute_attribution
)

# Visualization
from causalmmm.visualization.graphs import (
    plot_causal_graph,
    plot_graph_comparison,
    plot_temporal_effects
)
from causalmmm.visualization.analysis import (
    plot_channel_contribution,
    plot_saturation_curves,
    plot_response_curves
)

__all__ = [
    # Core
    'CausalMMM',
    'HybridCausalMMM',
    
    # Config
    'CausalMMMConfig',
    'HybridConfig',
    'EncoderConfig',
    'DecoderConfig',
    
    # Preprocessing
    'PanelDataLoader',
    'GroupStandardizer',
    'AdstockTransformer',
    
    # Discovery
    'CDNOTSDiscovery',
    'RegimeDetector',
    
    # Metrics
    'evaluate_forecast',
    'evaluate_causal_structure',
    'compute_attribution',
    
    # Visualization
    'plot_causal_graph',
    'plot_graph_comparison',
    'plot_temporal_effects',
    'plot_channel_contribution',
    'plot_saturation_curves',
    'plot_response_curves',
]

