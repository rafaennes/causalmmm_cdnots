"""
CAUSAL EDGES PATCH for mmm_data_generator
==========================================

This file contains the exact code to add inter-channel causal relationships
to the data generator. Three files need modification:

1. config.py - Add CausalEdgeConfig dataclass and causal_edges field to MMMDataConfig
2. core.py - Modify _generate_channel_spend_data to apply causal spillover
3. presets.py - Add new preset with causal edges

The causal edges create realistic marketing dynamics like:
- TV spend → increases Search volume (with 1-2 week delay)
- Social Media → increases Brand Search (with 1 week delay)
- Video → increases Social engagement (contemporaneous)

These relationships are known ground truth that CD-NOTS should discover.
"""

# ============================================================
# STEP 1: Add to config.py (before MMMDataConfig class)
# ============================================================

CONFIG_ADDITION = '''
@dataclass
class CausalEdgeConfig:
    """Configuration for a causal relationship between two channels.
    
    Models the phenomenon where spend in one channel causes increased
    spend/volume in another channel with a time delay. For example,
    TV advertising increases search queries 1-2 weeks later.
    
    Parameters
    ----------
    source_channel : str
        Name of the source channel (cause). Must match a ChannelConfig.name.
    target_channel : str
        Name of the target channel (effect). Must match a ChannelConfig.name.
    effect_size : float
        Strength of the causal effect. Fraction of source spend that 
        "spills over" to target. E.g., 0.15 means 15% of source spend
        is added to target spend after transformation.
    lag : int
        Delay in periods (weeks) before the effect manifests.
        lag=0 means contemporaneous, lag=1 means 1-week delay.
    decay : float
        Geometric decay of the spillover effect. 0.0 = no persistence
        (effect only at lag), 1.0 = permanent effect. Typical: 0.3-0.7.
    """
    source_channel: str
    target_channel: str
    effect_size: float = 0.15
    lag: int = 1
    decay: float = 0.5
    
    def __post_init__(self):
        if self.effect_size < 0 or self.effect_size > 1:
            raise ValueError("effect_size must be between 0 and 1")
        if self.lag < 0:
            raise ValueError("lag must be non-negative")
        if self.decay < 0 or self.decay > 1:
            raise ValueError("decay must be between 0 and 1")
        if self.source_channel == self.target_channel:
            raise ValueError("source and target channels must be different")
'''

# Add this field to MMMDataConfig (after control_variables):
MMMDATA_CONFIG_FIELD = '''
    # Inter-channel causal relationships
    causal_edges: List[CausalEdgeConfig] = field(default_factory=list)
'''

# ============================================================
# STEP 2: Replace _generate_channel_spend_data in core.py
# ============================================================

CORE_REPLACEMENT = '''
def _generate_channel_spend_data(
    config: MMMDataConfig, 
    time_index: pd.DatetimeIndex
) -> pd.DataFrame:
    """Generate spend data for all channels across all regions.
    
    If config.causal_edges is defined, applies inter-channel causal
    spillover effects: source channel spend at time t causes additional
    spend in target channel at time t + lag, with geometric decay.
    
    The causal edges create a known ground truth causal graph that
    can be used to evaluate causal discovery algorithms.
    """
    from .config import CausalEdgeConfig
    
    regional_dataframes = []
    
    for region_idx, region_name in enumerate(config.regions.region_names):
        regional_channels = generate_regional_channel_variations(
            config.regions, config.channels, region_idx, config.seed
        )
        
        region_data = pd.DataFrame(
            index=pd.MultiIndex.from_product(
                [time_index, [region_name]], names=['date', 'geo']
            )
        )
        
        # Phase 1: Generate independent (base) spend for each channel
        channel_spends = {}
        for i, channel in enumerate(regional_channels):
            region_seed = (
                config.seed + region_idx * 1000 + i * 100
                if config.seed is not None else None
            )
            base_spend = generate_channel_spend(channel, time_index, region_seed)
            col_name = f'x{i+1}_{channel.name}' if channel.name != "" else f'x{i+1}'
            channel_spends[col_name] = base_spend.copy()
        
        # Phase 2: Apply causal spillover effects (if any)
        if hasattr(config, 'causal_edges') and config.causal_edges:
            # Build name -> column mapping
            name_to_col = {}
            for i, channel in enumerate(config.channels):
                col_name = f'x{i+1}_{channel.name}' if channel.name != "" else f'x{i+1}'
                name_to_col[channel.name] = col_name
            
            for edge in config.causal_edges:
                source_col = name_to_col.get(edge.source_channel)
                target_col = name_to_col.get(edge.target_channel)
                
                if source_col is None or target_col is None:
                    continue
                
                source_spend = channel_spends[source_col]
                n_periods = len(source_spend)
                
                # Compute spillover: geometric adstock of source, shifted by lag
                spillover = np.zeros(n_periods)
                adstocked_source = np.zeros(n_periods)
                
                # Apply geometric decay to source
                for t in range(n_periods):
                    if t == 0:
                        adstocked_source[t] = source_spend[t]
                    else:
                        adstocked_source[t] = (
                            source_spend[t] + edge.decay * adstocked_source[t - 1]
                        )
                
                # Shift by lag and scale by effect_size
                for t in range(n_periods):
                    source_t = t - edge.lag
                    if source_t >= 0:
                        spillover[t] = edge.effect_size * adstocked_source[source_t]
                
                # Add spillover to target channel spend
                channel_spends[target_col] = channel_spends[target_col] + spillover
        
        # Store in DataFrame
        for col_name, spend_values in channel_spends.items():
            region_data[col_name] = spend_values
        
        regional_dataframes.append(region_data)
    
    return pd.concat(regional_dataframes, axis=0)
'''

# ============================================================
# STEP 3: Add ground truth causal graph to generate_mmm_dataset
# in core.py (add to ground_truth dict, after attribution_percentages)
# ============================================================

GROUND_TRUTH_ADDITION = '''
        # Build ground truth causal graph from config
        causal_graph_ground_truth = _build_causal_ground_truth(config)
        
        ground_truth['causal_graph'] = causal_graph_ground_truth
'''

GROUND_TRUTH_FUNCTION = '''
def _build_causal_ground_truth(config: MMMDataConfig) -> Dict[str, Any]:
    """Build ground truth causal adjacency matrix from config.
    
    Returns dict with:
        - adjacency_matrix: [n_channels+1, n_channels+1] binary matrix
          (last row/col is target y)
        - variable_names: ordered list matching matrix indices
        - edges: list of (source, target, effect_size, lag) tuples
    """
    n_channels = len(config.channels)
    n_vars = n_channels + 1  # channels + y
    
    # Variable names
    var_names = []
    for i, ch in enumerate(config.channels):
        col_name = f'x{i+1}_{ch.name}' if ch.name != "" else f'x{i+1}'
        var_names.append(col_name)
    var_names.append('y')
    
    # Adjacency matrix
    adj = np.zeros((n_vars, n_vars))
    
    # All channels with non-zero effectiveness cause y
    y_idx = n_vars - 1
    for i, ch in enumerate(config.channels):
        if ch.base_effectiveness > 0:
            adj[i, y_idx] = 1.0
    
    # Inter-channel causal edges
    edges = []
    name_to_idx = {ch.name: i for i, ch in enumerate(config.channels)}
    
    if hasattr(config, 'causal_edges'):
        for edge in config.causal_edges:
            src_idx = name_to_idx.get(edge.source_channel)
            tgt_idx = name_to_idx.get(edge.target_channel)
            if src_idx is not None and tgt_idx is not None:
                adj[src_idx, tgt_idx] = 1.0
                edges.append({
                    'source': edge.source_channel,
                    'target': edge.target_channel,
                    'effect_size': edge.effect_size,
                    'lag': edge.lag,
                    'decay': edge.decay,
                })
    
    return {
        'adjacency_matrix': adj,
        'variable_names': var_names,
        'edges': edges,
        'n_channels': n_channels,
    }
'''

# ============================================================
# STEP 4: Add new preset with causal edges to presets.py
# ============================================================

PRESET_WITH_CAUSAL_EDGES = '''
def _get_causal_business_preset(seed: int) -> MMMDataConfig:
    """Business preset with known inter-channel causal relationships.
    
    Causal structure (ground truth):
        TV → Search-Ads (lag=2, effect=0.20, decay=0.5)
            TV advertising drives search queries 2 weeks later
        Social-Media → Brand-Search (lag=1, effect=0.15, decay=0.4)
            Social engagement increases brand searches 1 week later
        Video → Social-Media (lag=1, effect=0.10, decay=0.3)
            Video content drives social media engagement
        All channels → Sales (direct, via adstock + saturation)
    
    This creates a realistic marketing funnel:
        Video → Social → Brand-Search → Sales
        TV → Search → Sales
        All channels also directly affect Sales
    """
    from .config import CausalEdgeConfig
    
    return MMMDataConfig(
        n_periods=156,  # 3 years
        channels=[
            ChannelConfig(
                name="Search-Ads",
                pattern="linear_trend",
                base_spend=3000.0,
                spend_trend=0.03,
                spend_volatility=0.15,
                base_effectiveness=0.7
            ),
            ChannelConfig(
                name="Brand-Search",
                pattern="seasonal",
                base_spend=1500.0,
                seasonal_amplitude=0.2,
                spend_volatility=0.12,
                base_effectiveness=0.5
            ),
            ChannelConfig(
                name="TV",
                pattern="seasonal",
                base_spend=5000.0,
                seasonal_amplitude=0.3,
                seasonal_phase=0.5,
                spend_volatility=0.08,
                base_effectiveness=0.8
            ),
            ChannelConfig(
                name="Video",
                pattern="linear_trend",
                base_spend=2000.0,
                spend_trend=0.06,
                spend_volatility=0.20,
                base_effectiveness=0.4
            ),
            ChannelConfig(
                name="Social-Media",
                pattern="seasonal",
                base_spend=2500.0,
                seasonal_amplitude=0.25,
                spend_volatility=0.18,
                base_effectiveness=0.5
            ),
            ChannelConfig(
                name="Display-Ads",
                pattern="on_off",
                base_spend=1800.0,
                activation_probability=0.6,
                spend_volatility=0.15,
                base_effectiveness=0.3
            ),
        ],
        causal_edges=[
            # TV drives search queries with 2-week delay
            CausalEdgeConfig(
                source_channel="TV",
                target_channel="Search-Ads",
                effect_size=0.20,
                lag=2,
                decay=0.5,
            ),
            # Social media drives brand search with 1-week delay
            CausalEdgeConfig(
                source_channel="Social-Media",
                target_channel="Brand-Search",
                effect_size=0.15,
                lag=1,
                decay=0.4,
            ),
            # Video drives social engagement with 1-week delay
            CausalEdgeConfig(
                source_channel="Video",
                target_channel="Social-Media",
                effect_size=0.10,
                lag=1,
                decay=0.3,
            ),
        ],
        regions=RegionConfig(
            n_regions=4,
            region_names=["geo_a", "geo_b", "geo_c", "geo_d"],
            base_sales_rate=15000.0,
            sales_trend=0.01,
            sales_volatility=0.02,
            seasonal_amplitude=0.15,
            baseline_variation=0.15,
            channel_param_variation=0.10,
            transform_variation=0.08,
        ),
        transforms=TransformConfig(
            adstock_fun="geometric_adstock",
            adstock_kwargs=[
                {"alpha": 0.5},   # Search-Ads: moderate decay
                {"alpha": 0.4},   # Brand-Search: faster decay
                {"alpha": 0.7},   # TV: slow decay
                {"alpha": 0.6},   # Video: moderate decay
                {"alpha": 0.5},   # Social-Media: moderate decay
                {"alpha": 0.3},   # Display-Ads: fast decay
            ],
            saturation_fun="hill_function",
            saturation_kwargs=[
                {"slope": 1.0, "kappa": 0.15},
                {"slope": 1.2, "kappa": 0.12},
                {"slope": 0.8, "kappa": 0.20},
                {"slope": 1.0, "kappa": 0.18},
                {"slope": 1.1, "kappa": 0.14},
                {"slope": 0.9, "kappa": 0.16},
            ],
        ),
        control_variables=[
            ControlConfig(
                name="price",
                pattern="linear_trend",
                base_value=10.0,
                value_volatility=1.0,
                value_trend=0.1,
                base_effectiveness=-0.3,
            ),
        ],
        seed=seed,
    )
'''

# Don't forget to add 'causal_business' to the presets dict in get_preset_config:
PRESET_REGISTRY_ADDITION = '''
        'causal_business': _get_causal_business_preset,
'''


# ============================================================
# Summary of changes
# ============================================================

if __name__ == "__main__":
    print("""
CAUSAL EDGES PATCH - Summary of Changes
========================================

Files to modify:
    
1. mmm_data_generator/config.py:
   - Add CausalEdgeConfig dataclass (before MMMDataConfig)
   - Add 'causal_edges' field to MMMDataConfig
   - Add CausalEdgeConfig to imports in __init__.py

2. mmm_data_generator/core.py:
   - Replace _generate_channel_spend_data() with version that applies spillover
   - Add _build_causal_ground_truth() function
   - Add causal_graph to ground_truth dict in generate_mmm_dataset()

3. mmm_data_generator/presets.py:
   - Add _get_causal_business_preset() function
   - Add 'causal_business' to presets dict
   - Import CausalEdgeConfig

4. mmm_data_generator/__init__.py:
   - Export CausalEdgeConfig

Usage after patching:
    
    from mmm_data_generator import get_preset_config, generate_mmm_dataset
    
    config = get_preset_config('causal_business')
    result = generate_mmm_dataset(config)
    
    # Ground truth causal graph
    gt = result['ground_truth']['causal_graph']
    print(gt['adjacency_matrix'])  # [n_channels+1, n_channels+1]
    print(gt['edges'])             # List of causal edges with params
    
    # Run benchmark
    python run_benchmark.py --datasets causal_business --samplers nutpie --cdnots

Expected causal graph (ground truth):
    TV ──(lag=2)──→ Search-Ads ──→ Sales
    Video ──(lag=1)──→ Social-Media ──(lag=1)──→ Brand-Search ──→ Sales
    All channels ──→ Sales (direct)
    Display-Ads ──→ Sales (direct, no inter-channel)

This creates 3 inter-channel edges + 6 channel→Sales edges = 9 true edges.
CD-NOTS should discover both the direct and mediated paths.
""")
