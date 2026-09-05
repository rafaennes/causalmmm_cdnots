## Copyright 2025 PyMC Labs
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Configuration presets for common MMM use cases.

This module provides pre-configured settings for typical MMM scenarios.
"""

import numpy as np
from math import pi
from typing import Dict, Any
from .config import ControlConfig, MMMDataConfig, ChannelConfig, RegionConfig, TransformConfig, CausalEdgeConfig


def get_preset_config(preset_name: str, seed: int = 2025_07_15) -> MMMDataConfig:
    """
    Get a named preset configuration.
    
    Parameters
    ----------
    preset_name : str
        Name of the preset configuration
        
    Returns
    -------
    MMMConfig
        Pre-configured MMM configuration
        
    Raises
    ------
    ValueError
        If preset name is not recognized
    """
    presets = {
        'basic': _get_basic_preset,
        'seasonal': _get_seasonal_preset,
        'multi_region': _get_multi_region_preset,
        'small_business': _get_small_business_preset,
        'growing_business': _get_growing_business_preset,
        'medium_business': _get_medium_business_preset,
        'large_business': _get_large_business_preset,
        'causal_business': _get_causal_business_preset,
        'causal_large': _get_causal_large_preset,
    }
    
    if preset_name not in presets:
        available_presets = ', '.join(presets.keys())
        raise ValueError(f"Unknown preset '{preset_name}'. Available presets: {available_presets}")
    
    return presets[preset_name](seed)


def _get_basic_preset(seed: int) -> MMMDataConfig:
    """Basic preset for learning and testing."""
    return MMMDataConfig(
        n_periods=52,  # 1 year of weekly data
        channels=[
            ChannelConfig(
                name="x-TV",
                pattern="linear_trend",
                base_spend=2000.0,
                spend_trend=0.02,
                base_effectiveness=0.6
            ),
            ChannelConfig(
                name="x-Digital",
                pattern="linear_trend",
                base_spend=1500.0,
                spend_trend=0.05,
                base_effectiveness=0.5
            ),
            ChannelConfig(
                name="x-Print",
                pattern="linear_trend",
                base_spend=800.0,
                spend_trend=-0.01,
                base_effectiveness=0.3
            )
        ],
        regions=RegionConfig(
            n_regions=1,
            region_names=["geo_a"]
        ),
        transforms=TransformConfig(
            adstock_fun="geometric_adstock",
            adstock_kwargs={"alpha": 0.5},
            saturation_fun="hill_function",
            saturation_kwargs={"slope": 1.0, "kappa": 0.15}
        ),
        seed=seed
    )


def _get_seasonal_preset(seed: int) -> MMMDataConfig:
    """Preset with strong seasonal patterns for seasonal business analysis."""
    return MMMDataConfig(
        n_periods=104,  # 2 years for seasonal analysis
        channels=[
            ChannelConfig(
                name="x-TV",
                pattern="seasonal",
                base_spend=3000.0,
                seasonal_amplitude=0.6,
                seasonal_phase=0.0,
                base_effectiveness=0.7
            ),
            ChannelConfig(
                name="x-Digital",
                pattern="seasonal",
                base_spend=2000.0,
                seasonal_amplitude=0.4,
                seasonal_phase=0.5,  # Offset phase
                base_effectiveness=0.5
            ),
            ChannelConfig(
                name="x-Radio",
                pattern="seasonal",
                base_spend=1000.0,
                seasonal_amplitude=0.3,
                seasonal_phase=1.0,
                base_effectiveness=0.3
            )
        ],
        regions=RegionConfig(
            n_regions=1,
            region_names=["geo_a"],
            seasonal_amplitude=0.3
        ),
        transforms=TransformConfig(
            adstock_fun="geometric_adstock",
            adstock_kwargs={"alpha": 0.6, "l_max": 8},
            saturation_fun="hill_function",
            saturation_kwargs={"slope": 1.0, "kappa": 0.2}
        ),
        seed=seed
    )


def _get_multi_region_preset(seed: int) -> MMMDataConfig:
    """Preset for multi-regional analysis with regional variations."""
    return MMMDataConfig(
        n_periods=78,  # 1.5 years
        channels=[
            ChannelConfig(
                name="x-TV",
                pattern="linear_trend",
                base_spend=2500.0,
                spend_trend=0.01,
                base_effectiveness=0.6
            ),
            ChannelConfig(
                name="x-Digital",
                pattern="linear_trend",
                base_spend=1800.0,
                spend_trend=0.08,
                base_effectiveness=0.5
            ),
            ChannelConfig(
                name="x-Print",
                pattern="linear_trend",
                base_spend=800.0,
                spend_trend=-0.03,
                base_effectiveness=0.3
            )
        ],
        regions=RegionConfig(
            n_regions=4,
            region_names=["geo_a", "geo_b", "geo_c", "geo_d"]
        ),
        transforms=TransformConfig(
            adstock_fun="geometric_adstock",
            adstock_kwargs={"alpha": 0.55},
            saturation_fun="hill_function",
            saturation_kwargs={"slope": 1.0, "kappa": 0.8}
        ),
        seed=seed
    )


def _get_small_business_preset(seed: int) -> MMMDataConfig:
    """Preset for small business with limited budget and channels."""
    return MMMDataConfig(
        n_periods=104,
        channels=[
            ChannelConfig(
                name="Search-Ads",
                pattern="linear_trend",
                spend_volatility=0.8,
                base_spend=100.0,
                spend_trend=0.004, # about 20% per year
                base_effectiveness=1.5,
            ),
            ChannelConfig(
                name="Social-Media",
                pattern="seasonal",
                base_spend=500.0,
                spend_volatility=0.3,
                seasonal_amplitude=0.4,
                seasonal_phase=0.5,
                base_effectiveness=1.2,
            ),
            ChannelConfig(
                name="Local-Ads",
                pattern="on_off",
                base_spend=500.0,
                spend_volatility=0.6,
                activation_probability=0.3,
                min_active_periods=2,
                max_active_periods=4,
                base_effectiveness=.9,
            ),
            ChannelConfig(
                name="Email",
                pattern="on_off",
                base_spend=100.0,
                spend_volatility=0.5,
                activation_probability=0.2,
                min_active_periods=1,
                max_active_periods=1,
                base_effectiveness=1.2,
            )
        ],
        control_variables=[
            ControlConfig(
                name="Event",
                pattern="on_off",
                base_value=100.0,
                value_volatility=0.5,
                base_effectiveness=0.5,
                activation_probability=0.04,
                min_active_periods=3,
                max_active_periods=5,
            ),
            ControlConfig(
                name="Sale",
                pattern="on_off",
                base_value=50.0,
                value_volatility=0.5,
                base_effectiveness=0.5,
                activation_probability=0.1,
                min_active_periods=1,
                max_active_periods=4,
            )
        ],
        regions=RegionConfig(
            n_regions=1,
            region_names=["Local"],
            base_sales_rate=5000.0,
            sales_volatility=0.1,
        ),
        transforms=TransformConfig(
            adstock_fun="geometric_adstock",
            adstock_kwargs=[
                {"alpha": 0, "l_max": 8}, 
                {"alpha": 0.2, "l_max": 8}, 
                {"alpha": 0.4, "l_max": 8}, 
                {"alpha": 0.3, "l_max": 8}
            ],
            saturation_fun="hill_function",
            saturation_kwargs = [
                {"slope": 1, "kappa": 1}, 
                {"slope": 1.5, "kappa": 0.8}, 
                {"slope": 1, "kappa": 1.5}, 
                {"slope": 2, "kappa": 0.5}
            ],
        ),
        seed=seed,
    )



def _get_growing_business_preset(seed: int) -> MMMDataConfig:
    """Preset for a growing business expanding to new market regions."""
    return MMMDataConfig(
        n_periods=131, # 2.5 years
        channels=[
            ChannelConfig(
                name="Search-Ads",
                pattern="linear_trend",
                spend_volatility=0.7,
                base_spend=300.0,
                spend_trend=0.002,# about 10% per year
                base_effectiveness=0.3,
            ),
            ChannelConfig(
                name="Search-Ads-Brand",
                pattern="linear_trend",
                spend_volatility=0.8,
                base_spend=100.0,
                spend_trend=0.002, # about 10% per year
                base_effectiveness=1.1,
            ),
            ChannelConfig(
                name="Video",
                pattern="delayed_start",
                base_spend=500.0,
                spend_volatility=0.5,
                start_period=25,
                ramp_up_periods=10,
                base_effectiveness=1.4,
            ),
            ChannelConfig(
                name="Social-Media",
                pattern="seasonal",
                base_spend=500.0,
                spend_volatility=0.6,
                seasonal_amplitude=0.4,
                seasonal_phase=0.5,
                base_effectiveness=0.7,
            ),
            ChannelConfig(
                name="Display-Ads",
                pattern="on_off",
                base_spend=500.0,
                spend_volatility=0.5,
                activation_probability=0.3,
                min_active_periods=2,
                max_active_periods=4,
                base_effectiveness=0.3,
            ),
            ChannelConfig(
                name="Email",
                pattern="on_off",
                base_spend=100.0,
                spend_volatility=0.4,
                activation_probability=0.2,
                min_active_periods=1,
                max_active_periods=1,
                base_effectiveness=0.4,
            )
        ],
        control_variables=[
            ControlConfig(
                name="Event",
                pattern="on_off",
                base_value=500.0,
                value_volatility=0.5,
                base_effectiveness=0.5,
                activation_probability=0.04,
                min_active_periods=3,
                max_active_periods=5,
            ),
            ControlConfig(
                name="Sale",
                pattern="on_off",
                base_value=25.0,
                value_volatility=0.5,
                base_effectiveness=1.5,
                activation_probability=0.1,
                min_active_periods=1,
                max_active_periods=4,
            )
        ],
        regions=RegionConfig(
            n_regions=2,
            region_names=["geo_a", "geo_b"],
            base_sales_rate=5000.0,
            sales_volatility=0.1,
        ),
        transforms=TransformConfig(
            adstock_fun="geometric_adstock",
            adstock_kwargs=[
                {"alpha": 0, "l_max": 8}, 
                {"alpha": 0.2, "l_max": 8}, 
                {"alpha": 0.4, "l_max": 8}, 
                {"alpha": 0.3, "l_max": 8}
            ],
            saturation_fun="hill_function",
            saturation_kwargs= [
                {"slope": 1, "kappa": 0.3},
                {"slope": 1.5, "kappa": 1.5},
                {"slope": 0.8, "kappa": 0.4},
                {"slope": 2, "kappa": 0.5},
                {"slope": .4, "kappa": 2}, # 
                {"slope": 1.5, "kappa": 1.25},
            ],
        ),
        seed=seed
    )

def _get_medium_business_preset(seed: int) -> MMMDataConfig:
    """
    Preset for a medium business with many channels and regions.
    """
    return MMMDataConfig(
        n_periods=156, # 3 years
        channels=[
            ChannelConfig(
                name="Search-Ads",
                pattern="linear_trend",
                spend_volatility=0.8,
                base_spend=300.0,
                spend_trend=0.002, # about 10% per year
                base_effectiveness=1.1,
            ),
            ChannelConfig(
                name="Search-Ads-Brand",
                pattern="linear_trend",
                spend_volatility=1.1,
                base_spend=100.0,
                spend_trend=0.002, # about 10% per year
                base_effectiveness=1.2,
            ),
            ChannelConfig(
                name="Video",
                pattern="delayed_start",
                base_spend=500.0,
                spend_volatility=0.9,
                start_period=25,
                ramp_up_periods=10,
                base_effectiveness=1.1,
            ),
            ChannelConfig(
                name="Video-2",
                pattern="delayed_start",
                base_spend=1000.0,
                spend_volatility=1.2,
                start_period=75,
                end_period=125,
                base_effectiveness=0.9,
            ),
            ChannelConfig(
                name="Social-Media",
                pattern="seasonal",
                base_spend=1000.0,
                spend_volatility=1.2,
                seasonal_amplitude=0.4,
                seasonal_phase=0.5,
                base_effectiveness=0.5,
            ),
            ChannelConfig(
                name="Social-Media-2",
                pattern="seasonal",
                base_spend=1000.0,
                spend_volatility=0.9,
                seasonal_amplitude=0.2,
                seasonal_phase=0.125 * 2 * pi,
                base_effectiveness=1.1,
            ),
            ChannelConfig(
                name="Display-Ads",
                pattern="on_off",
                base_spend=500.0,
                spend_volatility=0.7,
                activation_probability=0.3,
                min_active_periods=2,
                max_active_periods=4,
                base_effectiveness=0.8,
            ),
            ChannelConfig(
                name="Influencer",
                pattern="on_off",
                base_spend=500.0,
                spend_volatility=1.1,
                activation_probability=0.1,
                min_active_periods=2,
                max_active_periods=6,
                base_effectiveness=1.3,
            )
        ],
        control_variables=[
            ControlConfig(
                name="Event-A",
                pattern="on_off",
                base_value=500.0,
                value_volatility=0.5,
                base_effectiveness=0.5,
                activation_probability=0.04,
                min_active_periods=3,
                max_active_periods=5,
            ),
            ControlConfig(
                name="Event-B",
                pattern="on_off",
                base_value=1000.0,
                value_volatility=0.5,
                base_effectiveness=0.2,
                activation_probability=0.2,
                min_active_periods=1,
                max_active_periods=8,
            ),
            ControlConfig(
                name="Linear",
                pattern="linear_trend",
                base_value=1000.0,
                value_trend=0.005,
                value_volatility=1,
                base_effectiveness=0.8,
            ),
            ControlConfig(
                name="Sale",
                pattern="on_off",
                base_value=25.0,
                value_volatility=0.5,
                base_effectiveness=100.0,
                activation_probability=0.1,
                min_active_periods=1,
                max_active_periods=4,
            )
        ],
        regions=RegionConfig(
            n_regions=8,
            region_names=[
                "geo_a", "geo_b", "geo_c", "geo_d",
                "geo_e", "geo_f", "geo_g", "geo_h"
                ],
            base_sales_rate=100000.0,
            sales_volatility=0.1,
            seasonal_amplitude=0.6
        ),
        transforms=TransformConfig(
            adstock_fun="geometric_adstock",
            adstock_kwargs=[
                {"alpha": 0, "l_max": 8}, 
                {"alpha": 0.2, "l_max": 8}, 
                {"alpha": 0.4, "l_max": 8}, 
                {"alpha": 0.3, "l_max": 8},
                {"alpha": 0.2, "l_max": 8},
                {"alpha": 0.1, "l_max": 8},
                {"alpha": 0.05, "l_max": 8},
                {"alpha": 0.1, "l_max": 8},
                ],
            saturation_fun="hill_function",
            saturation_kwargs=[
                {"slope": 1, "kappa": 1},
                {"slope": 1.5, "kappa": 0.4},
                {"slope": 0.8, "kappa": 0.55},
                {"slope": 2, "kappa": 0.7},
                {"slope": 1, "kappa": 0.4},
                {"slope": 0.8, "kappa": 1.2},
                {"slope": 0.5, "kappa": 0.5},
                {"slope": 1.2, "kappa": 0.9},
            ],
        ),
        seed=seed
    )


def _get_large_business_preset(seed: int) -> MMMDataConfig:
    """
    Preset for a large business with 30 channels and 50 regions.
    """
    rng: np.random.Generator = np.random.default_rng(seed=seed)
    return MMMDataConfig(
        n_periods=4 * 52, # 4 years
        channels=[
            ChannelConfig(
                name=f"linear-{n}",
                pattern="linear_trend",
                spend_volatility=rng.uniform(0.8,1.2),
                base_spend=rng.uniform(500,5000),
                spend_trend=rng.uniform(0.0, 0.04),
                base_effectiveness=rng.uniform(0.5, 2),
            ) for n in range(10)] + [
            ChannelConfig(
                name=f"delayed-start-{n}",
                pattern="delayed_start",
                base_spend=rng.uniform(500,5000),
                spend_volatility=rng.uniform(0.8,1.2),
                seasonal_amplitude=rng.uniform(0.2,1),
                seasonal_phase=rng.uniform(0,2 * np.pi),
                base_effectiveness=rng.uniform(0.5, 2),
                start_period=rng.integers(10,2 * 52),
                ramp_up_periods=rng.integers(10, 52),
            ) for n in range(5)] + [
            ChannelConfig(
                name=f"seasonal-{n}",
                pattern="seasonal",
                base_spend=rng.uniform(500, 5000),
                spend_volatility=rng.uniform(0.8, 1.2),
                seasonal_amplitude=rng.uniform(0.2, 1),
                seasonal_phase=rng.uniform(0,np.pi),
                base_effectiveness=rng.uniform(0.5, 2),
            ) for n in range(5)] + [
            ChannelConfig(
                name=f"on-off-{n}",
                pattern="on_off",
                base_spend=rng.uniform(500, 5000),
                spend_volatility=rng.uniform(.8,1.2),
                base_effectiveness=rng.uniform(0.1, 2),
                activation_probability=rng.uniform(0.05,0.1),
                min_active_periods=rng.integers(1,n+2), # ensure min <= max
                max_active_periods=rng.integers(n+1,11),
            ) for n in range(10)],
        control_variables=[
            ControlConfig(
                name=f"on-off-{n}",
                pattern="on_off",
                base_value=rng.uniform(100, 1000),
                value_volatility=rng.uniform(.8,1.2),
                base_effectiveness=rng.uniform(0.1, 2),
                activation_probability=rng.uniform(0.01,0.1),
                min_active_periods=rng.integers(1,n+2), # ensure min <= max
                max_active_periods=rng.integers(n+1,9),
            ) for n in range(6)] + [
            ControlConfig(
                name=f"linear-{n}",
                pattern="linear_trend",
                base_value=rng.uniform(100, 1000),
                value_trend=rng.uniform(0, 0.01),
                value_volatility=rng.uniform(.8,1.2),
                base_effectiveness=rng.uniform(0.1, 2),
            ) for n in range(3)] + [
                ControlConfig("seasonal", "seasonal", base_value=1000.0)
            ],
        regions=RegionConfig(
            n_regions=50,
            region_names=[f"geo_{n + 1}" for n in range(50)],
            base_sales_rate=10000.0,
            sales_volatility=0.1,
            seasonal_amplitude=0.2,
        ),
        transforms=TransformConfig(
            adstock_fun="geometric_adstock",
            adstock_kwargs=[
                {"alpha": rng.uniform(0,0.5), "l_max": 8}
                for _ in range(30) 
                ],
            saturation_fun="hill_function",
            saturation_kwargs=[
                {
                    "slope": np.random.uniform(0,2),
                    "kappa": np.random.uniform(0,2),
                } for _ in range(30)
             ],
        ),
        seed=seed
    )


def list_available_presets() -> Dict[str, str]:
    """
    Get a list of available presets with descriptions.
    
    Returns
    -------
    Dict[str, str]
        Dictionary mapping preset names to descriptions
    """
    return {
        'basic': 'Simple configuration for learning and testing',
        'seasonal': 'Strong seasonal patterns for seasonal business analysis',
        'multi_region': 'Multi-regional analysis with regional variations',
        'small_business': 'Small business with limited budget and channels',
        'medium_business': 'Medium business with more budget and channels',
        'large_business': 'Large business with many channels and regions',
        'growing_business': 'Growing business with increasing spend patterns',
        'causal_business': 'Business with inter-channel causal spillover effects (CD-NOTS benchmark)',
        'causal_large': '1 geo, 20 media channels, 10 controls, 5 causal inter-channel edges (CD-NOTS at scale)',
        }


def customize_preset(preset_name: str, **overrides) -> MMMDataConfig:
    """
    Get a preset configuration with custom overrides.
    
    Parameters
    ----------
    preset_name : str
        Name of the preset configuration
    **overrides
        Parameters to override in the preset
        
    Returns
    -------
    MMMConfig
        Customized preset configuration
    """
    config = get_preset_config(preset_name)
    
    # Apply overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            # Try to set nested attributes
            parts = key.split('.')
            obj = config
            for part in parts[:-1]:
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    raise ValueError(f"Invalid override key: {key}")
            if hasattr(obj, parts[-1]):
                setattr(obj, parts[-1], value)
            else:
                raise ValueError(f"Invalid override key: {key}")
    
    return config 


def _get_causal_business_preset(seed: int) -> MMMDataConfig:
    """Business preset with known inter-channel causal relationships,
    endogeneity, and ghost channels.
    
    Causal structure (ground truth):
        TV → Search-Ads (lag=2, effect=0.20) — TV drives search queries
        Social-Media → Brand-Search (lag=1, effect=0.15) — social drives brand
        Video → Social-Media (lag=1, effect=0.10) — video drives social
        All real channels → Sales (direct, via adstock + saturation)
    
    Endogeneity test:
        Price (control) → Search-Ads (via causal_edges on control)
        This simulates: when price drops (promotion), search volume increases.
        Baseline models attribute this to Search-Ads effectiveness, but it's
        actually driven by price. CD-NOTS should detect Price → Search-Ads
        and shrink the Search-Ads prior.
    
    Ghost channels (noise):
        Ghost-A, Ghost-B: base_effectiveness=0.0 (pure noise)
        CD-NOTS + prior should regularize these toward zero without breaking
        convergence of real channels.
    """
    return MMMDataConfig(
        n_periods=156,
        channels=[
            ChannelConfig(
                name="Search-Ads",
                pattern="linear_trend",
                base_spend=3000.0,
                spend_trend=0.004,  # ponytail: ~20% annual growth (was 0.03 — 5.7× in 3yr, unrealistic)
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
                spend_trend=0.006,  # ponytail: ~25% annual growth (was 0.06 — 10.3× in 3yr, unrealistic)
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
            # ============================================================
            # Ghost channels: pure noise, zero real effectiveness
            # Tests if CD-NOTS + prior can correctly suppress these
            # ============================================================
            ChannelConfig(
                name="Ghost-A",
                pattern="seasonal",
                base_spend=2000.0,
                seasonal_amplitude=0.3,
                seasonal_phase=1.5708,  # ponytail: π/2 — A0.4 fix, orthogonal to phase=0 real channels
                spend_volatility=0.25,
                base_effectiveness=0.0,  # NO real effect on sales
            ),
            ChannelConfig(
                name="Ghost-B",
                pattern="on_off",
                base_spend=1500.0,
                activation_probability=0.5,
                spend_volatility=0.20,
                base_effectiveness=0.0,  # NO real effect on sales
            ),
        ],
        causal_edges=[
            # Inter-channel causal relationships
            CausalEdgeConfig(
                source_channel="TV",
                target_channel="Search-Ads",
                effect_size=0.20,
                lag=2,
                decay=0.5,
            ),
            CausalEdgeConfig(
                source_channel="Social-Media",
                target_channel="Brand-Search",
                effect_size=0.15,
                lag=1,
                decay=0.4,
            ),
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
                {"alpha": 0.5},   # Search-Ads
                {"alpha": 0.4},   # Brand-Search
                {"alpha": 0.7},   # TV
                {"alpha": 0.6},   # Video
                {"alpha": 0.5},   # Social-Media
                {"alpha": 0.3},   # Display-Ads
                {"alpha": 0.4},   # Ghost-A
                {"alpha": 0.3},   # Ghost-B
            ],
            saturation_fun="hill_function",
            saturation_kwargs=[
                {"slope": 1.0, "kappa": 0.15},
                {"slope": 1.2, "kappa": 0.12},
                {"slope": 0.8, "kappa": 0.20},
                {"slope": 1.0, "kappa": 0.18},
                {"slope": 1.1, "kappa": 0.14},
                {"slope": 0.9, "kappa": 0.16},
                {"slope": 1.0, "kappa": 0.15},  # Ghost-A
                {"slope": 1.0, "kappa": 0.15},  # Ghost-B
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


def _get_causal_large_preset(seed: int) -> MMMDataConfig:
    """1 geo, 22 media channels (20 real + 2 ghost), 10 controls, 5 inter-channel causal edges.

    Designed for CD-NOTS benchmarking at scale without multi-geo overhead.
    Single national geo keeps Granger tests tractable (23 vars x 20 = 460 tests).

    Causal structure (ground truth):
        TV           -> Paid-Search      (lag=2, effect=0.22, decay=0.55)
        Video        -> Social-Organic   (lag=1, effect=0.18, decay=0.40)
        OOH          -> Brand-Search     (lag=3, effect=0.14, decay=0.60)
        Social-Paid  -> Social-Organic   (lag=1, effect=0.12, decay=0.35)
        Display      -> Paid-Search      (lag=1, effect=0.10, decay=0.30)
        Real channels -> Sales (direct, via adstock + saturation)
        Ghost-A, Ghost-B -> NO effect (base_effectiveness=0.0, pure noise)
    """
    return MMMDataConfig(
        n_periods=208,  # 4 years — enough for Granger power on lagged effects
        channels=[
            # Upper-funnel brand channels
            ChannelConfig(name="TV",             pattern="seasonal",      base_spend=8000.0, seasonal_amplitude=0.35, seasonal_phase=0.4, spend_volatility=0.08, base_effectiveness=0.9),
            ChannelConfig(name="OOH",            pattern="seasonal",      base_spend=3000.0, seasonal_amplitude=0.20, seasonal_phase=0.8, spend_volatility=0.10, base_effectiveness=0.5),
            ChannelConfig(name="Radio",          pattern="seasonal",      base_spend=2000.0, seasonal_amplitude=0.25, seasonal_phase=0.2, spend_volatility=0.12, base_effectiveness=0.4),
            ChannelConfig(name="Podcast",        pattern="linear_trend",  base_spend=1200.0, spend_trend=0.005,                           spend_volatility=0.20, base_effectiveness=0.3),  # ponytail: ~21% annual (was 0.05)
            ChannelConfig(name="Video",          pattern="linear_trend",  base_spend=4000.0, spend_trend=0.006,                           spend_volatility=0.18, base_effectiveness=0.6),  # ponytail: ~25% annual (was 0.07)
            # Mid-funnel consideration channels
            ChannelConfig(name="Social-Paid",    pattern="seasonal",      base_spend=3500.0, seasonal_amplitude=0.20, seasonal_phase=0.6, spend_volatility=0.15, base_effectiveness=0.55),
            ChannelConfig(name="Social-Organic", pattern="seasonal",      base_spend=1800.0, seasonal_amplitude=0.30, seasonal_phase=0.5, spend_volatility=0.22, base_effectiveness=0.45),
            ChannelConfig(name="Display",        pattern="on_off",        base_spend=2500.0, activation_probability=0.70,                 spend_volatility=0.18, base_effectiveness=0.35),
            ChannelConfig(name="Programmatic",   pattern="on_off",        base_spend=2000.0, activation_probability=0.65,                 spend_volatility=0.20, base_effectiveness=0.30),
            ChannelConfig(name="Native-Ads",     pattern="linear_trend",  base_spend=1500.0, spend_trend=0.004,                           spend_volatility=0.16, base_effectiveness=0.28),  # ponytail: ~20% annual (was 0.03)
            # Lower-funnel performance channels
            ChannelConfig(name="Paid-Search",    pattern="linear_trend",  base_spend=5000.0, spend_trend=0.004,                           spend_volatility=0.12, base_effectiveness=0.80),  # ponytail: ~20% annual (was 0.04)
            ChannelConfig(name="Brand-Search",   pattern="seasonal",      base_spend=2500.0, seasonal_amplitude=0.18, seasonal_phase=0.3, spend_volatility=0.10, base_effectiveness=0.70),
            ChannelConfig(name="Shopping-Ads",   pattern="seasonal",      base_spend=3000.0, seasonal_amplitude=0.28, seasonal_phase=0.1, spend_volatility=0.14, base_effectiveness=0.65),
            ChannelConfig(name="Retargeting",    pattern="on_off",        base_spend=1800.0, activation_probability=0.75,                 spend_volatility=0.16, base_effectiveness=0.60),
            ChannelConfig(name="Affiliate",      pattern="linear_trend",  base_spend=1200.0, spend_trend=0.003,                           spend_volatility=0.14, base_effectiveness=0.50),  # ponytail: ~15% annual (was 0.02)
            # Emerging / digital channels
            ChannelConfig(name="CTV",            pattern="linear_trend",  base_spend=2000.0, spend_trend=0.008,                           spend_volatility=0.22, base_effectiveness=0.40),  # ponytail: ~30% annual (was 0.09)
            ChannelConfig(name="Streaming-Audio",pattern="linear_trend",  base_spend=800.0,  spend_trend=0.006,                           spend_volatility=0.25, base_effectiveness=0.25),  # ponytail: ~25% annual (was 0.06)
            ChannelConfig(name="Influencer",     pattern="on_off",        base_spend=1500.0, activation_probability=0.50,                 spend_volatility=0.30, base_effectiveness=0.35),
            ChannelConfig(name="Email",          pattern="seasonal",      base_spend=600.0,  seasonal_amplitude=0.15, seasonal_phase=0.0, spend_volatility=0.08, base_effectiveness=0.55),
            ChannelConfig(name="Push-Notif",     pattern="on_off",        base_spend=400.0,  activation_probability=0.60,                 spend_volatility=0.12, base_effectiveness=0.40),
            # Ghost channels: pure noise, zero real effectiveness
            ChannelConfig(name="Ghost-A",        pattern="seasonal",      base_spend=2000.0, seasonal_amplitude=0.30, seasonal_phase=1.5708, spend_volatility=0.25, base_effectiveness=0.0),  # A0.4: π/2
            ChannelConfig(name="Ghost-B",        pattern="on_off",        base_spend=1500.0, activation_probability=0.50,                 spend_volatility=0.20, base_effectiveness=0.0),
        ],
        causal_edges=[
            CausalEdgeConfig(source_channel="TV",          target_channel="Paid-Search",    effect_size=0.22, lag=2, decay=0.55),
            CausalEdgeConfig(source_channel="Video",       target_channel="Social-Organic", effect_size=0.18, lag=1, decay=0.40),
            CausalEdgeConfig(source_channel="OOH",         target_channel="Brand-Search",   effect_size=0.14, lag=3, decay=0.60),
            CausalEdgeConfig(source_channel="Social-Paid", target_channel="Social-Organic", effect_size=0.12, lag=1, decay=0.35),
            CausalEdgeConfig(source_channel="Display",     target_channel="Paid-Search",    effect_size=0.10, lag=1, decay=0.30),
        ],
        regions=RegionConfig(
            n_regions=1,
            region_names=["national"],
            base_sales_rate=50000.0,
            sales_trend=0.015,
            sales_volatility=0.02,
            seasonal_amplitude=0.12,
            baseline_variation=0.0,
            channel_param_variation=0.0,
            transform_variation=0.0,
        ),
        transforms=TransformConfig(
            adstock_fun="geometric_adstock",
            adstock_kwargs=[
                {"alpha": 0.70, "l_max": 8}, {"alpha": 0.55, "l_max": 8}, {"alpha": 0.50, "l_max": 8}, {"alpha": 0.40, "l_max": 8}, {"alpha": 0.65, "l_max": 8},
                {"alpha": 0.55, "l_max": 8}, {"alpha": 0.45, "l_max": 8}, {"alpha": 0.35, "l_max": 8}, {"alpha": 0.30, "l_max": 8}, {"alpha": 0.38, "l_max": 8},
                {"alpha": 0.60, "l_max": 8}, {"alpha": 0.55, "l_max": 8}, {"alpha": 0.58, "l_max": 8}, {"alpha": 0.50, "l_max": 8}, {"alpha": 0.45, "l_max": 8},
                {"alpha": 0.62, "l_max": 8}, {"alpha": 0.42, "l_max": 8}, {"alpha": 0.38, "l_max": 8}, {"alpha": 0.35, "l_max": 8}, {"alpha": 0.30, "l_max": 8},
                {"alpha": 0.40, "l_max": 8}, {"alpha": 0.30, "l_max": 8},  # Ghost-A, Ghost-B
            ],
            saturation_fun="hill_function",
            saturation_kwargs=[
                {"slope": 0.9, "kappa": 0.20}, {"slope": 0.7, "kappa": 0.25}, {"slope": 0.7, "kappa": 0.28}, {"slope": 0.6, "kappa": 0.30}, {"slope": 1.0, "kappa": 0.18},
                {"slope": 0.9, "kappa": 0.20}, {"slope": 0.8, "kappa": 0.22}, {"slope": 0.7, "kappa": 0.26}, {"slope": 0.6, "kappa": 0.28}, {"slope": 0.6, "kappa": 0.30},
                {"slope": 1.2, "kappa": 0.14}, {"slope": 1.1, "kappa": 0.15}, {"slope": 1.0, "kappa": 0.16}, {"slope": 1.0, "kappa": 0.18}, {"slope": 0.9, "kappa": 0.20},
                {"slope": 0.8, "kappa": 0.22}, {"slope": 0.6, "kappa": 0.30}, {"slope": 0.7, "kappa": 0.26}, {"slope": 0.9, "kappa": 0.18}, {"slope": 0.7, "kappa": 0.24},
                {"slope": 1.0, "kappa": 0.15}, {"slope": 1.0, "kappa": 0.15},  # Ghost-A, Ghost-B
            ],
        ),
        control_variables=[
            ControlConfig(name="price-index",         pattern="linear_trend", base_value=100.0, value_trend=0.02,  value_volatility=0.05, base_effectiveness=-0.40),
            ControlConfig(name="promotions",           pattern="on_off",       base_value=1.0,   activation_probability=0.15, value_volatility=0.80, base_effectiveness=0.25),
            ControlConfig(name="gdp-index",            pattern="linear_trend", base_value=100.0, value_trend=0.01,  value_volatility=0.02, base_effectiveness=0.30),
            ControlConfig(name="competitor-spend",     pattern="seasonal",     base_value=5000.0, seasonal_amplitude=0.20, value_volatility=0.15, base_effectiveness=-0.20),
            ControlConfig(name="consumer-confidence",  pattern="seasonal",     base_value=80.0,  seasonal_amplitude=0.10, value_volatility=0.04, base_effectiveness=0.15),
            ControlConfig(name="seasonality",          pattern="seasonal",     base_value=1.0,   seasonal_amplitude=0.30, value_volatility=0.02, base_effectiveness=0.20),
            ControlConfig(name="inventory-flag",       pattern="on_off",       base_value=1.0,   activation_probability=0.10, value_volatility=0.50, base_effectiveness=-0.15),
            ControlConfig(name="events",               pattern="on_off",       base_value=1.0,   activation_probability=0.08, value_volatility=1.00, base_effectiveness=0.35),
            ControlConfig(name="temp-index",           pattern="seasonal",     base_value=15.0,  seasonal_amplitude=0.60, value_volatility=0.05, base_effectiveness=0.10),
            ControlConfig(name="holiday-flag",         pattern="on_off",       base_value=1.0,   activation_probability=0.06, value_volatility=0.50, base_effectiveness=0.45),
        ],
        seed=seed,
    )