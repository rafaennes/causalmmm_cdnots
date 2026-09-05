# Copyright 2025 PyMC Labs
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
Core data generation functions for MMM Dataset Generator.

This module contains the main data generation function and core logic for
creating synthetic MMM datasets with known ground truth parameters.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings

from .config import MMMDataConfig, ChannelConfig, RegionConfig, TransformConfig, ControlConfig
from .channels import generate_channel_spend, generate_control_variable
from .regions import generate_regional_baseline, generate_regional_channel_variations, generate_regional_transform_variations
from .transforms import apply_transformations
from .ground_truth import calculate_roas_values, calculate_attribution_percentages
from .validation import validate_config, validate_output_schema


def generate_mmm_dataset(
    config: Optional[MMMDataConfig] = None,
    **kwargs
) -> Dict[str, Union[pd.DataFrame, Dict]]:
    """
    Generate a synthetic MMM dataset with known ground truth parameters.
    
    This is the main entry point for the MMM Dataset Generator. It creates
    a comprehensive dataset including channel spend data, regional sales data,
    transformed variables, and ground truth parameters for validation.
    
    Parameters
    ----------
    config : MMMDataConfig, optional
        Configuration object specifying all data generation parameters.
        If None, uses DEFAULT_CONFIG.
    **kwargs
        Additional parameters to override in the configuration.
        
    Returns
    -------
    Dict[str, Union[pd.DataFrame, Dict]]
        Dictionary containing:
        - 'data': Main dataset with spend, sales, and transformed variables
        - 'ground_truth': Dictionary with true parameters and contributions
        - 'config': The configuration used for generation
        
    Raises
    ------
    ValueError
        If configuration parameters are invalid or inconsistent.
    RuntimeError
        If data generation fails due to numerical issues.
        
    Examples
    --------
    >>> from mmm_data_generator import generate_mmm_dataset, DEFAULT_CONFIG
    >>> 
    >>> # Use default configuration
    >>> result = generate_mmm_dataset()
    >>> data = result['data']
    >>> ground_truth = result['ground_truth']
    >>> 
    >>> # Custom configuration
    >>> config = MMMDataConfig(
    ...     n_periods=52,
    ...     channels=[ChannelConfig(name="x_seasonal", base_spend=2000.0)],
    ...     seed=123
    ... )
    >>> result = generate_mmm_dataset(config)
    """
    
    # Set up configuration
    if config is None:
        from .config import DEFAULT_CONFIG
        config = DEFAULT_CONFIG
    
    # Override config with kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            warnings.warn(f"Unknown parameter '{key}' ignored")
    
    # Validate configuration
    try:
        validate_config(config)
    except ValueError as e:
        raise ValueError(f"Configuration validation failed: {e}")
    
    # Set random seed for reproducibility
    if config.seed is not None:
        np.random.seed(config.seed)
    
    try:
        # Generate time index
        time_index = _generate_time_index(config)
        
        # Generate baseline data for each region first
        baseline_data = _generate_baseline_data(config, time_index)
        
        # Generate channel spend data (without geo in column names)
        spend_data = _generate_channel_spend_data(config, time_index)
        
        # Generate control variables
        control_data = _generate_control_variables(config, time_index)
        
        # Apply transformations to spend data and track parameters
        transformed_data, transformation_params = _apply_transformations_to_data(
            config, spend_data, time_index
        )
        
        # Combine all data
        combined_data = pd.concat([spend_data, control_data], axis=1)

        # select control effect

        control_effects = control_data[[col for col in control_data.columns if col.endswith('_effect')]]
        
        # Calculate total sales using baseline_sales column
        # ponytail: clip to 0 — sales are non-negative; v2 controls (price<0) can push sum negative
        combined_data["y"] = np.clip(
            baseline_data["baseline_sales"] + control_effects.sum(axis=1) + transformed_data.sum(axis=1),
            0, None,
        )


        combined_data = combined_data.reset_index()

        # Calculate ground truth metrics
        roas_values = calculate_roas_values(spend_data, transformed_data, config)
        attribution_percentages = calculate_attribution_percentages(transformed_data, config)
        
        # Create ground truth summary
        ground_truth = {
            'transformation_parameters': transformation_params,
            'baseline_components': baseline_data,
            'roas_values': roas_values,
            'attribution_percentages': attribution_percentages,
            'transformed_spend': transformed_data,
            'causal_graph': _build_causal_ground_truth(config),
        }
        
        # Validate output schema
        schema_errors = validate_output_schema(combined_data, config)
        if schema_errors:
            error_message = "Output schema validation failed:\n" + "\n".join(f"- {error}" for error in schema_errors)
            raise ValueError(error_message)
        
        # Prepare output
        result = {
            'data': combined_data,
            'ground_truth': ground_truth,
            'config': config
        }
            
        return result
        
    except Exception as e:
        raise RuntimeError(f"Data generation failed: {e}")


def _generate_time_index(config: MMMDataConfig) -> pd.DatetimeIndex:
    """Generate time index for the dataset."""
    if config.start_date:
        start_date = pd.to_datetime(config.start_date)
    else:
        start_date = pd.Timestamp('2020-01-01')
    
    # Generate weekly periods
    time_index = pd.date_range(
        start=start_date,
        periods=config.n_periods,
        freq='W'
    )
    
    return time_index


def _generate_baseline_data(
    config: MMMDataConfig, 
    time_index: pd.DatetimeIndex
) -> pd.DataFrame:
    """Generate baseline sales data for all regions with separate components."""
    baseline_data = []
    
    for region_idx, region_name in enumerate(config.regions.region_names): # type: ignore
        baseline_components = generate_regional_baseline(
            config.regions, time_index, region_idx, config.seed
        )
        
        # Create DataFrame with separate columns for each component
        region_data = pd.DataFrame(
            {
                'base_sales': baseline_components['base_sales'],
                'trend': baseline_components['trend'],
                'seasonal': baseline_components['seasonal'],
                'baseline_sales': baseline_components['total']
            },
            index=pd.MultiIndex.from_product([time_index, [region_name]], names=['date', 'geo'])
        )
        baseline_data.append(region_data)
    
    return pd.concat(baseline_data, axis=0)


def _generate_channel_spend_data(
    config: MMMDataConfig, 
    time_index: pd.DatetimeIndex
) -> pd.DataFrame:
    """Generate spend data for all channels across all regions.
    
    If config.causal_edges is defined, applies inter-channel causal
    spillover: source channel spend at time t causes additional spend
    in target channel at time t+lag, with geometric decay.
    """
    regional_dataframes = []
    
    for region_idx, region_name in enumerate(config.regions.region_names): # type: ignore
        regional_channels = generate_regional_channel_variations(
            config.regions, config.channels, region_idx, config.seed
        )
        
        region_data = pd.DataFrame(index=pd.MultiIndex.from_product([time_index, [region_name]], names=['date', 'geo']))
        
        # Phase 1: Generate independent (base) spend for each channel
        channel_spends = {}
        col_names_ordered = []
        for i, channel in enumerate(regional_channels):
            region_seed = config.seed + region_idx * 1000 + i * 100 if config.seed is not None else None
            base_spend = generate_channel_spend(
                channel, time_index, region_seed,
                spec_version=getattr(config, 'spec_version', 'v1_legacy'),
            )
            column_name = f'x{i+1}_{channel.name}' if channel.name != "" else f'x{i+1}'
            channel_spends[column_name] = base_spend.copy()
            col_names_ordered.append(column_name)
        
        # Phase 2: Apply causal spillover effects (if any)
        if hasattr(config, 'causal_edges') and config.causal_edges:
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
                
                # Geometric adstock of source spend
                adstocked = np.zeros(n_periods)
                for t in range(n_periods):
                    adstocked[t] = source_spend[t] + (edge.decay * adstocked[t-1] if t > 0 else 0)
                
                # Shift by lag and scale by effect_size
                spillover = np.zeros(n_periods)
                for t in range(edge.lag, n_periods):
                    spillover[t] = edge.effect_size * adstocked[t - edge.lag]
                
                channel_spends[target_col] = channel_spends[target_col] + spillover
        
        # Store in DataFrame
        for col_name in col_names_ordered:
            region_data[col_name] = channel_spends[col_name]
        
        regional_dataframes.append(region_data)
    
    return pd.concat(regional_dataframes, axis=0)


def _generate_control_variables(
    config: MMMDataConfig, 
    time_index: pd.DatetimeIndex
) -> pd.DataFrame:
    """Generate control variables data with regional variations.
    
    Returns a DataFrame with multi-index (time_index, geo region) containing
    control variable effects for each region.
    """
    # Create a list to store regional DataFrames
    regional_dataframes = []
    
    for region_idx, region_name in enumerate(config.regions.region_names): # type: ignore
        regional_control_variables = generate_regional_channel_variations(
            config.regions, config.control_variables, region_idx, config.seed # type: ignore
        )
        region_data = pd.DataFrame(index=pd.MultiIndex.from_product([time_index, [region_name]], names=['date', 'geo']))
        
        for idx, _control in enumerate(regional_control_variables):
            # Generate independent control effect for this region
            # Use region-specific seed for independence
            region_seed = config.seed + region_idx * 1000 + idx * 100 if config.seed is not None else None
            
            # Convert ChannelConfig to ControlConfig using the new from_channel_config method
            control = ControlConfig.from_channel_config(_control)
            control_variable = generate_control_variable(
                control, time_index, region_seed,
                spec_version=getattr(config, 'spec_version', 'v1_legacy'),
            )
            region_data[f"c{idx+1}"] = control_variable
            region_data[f"c{idx+1}_effect"] = control_variable * control.base_effectiveness
        
        regional_dataframes.append(region_data)
    
    # Concatenate all regional DataFrames on axis=0
    control_data = pd.concat(regional_dataframes, axis=0)
    
    return control_data

def _build_causal_ground_truth(config: MMMDataConfig) -> Dict[str, Any]:
    """Build ground truth causal adjacency matrix from config.
    
    Returns dict with adjacency_matrix [n_channels+1, n_channels+1],
    variable_names, and edges list. Last variable is always 'y'.
    """
    n_channels = len(config.channels)
    n_vars = n_channels + 1
    var_names = []
    for i, ch in enumerate(config.channels):
        col_name = f'x{i+1}_{ch.name}' if ch.name != "" else f'x{i+1}'
        var_names.append(col_name)
    var_names.append('y')
    
    adj = np.zeros((n_vars, n_vars))
    y_idx = n_vars - 1
    
    # All channels with non-zero effectiveness cause y
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


def _apply_transformations_to_data(
    config: MMMDataConfig,
    spend_data: pd.DataFrame,
    time_index: pd.DatetimeIndex,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Apply adstock and saturation transformations to spend data and track parameters."""
    # Create a list to store regional DataFrames
    regional_dataframes = []
    transformation_params = {
        'channels': {},
    }
    
    for region_idx, region_name in enumerate(config.regions.region_names): # type: ignore
        # Get regional transform configuration
        regional_transform = generate_regional_transform_variations(
            config.regions,
            config.transforms,
            region_idx,
            config.seed
        )
        
        # Get spend data for this region
        region_mask = spend_data.index.get_level_values('geo') == region_name
        region_spend_data = spend_data[region_mask].copy()
        
        # Create DataFrame for this region with same structure
        region_data = pd.DataFrame(index=pd.MultiIndex.from_product([time_index, [region_name]], names=['date', 'geo']))
        
        
        # Apply transformations to each channel
        for i, channel in enumerate(config.channels):
            # Find the corresponding column in the spend data
            # Use consistent x{i+1} format for channel columns
            column_name = f'x{i+1}_{channel.name}' if channel.name != "" else f'x{i+1}'
        
            if column_name in region_spend_data.columns:
                # Apply transformations with channel index and regional transform config
                max_spend = region_spend_data[column_name].max()
                assert max_spend != 0, f"Max spend zero for {region_name}, {channel.name}"
                transformed_spend, channel_params = apply_transformations(
                    region_spend_data[column_name].values / max_spend,
                    regional_transform,
                    channel_idx=i
                )
                
                # Store in the region DataFrame with consistent column naming
                region_data["contribution_" + column_name] = channel.base_effectiveness * transformed_spend * config.regions.base_sales_rate
                
                # Track channel parameters
                if channel.name not in transformation_params['channels']:
                    transformation_params['channels'][channel.name] = {}
                transformation_params['channels'][channel.name][region_name] = channel_params
            else:
                # If column not found, use zeros
                region_data[f"contribution_{column_name}"] = 0.0
        
        regional_dataframes.append(region_data)
    
    # Concatenate all regional DataFrames on axis=0
    transformed_data = pd.concat(regional_dataframes, axis=0)
    
    return transformed_data, transformation_params
