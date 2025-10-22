"""
Data preprocessing utilities for CausalMMM.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict
from sklearn.preprocessing import StandardScaler


class PanelDataLoader:
    """
    Loader for panel/longitudinal marketing data.
    
    Handles:
    - Entity-level time series
    - Missing data
    - Temporal alignment
    
    Example:
        >>> loader = PanelDataLoader()
        >>> X, y, context, entities, time_idx = loader.load_from_dataframe(
        ...     df,
        ...     entity_col='store_id',
        ...     time_col='date',
        ...     channel_cols=['tv', 'radio', 'digital'],
        ...     target_col='sales',
        ...     context_cols=['is_holiday', 'temperature']
        ... )
    """
    
    def __init__(self):
        self.entity_ids = None
        self.channel_names = None
        self.target_name = None
    
    def load_from_dataframe(
        self,
        df: pd.DataFrame,
        entity_col: str = 'entity',
        time_col: str = 'time',
        channel_cols: Optional[List[str]] = None,
        target_col: str = 'target',
        context_cols: Optional[List[str]] = None,
        channel_prefix: Optional[str] = None,
        min_length: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], List[str], np.ndarray]:
        """
        Load panel data from DataFrame.
        
        Args:
            df: Input DataFrame
            entity_col: Column name for entity ID
            time_col: Column name for time
            channel_cols: List of channel columns (or use prefix)
            target_col: Target variable column
            context_cols: Context feature columns
            channel_prefix: Prefix to auto-detect channels (e.g., 'channel_')
            min_length: Minimum time series length
            
        Returns:
            X: [B, T, d] - Channel data
            y: [B, T, 1] - Target data
            context: [B, c] - Context features (or None)
            entities: List of entity IDs
            time_idx: [B, T] - Time indices
        """
        # Detect channel columns
        if channel_cols is None and channel_prefix is not None:
            channel_cols = [c for c in df.columns if c.startswith(channel_prefix)]
        elif channel_cols is None:
            raise ValueError("Must provide channel_cols or channel_prefix")
        
        self.channel_names = channel_cols
        self.target_name = target_col
        
        # Sort by entity and time
        df = df.sort_values([entity_col, time_col]).copy()
        
        # Get entities
        entities = df[entity_col].unique().tolist()
        self.entity_ids = entities
        
        # Determine minimum length
        entity_lengths = df.groupby(entity_col).size()
        if min_length is None:
            min_length = entity_lengths.min()
        else:
            entities = [e for e in entities if entity_lengths[e] >= min_length]
        
        # Extract data per entity
        X_list, y_list, context_list, time_list = [], [], [], []
        
        for entity in entities:
            entity_df = df[df[entity_col] == entity].tail(min_length)
            
            # Channels
            X_entity = entity_df[channel_cols].values.astype(np.float32)
            X_list.append(X_entity)
            
            # Target
            y_entity = entity_df[[target_col]].values.astype(np.float32)
            y_list.append(y_entity)
            
            # Time indices
            if pd.api.types.is_numeric_dtype(entity_df[time_col]):
                time_entity = entity_df[time_col].values.astype(np.int32)
            else:
                # Convert dates to integer indices
                time_entity = np.arange(len(entity_df), dtype=np.int32)
            time_list.append(time_entity)
            
            # Context (static features per entity)
            if context_cols:
                context_entity = entity_df[context_cols].iloc[0].values.astype(np.float32)
                context_list.append(context_entity)
        
        # Stack into arrays
        X = np.stack(X_list, axis=0)  # [B, T, d]
        y = np.stack(y_list, axis=0)  # [B, T, 1]
        time_idx = np.stack(time_list, axis=0)  # [B, T]
        context = np.stack(context_list, axis=0) if context_cols else None  # [B, c]
        
        return X, y, context, entities, time_idx


class GroupStandardizer:
    """
    Standardizes data per entity (group).
    
    Each entity gets its own mean/std to account for scale differences.
    
    Example:
        >>> scaler = GroupStandardizer()
        >>> scaler.fit(X_train, y_train, entity_ids)
        >>> X_scaled, y_scaled = scaler.transform(X_train, y_train, entity_ids)
    """
    
    def __init__(self):
        self.means: Dict[str, np.ndarray] = {}
        self.stds: Dict[str, np.ndarray] = {}
        self.entity_ids: List[str] = []
    
    def fit(self, X: np.ndarray, y: np.ndarray, entity_ids: List[str]):
        """
        Fit scaler on training data.
        
        Args:
            X: [B, T, d]
            y: [B, T, 1]
            entity_ids: List of B entity IDs
        """
        B, T, d = X.shape
        self.entity_ids = entity_ids
        
        for b, eid in enumerate(entity_ids):
            # Concatenate all variables
            xy = np.concatenate([X[b], y[b]], axis=-1)  # [T, d+1]
            
            # Compute statistics
            mean = xy.mean(axis=0)
            std = xy.std(axis=0) + 1e-8
            
            self.means[eid] = mean
            self.stds[eid] = std
    
    def transform(self, X: np.ndarray, y: np.ndarray, entity_ids: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform data using fitted statistics.
        
        Args:
            X: [B, T, d]
            y: [B, T, 1]
            entity_ids: List of entity IDs
            
        Returns:
            X_scaled: [B, T, d]
            y_scaled: [B, T, 1]
        """
        B, T, d = X.shape
        X_scaled = np.empty_like(X)
        y_scaled = np.empty_like(y)
        
        for b, eid in enumerate(entity_ids):
            mean = self.means[eid]
            std = self.stds[eid]
            
            xy = np.concatenate([X[b], y[b]], axis=-1)
            xy_scaled = (xy - mean) / std
            
            X_scaled[b] = xy_scaled[:, :d]
            y_scaled[b] = xy_scaled[:, d:]
        
        return X_scaled, y_scaled
    
    def inverse_transform_target(self, y_scaled: np.ndarray, entity_id: str) -> np.ndarray:
        """
        Inverse transform target variable.
        
        Args:
            y_scaled: Scaled target values
            entity_id: Entity ID
            
        Returns:
            y_original: Original scale
        """
        mean = self.means[entity_id][-1]
        std = self.stds[entity_id][-1]
        return y_scaled * std + mean


class AdstockTransformer:
    """
    Apply adstock (carryover) transformation to channels.
    
    Implements geometric adstock: x_t' = x_t + α*x_{t-1} + α²*x_{t-2} + ...
    
    Example:
        >>> transformer = AdstockTransformer(decay_rate=0.7, max_lag=8)
        >>> X_adstock = transformer.fit_transform(X)
    """
    
    def __init__(self, decay_rate: float = 0.5, max_lag: int = 8):
        """
        Args:
            decay_rate: Decay rate α ∈ (0, 1)
            max_lag: Maximum lag to consider
        """
        self.decay_rate = decay_rate
        self.max_lag = max_lag
        
        # Compute weights
        self.weights = decay_rate ** np.arange(max_lag)
        self.weights = self.weights / self.weights.sum()  # Normalize
    
    def fit(self, X: np.ndarray):
        """Fit (no-op for adstock)."""
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply adstock transformation.
        
        Args:
            X: [B, T, d] - Channel data
            
        Returns:
            X_adstock: [B, T, d] - Transformed data
        """
        from scipy.ndimage import convolve1d
        
        B, T, d = X.shape
        X_adstock = np.empty_like(X)
        
        for b in range(B):
            for c in range(d):
                # Convolve with decay weights
                X_adstock[b, :, c] = convolve1d(
                    X[b, :, c],
                    self.weights,
                    mode='constant',
                    cval=0.0,
                    origin=-self.max_lag // 2
                )
        
        return X_adstock
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform."""
        return self.fit(X).transform(X)


def temporal_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    time_idx: np.ndarray,
    context: Optional[np.ndarray] = None,
    train_frac: float = 0.7,
    val_frac: float = 0.15
) -> Tuple:
    """
    Split panel data temporally (preserves time order).
    
    Args:
        X, y, time_idx, context: Data arrays
        train_frac: Fraction for training
        val_frac: Fraction for validation
        
    Returns:
        (X_train, y_train, time_train, context_train),
        (X_val, y_val, time_val, context_val),
        (X_test, y_test, time_test, context_test)
    """
    B, T, d = X.shape
    
    t_train = int(T * train_frac)
    t_val = int(T * (train_frac + val_frac))
    
    def split(arr):
        if arr is None:
            return None, None, None
        return arr[:, :t_train], arr[:, t_train:t_val], arr[:, t_val:]
    
    X_train, X_val, X_test = split(X)
    y_train, y_val, y_test = split(y)
    time_train, time_val, time_test = split(time_idx)
    
    # Context is static per entity
    if context is not None and context.ndim == 2:
        context_train = context_val = context_test = context
    else:
        context_train = context_val = context_test = None
    
    return (
        (X_train, y_train, time_train, context_train),
        (X_val, y_val, time_val, context_val),
        (X_test, y_test, time_test, context_test)
    )
