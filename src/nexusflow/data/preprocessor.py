"""Enhanced tabular preprocessing pipeline for NexusFlow Phase 2."""
import pandas as pd
import torch
import torch.nn as nn
import pickle
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

class TabularPreprocessor:
    """
    Unified tabular preprocessing and tokenization pipeline.
    
    This class handles intelligent preprocessing of mixed categorical and numerical
    features, converting them into a unified embedding space suitable for 
    advanced transformers like FT-Transformer and TabNet.
    """
    
    def __init__(self):
        self.categorical_encoders = {}  # {col_name: LabelEncoder}
        self.numerical_scalers = {}     # {col_name: StandardScaler}
        self.categorical_columns = []
        self.numerical_columns = []
        self.vocab_sizes = {}           # {col_name: vocab_size}
        self.feature_stats = {}         # Column statistics for analysis
        self.is_fitted = False
        
    def fit(self, df: pd.DataFrame, categorical_cols: Optional[List[str]] = None, 
            numerical_cols: Optional[List[str]] = None) -> 'TabularPreprocessor':
        """
        Fit the preprocessor on training data.
        
        Args:
            df: Training DataFrame
            categorical_cols: List of categorical column names (auto-detect if None)
            numerical_cols: List of numerical column names (auto-detect if None)
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting TabularPreprocessor on data shape: {df.shape}")
        
        # Auto-detect column types if not provided
        if categorical_cols is None or numerical_cols is None:
            auto_cat, auto_num = self._auto_detect_types(df)
            categorical_cols = categorical_cols or auto_cat
            numerical_cols = numerical_cols or auto_num
        
        self.categorical_columns = categorical_cols
        self.numerical_columns = numerical_cols
        
        logger.info(f"Identified columns - categorical: {len(categorical_cols)}, "
                   f"numerical: {len(numerical_cols)}")
        
        # Fit categorical encoders
        for col in categorical_cols:
            if col in df.columns:
                # Handle missing values by adding special token
                col_data = df[col].fillna('<UNK>').astype(str)
                
                encoder = LabelEncoder()
                encoder.fit(col_data)
                
                self.categorical_encoders[col] = encoder
                self.vocab_sizes[col] = len(encoder.classes_)
                
                # Store statistics
                self.feature_stats[col] = {
                    'type': 'categorical',
                    'unique_values': len(encoder.classes_),
                    'missing_count': df[col].isnull().sum(),
                    'missing_pct': df[col].isnull().sum() / len(df) * 100
                }
                
                logger.debug(f"Categorical column {col}: {self.vocab_sizes[col]} unique values")
        
        # Fit numerical scalers
        for col in numerical_cols:
            if col in df.columns:
                # Handle missing values for fitting
                col_data = df[col].fillna(df[col].median())
                
                scaler = StandardScaler()
                scaler.fit(col_data.values.reshape(-1, 1))
                
                self.numerical_scalers[col] = scaler
                
                # Store statistics
                self.feature_stats[col] = {
                    'type': 'numerical',
                    'mean': float(col_data.mean()),
                    'std': float(col_data.std()),
                    'min': float(col_data.min()),
                    'max': float(col_data.max()),
                    'missing_count': df[col].isnull().sum(),
                    'missing_pct': df[col].isnull().sum() / len(df) * 100
                }
                
                logger.debug(f"Numerical column {col}: mean={self.feature_stats[col]['mean']:.3f}, "
                           f"std={self.feature_stats[col]['std']:.3f}")
        
        self.is_fitted = True
        logger.info("✅ TabularPreprocessor fitting complete")
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform DataFrame using fitted preprocessors.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            Transformed DataFrame with encoded categorical and scaled numerical features
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        logger.debug(f"Transforming DataFrame with shape: {df.shape}")
        
        transformed_df = df.copy()
        
        # Transform categorical columns
        for col in self.categorical_columns:
            if col in transformed_df.columns:
                # Handle missing values and unseen categories
                col_data = transformed_df[col].fillna('<UNK>').astype(str)
                
                encoder = self.categorical_encoders[col]
                
                # Handle unseen categories
                known_classes = set(encoder.classes_)
                col_data = col_data.apply(lambda x: x if x in known_classes else '<UNK>')
                
                # Transform to indices
                transformed_df[col] = encoder.transform(col_data)
                
        # Transform numerical columns
        for col in self.numerical_columns:
            if col in transformed_df.columns:
                # Handle missing values with median from training
                original_col = df[col] if col in df.columns else transformed_df[col]
                median_val = self.feature_stats[col]['mean']  # Use mean as fallback
                col_data = transformed_df[col].fillna(median_val)
                
                scaler = self.numerical_scalers[col]
                transformed_df[col] = scaler.transform(col_data.values.reshape(-1, 1)).flatten()
        
        logger.debug(f"Transform complete: {transformed_df.shape}")
        return transformed_df
    
    def fit_transform(self, df: pd.DataFrame, categorical_cols: Optional[List[str]] = None,
                     numerical_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df, categorical_cols, numerical_cols).transform(df)
    
    def _auto_detect_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Automatically detect categorical and numerical columns.
        
        Returns:
            Tuple of (categorical_columns, numerical_columns)
        """
        categorical_cols = []
        numerical_cols = []
        
        for col in df.columns:
            if df[col].dtype in ['object', 'category', 'bool']:
                categorical_cols.append(col)
            elif pd.api.types.is_numeric_dtype(df[col]):
                # Check if it's actually categorical (small number of unique values)
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < 0.05 and df[col].nunique() < 20:
                    categorical_cols.append(col)
                else:
                    numerical_cols.append(col)
        
        logger.info(f"Auto-detected types: {len(categorical_cols)} categorical, "
                   f"{len(numerical_cols)} numerical")
        
        return categorical_cols, numerical_cols
    
    def get_feature_info(self) -> Dict[str, Any]:
        """Get comprehensive feature information."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted first")
        
        return {
            'categorical_columns': self.categorical_columns,
            'numerical_columns': self.numerical_columns,
            'vocab_sizes': self.vocab_sizes,
            'feature_stats': self.feature_stats,
            'total_features': len(self.categorical_columns) + len(self.numerical_columns)
        }
    
    def save(self, path: str):
        """Save fitted preprocessor to file."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted preprocessor")
        
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)
        
        logger.info(f"Preprocessor saved to: {save_path}")
    
    @classmethod
    def load(cls, path: str) -> 'TabularPreprocessor':
        """Load preprocessor from file."""
        with open(path, 'rb') as f:
            preprocessor = pickle.load(f)
        
        logger.info(f"Preprocessor loaded from: {path}")
        return preprocessor


class FeatureTokenizer(nn.Module):
    """
    Neural feature tokenizer that converts preprocessed tabular data into embeddings.
    
    This module handles both categorical and numerical features, creating appropriate
    embeddings that can be processed by transformer architectures.
    """
    
    def __init__(self, feature_info: Dict[str, Any], embed_dim: int = 64):
        """
        Initialize feature tokenizer.
        
        Args:
            feature_info: Feature information from TabularPreprocessor
            embed_dim: Embedding dimension for all features
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.categorical_columns = feature_info['categorical_columns']
        self.numerical_columns = feature_info['numerical_columns']
        self.vocab_sizes = feature_info['vocab_sizes']
        
        # Create embedding layers for categorical features
        self.categorical_embeddings = nn.ModuleDict()
        for col in self.categorical_columns:
            vocab_size = self.vocab_sizes[col]
            # Add 1 for potential unseen categories
            self.categorical_embeddings[col] = nn.Embedding(vocab_size + 1, embed_dim)
        
        # Create linear projection for numerical features
        if self.numerical_columns:
            self.numerical_projection = nn.Linear(len(self.numerical_columns), embed_dim)
        
        # Feature normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        logger.info(f"FeatureTokenizer initialized: {len(self.categorical_columns)} categorical, "
                   f"{len(self.numerical_columns)} numerical features -> {embed_dim}D embeddings")
    
    def forward(self, x: torch.Tensor, column_info: Dict[str, List[str]]) -> torch.Tensor:
        """
        Forward pass: convert features to embeddings.
        
        Args:
            x: Input tensor [batch_size, num_features]
            column_info: Dict with 'categorical' and 'numerical' column names in order
            
        Returns:
            Feature embeddings [batch_size, total_features, embed_dim]
        """
        batch_size = x.size(0)
        feature_embeddings = []
        
        current_idx = 0
        
        # Process categorical features
        for col in column_info.get('categorical', []):
            if col in self.categorical_embeddings:
                # Extract categorical feature (should be integer indices)
                cat_feature = x[:, current_idx].long()
                
                # Clamp to valid range (handle unseen categories)
                vocab_size = self.vocab_sizes[col]
                cat_feature = torch.clamp(cat_feature, 0, vocab_size)
                
                # Get embedding
                cat_embedding = self.categorical_embeddings[col](cat_feature)  # [batch, embed_dim]
                feature_embeddings.append(cat_embedding.unsqueeze(1))  # [batch, 1, embed_dim]
                
            current_idx += 1
        
        # Process numerical features
        numerical_features = []
        for col in column_info.get('numerical', []):
            numerical_features.append(x[:, current_idx].unsqueeze(1))  # [batch, 1]
            current_idx += 1
        
        if numerical_features:
            # Combine all numerical features
            num_tensor = torch.cat(numerical_features, dim=1)  # [batch, num_numerical]
            
            # Project to embedding space
            num_embedding = self.numerical_projection(num_tensor)  # [batch, embed_dim]
            feature_embeddings.append(num_embedding.unsqueeze(1))  # [batch, 1, embed_dim]
        
        # Combine all feature embeddings
        if feature_embeddings:
            combined_embeddings = torch.cat(feature_embeddings, dim=1)  # [batch, total_features, embed_dim]
            
            # Apply normalization
            combined_embeddings = self.layer_norm(combined_embeddings)
            
            return combined_embeddings
        else:
            # No features case
            return torch.zeros(batch_size, 1, self.embed_dim, device=x.device)


def create_column_info_from_preprocessor(preprocessor: TabularPreprocessor) -> Dict[str, List[str]]:
    """
    Helper function to create column info dict for FeatureTokenizer.
    
    Args:
        preprocessor: Fitted TabularPreprocessor
        
    Returns:
        Dict with 'categorical' and 'numerical' keys
    """
    return {
        'categorical': preprocessor.categorical_columns,
        'numerical': preprocessor.numerical_columns
    }


# Integration helper functions
def enhance_config_with_preprocessing(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhance configuration dictionary with preprocessing metadata fields.
    
    Args:
        config_dict: Base configuration dictionary
        
    Returns:
        Enhanced configuration with preprocessing fields
    """
    enhanced_config = config_dict.copy()
    
    # Add preprocessing fields to dataset configs if they don't exist
    if 'datasets' in enhanced_config:
        for dataset_config in enhanced_config['datasets']:
            if 'categorical_columns' not in dataset_config:
                dataset_config['categorical_columns'] = None
            if 'numerical_columns' not in dataset_config:
                dataset_config['numerical_columns'] = None
    
    return enhanced_config