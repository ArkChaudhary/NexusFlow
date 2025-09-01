"""Enhanced data ingestion with unified preprocessing pipeline for NexusFlow Phase 2."""
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from typing import Dict, Tuple, List, Optional, Any
import os
import torch

from nexusflow.config import ConfigModel
from nexusflow.data.dataset import NexusFlowDataset
from nexusflow.data.preprocessor import TabularPreprocessor, FeatureTokenizer, create_column_info_from_preprocessor

def load_table(path: str) -> pd.DataFrame:
    """Load a single CSV table with enhanced validation."""
    if not os.path.exists(path):
        logger.error(f"Table not found: {path}")
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    
    # Enhanced logging with data quality metrics
    missing_summary = df.isnull().sum()
    missing_total = missing_summary.sum()
    missing_cols = missing_summary[missing_summary > 0]
    
    # Data type analysis
    categorical_count = sum(1 for col in df.columns if df[col].dtype in ['object', 'category', 'bool'])
    numerical_count = sum(1 for col in df.columns if pd.api.types.is_numeric_dtype(df[col]))
    
    logger.info(f"Loaded table: {path}")
    logger.info(f"  Shape: {len(df)} rows × {len(df.columns)} cols")
    logger.info(f"  Column types: {categorical_count} categorical, {numerical_count} numerical") 
    logger.info(f"  Missing values: {missing_total} total ({missing_total/df.size*100:.1f}%)")
    
    if len(missing_cols) > 0:
        logger.debug(f"  Missing by column: {dict(missing_cols.head())}")
    
    return df

def validate_primary_key(df: pd.DataFrame, key: str) -> bool:
    """Enhanced primary key validation."""
    if key not in df.columns:
        logger.error(f"Primary key '{key}' not found in DataFrame columns: {list(df.columns)}")
        raise KeyError(key)
    
    dups = df[key].duplicated()
    nulls = df[key].isna()
    
    if dups.any() or nulls.any():
        logger.warning(f"Primary key '{key}' issues: {int(dups.sum())} duplicates, {int(nulls.sum())} nulls")
        return False
    
    logger.debug(f"Primary key '{key}' validation passed: {len(df[key].unique())} unique values")
    return True

def load_and_preprocess_datasets(cfg: ConfigModel) -> Tuple[Dict[str, pd.DataFrame], Dict[str, TabularPreprocessor]]:
    """
    Enhanced dataset loading with preprocessing pipeline.
    
    Returns:
        Tuple of (processed_datasets, fitted_preprocessors)
    """
    logger.info("🔄 Loading datasets with advanced preprocessing...")
    
    raw_datasets = {}
    preprocessors = {}
    
    # Load all datasets
    for dataset_cfg in cfg.datasets:
        path = f"datasets/{dataset_cfg.name}"
        df = load_table(path)
        validate_primary_key(df, cfg.primary_key)
        raw_datasets[dataset_cfg.name] = df
    
    # Align datasets by primary key
    aligned_datasets = align_datasets(raw_datasets, cfg.primary_key)
    
    if not cfg.training.use_advanced_preprocessing:
        logger.info("Using simple preprocessing (fillna)")
        return aligned_datasets, {}
    
    # Apply advanced preprocessing per dataset
    processed_datasets = {}
    
    for dataset_cfg in cfg.datasets:
        dataset_name = dataset_cfg.name
        df = aligned_datasets[dataset_name]
        
        logger.info(f"Preprocessing dataset: {dataset_name}")
        
        # Create and fit preprocessor
        preprocessor = TabularPreprocessor()
        
        # Exclude primary key and target from preprocessing
        feature_df = df.drop(columns=[cfg.primary_key], errors='ignore')
        target_col = cfg.target['target_column']
        if target_col and target_col in feature_df.columns:
            feature_df = feature_df.drop(columns=[target_col])
        
        # Fit preprocessor with explicit column specification or auto-detection
        categorical_cols = dataset_cfg.categorical_columns
        numerical_cols = dataset_cfg.numerical_columns
        
        if cfg.training.auto_detect_types and (categorical_cols is None or numerical_cols is None):
            logger.info(f"Auto-detecting column types for {dataset_name}")
            preprocessor.fit(feature_df, categorical_cols, numerical_cols)
        else:
            logger.info(f"Using explicit column types for {dataset_name}")
            # Validate specified columns exist
            if categorical_cols:
                missing_cats = [col for col in categorical_cols if col not in feature_df.columns]
                if missing_cats:
                    logger.warning(f"Categorical columns not found: {missing_cats}")
                    categorical_cols = [col for col in categorical_cols if col in feature_df.columns]
            
            if numerical_cols:
                missing_nums = [col for col in numerical_cols if col not in feature_df.columns]
                if missing_nums:
                    logger.warning(f"Numerical columns not found: {missing_nums}")
                    numerical_cols = [col for col in numerical_cols if col in feature_df.columns]
            
            preprocessor.fit(feature_df, categorical_cols, numerical_cols)
        
        # Transform features
        processed_features = preprocessor.transform(feature_df)
        
        # Reconstruct full dataset with primary key and target, using ONLY processed columns
        final_feature_cols = preprocessor.categorical_columns + preprocessor.numerical_columns
        processed_df = processed_features[final_feature_cols].copy()
        
        processed_df[cfg.primary_key] = df[cfg.primary_key]
        if target_col and target_col in df.columns:
            processed_df[target_col] = df[target_col]
        
        processed_datasets[dataset_name] = processed_df
        preprocessors[dataset_name] = preprocessor
        
        # Log preprocessing results
        feature_info = preprocessor.get_feature_info()
        logger.info(f"  ✅ {dataset_name} processed:")
        logger.info(f"     Categorical features: {len(feature_info['categorical_columns'])}")
        logger.info(f"     Numerical features: {len(feature_info['numerical_columns'])}")
        logger.info(f"     Total vocabulary size: {sum(feature_info['vocab_sizes'].values())}")
    
    logger.info("🎯 Advanced preprocessing complete")
    return processed_datasets, preprocessors

def align_datasets(datasets: Dict[str, pd.DataFrame], primary_key: str) -> Dict[str, pd.DataFrame]:
    """Enhanced dataset alignment with better logging."""
    if not datasets:
        raise ValueError("No datasets provided for alignment")
    
    # Find common primary key values across all datasets
    common_keys = None
    alignment_stats = {}
    
    for name, df in datasets.items():
        keys = set(df[primary_key].unique())
        alignment_stats[name] = {
            'original_keys': len(keys),
            'total_rows': len(df),
            'duplicates': df[primary_key].duplicated().sum()
        }
        
        if common_keys is None:
            common_keys = keys
        else:
            common_keys = common_keys.intersection(keys)
        
        logger.debug(f"Dataset {name}: {len(keys)} unique keys, {len(df)} rows")
    
    logger.info(f"Alignment analysis:")
    for name, stats in alignment_stats.items():
        coverage = len(common_keys) / stats['original_keys'] * 100 if stats['original_keys'] > 0 else 0
        logger.info(f"  {name}: {stats['original_keys']} keys, {coverage:.1f}% coverage")
    
    logger.info(f"Common keys across all datasets: {len(common_keys)}")
    
    if len(common_keys) == 0:
        raise ValueError("No common primary key values found across datasets")
    
    # Filter each dataset to only include common keys
    aligned_datasets = {}
    for name, df in datasets.items():
        aligned_df = df[df[primary_key].isin(common_keys)].copy()
        aligned_df = aligned_df.sort_values(primary_key).reset_index(drop=True)
        aligned_datasets[name] = aligned_df
        
        retention_rate = len(aligned_df) / len(df) * 100
        logger.debug(f"Aligned {name}: {len(aligned_df)} rows ({retention_rate:.1f}% retained)")
    
    return aligned_datasets

def create_multi_table_dataset(datasets: Dict[str, pd.DataFrame], 
                                       preprocessors: Dict[str, TabularPreprocessor],
                                       cfg: ConfigModel) -> Tuple[NexusFlowDataset, Dict[str, Any]]:
    """
    Create enhanced multi-table dataset with preprocessing metadata.
    
    Args:
        datasets: Aligned and preprocessed datasets
        preprocessors: Fitted preprocessors for each dataset
        cfg: Configuration object
        
    Returns:
        Tuple of (enhanced_dataset, preprocessing_metadata)
    """
    logger.info("Creating enhanced multi-table dataset...")
    
    # Find the target table and column
    target_table_name = cfg.target['target_table']
    if target_table_name not in datasets:
        raise KeyError(f"Target table '{target_table_name}' not found in datasets")
    
    target_df = datasets[target_table_name]
    target_column = cfg.target['target_column']
    
    if target_column not in target_df.columns:
        raise KeyError(f"Target column '{target_column}' not found in target table")
    
    # Combine all features with preprocessing metadata
    combined_data = []
    feature_start_indices = []
    preprocessing_metadata = {
        'dataset_order': [],
        'feature_dimensions': [],
        'preprocessor_info': {},
        'column_mappings': {}
    }
    
    current_idx = 0
    for dataset_config in cfg.datasets:
        dataset_name = dataset_config.name
        df = datasets[dataset_name]
        
        # Get feature columns (exclude primary key and target)
        excluded_cols = {cfg.primary_key}
        if target_column in df.columns:
            excluded_cols.add(target_column)
        
        feature_cols = [col for col in df.columns if col not in excluded_cols]
        
        if feature_cols:
            feature_start_indices.append(current_idx)
            combined_data.append(df[feature_cols])
            
            # Store preprocessing metadata
            preprocessing_metadata['dataset_order'].append(dataset_name)
            preprocessing_metadata['feature_dimensions'].append(len(feature_cols))
            
            if dataset_name in preprocessors:
                preprocessor = preprocessors[dataset_name]
                feature_info = preprocessor.get_feature_info()
                
                preprocessing_metadata['preprocessor_info'][dataset_name] = {
                    'categorical_columns': feature_info['categorical_columns'],
                    'numerical_columns': feature_info['numerical_columns'],
                    'vocab_sizes': feature_info['vocab_sizes'],
                    'transformer_type': dataset_config.transformer_type
                }
                
                # Create column mapping for tokenizer
                preprocessing_metadata['column_mappings'][dataset_name] = {
                    'categorical': [col for col in feature_cols if col in feature_info['categorical_columns']],
                    'numerical': [col for col in feature_cols if col in feature_info['numerical_columns']]
                }
            
            current_idx += len(feature_cols)
    
    # Combine all features horizontally
    if combined_data:
        combined_features = pd.concat(combined_data, axis=1)
    else:
        raise ValueError("No feature columns found across all datasets")
    
    # Add target column
    combined_features[target_column] = target_df[target_column]
    
    # Create the enhanced dataset
    dataset = NexusFlowDataset(combined_features, target_col=target_column)
    
    # Store enhanced metadata
    dataset.feature_start_indices = feature_start_indices
    dataset.feature_dimensions = preprocessing_metadata['feature_dimensions']
    dataset.preprocessing_metadata = preprocessing_metadata
    
    # Enhanced attributes for advanced transformers
    dataset.transformer_types = [d.transformer_type for d in cfg.datasets]
    dataset.complexities = [d.complexity for d in cfg.datasets]
    dataset.context_weights = [d.context_weight for d in cfg.datasets]
    
    logger.info(f"Enhanced dataset created:")
    logger.info(f"  Total features: {sum(preprocessing_metadata['feature_dimensions'])}")
    logger.info(f"  Datasets: {len(preprocessing_metadata['dataset_order'])}")
    logger.info(f"  Preprocessing: {'enabled' if preprocessors else 'disabled'}")
    
    return dataset, preprocessing_metadata

def make_dataloaders(cfg: ConfigModel, datasets: Dict[str, pd.DataFrame], 
                            preprocessors: Dict[str, TabularPreprocessor] = None):
    """
    Create enhanced dataloaders with preprocessing support.
    
    Args:
        cfg: Configuration object
        datasets: Processed datasets
        preprocessors: Fitted preprocessors (optional)
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, preprocessing_metadata)
    """
    logger.info("Creating enhanced dataloaders with preprocessing support...")
    
    # Get reference dataset for splitting
    reference_df = list(datasets.values())[0]
    
    # Split indices - access Pydantic model attributes directly
    train_indices, val_indices, test_indices = split_df(
        reference_df,
        test_size=cfg.training.split_config.test_size,
        val_size=cfg.training.split_config.validation_size,
        randomize=cfg.training.split_config.randomize,
    )
    
    # Split all datasets using the same indices
    train_datasets = {name: df.iloc[train_indices.index] for name, df in datasets.items()}
    val_datasets = {name: df.iloc[val_indices.index] for name, df in datasets.items()} if len(val_indices) > 0 else {}
    test_datasets = {name: df.iloc[test_indices.index] for name, df in datasets.items()}
    
    # Create enhanced datasets with preprocessing metadata
    train_dataset, preprocessing_metadata = create_multi_table_dataset(
        train_datasets, preprocessors or {}, cfg
    )
    
    val_dataset = None
    if val_datasets:
        val_dataset, _ = create_multi_table_dataset(
            val_datasets, preprocessors or {}, cfg
        )
    
    test_dataset, _ = create_multi_table_dataset(
        test_datasets, preprocessors or {}, cfg
    )
    
    # Create DataLoaders
    batch_size = cfg.training.batch_size
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f"Enhanced DataLoaders created:")
    logger.info(f"  Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
    logger.info(f"  Val: {len(val_loader) if val_loader else 0} batches ({len(val_dataset) if val_dataset else 0} samples)")
    logger.info(f"  Test: {len(test_loader)} batches ({len(test_dataset)} samples)")
    
    return train_loader, val_loader, test_loader, preprocessing_metadata

def split_df(df: pd.DataFrame, test_size: float = 0.15, val_size: float = 0.15, randomize: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Enhanced dataset splitting with better logging."""
    logger.debug(f"Splitting dataset: {len(df)} samples, test={test_size}, val={val_size}, random={randomize}")
    
    if randomize:
        train_val, test = train_test_split(df, test_size=test_size, random_state=42)
        if val_size > 0:
            train, val = train_test_split(train_val, test_size=val_size/(1-test_size), random_state=42)
        else:
            train, val = train_val, pd.DataFrame()
    else:
        n = len(df)
        n_test = int(n * test_size)
        n_val = int(n * val_size)
        train = df[:-n_test-n_val].copy()
        val = df[-n_test-n_val:-n_test].copy() if n_val > 0 else pd.DataFrame()
        test = df[-n_test:].copy()
    
    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True) if len(val) > 0 else val
    test = test.reset_index(drop=True)
    
    logger.debug(f"Split complete: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test

# Legacy compatibility functions
def load_datasets(cfg: ConfigModel) -> Dict[str, pd.DataFrame]:
    """Legacy function for backward compatibility."""
    logger.warning("Using legacy load_datasets - consider upgrading to load_and_preprocess_datasets")
    
    datasets = {}
    for dataset_cfg in cfg.datasets:
        path = f"datasets/{dataset_cfg.name}"
        df = load_table(path)
        validate_primary_key(df, cfg.primary_key)
        datasets[dataset_cfg.name] = df
    
    return datasets

def get_feature_dimensions(datasets: Dict[str, pd.DataFrame], primary_key: str, target_column: str) -> List[int]:
    """Legacy function for feature dimension calculation."""
    dimensions = []
    
    for name, df in datasets.items():
        excluded_cols = {primary_key}
        if target_column in df.columns:
            excluded_cols.add(target_column)
        
        feature_cols = [col for col in df.columns if col not in excluded_cols]
        dimensions.append(len(feature_cols))
        logger.debug(f"Dataset {name}: {len(feature_cols)} features")
    
    return dimensions