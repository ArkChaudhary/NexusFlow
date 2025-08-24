"""Enhanced data ingestion utilities: load multiple CSVs, align by primary key, and create DataLoaders."""
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from typing import Dict, Tuple, List
import os

from nexusflow.config import ConfigModel
from nexusflow.data.dataset import NexusFlowDataset

def load_table(path: str) -> pd.DataFrame:
    """Load a single CSV table with validation."""
    if not os.path.exists(path):
        logger.error(f"Table not found: {path}")
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    
    # Enhanced logging: rows/cols and missing value summary
    missing_summary = df.isnull().sum()
    missing_total = missing_summary.sum()
    missing_cols = missing_summary[missing_summary > 0]
    
    logger.info(f"Loaded table: {path} rows={len(df)} cols={len(df.columns)}")
    logger.info(f"Missing values: total={missing_total} ({missing_total/df.size*100:.1f}%)")
    
    if len(missing_cols) > 0:
        logger.info(f"Missing by column: {missing_cols.to_dict()}")
    else:
        logger.debug("No missing values found")
    
    return df

def validate_primary_key(df: pd.DataFrame, key: str) -> bool:
    """Validate that primary key column exists and has no duplicates or nulls."""
    if key not in df.columns:
        logger.error(f"Primary key '{key}' not found in DataFrame columns: {list(df.columns)}")
        raise KeyError(key)
    dups = df[key].duplicated()
    nulls = df[key].isna()
    if dups.any() or nulls.any():
        logger.warning(
            f"Primary key '{key}' invalid: duplicates={int(dups.sum())}, nulls={int(nulls.sum())}"
        )
        return False
    return True

def load_datasets(cfg: ConfigModel) -> Dict[str, pd.DataFrame]:
    """
    Load all datasets specified in config and validate primary keys.
    
    Returns:
        Dict mapping dataset name to DataFrame
    """
    datasets = {}
    
    for dataset_cfg in cfg.datasets:
        # Load the CSV
        path = f"datasets/{dataset_cfg.name}"
        df = load_table(path)
        
        # Validate primary key
        validate_primary_key(df, cfg.primary_key)
        
        # Log dataset info
        missing_values = df.isnull().sum().sum()
        logger.info(f"Dataset {dataset_cfg.name}: shape={df.shape}, missing_values={missing_values}")
        
        if missing_values > 0:
            logger.debug(f"Missing values by column in {dataset_cfg.name}: {df.isnull().sum().to_dict()}")
        
        datasets[dataset_cfg.name] = df
    
    return datasets

def align_datasets(datasets: Dict[str, pd.DataFrame], primary_key: str) -> Dict[str, pd.DataFrame]:
    """
    Align datasets by primary key, keeping only rows that exist in all datasets.
    
    Args:
        datasets: Dict of dataset_name -> DataFrame
        primary_key: Column name to align on
        
    Returns:
        Dict of aligned DataFrames with same row indices
    """
    if not datasets:
        raise ValueError("No datasets provided for alignment")
    
    # Find common primary key values across all datasets
    common_keys = None
    for name, df in datasets.items():
        keys = set(df[primary_key].unique())
        if common_keys is None:
            common_keys = keys
        else:
            common_keys = common_keys.intersection(keys)
        logger.debug(f"Dataset {name}: {len(keys)} unique keys")
    
    logger.info(f"Found {len(common_keys)} common keys across all datasets")
    
    if len(common_keys) == 0:
        raise ValueError("No common primary key values found across datasets")
    
    # Filter each dataset to only include common keys
    aligned_datasets = {}
    for name, df in datasets.items():
        aligned_df = df[df[primary_key].isin(common_keys)].copy()
        aligned_df = aligned_df.sort_values(primary_key).reset_index(drop=True)
        aligned_datasets[name] = aligned_df
        logger.debug(f"Aligned {name}: {len(aligned_df)} rows")
    
    return aligned_datasets

def split_df(df: pd.DataFrame, test_size: float = 0.15, val_size: float = 0.15, randomize: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split DataFrame into train/val/test splits."""
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
    
    logger.debug(f"Split sizes -> train={len(train)} val={len(val)} test={len(test)}")
    return train, val, test

def get_feature_dimensions(datasets: Dict[str, pd.DataFrame], primary_key: str, target_column: str) -> List[int]:
    """
    Calculate the number of feature columns for each dataset.
    
    Args:
        datasets: Dict of dataset_name -> DataFrame
        primary_key: Primary key column to exclude
        target_column: Target column to exclude
        
    Returns:
        List of feature dimensions for each dataset
    """
    dimensions = []
    
    for name, df in datasets.items():
        # Exclude primary key and target column (if present)
        excluded_cols = {primary_key}
        if target_column in df.columns:
            excluded_cols.add(target_column)
        
        feature_cols = [col for col in df.columns if col not in excluded_cols]
        num_features = len(feature_cols)
        dimensions.append(num_features)
        
        logger.debug(f"Dataset {name}: {num_features} features from columns {feature_cols}")
    
    return dimensions

def create_multi_table_dataset(datasets: Dict[str, pd.DataFrame], cfg: ConfigModel) -> NexusFlowDataset:
    """
    Create a NexusFlowDataset that handles multiple aligned tables.
    
    Args:
        datasets: Dict of aligned DataFrames
        cfg: Configuration object
        
    Returns:
        NexusFlowDataset instance
    """
    # Find the target table
    target_table_name = cfg.target['target_table']
    if target_table_name not in datasets:
        raise KeyError(f"Target table '{target_table_name}' not found in datasets")
    
    target_df = datasets[target_table_name]
    target_column = cfg.target['target_column']
    
    if target_column not in target_df.columns:
        raise KeyError(f"Target column '{target_column}' not found in target table")
    
    # Create a combined DataFrame with all features and target
    combined_data = []
    feature_start_indices = []  # Track where each dataset's features start
    
    current_idx = 0
    for dataset_name in [d.name for d in cfg.datasets]:
        df = datasets[dataset_name]
        
        # Get feature columns (exclude primary key and target)
        excluded_cols = {cfg.primary_key}
        if target_column in df.columns:
            excluded_cols.add(target_column)
        
        feature_cols = [col for col in df.columns if col not in excluded_cols]
        
        if feature_cols:
            feature_start_indices.append(current_idx)
            combined_data.append(df[feature_cols])
            current_idx += len(feature_cols)
    
    # Combine all features horizontally
    if combined_data:
        combined_features = pd.concat(combined_data, axis=1)
    else:
        raise ValueError("No feature columns found across all datasets")
    
    # Add target column
    combined_features[target_column] = target_df[target_column]
    
    # Create the dataset
    dataset = NexusFlowDataset(combined_features, target_col=target_column)
    
    # Store metadata for splitting features back into per-dataset groups
    dataset.feature_start_indices = feature_start_indices
    dataset.feature_dimensions = get_feature_dimensions(datasets, cfg.primary_key, target_column)
    
    return dataset

def make_dataloaders(cfg: ConfigModel, datasets: Dict[str, pd.DataFrame]) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders from aligned datasets.
    
    Args:
        cfg: Configuration object
        datasets: Dict of aligned DataFrames
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Split the data
    # Use the first dataset as reference for splitting (all should have same rows after alignment)
    reference_df = list(datasets.values())[0]
    
    train_indices, val_indices, test_indices = split_df(
        reference_df,
        test_size=cfg.training.split_config.get("test_size", 0.15),
        val_size=cfg.training.split_config.get("validation_size", 0.15),
        randomize=cfg.training.split_config.get("randomize", True),
    )
    
    # Split all datasets using the same indices
    train_datasets = {name: df.iloc[train_indices.index] for name, df in datasets.items()}
    val_datasets = {name: df.iloc[val_indices.index] for name, df in datasets.items()} if len(val_indices) > 0 else {}
    test_datasets = {name: df.iloc[test_indices.index] for name, df in datasets.items()}
    
    # Create datasets
    train_dataset = create_multi_table_dataset(train_datasets, cfg)
    val_dataset = create_multi_table_dataset(val_datasets, cfg) if val_datasets else None
    test_dataset = create_multi_table_dataset(test_datasets, cfg)
    
    # Create DataLoaders
    batch_size = cfg.training.batch_size
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f"Created DataLoaders: train={len(train_loader)} val={len(val_loader) if val_loader else 0} test={len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader