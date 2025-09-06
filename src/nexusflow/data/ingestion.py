"""Enhanced data ingestion with unified preprocessing pipeline for NexusFlow Phase 2."""
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from typing import Dict, Tuple, List, Optional, Any
import os
import torch

from nexusflow.config import ConfigModel, DatasetConfig
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
    Load datasets with multi-table support and individual preprocessing.
    RESTORED: No flattening - maintains separate tables for multi-agent architecture.
    """
    logger.info("🔄 Loading datasets with multi-table support...")
    
    raw_datasets = {}
    preprocessors = {}
    
    # Load all datasets
    for dataset_cfg in cfg.datasets:
        path = f"datasets/{dataset_cfg.name}"
        df = load_table(path)
        
        # Validate primary key for this specific dataset
        pk = dataset_cfg.primary_key
        pk_cols = pk if isinstance(pk, list) else [pk]
        
        for pk_col in pk_cols:
            if pk_col not in df.columns:
                logger.error(f"Primary key '{pk_col}' not found in {dataset_cfg.name}")
                raise KeyError(f"Primary key '{pk_col}' missing from {dataset_cfg.name}")
        
        raw_datasets[dataset_cfg.name] = df
    
    # Use align_datasets instead of flatten_relational_data
    aligned_datasets = align_datasets(raw_datasets, cfg.primary_key)
    
    if not cfg.training.use_advanced_preprocessing:
        logger.info("Using simple preprocessing (fillna)")
        return aligned_datasets, {}
    
    # Apply preprocessing to each aligned DataFrame individually
    logger.info("Applying individual preprocessing to each aligned dataset...")
    processed_datasets = {}
    
    for dataset_cfg in cfg.datasets:
        dataset_name = dataset_cfg.name
        if dataset_name not in aligned_datasets:
            continue
            
        df = aligned_datasets[dataset_name]
        
        # Create individual preprocessor for this dataset
        preprocessor = TabularPreprocessor()
        
        # Exclude target column and primary key from preprocessing
        target_col = cfg.target.get('target_column')
        feature_df = df.copy()
        excluded_cols = {cfg.primary_key}
        if target_col and target_col in feature_df.columns:
            excluded_cols.add(target_col)
        
        # Remove excluded columns for preprocessing
        for col in excluded_cols:
            if col in feature_df.columns:
                feature_df = feature_df.drop(columns=[col])
        
        # Get categorical/numerical columns for this specific dataset
        categorical_cols = dataset_cfg.categorical_columns or []
        numerical_cols = dataset_cfg.numerical_columns or []
        
        # Filter to only columns that exist in this dataset's feature_df
        existing_categorical = [col for col in categorical_cols if col in feature_df.columns]
        existing_numerical = [col for col in numerical_cols if col in feature_df.columns]
        
        # Auto-detect remaining columns if enabled
        if cfg.training.auto_detect_types:
            remaining_cols = [col for col in feature_df.columns 
                             if col not in existing_categorical and col not in existing_numerical]
            
            for col in remaining_cols:
                if feature_df[col].dtype in ['object', 'category', 'bool']:
                    existing_categorical.append(col)
                elif pd.api.types.is_numeric_dtype(feature_df[col]):
                    existing_numerical.append(col)
        
        # Fit and transform this dataset individually
        if existing_categorical or existing_numerical:
            preprocessor.fit(feature_df, existing_categorical, existing_numerical)
            processed_features = preprocessor.transform(feature_df)
            
            # Reconstruct dataset with processed features + excluded columns
            final_cols = preprocessor.categorical_columns + preprocessor.numerical_columns
            processed_df = processed_features[final_cols].copy()
            
            # Add back excluded columns
            for col in excluded_cols:
                if col in df.columns:
                    processed_df[col] = df[col]
            
            processed_datasets[dataset_name] = processed_df
            preprocessors[dataset_name] = preprocessor
            
            logger.info(f"✅ Processed {dataset_name}: {len(final_cols)} features")
        else:
            # No features to process, keep original
            processed_datasets[dataset_name] = df
            logger.info(f"⚠️ No features to process in {dataset_name}")
    
    logger.info("🎯 Multi-table preprocessing complete - datasets remain separate")
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
    Create multi-table dataset that preserves separate table structure.
    RESTORED: Creates feature dimension mapping for proper tensor splitting.
    """
    logger.info("Creating multi-table dataset with preserved table boundaries...")
    
    # Find the target table and column
    target_table_name = cfg.target['target_table']
    if target_table_name not in datasets:
        raise KeyError(f"Target table '{target_table_name}' not found in datasets")
    
    target_df = datasets[target_table_name]
    target_column = cfg.target['target_column']
    
    if target_column not in target_df.columns:
        raise KeyError(f"Target column '{target_column}' not found in target table")
    
    # Combine features while tracking dimensions for each dataset
    combined_data = []
    feature_dimensions = []  # This is the key mapping for tensor splitting
    preprocessing_metadata = {
        'dataset_order': [],
        'feature_dimensions': [],
        'preprocessor_info': {},
        'column_mappings': {}
    }
    
    for dataset_config in cfg.datasets:
        dataset_name = dataset_config.name
        df = datasets[dataset_name]
        
        # Get feature columns (exclude primary key and target)
        excluded_cols = {cfg.primary_key}
        if target_column in df.columns:
            excluded_cols.add(target_column)
        
        feature_cols = [col for col in df.columns if col not in excluded_cols]
        
        if feature_cols:
            # Store the feature dimension for this dataset
            feature_dimensions.append(len(feature_cols))
            combined_data.append(df[feature_cols])
            
            # Store metadata
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
                
                preprocessing_metadata['column_mappings'][dataset_name] = {
                    'categorical': [col for col in feature_cols if col in feature_info['categorical_columns']],
                    'numerical': [col for col in feature_cols if col in feature_info['numerical_columns']]
                }
            
            logger.info(f"📊 Dataset {dataset_name}: {len(feature_cols)} features")
    
    # Combine all features horizontally (this creates the wide format)
    if combined_data:
        combined_features = pd.concat(combined_data, axis=1)
    else:
        raise ValueError("No feature columns found across all datasets")
    
    # Add target column
    combined_features[target_column] = target_df[target_column]
    
    # Create the dataset
    dataset = NexusFlowDataset(combined_features, target_col=target_column)
    
    # CRITICAL: Store the feature dimensions for tensor splitting
    dataset.feature_dimensions = feature_dimensions
    dataset.preprocessing_metadata = preprocessing_metadata
    
    # Enhanced attributes for advanced transformers
    dataset.transformer_types = [d.transformer_type for d in cfg.datasets]
    dataset.complexities = [d.complexity for d in cfg.datasets]
    dataset.context_weights = [d.context_weight for d in cfg.datasets]
    
    logger.info(f"Multi-table dataset created:")
    logger.info(f"  Total features: {sum(feature_dimensions)}")
    logger.info(f"  Dataset dimensions: {feature_dimensions}")
    logger.info(f"  Datasets: {len(preprocessing_metadata['dataset_order'])}")
    
    return dataset, preprocessing_metadata

def make_dataloaders(cfg: ConfigModel, datasets: Dict[str, pd.DataFrame], 
                     preprocessors: Dict[str, TabularPreprocessor] = None):
    """
    Create dataloaders with multi-table support.
    RESTORED: Uses create_multi_table_dataset to maintain separate table structure.
    """
    logger.info("Creating dataloaders for multi-table data...")
    
    # Use the restored multi-table dataset creation
    dataset, preprocessing_metadata = create_multi_table_dataset(
        datasets, preprocessors or {}, cfg
    )
    
    # Split indices from the combined dataset
    combined_df = dataset.df
    train_indices, val_indices, test_indices = split_df(
        combined_df,
        test_size=cfg.training.split_config.test_size,
        val_size=cfg.training.split_config.validation_size,
        randomize=cfg.training.split_config.randomize,
    )
    
    # Create split datasets
    train_df = combined_df.iloc[train_indices.index].reset_index(drop=True)
    val_df = combined_df.iloc[val_indices.index].reset_index(drop=True) if len(val_indices) > 0 else pd.DataFrame()
    test_df = combined_df.iloc[test_indices.index].reset_index(drop=True)
    
    target_column = cfg.target['target_column']
    
    # Create NexusFlowDataset instances with preserved feature_dimensions
    train_dataset = NexusFlowDataset(train_df, target_col=target_column)
    train_dataset.feature_dimensions = dataset.feature_dimensions  # CRITICAL: Copy the mapping
    train_dataset.preprocessing_metadata = preprocessing_metadata
    
    val_dataset = None
    if len(val_df) > 0:
        val_dataset = NexusFlowDataset(val_df, target_col=target_column)
        val_dataset.feature_dimensions = dataset.feature_dimensions
        val_dataset.preprocessing_metadata = preprocessing_metadata
    
    test_dataset = NexusFlowDataset(test_df, target_col=target_column)
    test_dataset.feature_dimensions = dataset.feature_dimensions
    test_dataset.preprocessing_metadata = preprocessing_metadata
    
    # Create DataLoaders with custom collate for multi-table data
    batch_size = cfg.training.batch_size
    
    # Use MultiTableDataLoader if we have multi-table data
    if dataset.feature_dimensions and len(dataset.feature_dimensions) > 1:
        from nexusflow.data.dataset import MultiTableDataLoader
        
        train_loader = MultiTableDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = MultiTableDataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None
        test_loader = MultiTableDataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        logger.info("🔀 Using MultiTableDataLoader for multi-agent batching")
    else:
        # Standard DataLoader for single table
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f"Multi-table DataLoaders created:")
    logger.info(f"  Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
    logger.info(f"  Val: {len(val_loader) if val_loader else 0} batches ({len(val_dataset) if val_dataset else 0} samples)")
    logger.info(f"  Test: {len(test_loader)} batches ({len(test_dataset)} samples)")
    logger.info(f"  Feature dimensions per table: {dataset.feature_dimensions}")
    
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

def build_join_graph(datasets_config: List[DatasetConfig]) -> Dict[str, Any]:
    """Build a dependency graph for table joins based on foreign key relationships."""
    graph = {
        'nodes': {},  # table_name -> DatasetConfig
        'edges': [],  # (from_table, to_table, join_info)
        'dependencies': {}  # table -> list of tables it depends on
    }
    
    # Build nodes
    for dataset in datasets_config:
        graph['nodes'][dataset.name] = dataset
        graph['dependencies'][dataset.name] = []
    
    # Build edges from foreign key relationships
    for dataset in datasets_config:
        if dataset.foreign_keys:
            for fk in dataset.foreign_keys:
                edge = {
                    'from_table': dataset.name,
                    'to_table': fk.references_table,
                    'from_columns': fk.columns if isinstance(fk.columns, list) else [fk.columns],
                    'to_columns': fk.references_columns if isinstance(fk.references_columns, list) else [fk.references_columns]
                }
                graph['edges'].append(edge)
                graph['dependencies'][dataset.name].append(fk.references_table)
    
    logger.info(f"Join graph built: {len(graph['nodes'])} tables, {len(graph['edges'])} relationships")
    return graph

def flatten_relational_data(datasets: Dict[str, pd.DataFrame], cfg: ConfigModel) -> pd.DataFrame:
    """
    Intelligent relational data flattening using proper foreign key relationships.
    
    This replaces the old single-key alignment with true relational joins.
    """
    if cfg.training.use_synthetic or not cfg.datasets:
        # Fallback for synthetic data or simple cases
        return list(datasets.values())[0] if datasets else pd.DataFrame()
    
    logger.info("🔗 Starting intelligent relational data flattening...")
    
    # Build join graph
    join_graph = build_join_graph(cfg.datasets)
    
    # Start with target table as base
    target_table = cfg.target.get('target_table')
    if not target_table or target_table not in datasets:
        logger.warning("No target table specified, using first dataset as base")
        target_table = list(datasets.keys())[0]
    
    base_df = datasets[target_table].copy()
    logger.info(f"Base table: {target_table} ({len(base_df)} rows)")
    
    # Track which tables we've already joined
    joined_tables = {target_table}
    result_df = base_df
    
    # Iteratively join tables based on dependencies
    max_iterations = len(datasets) * 2  # Prevent infinite loops
    iteration = 0
    
    while len(joined_tables) < len(datasets) and iteration < max_iterations:
        iteration += 1
        progress_made = False
        
        for dataset_config in cfg.datasets:
            table_name = dataset_config.name
            
            # Skip if already joined or no foreign keys
            if table_name in joined_tables or not dataset_config.foreign_keys:
                continue
            
            # Check if all referenced tables are already joined
            can_join = True
            for fk in dataset_config.foreign_keys:
                if fk.references_table not in joined_tables:
                    can_join = False
                    break
            
            if can_join:
                # Perform the join
                table_df = datasets[table_name].copy()
                
                # Join with primary foreign key relationship
                primary_fk = dataset_config.foreign_keys[0]  # Use first FK as primary
                
                from_cols = primary_fk.columns if isinstance(primary_fk.columns, list) else [primary_fk.columns]
                to_cols = primary_fk.references_columns if isinstance(primary_fk.references_columns, list) else [primary_fk.references_columns]
                
                # Detect relationship type
                ref_table_name = primary_fk.references_table
                ref_key_counts = result_df[to_cols].drop_duplicates()
                table_key_counts = table_df[from_cols].drop_duplicates()
                
                is_one_to_many = len(table_df) > len(table_key_counts)
                
                if is_one_to_many:
                    logger.info(f"Aggregating {table_name} (one-to-many relationship)")
                    table_df = _aggregate_for_join(table_df, from_cols, dataset_config)
                
                # Perform the join
                result_df = result_df.merge(
                    table_df, 
                    left_on=to_cols, 
                    right_on=from_cols, 
                    how='left',
                    suffixes=('', f'_{table_name}')
                )
                
                # Clean up duplicate columns
                duplicate_cols = [col for col in result_df.columns if col.endswith(f'_{table_name}')]
                for dup_col in duplicate_cols:
                    original_col = dup_col.replace(f'_{table_name}', '')
                    if original_col in result_df.columns:
                        result_df = result_df.drop(columns=[dup_col])
                
                joined_tables.add(table_name)
                progress_made = True
                
                logger.info(f"✅ Joined {table_name}: {len(result_df)} rows, {len(result_df.columns)} columns")
    
    if len(joined_tables) < len(datasets):
        missing_tables = set(d.name for d in cfg.datasets) - joined_tables
        logger.warning(f"Could not join all tables. Missing: {missing_tables}")
    
    logger.info(f"🎯 Relational flattening complete: {len(result_df)} rows, {len(result_df.columns)} columns")
    return result_df

def _aggregate_for_join(df: pd.DataFrame, key_columns: List[str], dataset_config: DatasetConfig) -> pd.DataFrame:
    """Aggregate one-to-many data before joining."""
    # Identify feature columns (exclude keys)
    feature_cols = [col for col in df.columns if col not in key_columns]
    
    # Separate categorical and numerical for different aggregation strategies
    categorical_cols = dataset_config.categorical_columns or []
    numerical_cols = dataset_config.numerical_columns or []
    
    # Auto-detect if not specified
    if not categorical_cols and not numerical_cols:
        for col in feature_cols:
            if df[col].dtype in ['object', 'category', 'bool']:
                categorical_cols.append(col)
            elif pd.api.types.is_numeric_dtype(df[col]):
                numerical_cols.append(col)
    
    agg_dict = {}
    
    # Aggregate numerical columns with multiple statistics
    for col in numerical_cols:
        if col in feature_cols:
            agg_dict[f'{col}_mean'] = pd.NamedAgg(column=col, aggfunc='mean')
            agg_dict[f'{col}_sum'] = pd.NamedAgg(column=col, aggfunc='sum')
            agg_dict[f'{col}_std'] = pd.NamedAgg(column=col, aggfunc='std')
            agg_dict[f'{col}_count'] = pd.NamedAgg(column=col, aggfunc='count')
    
    # Aggregate categorical columns with mode and count
    for col in categorical_cols:
        if col in feature_cols:
            agg_dict[f'{col}_mode'] = pd.NamedAgg(column=col, aggfunc=lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0])
            agg_dict[f'{col}_nunique'] = pd.NamedAgg(column=col, aggfunc='nunique')
    
    if agg_dict:
        aggregated = df.groupby(key_columns).agg(agg_dict).reset_index()
        # Flatten column names
        aggregated.columns = [col if isinstance(col, str) else col for col in aggregated.columns]
    else:
        # If no features to aggregate, just get unique keys
        aggregated = df[key_columns].drop_duplicates().reset_index(drop=True)
    
    logger.debug(f"Aggregated {df.shape} -> {aggregated.shape}")
    return aggregated