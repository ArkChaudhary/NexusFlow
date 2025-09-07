"""Enhanced data ingestion with unified preprocessing pipeline for NexusFlow Phase 2."""
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from typing import Dict, Tuple, List, Optional, Any
import os
import torch

from nexusflow.config import ConfigModel, DatasetConfig, ForeignKeyConfig
from nexusflow.data.dataset import NexusFlowDataset, AlignedData
from nexusflow.data.preprocessor import TabularPreprocessor, FeatureTokenizer, create_column_info_from_preprocessor
from dataclasses import dataclass
import uuid


def align_relational_data(datasets: Dict[str, pd.DataFrame], cfg: ConfigModel) -> AlignedData:
    """
    True relational alignment that produces multiple aligned but separate tables.
    
    This function performs proper relational joins without merging features,
    creating aligned tables that maintain separate feature spaces.
    """
    logger.info("🔗 Starting true multi-table relational alignment...")
    
    if cfg.training.use_synthetic or not cfg.datasets:
        return _create_fallback_aligned_data(datasets)
    
    # Step 1: Identify target table and build dependency graph
    target_table_name = cfg.target.get('target_table')
    if not target_table_name or target_table_name not in datasets:
        logger.warning("No valid target table, using first dataset")
        target_table_name = list(datasets.keys())[0]
    
    base_df = datasets[target_table_name].copy()
    logger.info(f"Base table: {target_table_name} ({len(base_df)} rows)")
    
    # Step 2: Assign global_id to base table
    base_df['global_id'] = [str(uuid.uuid4()) for _ in range(len(base_df))]
    
    # Initialize key mapping
    target_dataset_config = next(d for d in cfg.datasets if d.name == target_table_name)
    target_pk = target_dataset_config.primary_key
    target_pk_cols = target_pk if isinstance(target_pk, list) else [target_pk]
    
    key_map_data = []
    for _, row in base_df.iterrows():
        key_map_entry = {'global_id': row['global_id'], 'source_table': target_table_name}
        for pk_col in target_pk_cols:
            key_map_entry[f'{target_table_name}_{pk_col}'] = row[pk_col]
        key_map_data.append(key_map_entry)
    
    # Step 3: Calculate expansion requirements from one-to-many joins
    expansion_map = {}  # Maps global_id to how many times it should be replicated
    join_info = {}     # Stores join details for each table
    
    for table_name in datasets:
        if table_name == target_table_name:
            continue
            
        table_df = datasets[table_name].copy()
        dataset_config = next((d for d in cfg.datasets if d.name == table_name), None)
        if not dataset_config or not dataset_config.foreign_keys:
            continue
        
        # Find foreign key connecting to target table
        connecting_fk = None
        for fk in dataset_config.foreign_keys:
            if fk.references_table == target_table_name:
                connecting_fk = fk
                break
        
        if not connecting_fk:
            continue
        
        # Calculate expansion requirements
        fk_cols = connecting_fk.columns if isinstance(connecting_fk.columns, list) else [connecting_fk.columns]
        ref_cols = connecting_fk.references_columns if isinstance(connecting_fk.references_columns, list) else [connecting_fk.references_columns]
        
        # Count how many related records each base record has
        join_counts = table_df.groupby(fk_cols).size().to_dict()
        
        # Map to base table records
        for _, base_row in base_df.iterrows():
            base_key = tuple(base_row[col] for col in ref_cols) if len(ref_cols) > 1 else base_row[ref_cols[0]]
            count = join_counts.get(base_key, 0)
            
            global_id = base_row['global_id']
            if global_id not in expansion_map:
                expansion_map[global_id] = 1
            expansion_map[global_id] = max(expansion_map[global_id], count)
        
        join_info[table_name] = {
            'fk_config': connecting_fk,
            'table_df': table_df,
            'fk_cols': fk_cols,
            'ref_cols': ref_cols
        }
    
    # Step 4: Create aligned tables with proper row expansion
    aligned_tables = {}
    total_expansions = 0
    
    # Expand base table first
    expanded_base_rows = []
    for _, base_row in base_df.iterrows():
        global_id = base_row['global_id']
        expansion_count = expansion_map.get(global_id, 1)
        
        for i in range(expansion_count):
            expanded_row = base_row.copy()
            expanded_row['_row_index'] = i  # Track which expansion this is
            expanded_base_rows.append(expanded_row)
        
        if expansion_count > 1:
            total_expansions += expansion_count - 1
    
    expanded_base_df = pd.DataFrame(expanded_base_rows).reset_index(drop=True)
    aligned_tables[target_table_name] = expanded_base_df
    
    # Create aligned versions of other tables
    for table_name, info in join_info.items():
        table_df = info['table_df']
        fk_cols = info['fk_cols']
        ref_cols = info['ref_cols']
        
        # Create aligned version by matching to expanded base table
        aligned_rows = []
        
        for _, expanded_base_row in expanded_base_df.iterrows():
            global_id = expanded_base_row['global_id']
            row_index = expanded_base_row['_row_index']
            
            # Find matching records in the related table
            base_key = tuple(expanded_base_row[col] for col in ref_cols) if len(ref_cols) > 1 else expanded_base_row[ref_cols[0]]
            matching_records = table_df[
                (table_df[fk_cols] == base_key).all(axis=1) if len(fk_cols) > 1 
                else table_df[fk_cols[0]] == base_key
            ]
            
            if len(matching_records) > row_index:
                # Use the specific record for this row index
                selected_record = matching_records.iloc[row_index].copy()
            elif len(matching_records) > 0:
                # Use the last available record if we've run out
                selected_record = matching_records.iloc[-1].copy()
            else:
                # Create null record if no matches
                selected_record = pd.Series({col: None for col in table_df.columns})
            
            # Add global_id to maintain alignment
            selected_record['global_id'] = global_id
            aligned_rows.append(selected_record)
        
        aligned_table_df = pd.DataFrame(aligned_rows).reset_index(drop=True)
        aligned_tables[table_name] = aligned_table_df
    
    # Clean up temporary columns
    if '_row_index' in aligned_tables[target_table_name].columns:
        aligned_tables[target_table_name] = aligned_tables[target_table_name].drop(columns=['_row_index'])
    
    # Step 5: Update key mapping for expanded structure
    expanded_key_map_data = []
    for _, row in expanded_base_df.iterrows():
        key_entry = {'global_id': row['global_id'], 'source_table': target_table_name}
        for pk_col in target_pk_cols:
            key_entry[f'{target_table_name}_{pk_col}'] = row[pk_col]
        expanded_key_map_data.append(key_entry)
    
    key_map_df = pd.DataFrame(expanded_key_map_data)
    
    logger.info(f"🎯 Multi-table relational alignment complete:")
    logger.info(f"   Total row expansions: {total_expansions}")
    logger.info(f"   Target table final size: {len(aligned_tables[target_table_name])}")
    logger.info(f"   Aligned tables: {list(aligned_tables.keys())}")
    
    return AlignedData(
        aligned_tables=aligned_tables,
        key_map=key_map_df,
        metadata={
            'target_table': target_table_name,
            'join_stats': {
                'total_expansions': total_expansions,
                'join_operations': list(join_info.keys())
            },
            'original_sizes': {name: len(df) for name, df in datasets.items()},
            'alignment_mode': 'multi_table_preserved'
        }
    )

def _perform_expansion_join(base_df: pd.DataFrame, join_df: pd.DataFrame, 
                           fk_config: 'ForeignKeyConfig', join_table_name: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Perform a join that expands the base table for one-to-many relationships.
    
    Returns:
        Tuple of (expanded_dataframe, join_statistics)
    """
    # Extract join columns
    fk_cols = fk_config.columns if isinstance(fk_config.columns, list) else [fk_config.columns]
    ref_cols = fk_config.references_columns if isinstance(fk_config.references_columns, list) else [fk_config.references_columns]
    
    # Detect relationship cardinality
    base_key_counts = base_df[ref_cols].value_counts()
    join_key_counts = join_df[fk_cols].value_counts()
    
    is_one_to_many = any(join_key_counts > 1)
    
    if is_one_to_many:
        # One-to-many: Expand base table by duplicating rows
        logger.debug(f"One-to-many join detected for {join_table_name}")
        
        # Perform left join which will naturally duplicate base rows
        expanded_df = base_df.merge(
            join_df,
            left_on=ref_cols,
            right_on=fk_cols,
            how='left',
            suffixes=('', f'_from_{join_table_name}')
        )
        
        # Count expansions
        original_rows = len(base_df)
        final_rows = len(expanded_df)
        expansions = final_rows - original_rows
        
        join_stats = {
            'join_type': 'one_to_many',
            'expansions': expansions,
            'original_base_rows': original_rows,
            'final_rows': final_rows
        }
    else:
        # One-to-one: No expansion needed
        logger.debug(f"One-to-one join detected for {join_table_name}")
        
        expanded_df = base_df.merge(
            join_df,
            left_on=ref_cols,
            right_on=fk_cols,
            how='left',
            suffixes=('', f'_from_{join_table_name}')
        )
        
        join_stats = {
            'join_type': 'one_to_one',
            'expansions': 0,
            'original_base_rows': len(base_df),
            'final_rows': len(expanded_df)
        }
    
    return expanded_df, join_stats

def _update_key_mapping(key_map_data: List[Dict], expanded_df: pd.DataFrame, 
                       table_name: str, dataset_config: 'DatasetConfig'):
    """Update key mapping with new table's primary keys."""
    pk = dataset_config.primary_key
    pk_cols = pk if isinstance(pk, list) else [pk]
    
    # Create a mapping from global_id to the new table's keys
    global_id_to_keys = {}
    
    for _, row in expanded_df.iterrows():
        global_id = row['global_id']
        if global_id not in global_id_to_keys:
            global_id_to_keys[global_id] = []
        
        # Store this row's keys from the joined table
        row_keys = {}
        for pk_col in pk_cols:
            col_name = f'{table_name}_{pk_col}'
            if col_name in row or pk_col in row:
                row_keys[f'{table_name}_{pk_col}'] = row.get(col_name, row.get(pk_col))
        
        if row_keys:  # Only add if we found valid keys
            global_id_to_keys[global_id].append(row_keys)
    
    # Update the key_map_data with the new keys
    for entry in key_map_data:
        global_id = entry['global_id']
        if global_id in global_id_to_keys:
            # For now, take the first set of keys (could be enhanced later)
            keys = global_id_to_keys[global_id][0] if global_id_to_keys[global_id] else {}
            entry.update(keys)

def _build_join_dependency_graph(datasets_config: List['DatasetConfig']) -> Dict[str, List[str]]:
    """Build dependency graph for determining join order."""
    graph = {}
    
    for dataset in datasets_config:
        graph[dataset.name] = []
        if dataset.foreign_keys:
            for fk in dataset.foreign_keys:
                graph[dataset.name].append(fk.references_table)
    
    return graph

def _determine_join_order(graph: Dict[str, List[str]], target_table: str) -> List[str]:
    """Determine optimal join order using topological sort."""
    # Simple implementation: tables with no dependencies first
    no_deps = [table for table, deps in graph.items() if not deps]
    has_deps = [table for table, deps in graph.items() if deps]
    
    # Ensure target table is first
    join_order = [target_table] if target_table in no_deps else []
    join_order.extend([t for t in no_deps if t != target_table])
    join_order.extend(has_deps)
    
    return join_order

def _create_fallback_aligned_data(datasets: Dict[str, pd.DataFrame]) -> AlignedData:
    """Fallback for synthetic data or simple cases."""
    if not datasets:
        return AlignedData(aligned_tables={}, key_map=pd.DataFrame(), metadata={})
    
    # Simple case: just add global_id to first dataset
    first_table_name = list(datasets.keys())[0]
    first_df = datasets[first_table_name].copy()
    first_df['global_id'] = [str(uuid.uuid4()) for _ in range(len(first_df))]
    
    key_map = pd.DataFrame({
        'global_id': first_df['global_id'],
        'source_table': [first_table_name] * len(first_df)
    })
    
    return AlignedData(
        aligned_tables={first_table_name: first_df},
        key_map=key_map,
        metadata={'target_table': first_table_name, 'fallback_mode': True}
    )

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

def load_and_preprocess_datasets(cfg: ConfigModel) -> Tuple[AlignedData, Dict[str, TabularPreprocessor]]:
    """
    UPDATED: Load datasets with true relational alignment and preprocessing.
    
    Returns AlignedData instead of separate datasets dictionary.
    """
    logger.info("🔄 Loading datasets with TRUE relational alignment...")
    
    # Load raw datasets
    raw_datasets = {}
    for dataset_cfg in cfg.datasets:
        path = f"datasets/{dataset_cfg.name}"
        df = load_table(path)
        
        # Validate primary key for this dataset
        pk = dataset_cfg.primary_key
        pk_cols = pk if isinstance(pk, list) else [pk]
        
        for pk_col in pk_cols:
            if pk_col not in df.columns:
                logger.error(f"Primary key '{pk_col}' not found in {dataset_cfg.name}")
                raise KeyError(f"Primary key '{pk_col}' missing from {dataset_cfg.name}")
        
        raw_datasets[dataset_cfg.name] = df
    
    # Perform TRUE relational alignment
    aligned_data = align_relational_data(raw_datasets, cfg)
    
    preprocessors = {}
    
    if not cfg.training.use_advanced_preprocessing:
        logger.info("Using simple preprocessing")
        return aligned_data, preprocessors
    
    # Apply preprocessing to the aligned data
    logger.info("Applying preprocessing to aligned relational data...")
    
    main_table_name = aligned_data['metadata']['target_table']
    main_df = aligned_data['aligned_tables'][main_table_name]
    
    # Create preprocessor for the expanded table
    preprocessor = TabularPreprocessor()
    
    # Exclude system and target columns from preprocessing
    target_col = cfg.target.get('target_column')
    excluded_cols = {'global_id'}
    if target_col and target_col in main_df.columns:
        excluded_cols.add(target_col)
    
    feature_df = main_df.copy()
    for col in excluded_cols:
        if col in feature_df.columns:
            feature_df = feature_df.drop(columns=[col])
    
    # Aggregate column type information from original dataset configs
    categorical_cols = []
    numerical_cols = []
    
    for dataset_cfg in cfg.datasets:
        if dataset_cfg.categorical_columns:
            existing_cat = [col for col in dataset_cfg.categorical_columns if col in feature_df.columns]
            categorical_cols.extend(existing_cat)
        
        if dataset_cfg.numerical_columns:
            existing_num = [col for col in dataset_cfg.numerical_columns if col in feature_df.columns]
            numerical_cols.extend(existing_num)
    
    # Auto-detect remaining columns if enabled
    if cfg.training.auto_detect_types:
        remaining_cols = [col for col in feature_df.columns 
                         if col not in categorical_cols and col not in numerical_cols]
        
        for col in remaining_cols:
            if feature_df[col].dtype in ['object', 'category', 'bool']:
                categorical_cols.append(col)
            elif pd.api.types.is_numeric_dtype(feature_df[col]):
                numerical_cols.append(col)
    
    # Fit and transform
    if categorical_cols or numerical_cols:
        preprocessor.fit(feature_df, categorical_cols, numerical_cols)
        processed_features = preprocessor.transform(feature_df)
        
        # Reconstruct the main table with processed features
        final_cols = preprocessor.categorical_columns + preprocessor.numerical_columns
        processed_df = processed_features[final_cols].copy()
        
        # Add back excluded columns
        for col in excluded_cols:
            if col in main_df.columns:
                processed_df[col] = main_df[col]
        
        # Update the aligned data
        aligned_data['aligned_tables'][main_table_name] = processed_df
        preprocessors[main_table_name] = preprocessor
        
        # Update metadata
        aligned_data['metadata']['preprocessing_applied'] = True
        aligned_data['metadata']['processed_features'] = len(final_cols)
        
        logger.info(f"✅ Processed expanded table: {len(final_cols)} features")
    else:
        logger.info("⚠️ No features to process in expanded table")
    
    logger.info("🎯 TRUE relational preprocessing complete")
    return aligned_data, preprocessors

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

def make_dataloaders(cfg: ConfigModel, aligned_data: AlignedData, 
                     preprocessors: Dict[str, TabularPreprocessor] = None):
    """
    Create dataloaders from AlignedData structure.
    
    Args:
        cfg: Configuration model
        aligned_data: AlignedData structure with relational tables
        preprocessors: Optional preprocessors dictionary
    """
    logger.info("Creating dataloaders from AlignedData...")
    
    # Import here to avoid circular imports
    from nexusflow.data.dataset import NexusFlowDataset, MultiTableDataLoader
    
    # Create dataset from AlignedData
    dataset = NexusFlowDataset(aligned_data, target_col=cfg.target['target_column'])
    
    # Split the dataset
    total_samples = len(dataset)
    test_size = int(total_samples * cfg.training.split_config.test_size)
    val_size = int(total_samples * cfg.training.split_config.validation_size)
    train_size = total_samples - test_size - val_size
    
    # Create indices
    if cfg.training.split_config.randomize:
        import torch
        indices = torch.randperm(total_samples).tolist()
    else:
        indices = list(range(total_samples))
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size] if val_size > 0 else []
    test_indices = indices[train_size + val_size:]
    
    # Create subset datasets
    from torch.utils.data import Subset
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices) if val_indices else None
    test_dataset = Subset(dataset, test_indices)
    
    # Create dataloaders
    batch_size = cfg.training.batch_size
    
    train_loader = MultiTableDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = MultiTableDataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None
    test_loader = MultiTableDataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create preprocessing metadata for backward compatibility
    preprocessing_metadata = {
        'aligned_data_metadata': aligned_data['metadata'],
        'preprocessors': preprocessors or {},
        'feature_dimensions': dataset.feature_dimensions,
        'total_samples': total_samples
    }
    
    logger.info(f"AlignedData DataLoaders created:")
    logger.info(f"  Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
    logger.info(f"  Val: {len(val_loader) if val_loader else 0} batches")
    logger.info(f"  Test: {len(test_loader)} batches ({len(test_dataset)} samples)")
    logger.info(f"  Source: {aligned_data['metadata']['target_table']} (expanded)")
    
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

# Mark legacy functions as deprecated
def flatten_relational_data(datasets: Dict[str, pd.DataFrame], cfg: ConfigModel) -> pd.DataFrame:
    """
    DEPRECATED: Use align_relational_data instead.
    
    This function performs legacy flattening and should be replaced.
    """
    import warnings
    warnings.warn(
        "flatten_relational_data is deprecated. Use align_relational_data for true relational support.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Fallback to simple join for backward compatibility
    if not datasets or len(datasets) == 1:
        return list(datasets.values())[0] if datasets else pd.DataFrame()
    
    # Simple left join approach
    base_table_name = cfg.target.get('target_table', list(datasets.keys())[0])
    result_df = datasets[base_table_name].copy()
    
    for dataset_config in cfg.datasets:
        if dataset_config.name == base_table_name or not dataset_config.foreign_keys:
            continue
        
        table_df = datasets[dataset_config.name]
        fk = dataset_config.foreign_keys[0]
        
        result_df = result_df.merge(
            table_df,
            left_on=fk.references_columns,
            right_on=fk.columns,
            how='left',
            suffixes=('', f'_{dataset_config.name}')
        )
    
    logger.warning("Used deprecated flattening - consider upgrading to align_relational_data")
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