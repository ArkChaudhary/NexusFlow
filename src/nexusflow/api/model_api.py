import torch
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from typing import Dict, Any, List, Union, Optional
import torch.nn as nn
import json

class NexusFlowModel:
    """
    Complete NexusFlow model that implements the required .predict() and .get_params() interface.
    This is what gets saved as a .nxf file and provides the end-user API.
    Combines the functionality of both NexusFlowModelArtifact and ModelAPI.
    """
    
    def __init__(self, model: torch.nn.Module, preprocess_meta: Dict[str, Any]):
        """
        Initialize the NexusFlow model.
        
        Args:
            model: Trained NexusFormer model (must be torch.nn.Module)
            preprocess_meta: Metadata including config, input dimensions, training info
        """
        # Add type checking to prevent nested artifacts
        if not isinstance(model, torch.nn.Module):
            raise TypeError(f"Expected torch.nn.Module, got {type(model)}. "
                        f"If passing a NexusFlowModel, use model.model instead.")
        
        self.model = model
        self.model.eval()  # Always in eval mode for inference
        self.meta = preprocess_meta
        
        # Extract key information from metadata
        self.config = preprocess_meta.get('config', {})
        self.input_dims = preprocess_meta.get('input_dims', [])
        self.primary_key = self.config.get('primary_key', 'id')
        self.target_info = self.config.get('target', {})
        self.datasets_config = self.config.get('datasets', [])
        
        # Extract preprocessors if available
        self.preprocessors = preprocess_meta.get('preprocessors', {})
        
        # Device management
        self.device = torch.device('cpu')  # Default to CPU for artifacts
        
        logger.info(f"NexusFlow model initialized with {len(self.input_dims)} input dimensions")
        if self.preprocessors:
            logger.info(f"Preprocessors available for: {list(self.preprocessors.keys())}")

    def save(self, path: str):
        """Save model to .nxf file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        
        # Ensure .nxf extension
        if not p.suffix == '.nxf':
            p = p.with_suffix('.nxf')
        
        # Check if model has been optimized and update metadata accordingly
        meta = self.meta.copy()
        if 'optimization' in meta:
            optimization_info = meta['optimization']
            logger.info(f"Saving optimized model with {optimization_info['method']}")
            logger.info(f"  Size reduction: {optimization_info['size_reduction']:.1%}")
            logger.info(f"  Parameter reduction: {optimization_info['parameter_reduction']:.1%}")
        
        save_data = {
            'model_state': self.model.state_dict(),
            'meta': meta,
            'artifact_version': '1.0'
        }
        
        torch.save(save_data, p)
        logger.info(f"NexusFlow model saved to: {p}")
    
    def predict(self, data: Union[Dict[str, pd.DataFrame], List[pd.DataFrame], pd.DataFrame]) -> np.ndarray:
        """
        Make predictions on new data with Phase 4 relational aggregation support.
        """
        logger.info("Making predictions on new data with Phase 4 relational support...")
        
        # Enhanced relational detection based on training logs
        # Check multiple indicators that this model was trained with relational data
        
        # 1. Check if model has preprocessors (indicates Phase 2+ training)
        has_preprocessors = bool(self.preprocessors)
        
        # 2. Check metadata indicators
        relational_features = self.meta.get('relational_features', {})
        has_relational_support = relational_features.get('relational_data_support', False)
        
        # 3. Check phase_2_features
        phase_2_features = self.meta.get('phase_2_features', {})
        has_relational_joins = phase_2_features.get('relational_joins', False)
        
        # 4. Check if config indicates multiple datasets but model expects only 1 input dimension
        config_dict = self.meta.get('config', {})
        datasets_config = config_dict.get('datasets', [])
        multiple_datasets = len(datasets_config) > 1
        single_input_dim = len(self.input_dims) == 1
        
        # 5. Check for aligned_data_metadata in preprocessing_metadata
        preprocessing_metadata = self.meta.get('preprocessing_metadata', {})
        has_aligned_metadata = 'aligned_data_metadata' in preprocessing_metadata
        
        # Decision logic: Use relational pipeline if ANY of these conditions are true
        should_use_relational = (
            # Core indicators
            (has_relational_support and isinstance(data, dict)) or
            (has_relational_joins and isinstance(data, dict)) or
            (has_aligned_metadata and isinstance(data, dict) and len(data) > 1) or
            # Strong indicators: multiple datasets in config but single input dimension
            (multiple_datasets and single_input_dim and isinstance(data, dict) and len(data) > 1) or
            # Preprocessor indicator: has preprocessors and multiple input tables
            (has_preprocessors and isinstance(data, dict) and len(data) > 1)
        )
        
        # Debug logging
        logger.info(f"Relational detection analysis:")
        logger.info(f"  - Has preprocessors: {has_preprocessors}")
        logger.info(f"  - Has relational support: {has_relational_support}")
        logger.info(f"  - Has relational joins: {has_relational_joins}")
        logger.info(f"  - Multiple datasets in config: {multiple_datasets}")
        logger.info(f"  - Single input dimension: {single_input_dim}")
        logger.info(f"  - Has aligned metadata: {has_aligned_metadata}")
        logger.info(f"  - Input is multi-table dict: {isinstance(data, dict) and len(data) > 1}")
        logger.info(f"  - Decision: Use {'RELATIONAL' if should_use_relational else 'STANDARD'} pipeline")
        
        if should_use_relational:
            return self._predict_relational_with_aggregation(data)
        else:
            return self._predict_standard(data)
    
    def _preprocess_input(self, data: Union[Dict[str, pd.DataFrame], List[pd.DataFrame], pd.DataFrame]) -> List[torch.Tensor]:
        """
        Preprocess input data into the format expected by the model.
        
        Returns:
            List of feature tensors, one per dataset
        """
        if isinstance(data, dict):
            # Multi-table format with names
            return self._process_named_tables(data)
        elif isinstance(data, list):
            # Multi-table format as list
            return self._process_table_list(data)
        elif isinstance(data, pd.DataFrame):
            # Single flattened table - need to split back into components
            return self._process_flattened_table(data)
        else:
            raise ValueError(f"Unsupported data format: {type(data)}")
    
    def _process_named_tables(self, data: Dict[str, pd.DataFrame]) -> List[torch.Tensor]:
        """Process data provided as named tables dictionary with categorical encoding."""
        feature_tensors = []
        
        for dataset_config in self.datasets_config:
            dataset_name = dataset_config['name']
            
            if dataset_name not in data:
                raise KeyError(f"Required dataset '{dataset_name}' not found in input data")
            
            df = data[dataset_name].copy()

            # --- FIX: Convert datetime columns to numeric before processing ---
            for col in df.select_dtypes(include=['datetime64[ns]']).columns:
                logger.info(f"Converting datetime column '{col}' to numeric timestamp.")
                df[col] = df[col].astype(np.int64) // 10**9 # Convert to seconds since epoch
            
            # Apply preprocessing if available
            if dataset_name in self.preprocessors:
                logger.info(f"Applying preprocessing to {dataset_name}")
                preprocessor = self.preprocessors[dataset_name]
                
                # Transform the dataframe using the trained preprocessor
                try:
                    # Use the preprocessor's transform method (assumes it has one)
                    if hasattr(preprocessor, 'transform'):
                        processed_df = preprocessor.transform(df)
                    elif hasattr(preprocessor, 'apply_transformations'):
                        processed_df = preprocessor.apply_transformations(df)
                    else:
                        logger.warning(f"Preprocessor for {dataset_name} has no transform method, using manual encoding")
                        processed_df = self._manual_categorical_encoding(df, dataset_config)
                except Exception as e:
                    logger.warning(f"Preprocessor failed for {dataset_name}: {e}, using manual encoding")
                    processed_df = self._manual_categorical_encoding(df, dataset_config)
                
                df = processed_df
            else:
                # Manual categorical encoding fallback
                logger.info(f"No preprocessor found for {dataset_name}, using manual categorical encoding")
                df = self._manual_categorical_encoding(df, dataset_config)
            
            # Extract feature columns (exclude primary key and target if present)
            excluded_cols = {self.primary_key}
            target_col = self.target_info.get('target_column')
            if target_col and target_col in df.columns:
                excluded_cols.add(target_col)
            
            feature_cols = [col for col in df.columns if col not in excluded_cols]
            
            if not feature_cols:
                raise ValueError(f"No feature columns found in dataset '{dataset_name}'")
            
            # Convert to tensor
            features_df = df[feature_cols].fillna(0)  # Handle missing values
            
            # Ensure all data is numeric
            try:
                tensor = torch.tensor(features_df.values.astype(np.float32))
            except ValueError as e:
                # If conversion still fails, show which columns have issues
                non_numeric_cols = []
                for col in feature_cols:
                    try:
                        pd.to_numeric(features_df[col])
                    except (ValueError, TypeError):
                        non_numeric_cols.append(col)
                
                raise ValueError(f"Cannot convert columns to numeric in {dataset_name}: {non_numeric_cols}. "
                               f"Original error: {e}")
            
            feature_tensors.append(tensor)
            logger.info(f"Processed {dataset_name}: {tensor.shape}")
        
        return feature_tensors
    
    def _manual_categorical_encoding(self, df: pd.DataFrame, dataset_config: Dict) -> pd.DataFrame:
        """
        Manual categorical encoding fallback when preprocessors are not available.
        """
        df = df.copy()
        
        # Get categorical columns from config
        categorical_columns = dataset_config.get('categorical_columns', [])
        
        for col in categorical_columns:
            if col in df.columns:
                logger.info(f"Encoding categorical column: {col}")
                
                # Simple label encoding for categorical columns
                if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                    # Create a simple label encoding
                    unique_values = df[col].dropna().unique()
                    value_map = {val: idx for idx, val in enumerate(unique_values)}
                    
                    # Map values and handle unseen categories
                    df[col] = df[col].map(value_map).fillna(-1)  # -1 for unseen categories
                    
                    logger.info(f"  Encoded {col}: {len(unique_values)} categories -> numeric")
        
        # Ensure all remaining object columns are handled
        for col in df.columns:
            if df[col].dtype == 'object':
                logger.warning(f"Converting remaining object column {col} to numeric")
                # Try to convert to numeric, fallback to label encoding
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    # Final fallback: simple label encoding
                    unique_values = df[col].dropna().unique()
                    value_map = {val: idx for idx, val in enumerate(unique_values)}
                    df[col] = df[col].map(value_map).fillna(-1)
        
        return df
    
    def _process_table_list(self, data: List[pd.DataFrame]) -> List[torch.Tensor]:
        """Process data provided as list of DataFrames."""
        if len(data) != len(self.datasets_config):
            raise ValueError(f"Expected {len(self.datasets_config)} DataFrames, got {len(data)}")
        
        feature_tensors = []
        
        for i, df in enumerate(data):
            dataset_config = self.datasets_config[i]
            dataset_name = dataset_config['name']
            df_copy = df.copy()
            
            # Apply preprocessing if available
            if dataset_name in self.preprocessors:
                logger.info(f"Applying preprocessing to dataset {i} ({dataset_name})")
                preprocessor = self.preprocessors[dataset_name]
                
                try:
                    processed_df = preprocessor.transform(df_copy)
                    df_copy = processed_df
                    logger.info(f"Successfully applied trained preprocessor to {dataset_name}")
                except Exception as e:
                    logger.warning(f"Preprocessor failed for dataset {i}: {e}, using manual encoding")
                    df_copy = self._manual_categorical_encoding(df_copy, dataset_config)
            else:
                # Manual categorical encoding fallback
                df_copy = self._manual_categorical_encoding(df_copy, dataset_config)
            
            # Extract feature columns (exclude primary key and target if present)
            excluded_cols = {self.primary_key}
            target_col = self.target_info.get('target_column')
            if target_col and target_col in df_copy.columns:
                excluded_cols.add(target_col)
            
            feature_cols = [col for col in df_copy.columns if col not in excluded_cols]
            
            if not feature_cols:
                raise ValueError(f"No feature columns found in DataFrame {i}")
            
            # Convert to tensor
            features_df = df_copy[feature_cols].fillna(0)
            
            try:
                tensor = torch.tensor(features_df.values.astype(np.float32))
            except ValueError as e:
                # Debug info for remaining non-numeric columns
                non_numeric_cols = []
                for col in feature_cols:
                    try:
                        pd.to_numeric(features_df[col])
                    except:
                        non_numeric_cols.append(col)
                
                raise ValueError(f"Cannot convert columns to numeric in dataset {i}: {non_numeric_cols}. "
                               f"Original error: {e}")
            
            feature_tensors.append(tensor)
            logger.info(f"Processed dataset {i}: {tensor.shape}")
        
        return feature_tensors
    
    def _process_flattened_table(self, data: pd.DataFrame) -> List[torch.Tensor]:
        """Process data provided as single flattened DataFrame."""
        df = data.copy()
        
        # Apply categorical encoding to the entire dataframe
        # Since we don't know which dataset each column belongs to, use a general approach
        for col in df.columns:
            if df[col].dtype == 'object':
                logger.info(f"Encoding categorical column in flattened data: {col}")
                # Simple label encoding
                unique_values = df[col].dropna().unique()
                value_map = {val: idx for idx, val in enumerate(unique_values)}
                df[col] = df[col].map(value_map).fillna(-1)
        
        # Exclude primary key and target columns
        excluded_cols = {self.primary_key}
        target_col = self.target_info.get('target_column')
        if target_col and target_col in df.columns:
            excluded_cols.add(target_col)
        
        feature_cols = [col for col in df.columns if col not in excluded_cols]
        features_df = df[feature_cols].fillna(0)
        
        # Split features according to input dimensions
        feature_tensors = []
        start_idx = 0
        
        for dim in self.input_dims:
            end_idx = start_idx + dim
            
            if end_idx > len(feature_cols):
                raise ValueError(f"Not enough features in flattened table. Expected {sum(self.input_dims)}, got {len(feature_cols)}")
            
            subset_cols = feature_cols[start_idx:end_idx]
            tensor = torch.tensor(features_df[subset_cols].values.astype(np.float32))
            feature_tensors.append(tensor)
            
            start_idx = end_idx
        
        return feature_tensors

    def _predict_relational_with_aggregation(self, data: Dict[str, pd.DataFrame]) -> np.ndarray:
        """
        Phase 4 relational prediction with proper aggregation.
        Uses the same align_relational_data function as training.
        """
        logger.info("🔄 Applying Phase 4 relational prediction with aggregation...")
        
        try:
            # Import here to avoid circular dependencies
            from nexusflow.data.ingestion import align_relational_data
            from nexusflow.config import ConfigModel
            
            # Reconstruct config from metadata
            config_dict = self.meta.get('config', {})
            
            # Handle different pydantic versions
            if hasattr(ConfigModel, 'model_validate'):
                config = ConfigModel.model_validate(config_dict)
            elif hasattr(ConfigModel, 'parse_obj'):
                config = ConfigModel.parse_obj(config_dict)
            else:
                config = ConfigModel(**config_dict)
            
            # Ensure proper table naming (add .csv extension if missing)
            data_with_csv = {}
            for table_name, df in data.items():
                if not table_name.endswith('.csv'):
                    data_with_csv[f"{table_name}.csv"] = df
                else:
                    data_with_csv[table_name] = df
            
            logger.info(f"Processing tables: {list(data_with_csv.keys())}")
            
            # Apply the SAME relational alignment as used in training
            logger.info("Applying relational alignment for prediction...")
            aligned_data = align_relational_data(data_with_csv, config)
            
            # Apply preprocessing if available
            if self.preprocessors:
                aligned_data = self._apply_prediction_preprocessing(aligned_data, config)
            
            # Get the main expanded table
            target_table = aligned_data['metadata']['target_table']
            main_df = aligned_data['aligned_tables'][target_table]
            
            logger.info(f"Aligned data shape: {main_df.shape}")
            logger.info(f"Aligned data columns: {list(main_df.columns)}")
            
            # Extract features (exclude system columns)
            excluded_cols = {'global_id'}
            target_col = config.target.get('target_column')
            if target_col and target_col in main_df.columns:
                excluded_cols.add(target_col)
            
            feature_cols = [col for col in main_df.columns if col not in excluded_cols]
            features_df = main_df[feature_cols].fillna(0)
            
            logger.info(f"Features for prediction: {len(feature_cols)} columns, {len(features_df)} rows")
            logger.info(f"Expected input dims: {self.input_dims}")
            logger.info(f"Actual feature dimensions: {len(feature_cols)}")
            
            # Validate and adjust feature dimensions if necessary
            expected_features = self.input_dims[0] if len(self.input_dims) > 0 else len(feature_cols)
            
            if len(feature_cols) != expected_features:
                logger.warning(f"Feature dimension mismatch: expected {expected_features}, got {len(feature_cols)}")
                logger.warning(f"Feature columns: {feature_cols[:10]}...")  # Show first 10
                
                # Try to match features by selecting the expected number
                if len(feature_cols) > expected_features:
                    logger.info(f"Truncating features from {len(feature_cols)} to {expected_features}")
                    feature_cols = feature_cols[:expected_features]
                    features_df = features_df[feature_cols]
                elif len(feature_cols) < expected_features:
                    logger.error(f"Not enough features: need {expected_features}, have {len(feature_cols)}")
                    # Pad with zeros
                    missing_features = expected_features - len(feature_cols)
                    for i in range(missing_features):
                        features_df[f'padded_feature_{i}'] = 0.0
                    logger.info(f"Padded with {missing_features} zero features")
            
            # Convert to tensors
            feature_tensor = torch.tensor(features_df.values.astype(np.float32))
            logger.info(f"Feature tensor shape: {feature_tensor.shape}")
            
            # Create key features tensor (global_id + other key columns)
            key_cols = ['global_id']
            for col in main_df.columns:
                if any(keyword in col.lower() for keyword in ['_id', 'key', 'pk', 'fk']) and col != 'global_id':
                    key_cols.append(col)
            
            logger.info(f"Key columns: {key_cols}")
            
            if len(key_cols) > 1:
                key_data = []
                for col in key_cols:
                    if col == 'global_id':
                        # Convert global_id to hash for numerical processing
                        key_data.append([hash(str(val)) % (2**31) for val in main_df[col]])
                    else:
                        # Handle other key columns
                        col_data = []
                        for val in main_df[col]:
                            if pd.isna(val):
                                col_data.append(0.0)
                            elif isinstance(val, (int, float)):
                                col_data.append(float(val))
                            else:
                                col_data.append(float(hash(str(val)) % (2**31)))
                        key_data.append(col_data)
                
                key_tensor = torch.tensor(np.array(key_data).T, dtype=torch.float32)
            else:
                # Only global_id
                key_data = [hash(str(val)) % (2**31) for val in main_df['global_id']]
                key_tensor = torch.tensor(key_data, dtype=torch.float32).unsqueeze(1)
            
            logger.info(f"Key tensor shape: {key_tensor.shape}")
            
            # Run model prediction
            self.model.eval()
            with torch.no_grad():
                predictions = self.model([feature_tensor], key_features=key_tensor)
            
            # Convert to numpy
            predictions_np = predictions.cpu().numpy()
            logger.info(f"Raw predictions shape: {predictions_np.shape}")
            
            # Apply task-specific post-processing
            target_col = self.target_info.get('target_column', 'label')
            if target_col == 'label':
                # Classification: apply sigmoid for binary classification
                predictions_np = 1 / (1 + np.exp(-predictions_np))  # sigmoid
            
            # PHASE 4 AGGREGATION: Group predictions by original primary key
            logger.info("🎯 Performing Phase 4 prediction aggregation...")
            
            if 'key_map' in aligned_data and len(aligned_data['key_map']) > 0:
                key_map_df = aligned_data['key_map']
                logger.info(f"Key map shape: {key_map_df.shape}")
                logger.info(f"Key map columns: {list(key_map_df.columns)}")
                
                # Create aggregation mapping
                aggregation_df = pd.DataFrame({
                    'global_id': main_df['global_id'].values,
                    'prediction': predictions_np
                })
                
                logger.info(f"Aggregation df shape: {aggregation_df.shape}")
                
                # Merge with key_map to get original keys
                merged_df = aggregation_df.merge(key_map_df, on='global_id', how='left')
                logger.info(f"Merged df shape: {merged_df.shape}")
                logger.info(f"Merged df columns: {list(merged_df.columns)}")
                
                # Find the primary key column for the target table
                target_dataset_config = None
                target_table_clean = target_table.replace('.csv', '')
                
                for ds_config in config.datasets:
                    if ds_config.name == target_table_clean:
                        target_dataset_config = ds_config
                        break
                
                if target_dataset_config and target_dataset_config.primary_key:
                    primary_key = target_dataset_config.primary_key
                    if isinstance(primary_key, list):
                        primary_key_name = primary_key[0]
                    else:
                        primary_key_name = primary_key
                    
                    # Look for the primary key column in merged_df
                    possible_key_cols = [
                        f"{target_table}_{primary_key_name}",
                        f"{target_table_clean}_{primary_key_name}",
                        primary_key_name,
                        f"users.csv_{primary_key_name}",  # specific for this case
                        f"users_{primary_key_name}"
                    ]
                    
                    available_key_cols = [col for col in possible_key_cols if col in merged_df.columns]
                    
                    logger.info(f"Looking for primary key: {primary_key_name}")
                    logger.info(f"Possible columns: {possible_key_cols}")
                    logger.info(f"Available columns: {available_key_cols}")
                    
                    if available_key_cols:
                        key_col = available_key_cols[0]
                        logger.info(f"Aggregating by: {key_col}")
                        
                        # Group by original primary key and aggregate (mean strategy)
                        aggregated = merged_df.groupby(key_col)['prediction'].mean().reset_index()
                        final_predictions = aggregated['prediction'].values
                        
                        logger.info(f"✅ Aggregated {len(predictions_np)} expanded predictions to {len(final_predictions)} final predictions")
                        return final_predictions
                    else:
                        logger.warning(f"Primary key column not found. Available columns: {list(merged_df.columns)}")
                else:
                    logger.warning("No primary key configuration found for aggregation")
            
            # Fallback: return raw predictions if aggregation fails
            logger.warning("⚠️ Could not perform prediction aggregation, returning raw predictions")
            return predictions_np
            
        except Exception as e:
            logger.error(f"Relational prediction failed: {e}")
            logger.exception("Full error details:")
            logger.info("Falling back to standard prediction pipeline")
            return self._predict_standard(data)

    def _predict_standard(self, data: Union[Dict[str, pd.DataFrame], List[pd.DataFrame], pd.DataFrame]) -> np.ndarray:
        """Standard prediction pipeline (original logic)."""
        # Convert input to standardized format
        feature_tensors = self._preprocess_input(data)
        
        # Validate input dimensions
        if len(feature_tensors) != len(self.input_dims):
            raise ValueError(f"Expected {len(self.input_dims)} feature groups, got {len(feature_tensors)}")
        
        for i, (tensor, expected_dim) in enumerate(zip(feature_tensors, self.input_dims)):
            if tensor.size(-1) != expected_dim:
                raise ValueError(f"Feature group {i} has {tensor.size(-1)} features, expected {expected_dim}")
        
        # Run inference
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(feature_tensors)
        
        # Convert to numpy and apply post-processing based on task type
        predictions_np = predictions.cpu().numpy()
        
        # Apply task-specific post-processing
        target_col = self.target_info.get('target_column', 'label')
        if target_col == 'label':
            # Classification: apply sigmoid for binary classification
            predictions_np = 1 / (1 + np.exp(-predictions_np))  # sigmoid
        
        logger.info(f"Generated {len(predictions_np)} predictions")
        return predictions_np

    def _apply_prediction_preprocessing(self, aligned_data, config):
        """Apply preprocessing to aligned data for prediction."""
        target_table = aligned_data['metadata']['target_table']
        
        if target_table in self.preprocessors:
            logger.info(f"Applying trained preprocessor to {target_table}")
            
            main_df = aligned_data['aligned_tables'][target_table].copy()
            preprocessor = self.preprocessors[target_table]
            
            # Exclude system columns
            excluded_cols = {'global_id'}
            target_col = config.target.get('target_column')
            if target_col and target_col in main_df.columns:
                excluded_cols.add(target_col)
            
            feature_df = main_df.drop(columns=excluded_cols, errors='ignore')
            
            # Transform features
            try:
                processed_features = preprocessor.transform(feature_df)
                final_cols = preprocessor.categorical_columns + preprocessor.numerical_columns
                processed_df = processed_features[final_cols].copy()
                
                # Add back excluded columns
                for col in excluded_cols:
                    if col in main_df.columns:
                        processed_df[col] = main_df[col].values
                
                aligned_data['aligned_tables'][target_table] = processed_df
                
            except Exception as e:
                logger.warning(f"Preprocessing failed during prediction: {e}")
        
        return aligned_data
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters and configuration.
        
        Returns:
            Dictionary containing all model parameters and training configuration
        """
        params = {
            'model_class': 'NexusFormer',
            'input_dimensions': self.input_dims,
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'config': self.config,
            'training_info': {
                'best_epoch': self.meta.get('best_epoch'),
                'best_val_metric': self.meta.get('best_val_metric'),
                'training_complete': self.meta.get('training_complete', False)
            },
            'architecture': {
                'embed_dim': self.config.get('architecture', {}).get('global_embed_dim', 64),
                'refinement_iterations': self.config.get('architecture', {}).get('refinement_iterations', 3),
                'num_encoders': len(self.input_dims)
            },
            'datasets_info': {
                'primary_key': self.primary_key,
                'target_column': self.target_info.get('target_column'),
                'target_table': self.target_info.get('target_table'),
                'num_datasets': len(self.datasets_config),
                'dataset_names': [d['name'] for d in self.datasets_config] if self.datasets_config else [],
                'has_preprocessors': bool(self.preprocessors)
            }
        }
        
        # Add optimization info if model was optimized
        if 'optimization' in self.meta:
            params['optimization'] = self.meta['optimization']
        
        return params
    
    def evaluate(self) -> Dict[str, float]:
        """
        Return evaluation metrics from training if available.
        
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        if 'best_val_metric' in self.meta:
            metrics['best_validation_metric'] = self.meta['best_val_metric']
        
        if 'best_epoch' in self.meta:
            metrics['best_epoch'] = self.meta['best_epoch']
        
        # Add any other stored metrics
        stored_metrics = self.meta.get('final_metrics', {})
        metrics.update(stored_metrics)
        
        return metrics
    
    def get_preprocessing_info(self) -> Dict[str, Any]:
        """
        Get information about preprocessing applied to each dataset.
        
        Returns:
            Dictionary with preprocessing information
        """
        preprocessing_info = {
            'has_preprocessors': bool(self.preprocessors),
            'datasets': {}
        }
        
        for dataset_config in self.datasets_config:
            dataset_name = dataset_config['name']
            dataset_info = {
                'categorical_columns': dataset_config.get('categorical_columns', []),
                'numerical_columns': dataset_config.get('numerical_columns', []),
                'transformer_type': dataset_config.get('transformer_type', 'standard'),
                'has_trained_preprocessor': dataset_name in self.preprocessors
            }
            
            if dataset_name in self.preprocessors:
                preprocessor = self.preprocessors[dataset_name]
                if hasattr(preprocessor, 'get_feature_info'):
                    try:
                        feature_info = preprocessor.get_feature_info()
                        dataset_info['preprocessor_info'] = feature_info
                    except:
                        pass
            
            preprocessing_info['datasets'][dataset_name] = dataset_info
        
        return preprocessing_info
    
    def visualize_flow(self):
        """
        Launch the interactive visualization dashboard.
        Note: This is a placeholder - full implementation would require web interface.
        """
        logger.warning("Interactive visualization not yet implemented")
        logger.info("Model architecture summary:")
        logger.info(f"  - {len(self.input_dims)} contextual encoders")
        logger.info(f"  - Input dimensions: {self.input_dims}")
        logger.info(f"  - Refinement iterations: {self.config.get('architecture', {}).get('refinement_iterations', 3)}")
        logger.info(f"  - Total parameters: {sum(p.numel() for p in self.model.parameters())}")
        logger.info(f"  - Preprocessing: {'Available' if self.preprocessors else 'Manual fallback'}")
        
        return {
            'architecture': 'NexusFormer',
            'num_encoders': len(self.input_dims),
            'input_dims': self.input_dims,
            'refinement_iterations': self.config.get('architecture', {}).get('refinement_iterations', 3),
            'has_preprocessors': bool(self.preprocessors)
        }
    
    def to(self, device: str):
        """Move model to specified device."""
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        logger.info(f"Model moved to device: {device}")
        return self
    
    def summary(self) -> str:
        """Get a human-readable summary of the model."""
        params = self.get_params()
        
        summary = f"""
NexusFlow Model Summary
======================
Model Class: {params['model_class']}
Total Parameters: {params['total_parameters']:,}

Architecture:
  - Encoders: {params['architecture']['num_encoders']}
  - Embedding Dimension: {params['architecture']['embed_dim']}
  - Refinement Iterations: {params['architecture']['refinement_iterations']}
  - Input Dimensions: {params['input_dimensions']}

Training Info:
  - Best Epoch: {params['training_info']['best_epoch']}
  - Best Validation Metric: {params['training_info']['best_val_metric']:.6f}
  - Training Complete: {params['training_info']['training_complete']}

Data Info:
  - Primary Key: {params['datasets_info']['primary_key']}
  - Target Column: {params['datasets_info']['target_column']}
  - Datasets: {params['datasets_info']['num_datasets']}
  - Preprocessors: {params['datasets_info']['has_preprocessors']}"""

        # Add optimization info if available
        if 'optimization' in params:
            opt_info = params['optimization']
            summary += f"""

Optimization Info:
  - Method: {opt_info['method']}
  - Size Reduction: {opt_info['size_reduction']:.1%}
  - Parameter Reduction: {opt_info['parameter_reduction']:.1%}
  - Original Size: {opt_info['original_size_mb']:.2f} MB
  - Optimized Size: {opt_info['optimized_size_mb']:.2f} MB"""

        return summary.strip()


def extract_pytorch_model(obj):
    """
    Safely extract the actual PyTorch model from any object.
    
    Args:
        obj: Could be a torch.nn.Module, NexusFlowModel, or nested artifact
    
    Returns:
        torch.nn.Module: The actual PyTorch model
    """
    # If it's already a PyTorch model, return it
    if isinstance(obj, torch.nn.Module) and not hasattr(obj, 'model'):
        return obj
    
    # If it's an artifact, extract the model
    current = obj
    max_depth = 10  # Prevent infinite loops
    depth = 0
    
    while hasattr(current, 'model') and depth < max_depth:
        current = current.model
        depth += 1
        
        # If we've found a PyTorch model, return it
        if isinstance(current, torch.nn.Module) and not hasattr(current, 'model'):
            return current
    
    # If we still don't have a proper model, raise an error
    raise ValueError(f"Could not extract PyTorch model from {type(obj)}")


def create_optimized_artifact(original_artifact, optimized_model, optimization_metadata):
    """
    Create a new NexusFlowModel with an optimized model.
    Maintains backward compatibility with the original function name.
    
    Args:
        original_artifact: Original NexusFlowModel
        optimized_model: Optimized PyTorch model
        optimization_metadata: Metadata about the optimization
    
    Returns:
        NexusFlowModel: New model with optimized model
    """
    # Ensure we have the actual PyTorch model
    pytorch_model = extract_pytorch_model(optimized_model)
    
    # Copy metadata and add optimization info
    new_meta = original_artifact.meta.copy()
    new_meta['optimization'] = optimization_metadata
    
    # Create new model
    return NexusFlowModel(pytorch_model, new_meta)


def create_optimized_model(original_model, optimized_pytorch_model, optimization_metadata):
    """
    Alias for create_optimized_artifact for consistency.
    
    Args:
        original_model: Original NexusFlowModel
        optimized_pytorch_model: Optimized PyTorch model
        optimization_metadata: Metadata about the optimization
    
    Returns:
        NexusFlowModel: New model with optimized PyTorch model
    """
    return create_optimized_artifact(original_model, optimized_pytorch_model, optimization_metadata)


def _create_model_from_meta(meta: Dict[str, Any]) -> nn.Module:
    """Helper function to create model from metadata."""
    config = meta.get('config', {})
    input_dims = meta.get('input_dims', [])
    
    if not input_dims:
        raise ValueError("Model metadata missing input dimensions")
    
    # Import and create model
    from nexusflow.model.nexus_former import NexusFormer
    
    # Extract architecture configuration
    arch_config = config.get('architecture', {})
    embed_dim = arch_config.get('global_embed_dim', 64)
    refinement_iterations = arch_config.get('refinement_iterations', 3)
    
    # Extract MoE and FlashAttention settings
    use_moe = arch_config.get('use_moe', False)
    num_experts = arch_config.get('num_experts', 4)
    use_flash_attn = arch_config.get('use_flash_attn', True)
    
    # Determine encoder type
    encoder_type = 'standard'
    if 'architecture_features' in meta:
        encoder_type = meta['architecture_features'].get('encoder_type', 'standard')
    elif 'datasets' in config and config['datasets']:
        transformer_types = {d.get('transformer_type', 'standard') for d in config['datasets']}
        if len(transformer_types) == 1:
            encoder_type = list(transformer_types)[0]
    
    logger.info(f"Loading model with: encoder_type={encoder_type}, use_moe={use_moe}, "
                f"num_experts={num_experts}, use_flash_attn={use_flash_attn}")
    
    model = NexusFormer(
        input_dims=input_dims,
        embed_dim=embed_dim, 
        refinement_iterations=refinement_iterations,
        encoder_type=encoder_type,
        use_moe=use_moe,
        num_experts=num_experts,
        use_flash_attn=use_flash_attn
    )
    
    return model


def load_model(path: str) -> NexusFlowModel:
    """
    Load a NexusFlow model from a .nxf file.
    This is the primary entry point for loading trained models.
    
    Args:
        path: Path to .nxf file
        
    Returns:
        NexusFlowModel: Loaded model ready for inference
    """
    p = Path(path)
    if not p.exists():
        logger.error(f"Model file not found: {p}")
        raise FileNotFoundError(p)
    
    # Load checkpoint
    ckpt = torch.load(p, map_location='cpu', weights_only=False)
    meta = ckpt.get('meta', {})
    
    # Check if this is an optimized model
    is_optimized = 'optimization' in meta
    optimization_method = meta.get('optimization', {}).get('method', '')
    
    if is_optimized:
        logger.info(f"Loading optimized model with {optimization_method}")
        
        if 'quantization' in optimization_method.lower():
            # For quantized models, we need to create the model and then apply quantization
            logger.info("Detected quantized model - applying post-load quantization")
            
            # First create the original model structure
            original_model = _create_model_from_meta(meta)
            
            # Apply quantization to get the right structure
            from torch.quantization import quantize_dynamic
            quantized_model = quantize_dynamic(
                original_model, 
                {nn.Linear}, 
                dtype=torch.qint8
            )
            
            # Now try to load the quantized state dict
            try:
                quantized_model.load_state_dict(ckpt['model_state'], strict=False)
                logger.info("Successfully loaded quantized model state")
            except Exception as e:
                logger.warning(f"Failed to load quantized state: {e}")
                # Fallback: use the quantized model structure but retrain might be needed
                logger.info("Using freshly quantized model structure")
            
            return NexusFlowModel(quantized_model, meta)
        
        elif 'pruning' in optimization_method.lower():
            # For pruned models, create original model and load state normally
            model = _create_model_from_meta(meta)
            model.load_state_dict(ckpt['model_state'])
            return NexusFlowModel(model, meta)
    
    else:
        # Original unoptimized model
        model = _create_model_from_meta(meta)
        model.load_state_dict(ckpt['model_state'])
        return NexusFlowModel(model, meta)