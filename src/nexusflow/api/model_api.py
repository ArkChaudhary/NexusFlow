import torch
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from typing import Dict, Any, List, Union, Optional
import json

class NexusFlowModelArtifact:
    """
    Complete model artifact that implements the required .predict() and .get_params() interface.
    This is what gets saved as a .nxf file and provides the end-user API.
    """
    
    def __init__(self, model: torch.nn.Module, preprocess_meta: Dict[str, Any]):
        """
        Initialize the model artifact.
        
        Args:
            model: Trained NexusFormer model
            preprocess_meta: Metadata including config, input dimensions, training info
        """
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
        
        logger.info(f"NexusFlow model artifact initialized with {len(self.input_dims)} input dimensions")
        if self.preprocessors:
            logger.info(f"Preprocessors available for: {list(self.preprocessors.keys())}")
    
    def predict(self, data: Union[Dict[str, pd.DataFrame], List[pd.DataFrame], pd.DataFrame]) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            data: Input data in one of three formats:
                1. Dict[str, DataFrame] - Multiple tables keyed by dataset name
                2. List[DataFrame] - Multiple tables in order matching training
                3. DataFrame - Single flattened table (backwards compatibility)
        
        Returns:
            numpy array of predictions
        """
        logger.info("Making predictions on new data...")
        
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
NexusFlow Model Artifact Summary
================================
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

class ModelAPI:
    """
    Enhanced ModelAPI for saving/loading .nxf artifacts.
    """
    
    def __init__(self, model: torch.nn.Module, preprocess_meta: Optional[Dict[str, Any]] = None):
        """Initialize ModelAPI with model and metadata."""
        self.artifact = NexusFlowModelArtifact(model, preprocess_meta or {})

    def save(self, path: str):
        """Save model artifact to .nxf file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        
        # Ensure .nxf extension
        if not p.suffix == '.nxf':
            p = p.with_suffix('.nxf')
        
        # Check if model has been optimized and update metadata accordingly
        meta = self.artifact.meta.copy()
        if 'optimization' in meta:
            optimization_info = meta['optimization']
            logger.info(f"Saving optimized model with {optimization_info['method']}")
            logger.info(f"  Size reduction: {optimization_info['size_reduction']:.1%}")
            logger.info(f"  Parameter reduction: {optimization_info['parameter_reduction']:.1%}")
        
        save_data = {
            'model_state': self.artifact.model.state_dict(),
            'meta': meta,
            'artifact_version': '1.0'
        }
        
        torch.save(save_data, p)
        logger.info(f"NexusFlow model artifact saved to: {p}")

    @staticmethod
    def load(path: str) -> NexusFlowModelArtifact:
        """Load model artifact from .nxf file."""
        p = Path(path)
        if not p.exists():
            logger.error(f"Model file not found: {p}")
            raise FileNotFoundError(p)
        
        # Load checkpoint
        ckpt = torch.load(p, map_location='cpu', weights_only=False)
        meta = ckpt.get('meta', {})
        
        # Reconstruct model from metadata
        config = meta.get('config', {})
        input_dims = meta.get('input_dims', [])
        
        if not input_dims:
            raise ValueError("Model metadata missing input dimensions")
        
        # Import and create model (assuming NexusFormer is available)
        from nexusflow.model.nexus_former import NexusFormer
        
        # Extract architecture configuration
        arch_config = config.get('architecture', {})
        embed_dim = arch_config.get('global_embed_dim', 64)
        refinement_iterations = arch_config.get('refinement_iterations', 3)
        
        # Extract MoE and FlashAttention settings
        use_moe = arch_config.get('use_moe', False)
        num_experts = arch_config.get('num_experts', 4)
        use_flash_attn = arch_config.get('use_flash_attn', True)
        
        # Determine encoder type from metadata
        encoder_type = 'standard'  # default
        if 'architecture_features' in meta:
            encoder_type = meta['architecture_features'].get('encoder_type', 'standard')
        elif 'datasets' in config and config['datasets']:
            # Infer from dataset configs
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
        
        model.load_state_dict(ckpt['model_state'])
        
        logger.info(f"NexusFlow model artifact loaded from: {p}")
        if 'preprocessors' in meta:
            logger.info(f"Loaded with preprocessors for: {list(meta['preprocessors'].keys())}")
        
        return NexusFlowModelArtifact(model, meta)

    def predict(self, inputs):
        """Delegate to artifact predict method."""
        return self.artifact.predict(inputs)
    
    def get_params(self):
        """Delegate to artifact get_params method."""
        return self.artifact.get_params()

# Convenience function for loading .nxf files
def load_model(path: str) -> NexusFlowModelArtifact:
    """
    Convenience function to load a NexusFlow model artifact.
    
    Args:
        path: Path to .nxf file
        
    Returns:
        NexusFlowModelArtifact instance ready for inference
    """
    return ModelAPI.load(path)