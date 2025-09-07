from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
from typing import List, Tuple, Optional, Union, TypedDict, Dict, Any
from loguru import logger

class AlignedData(TypedDict):
    """Structure for aligned relational data."""
    aligned_tables: Dict[str, pd.DataFrame]  # Tables with global_id column
    key_map: pd.DataFrame  # Mapping global_id to original keys
    metadata: Dict[str, Any]  # Join metadata and statistics

class NexusFlowDataset(Dataset):
    """
    Enhanced dataset that handles AlignedData structure with global_id indexing.
    """

    def __init__(self, aligned_data: AlignedData, target_col: str = 'churn_risk'):
        self.aligned_data = aligned_data
        self.target_col = target_col
        
        # Get the main table (should be the target table with expansions)
        self.main_table_name = aligned_data['metadata']['target_table']
        self.main_df = aligned_data['aligned_tables'][self.main_table_name]
        self.key_map = aligned_data['key_map']
        
        if self.target_col not in self.main_df.columns:
            raise KeyError(f"Target column '{self.target_col}' missing from main table")
        
        # Extract feature columns (exclude target and global_id)
        self.feature_cols = [c for c in self.main_df.columns 
                           if c not in [self.target_col, 'global_id']]
        
        # Extract key feature columns (global_id + all PK/FK columns)
        key_cols = ['global_id']
        for col in self.main_df.columns:
            if any(keyword in col.lower() for keyword in ['_id', 'key', 'pk', 'fk']):
                if col not in key_cols and col != 'global_id':
                    key_cols.append(col)
        
        self.key_feature_cols = key_cols
        
        # Simple fill for NaNs
        self.main_df = self.main_df.fillna(0)
        
        # Determine target dtype
        if pd.api.types.is_integer_dtype(self.main_df[self.target_col]):
            self.target_dtype = torch.long
        else:
            self.target_dtype = torch.float32
        
        # Multi-table metadata (for backward compatibility)
        self.feature_dimensions = self._calculate_feature_dimensions()
        
        logger.info(f"NexusFlowDataset initialized with {len(self)} samples")
        logger.info(f"  Feature columns: {len(self.feature_cols)}")
        logger.info(f"  Key feature columns: {len(self.key_feature_cols)}")

    def __len__(self):
        return len(self.main_df)

    def __getitem__(self, idx) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Returns separate feature tensors for each table, key_features, and target.
        
        Returns:
            Tuple of (feature_tensors_list, key_features_tensor, target_tensor)
        """
        # Get the main table row first
        main_row = self.main_df.iloc[idx]
        global_id = main_row['global_id']
        
        # Extract features from each aligned table separately
        feature_tensors = []
        
        # Process each aligned table
        for table_name, table_df in self.aligned_data['aligned_tables'].items():
            # Find the corresponding row in this table by global_id
            matching_rows = table_df[table_df['global_id'] == global_id]
            
            if len(matching_rows) == 0:
                # Create zero tensor if no matching row
                table_feature_count = len([col for col in table_df.columns 
                                        if col not in [self.target_col, 'global_id']])
                features = torch.zeros(table_feature_count, dtype=torch.float32)
            else:
                # Use the first matching row
                table_row = matching_rows.iloc[0]
                
                # Get feature columns for this table (exclude target and global_id)
                table_feature_cols = [col for col in table_df.columns 
                                    if col not in [self.target_col, 'global_id']]
                
                if table_feature_cols:
                    # Fill NaNs and convert to tensor
                    feature_values = table_row[table_feature_cols].fillna(0).values.astype('float32')
                    features = torch.tensor(feature_values)
                else:
                    # Empty feature tensor if no features in this table
                    features = torch.zeros(0, dtype=torch.float32)
            
            feature_tensors.append(features)
        
        # Extract key features (global_id and relational keys)
        key_features = []
        for col in self.key_feature_cols:
            if col == 'global_id':
                # Convert global_id to hash for numerical processing
                key_features.append(hash(main_row[col]) % (2**31))
            else:
                # Handle other key columns
                val = main_row[col] if not pd.isna(main_row[col]) else 0
                key_features.append(float(val) if isinstance(val, (int, float)) else hash(str(val)) % (2**31))
        
        key_features_tensor = torch.tensor(key_features, dtype=torch.float32)
        
        # Extract target
        target = torch.tensor(main_row[self.target_col], dtype=self.target_dtype)
        
        return feature_tensors, key_features_tensor, target

    def _calculate_feature_dimensions(self) -> List[int]:
        """Calculate feature dimensions for each table separately."""
        dimensions = []
        
        for table_name, table_df in self.aligned_data['aligned_tables'].items():
            # Count feature columns (exclude target and global_id)
            feature_count = len([col for col in table_df.columns 
                            if col not in [self.target_col, 'global_id']])
            dimensions.append(feature_count)
        
        logger.debug(f"Feature dimensions per table: {dimensions}")
        return dimensions

class MultiTableDataLoader:
    """
    Enhanced DataLoader that handles the new three-tensor output structure.
    """
    
    def __init__(self, dataset: NexusFlowDataset, batch_size: int = 32, shuffle: bool = False):
        from torch.utils.data import DataLoader
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Always use custom collate function for the enhanced structure
        self.dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            collate_fn=self._collate_enhanced_batch
        )
    
    def _collate_enhanced_batch(self, batch):
        """
        Enhanced collate function for the three-tensor structure.
        
        Args:
            batch: List of (feature_list, key_features, target) tuples
            
        Returns:
            (feature_tensors_list, key_features_batch, target_batch)
        """
        if not batch:
            return [], torch.tensor([]), torch.tensor([])
        
        feature_lists, key_features_list, targets = zip(*batch)
        
        # Stack features for each table separately
        num_tables = len(feature_lists[0])
        batched_features = []
        
        for table_idx in range(num_tables):
            table_features = [sample[table_idx] for sample in feature_lists]
            batched_table = torch.stack(table_features, dim=0)
            batched_features.append(batched_table)
        
        # Stack key features
        batched_key_features = torch.stack(key_features_list, dim=0)
        
        # Stack targets
        batched_targets = torch.stack(targets, dim=0)
        
        return batched_features, batched_key_features, batched_targets
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)