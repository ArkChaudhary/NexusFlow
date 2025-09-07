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

    def _calculate_feature_dimensions(self) -> List[int]:
        """Calculate feature dimensions for backward compatibility."""
        # For now, return single dimension representing all features
        return [len(self.feature_cols)]

    def __len__(self):
        return len(self.main_df)

    def __getitem__(self, idx) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Returns features, key_features, and target for a single sample.
        
        Returns:
            Tuple of (feature_tensors_list, key_features_tensor, target_tensor)
        """
        row = self.main_df.iloc[idx]
        
        # Extract regular features
        features = torch.tensor(row[self.feature_cols].values.astype('float32'))
        
        # Extract key features (global_id and relational keys)
        key_features = []
        for col in self.key_feature_cols:
            if col == 'global_id':
                # Convert global_id to hash for numerical processing
                key_features.append(hash(row[col]) % (2**31))
            else:
                # Handle other key columns
                val = row[col] if not pd.isna(row[col]) else 0
                key_features.append(float(val) if isinstance(val, (int, float)) else hash(str(val)) % (2**31))
        
        key_features_tensor = torch.tensor(key_features, dtype=torch.float32)
        
        # Extract target
        target = torch.tensor(row[self.target_col], dtype=self.target_dtype)
        
        # Return as list for backward compatibility with MultiTableDataLoader
        feature_list = [features]  # Single table for now
        
        return feature_list, key_features_tensor, target

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