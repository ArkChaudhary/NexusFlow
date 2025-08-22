from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
from typing import List, Tuple, Optional

class NexusFlowDataset(Dataset):
    """
    Enhanced dataset that handles multiple tables for NexusFlow.
    
    This dataset can work in two modes:
    1. Single table mode (backwards compatible)
    2. Multi-table mode where features from different tables are split back into separate tensors
    """

    def __init__(self, df: pd.DataFrame, target_col: str = 'label'):
        self.df = df.copy().reset_index(drop=True)
        self.target_col = target_col

        if self.target_col not in self.df.columns:
            raise KeyError(f"Target column '{self.target_col}' missing from DataFrame")

        # Separate feature columns (exclude target)
        self.feature_cols = [c for c in self.df.columns if c != self.target_col]

        # Simple fill for NaNs
        self.df = self.df.fillna(0)

        # Decide target dtype once (classification vs regression)
        if pd.api.types.is_integer_dtype(self.df[self.target_col]):
            self.target_dtype = torch.long
        else:
            self.target_dtype = torch.float32
            
        # Multi-table metadata (set by create_multi_table_dataset if needed)
        self.feature_start_indices = None
        self.feature_dimensions = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns features and target for a single sample.
        
        If this is a multi-table dataset (has feature_dimensions), returns a list of tensors.
        Otherwise returns a single tensor (backwards compatible).
        """
        row = self.df.iloc[idx]
        features = torch.tensor(row[self.feature_cols].values.astype('float32'))
        target = torch.tensor(row[self.target_col], dtype=self.target_dtype)
        
        # If we have multi-table metadata, split features into separate tensors
        if self.feature_dimensions is not None:
            feature_list = []
            start_idx = 0
            for dim in self.feature_dimensions:
                end_idx = start_idx + dim
                feature_list.append(features[start_idx:end_idx])
                start_idx = end_idx
            return feature_list, target
        
        return features, target

class MultiTableDataLoader:
    """
    Custom DataLoader wrapper that handles batching of multi-table data.
    """
    
    def __init__(self, dataset: NexusFlowDataset, batch_size: int = 32, shuffle: bool = False):
        from torch.utils.data import DataLoader
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Use custom collate function for multi-table data
        if dataset.feature_dimensions is not None:
            self.dataloader = DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=shuffle,
                collate_fn=self._collate_multi_table
            )
        else:
            self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def _collate_multi_table(self, batch):
        """
        Custom collate function for multi-table batches.
        
        Args:
            batch: List of (feature_list, target) tuples
            
        Returns:
            (feature_tensors_list, target_tensor) where feature_tensors_list[i] 
            contains all samples for table i
        """
        if not batch:
            return [], torch.tensor([])
            
        feature_lists, targets = zip(*batch)
        
        # Stack features for each table separately
        num_tables = len(feature_lists[0])
        batched_features = []
        
        for table_idx in range(num_tables):
            table_features = [sample[table_idx] for sample in feature_lists]
            batched_table = torch.stack(table_features, dim=0)
            batched_features.append(batched_table)
        
        # Stack targets
        batched_targets = torch.stack(targets, dim=0)
        
        return batched_features, batched_targets
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)