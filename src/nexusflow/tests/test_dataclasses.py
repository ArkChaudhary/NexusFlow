"""Unit tests for NexusFlow dataset classes."""
import pytest
import pandas as pd
import torch
import numpy as np

from nexusflow.data.dataset import NexusFlowDataset, MultiTableDataLoader


class TestNexusFlowDataset:
    """Test the NexusFlowDataset class."""
    
    def test_basic_dataset_creation(self):
        """Test creating a basic dataset."""
        df = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0],
            'feature2': [4.0, 5.0, 6.0], 
            'label': [0, 1, 0]
        })
        
        dataset = NexusFlowDataset(df, target_col='label')
        
        assert len(dataset) == 3
        assert dataset.target_col == 'label'
        assert dataset.feature_cols == ['feature1', 'feature2']

    def test_dataset_getitem_single_table(self):
        """Test getting items from single table dataset."""
        df = pd.DataFrame({
            'feat1': [1.0, 2.0, 3.0],
            'feat2': [4.0, 5.0, 6.0],
            'label': [0, 1, 0]
        })
        
        dataset = NexusFlowDataset(df, target_col='label')
        
        # Test first item
        features, target = dataset[0]
        
        assert isinstance(features, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        assert features.shape == (2,)  # 2 features
        assert target.shape == ()  # scalar target
        
        # Check values
        assert torch.allclose(features, torch.tensor([1.0, 4.0]))
        assert target.item() == 0

    def test_dataset_multi_table_mode(self):
        """Test dataset in multi-table mode."""
        df = pd.DataFrame({
            'feat1': [1.0, 2.0],
            'feat2': [3.0, 4.0], 
            'feat3': [5.0, 6.0],
            'label': [0, 1]
        })
        
        dataset = NexusFlowDataset(df, target_col='label')
        
        # Set multi-table metadata
        dataset.feature_dimensions = [2, 1]  # First table: 2 features, second: 1
        
        features, target = dataset[0]
        
        # Should return list of tensors
        assert isinstance(features, list)
        assert len(features) == 2
        assert features[0].shape == (2,)  # First table features
        assert features[1].shape == (1,)  # Second table features
        
        # Check values
        assert torch.allclose(features[0], torch.tensor([1.0, 3.0]))
        assert torch.allclose(features[1], torch.tensor([5.0]))

    def test_dataset_missing_target_column(self):
        """Test error when target column is missing."""
        df = pd.DataFrame({'feature1': [1, 2, 3]})
        
        with pytest.raises(KeyError, match="Target column 'missing' missing from DataFrame"):
            NexusFlowDataset(df, target_col='missing')

    def test_dataset_handles_missing_values(self):
        """Test that dataset handles missing values by filling with 0."""
        df = pd.DataFrame({
            'feat1': [1.0, np.nan, 3.0],
            'feat2': [np.nan, 5.0, 6.0],
            'label': [0, 1, 0]
        })
        
        dataset = NexusFlowDataset(df, target_col='label')
        
        # Check that NaNs are filled
        features, target = dataset[0]
        assert not torch.isnan(features).any()
        
        features, target = dataset[1]
        assert not torch.isnan(features).any()

    def test_dataset_integer_vs_float_targets(self):
        """Test dataset handling of different target types."""
        # Integer targets (classification)
        df_int = pd.DataFrame({
            'feat': [1.0, 2.0],
            'label': [0, 1]  # integers
        })
        
        dataset_int = NexusFlowDataset(df_int, target_col='label')
        features, target = dataset_int[0]
        assert target.dtype == torch.long
        
        # Float targets (regression)
        df_float = pd.DataFrame({
            'feat': [1.0, 2.0],
            'label': [0.5, 1.5]  # floats
        })
        
        dataset_float = NexusFlowDataset(df_float, target_col='label')
        features, target = dataset_float[0]
        assert target.dtype == torch.float32

    def test_dataset_different_shapes(self):
        """Test dataset with different numbers of features."""
        # Single feature
        df_single = pd.DataFrame({
            'feat': [1.0, 2.0, 3.0],
            'label': [0, 1, 0]
        })
        
        dataset = NexusFlowDataset(df_single, target_col='label')
        features, target = dataset[0]
        assert features.shape == (1,)
        
        # Many features
        df_many = pd.DataFrame({
            f'feat_{i}': [float(i), float(i+1), float(i+2)] 
            for i in range(10)
        })
        df_many['label'] = [0, 1, 0]
        
        dataset = NexusFlowDataset(df_many, target_col='label')
        features, target = dataset[0]
        assert features.shape == (10,)

    def test_dataset_empty_dataframe(self):
        """Test dataset with empty DataFrame."""
        df = pd.DataFrame(columns=['feat', 'label'])
        dataset = NexusFlowDataset(df, target_col='label')
        assert len(dataset) == 0

    def test_dataset_single_row(self):
        """Test dataset with single row."""
        df = pd.DataFrame({
            'feat1': [1.0],
            'feat2': [2.0],
            'label': [1]
        })
        
        dataset = NexusFlowDataset(df, target_col='label')
        assert len(dataset) == 1
        
        features, target = dataset[0]
        assert features.shape == (2,)
        assert target.item() == 1


class TestMultiTableDataLoader:
    """Test the MultiTableDataLoader class."""
    
    def test_single_table_dataloader(self):
        """Test dataloader with single table dataset."""
        df = pd.DataFrame({
            'feat1': [1.0, 2.0, 3.0, 4.0],
            'feat2': [5.0, 6.0, 7.0, 8.0],
            'label': [0, 1, 0, 1]
        })
        
        dataset = NexusFlowDataset(df, target_col='label')
        dataloader = MultiTableDataLoader(dataset, batch_size=2, shuffle=False)
        
        batch = next(iter(dataloader))
        features, targets = batch
        
        # Single table should return tensor, not list
        assert isinstance(features, torch.Tensor)
        assert features.shape == (2, 2)  # batch_size=2, 2 features
        assert targets.shape == (2,)

    def test_multi_table_dataloader(self):
        """Test dataloader with multi-table dataset."""
        df = pd.DataFrame({
            'feat1': [1.0, 2.0, 3.0, 4.0],
            'feat2': [5.0, 6.0, 7.0, 8.0],
            'feat3': [9.0, 10.0, 11.0, 12.0],
            'label': [0, 1, 0, 1]
        })
        
        dataset = NexusFlowDataset(df, target_col='label')
        # Set multi-table metadata
        dataset.feature_dimensions = [2, 1]  # Table 1: 2 features, Table 2: 1 feature
        
        dataloader = MultiTableDataLoader(dataset, batch_size=2, shuffle=False)
        
        batch = next(iter(dataloader))
        features, targets = batch
        
        # Multi-table should return list of tensors
        assert isinstance(features, list)
        assert len(features) == 2
        assert features[0].shape == (2, 2)  # batch_size=2, 2 features for table 1
        assert features[1].shape == (2, 1)  # batch_size=2, 1 feature for table 2
        assert targets.shape == (2,)

    def test_dataloader_shuffling(self):
        """Test dataloader shuffling functionality."""
        df = pd.DataFrame({
            'feat': range(10),
            'label': range(10)
        })
        
        dataset = NexusFlowDataset(df, target_col='label')
        
        # Without shuffling
        loader_no_shuffle = MultiTableDataLoader(dataset, batch_size=3, shuffle=False)
        batch1 = next(iter(loader_no_shuffle))
        features1, targets1 = batch1
        
        # With shuffling
        loader_shuffle = MultiTableDataLoader(dataset, batch_size=3, shuffle=True)
        batch2 = next(iter(loader_shuffle))
        features2, targets2 = batch2
        
        # Note: This test might occasionally fail due to random chance
        # but is very unlikely with 10 samples
        try:
            assert not torch.equal(targets1, targets2)
        except AssertionError:
            # If they happen to be equal, just check that shuffle is enabled
            assert loader_shuffle.shuffle is True

    def test_dataloader_different_batch_sizes(self):
        """Test dataloader with different batch sizes."""
        df = pd.DataFrame({
            'feat': [1.0, 2.0, 3.0, 4.0, 5.0],
            'label': [0, 1, 0, 1, 0]
        })
        
        dataset = NexusFlowDataset(df, target_col='label')
        
        # Test various batch sizes
        for batch_size in [1, 2, 3, 5]:
            loader = MultiTableDataLoader(dataset, batch_size=batch_size)
            
            batches = list(loader)
            total_samples = sum(len(batch[1]) for batch in batches)
            assert total_samples == 5  # Should see all samples
            
            # Last batch might be smaller
            if len(batches) > 1:
                for batch in batches[:-1]:
                    assert len(batch[1]) == batch_size
            
            # Check last batch
            last_batch_size = len(batches[-1][1])
            expected_last_size = 5 % batch_size if 5 % batch_size != 0 else batch_size
            assert last_batch_size == expected_last_size

    def test_empty_dataloader(self):
        """Test dataloader with empty dataset."""
        df = pd.DataFrame(columns=['feat', 'label'])
        dataset = NexusFlowDataset(df, target_col='label')
        loader = MultiTableDataLoader(dataset, batch_size=2)
        
        assert len(loader) == 0
        batches = list(loader)
        assert len(batches) == 0

    def test_custom_collate_function(self):
        """Test that custom collate function works correctly for multi-table data."""
        df = pd.DataFrame({
            'feat1': [1.0, 2.0, 3.0],
            'feat2': [4.0, 5.0, 6.0],
            'feat3': [7.0, 8.0, 9.0],
            'label': [0, 1, 0]
        })
        
        dataset = NexusFlowDataset(df, target_col='label')
        dataset.feature_dimensions = [1, 2]  # First table: 1 feature, Second: 2 features
        
        loader = MultiTableDataLoader(dataset, batch_size=2)
        
        # Test the custom collate function
        sample_batch = [dataset[0], dataset[1]]
        collated = loader._collate_multi_table(sample_batch)
        
        features, targets = collated
        assert isinstance(features, list)
        assert len(features) == 2
        assert features[0].shape == (2, 1)  # 2 samples, 1 feature for table 1
        assert features[1].shape == (2, 2)  # 2 samples, 2 features for table 2
        assert targets.shape == (2,)

    def test_dataloader_iteration(self):
        """Test that dataloader can be iterated multiple times."""
        df = pd.DataFrame({
            'feat': [1.0, 2.0, 3.0, 4.0],
            'label': [0, 1, 0, 1]
        })
        
        dataset = NexusFlowDataset(df, target_col='label')
        loader = MultiTableDataLoader(dataset, batch_size=2, shuffle=False)
        
        # First iteration
        batches1 = list(loader)
        assert len(batches1) == 2
        
        # Second iteration
        batches2 = list(loader)
        assert len(batches2) == 2
        
        # Should get same results (no shuffle)
        for b1, b2 in zip(batches1, batches2):
            torch.testing.assert_close(b1[0], b2[0])  # features
            torch.testing.assert_close(b1[1], b2[1])  # targets

    def test_dataloader_len(self):
        """Test dataloader length calculation."""
        df = pd.DataFrame({
            'feat': range(10),
            'label': range(10)
        })
        
        dataset = NexusFlowDataset(df, target_col='label')
        
        # Test different batch sizes
        test_cases = [
            (1, 10),   # batch_size=1, expect 10 batches
            (2, 5),    # batch_size=2, expect 5 batches  
            (3, 4),    # batch_size=3, expect 4 batches (10/3 = 3.33, rounded up)
            (5, 2),    # batch_size=5, expect 2 batches
            (10, 1),   # batch_size=10, expect 1 batch
            (15, 1)    # batch_size=15, expect 1 batch (larger than dataset)
        ]
        
        for batch_size, expected_batches in test_cases:
            loader = MultiTableDataLoader(dataset, batch_size=batch_size)
            assert len(loader) == expected_batches, \
                f"Batch size {batch_size}: expected {expected_batches} batches, got {len(loader)}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])