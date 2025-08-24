"""Unit tests for NexusFlow data ingestion utilities."""
import pytest
import pandas as pd
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from nexusflow.data.ingestion import (
    load_table, validate_primary_key, load_datasets, align_datasets,
    split_df, get_feature_dimensions, create_multi_table_dataset, make_dataloaders
)
from nexusflow.config import ConfigModel
from nexusflow.data.dataset import NexusFlowDataset


class TestLoadTable:
    """Test table loading functionality."""
    
    def test_load_valid_csv(self):
        """Test loading a valid CSV file."""
        data = {'id': [1, 2, 3], 'name': ['A', 'B', 'C'], 'value': [10, 20, 30]}
        df = pd.DataFrame(data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            loaded_df = load_table(temp_path)
            pd.testing.assert_frame_equal(loaded_df, df)
        finally:
            os.unlink(temp_path)

    def test_load_nonexistent_file(self):
        """Test that loading non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_table('nonexistent.csv')

    def test_load_csv_with_missing_values(self):
        """Test loading CSV with missing values."""
        data = {'id': [1, 2, 3], 'name': ['A', None, 'C'], 'value': [10, 20, None]}
        df = pd.DataFrame(data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            loaded_df = load_table(temp_path)
            assert loaded_df.isnull().sum().sum() == 2  # 2 missing values
        finally:
            os.unlink(temp_path)


class TestValidatePrimaryKey:
    """Test primary key validation."""
    
    def test_valid_primary_key(self):
        """Test validation of a valid primary key."""
        df = pd.DataFrame({'id': [1, 2, 3], 'value': [10, 20, 30]})
        assert validate_primary_key(df, 'id') is True

    def test_missing_primary_key_column(self):
        """Test that missing primary key column raises KeyError."""
        df = pd.DataFrame({'value': [10, 20, 30]})
        with pytest.raises(KeyError):
            validate_primary_key(df, 'id')

    def test_duplicate_primary_key_values(self):
        """Test that duplicate primary key values return False."""
        df = pd.DataFrame({'id': [1, 1, 2], 'value': [10, 20, 30]})
        assert validate_primary_key(df, 'id') is False

    def test_null_primary_key_values(self):
        """Test that null primary key values return False."""
        df = pd.DataFrame({'id': [1, None, 3], 'value': [10, 20, 30]})
        assert validate_primary_key(df, 'id') is False


class TestAlignDatasets:
    """Test dataset alignment functionality."""
    
    def test_align_datasets_with_common_keys(self):
        """Test aligning datasets with common primary keys."""
        df1 = pd.DataFrame({'id': [1, 2, 3], 'feature_a': [10, 20, 30]})
        df2 = pd.DataFrame({'id': [2, 3, 4], 'feature_b': [100, 200, 300]})
        
        datasets = {'table1': df1, 'table2': df2}
        aligned = align_datasets(datasets, 'id')
        
        # Should only keep rows with id 2, 3 (common between both)
        assert len(aligned['table1']) == 2
        assert len(aligned['table2']) == 2
        assert list(aligned['table1']['id']) == [2, 3]
        assert list(aligned['table2']['id']) == [2, 3]

    def test_align_datasets_no_common_keys(self):
        """Test that datasets with no common keys raise ValueError."""
        df1 = pd.DataFrame({'id': [1, 2], 'feature_a': [10, 20]})
        df2 = pd.DataFrame({'id': [3, 4], 'feature_b': [100, 200]})
        
        datasets = {'table1': df1, 'table2': df2}
        with pytest.raises(ValueError, match="No common primary key values"):
            align_datasets(datasets, 'id')

    def test_align_empty_datasets(self):
        """Test that empty datasets dict raises ValueError."""
        with pytest.raises(ValueError, match="No datasets provided"):
            align_datasets({}, 'id')


class TestSplitDf:
    """Test DataFrame splitting functionality."""
    
    def test_split_with_randomization(self):
        """Test splitting DataFrame with randomization."""
        df = pd.DataFrame({'id': range(100), 'value': range(100)})
        train, val, test = split_df(df, test_size=0.2, val_size=0.1, randomize=True)
        
        # Check sizes
        assert len(train) == 70  # 100 * (1 - 0.2 - 0.1) = 70
        assert len(val) == 10    # 100 * 0.1 = 10  
        assert len(test) == 20   # 100 * 0.2 = 20
        
        # Check no overlap
        all_ids = set(train['id']).union(set(val['id'])).union(set(test['id']))
        assert len(all_ids) == 100

    def test_split_without_randomization(self):
        """Test splitting DataFrame without randomization."""
        df = pd.DataFrame({'id': range(10), 'value': range(10)})
        train, val, test = split_df(df, test_size=0.2, val_size=0.1, randomize=False)
        
        # Check sizes
        assert len(train) == 7
        assert len(val) == 1
        assert len(test) == 2
        
        # Check order (should be sequential)
        assert list(train['id']) == list(range(7))
        assert list(val['id']) == [7]
        assert list(test['id']) == [8, 9]

    def test_split_no_validation(self):
        """Test splitting with no validation set."""
        df = pd.DataFrame({'id': range(10), 'value': range(10)})
        train, val, test = split_df(df, test_size=0.2, val_size=0.0, randomize=False)
        
        assert len(train) == 8
        assert len(val) == 0
        assert len(test) == 2


class TestGetFeatureDimensions:
    """Test feature dimension calculation."""
    
    def test_get_feature_dimensions_basic(self):
        """Test calculating feature dimensions for basic datasets."""
        df1 = pd.DataFrame({'id': [1, 2], 'feat1': [10, 20], 'feat2': [30, 40]})
        df2 = pd.DataFrame({'id': [1, 2], 'feat3': [50, 60]})
        
        datasets = {'table1': df1, 'table2': df2}
        dims = get_feature_dimensions(datasets, 'id', 'target')
        
        # df1 has 2 features (feat1, feat2), df2 has 1 feature (feat3)
        assert dims == [2, 1]

    def test_get_feature_dimensions_with_target(self):
        """Test calculating feature dimensions when target column is present."""
        df1 = pd.DataFrame({
            'id': [1, 2], 
            'feat1': [10, 20], 
            'feat2': [30, 40], 
            'target': [0, 1]
        })
        df2 = pd.DataFrame({'id': [1, 2], 'feat3': [50, 60]})
        
        datasets = {'table1': df1, 'table2': df2}
        dims = get_feature_dimensions(datasets, 'id', 'target')
        
        # df1 has 2 features (excluding id and target), df2 has 1 feature
        assert dims == [2, 1]


class TestCreateMultiTableDataset:
    """Test multi-table dataset creation."""
    
    def test_create_multi_table_dataset_basic(self):
        """Test creating a basic multi-table dataset."""
        # Create mock datasets
        df1 = pd.DataFrame({
            'id': [1, 2, 3],
            'feat1': [10, 20, 30],
            'feat2': [40, 50, 60],
            'label': [0, 1, 0]
        })
        df2 = pd.DataFrame({
            'id': [1, 2, 3], 
            'feat3': [70, 80, 90]
        })
        
        datasets = {'table1.csv': df1, 'table2.csv': df2}
        
        # Create mock config with ALL required fields
        config_data = {
            'project_name': 'test_project',  # Added required field
            'primary_key': 'id',
            'target': {
                'target_table': 'table1.csv',
                'target_column': 'label'
            },
            'architecture': {  # Added required field
                'global_embed_dim': 64,
                'refinement_iterations': 2
            },
            'datasets': [
                {'name': 'table1.csv'},
                {'name': 'table2.csv'}
            ]
        }
        config = ConfigModel.model_validate(config_data)
        
        dataset = create_multi_table_dataset(datasets, config)
        
        assert isinstance(dataset, NexusFlowDataset)
        assert len(dataset) == 3
        assert dataset.feature_dimensions == [2, 1]  # table1: 2 features, table2: 1 feature

    def test_create_multi_table_dataset_missing_target_table(self):
        """Test error when target table is missing."""
        datasets = {'table1.csv': pd.DataFrame({'id': [1], 'feat': [10]})}
        
        config_data = {
            'project_name': 'test_project',  # Added required field
            'primary_key': 'id',
            'target': {
                'target_table': 'missing.csv',
                'target_column': 'label'
            },
            'architecture': {  # Added required field
                'global_embed_dim': 64,
                'refinement_iterations': 2
            },
            'datasets': [{'name': 'table1.csv'}]
        }
        config = ConfigModel.model_validate(config_data)
        
        with pytest.raises(KeyError, match="Target table 'missing.csv' not found"):
            create_multi_table_dataset(datasets, config)

    def test_create_multi_table_dataset_missing_target_column(self):
        """Test error when target column is missing."""
        df = pd.DataFrame({'id': [1], 'feat': [10]})
        datasets = {'table1.csv': df}
        
        config_data = {
            'project_name': 'test_project',  # Added required field
            'primary_key': 'id',
            'target': {
                'target_table': 'table1.csv',
                'target_column': 'missing_label'
            },
            'architecture': {  # Added required field
                'global_embed_dim': 64,
                'refinement_iterations': 2
            },
            'datasets': [{'name': 'table1.csv'}]
        }
        config = ConfigModel.model_validate(config_data)
        
        with pytest.raises(KeyError, match="Target column 'missing_label' not found"):
            create_multi_table_dataset(datasets, config)


class TestLoadDatasets:
    """Test dataset loading with mocks."""
    
    @patch('nexusflow.data.ingestion.load_table')
    @patch('nexusflow.data.ingestion.validate_primary_key')
    def test_load_datasets_success(self, mock_validate, mock_load):
        """Test successful dataset loading."""
        # Mock return values
        df1 = pd.DataFrame({'id': [1, 2], 'feat1': [10, 20]})
        df2 = pd.DataFrame({'id': [1, 2], 'feat2': [30, 40]})
        mock_load.side_effect = [df1, df2]
        mock_validate.return_value = True
        
        config_data = {
            'project_name': 'test_project',  # Added required field
            'primary_key': 'id',
            'target': {  # Added required field
                'target_table': 'table1.csv',
                'target_column': 'label'
            },
            'architecture': {  # Added required field
                'global_embed_dim': 64,
                'refinement_iterations': 2
            },
            'datasets': [
                {'name': 'table1.csv'},
                {'name': 'table2.csv'}
            ]
        }
        config = ConfigModel.model_validate(config_data)
        
        datasets = load_datasets(config)
        
        assert len(datasets) == 2
        assert 'table1.csv' in datasets
        assert 'table2.csv' in datasets
        assert mock_load.call_count == 2
        assert mock_validate.call_count == 2

    @patch('nexusflow.data.ingestion.load_table')
    def test_load_datasets_file_not_found(self, mock_load):
        """Test dataset loading with missing file."""
        mock_load.side_effect = FileNotFoundError("File not found")
        
        config_data = {
            'project_name': 'test_project',  # Added required field
            'primary_key': 'id',
            'target': {  # Added required field
                'target_table': 'missing.csv',
                'target_column': 'label'
            },
            'architecture': {  # Added required field
                'global_embed_dim': 64,
                'refinement_iterations': 2
            },
            'datasets': [{'name': 'missing.csv'}]
        }
        config = ConfigModel.model_validate(config_data)
        
        with pytest.raises(FileNotFoundError):
            load_datasets(config)


class TestMakeDataLoaders:
    """Test DataLoader creation."""
    
    def test_make_dataloaders_basic(self):
        """Test creating basic DataLoaders."""
        # Create aligned datasets
        df1 = pd.DataFrame({
            'id': range(10),
            'feat1': range(10, 20), 
            'label': [0, 1] * 5
        })
        df2 = pd.DataFrame({
            'id': range(10),
            'feat2': range(20, 30)
        })
        
        datasets = {'table1.csv': df1, 'table2.csv': df2}
        
        config_data = {
            'project_name': 'test_project',  # Added required field
            'primary_key': 'id',
            'target': {
                'target_table': 'table1.csv',
                'target_column': 'label'
            },
            'architecture': {  # Added required field
                'global_embed_dim': 64,
                'refinement_iterations': 2
            },
            'datasets': [
                {'name': 'table1.csv'},
                {'name': 'table2.csv'}
            ],
            'training': {
                'batch_size': 4,
                'split_config': {
                    'test_size': 0.2,
                    'validation_size': 0.2,
                    'randomize': False
                }
            }
        }
        config = ConfigModel.model_validate(config_data)
        
        train_loader, val_loader, test_loader = make_dataloaders(config, datasets)
        
        assert train_loader is not None
        assert val_loader is not None  
        assert test_loader is not None
        
        # Check that we can iterate through the loaders
        train_batch = next(iter(train_loader))
        assert len(train_batch) == 2  # features and targets
        features, targets = train_batch
        assert len(features) == 2  # Two datasets


if __name__ == '__main__':
    pytest.main([__file__, '-v'])