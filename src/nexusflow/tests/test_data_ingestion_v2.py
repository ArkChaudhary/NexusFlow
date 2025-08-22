import pytest
import pandas as pd
import tempfile
import os
from pathlib import Path

from nexusflow.data.ingestion import (
    load_datasets, align_datasets, get_feature_dimensions, 
    make_dataloaders, create_multi_table_dataset
)
from nexusflow.config import ConfigModel

@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    return ConfigModel(
        project_name="test_project",
        primary_key="id",
        target={"target_table": "table_a.csv", "target_column": "label"},
        architecture={"global_embed_dim": 32, "refinement_iterations": 1},
        datasets=[
            {"name": "table_a.csv"},
            {"name": "table_b.csv"}
        ],
        training={"batch_size": 4, "epochs": 1},
        mlops={"logging_provider": "stdout"}
    )

@pytest.fixture
def sample_csvs(tmp_path):
    """Create sample CSV files for testing."""
    # Table A with target column
    df_a = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'feature_a1': [0.1, 0.2, 0.3, 0.4, 0.5],
        'feature_a2': [1.1, 1.2, 1.3, 1.4, 1.5],
        'label': [0, 1, 0, 1, 1]
    })
    
    # Table B with overlapping IDs
    df_b = pd.DataFrame({
        'id': [1, 2, 3, 6, 7],  # IDs 6,7 won't align
        'feature_b1': [2.1, 2.2, 2.3, 2.4, 2.5],
        'feature_b2': [3.1, 3.2, 3.3, 3.4, 3.5]
    })
    
    # Create datasets directory
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    
    # Save CSV files
    path_a = datasets_dir / "table_a.csv"
    path_b = datasets_dir / "table_b.csv"
    df_a.to_csv(path_a, index=False)
    df_b.to_csv(path_b, index=False)
    
    # Change working directory to tmp_path so relative paths work
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    
    yield path_a, path_b, df_a, df_b
    
    # Restore working directory
    os.chdir(old_cwd)

def test_load_datasets(sample_config, sample_csvs):
    """Test loading multiple datasets."""
    path_a, path_b, df_a, df_b = sample_csvs
    
    datasets = load_datasets(sample_config)
    
    assert len(datasets) == 2
    assert "table_a.csv" in datasets
    assert "table_b.csv" in datasets
    
    # Check that data loaded correctly
    assert len(datasets["table_a.csv"]) == 5
    assert len(datasets["table_b.csv"]) == 5
    assert list(datasets["table_a.csv"]["id"]) == [1, 2, 3, 4, 5]
    assert list(datasets["table_b.csv"]["id"]) == [1, 2, 3, 6, 7]

def test_align_datasets(sample_config, sample_csvs):
    """Test dataset alignment by primary key."""
    path_a, path_b, df_a, df_b = sample_csvs
    
    datasets = load_datasets(sample_config)
    aligned = align_datasets(datasets, sample_config.primary_key)
    
    # Should only have 3 common IDs: 1, 2, 3
    assert len(aligned["table_a.csv"]) == 3
    assert len(aligned["table_b.csv"]) == 3
    
    # Check that IDs match across datasets
    ids_a = set(aligned["table_a.csv"]["id"])
    ids_b = set(aligned["table_b.csv"]["id"])
    assert ids_a == ids_b == {1, 2, 3}
    
    # Check that rows are sorted by ID
    assert list(aligned["table_a.csv"]["id"]) == [1, 2, 3]
    assert list(aligned["table_b.csv"]["id"]) == [1, 2, 3]

def test_get_feature_dimensions(sample_config, sample_csvs):
    """Test calculation of feature dimensions for each dataset."""
    path_a, path_b, df_a, df_b = sample_csvs
    
    datasets = load_datasets(sample_config)
    aligned = align_datasets(datasets, sample_config.primary_key)
    
    dimensions = get_feature_dimensions(
        aligned, 
        sample_config.primary_key, 
        sample_config.target['target_column']
    )
    
    # Table A: 2 features (feature_a1, feature_a2) - excludes id and label
    # Table B: 2 features (feature_b1, feature_b2) - excludes id
    assert dimensions == [2, 2]

def test_create_multi_table_dataset(sample_config, sample_csvs):
    """Test creation of multi-table dataset."""
    path_a, path_b, df_a, df_b = sample_csvs
    
    datasets = load_datasets(sample_config)
    aligned = align_datasets(datasets, sample_config.primary_key)
    
    dataset = create_multi_table_dataset(aligned, sample_config)
    
    # Should have 3 aligned samples
    assert len(dataset) == 3
    
    # Check that feature dimensions are stored
    assert dataset.feature_dimensions == [2, 2]
    
    # Test getting a sample
    features, target = dataset[0]
    assert isinstance(features, list)
    assert len(features) == 2  # Two datasets
    assert features[0].shape == (2,)  # Table A features
    assert features[1].shape == (2,)  # Table B features
    assert target.item() in [0, 1]  # Binary target

def test_make_dataloaders(sample_config, sample_csvs):
    """Test creation of DataLoaders."""
    path_a, path_b, df_a, df_b = sample_csvs
    
    datasets = load_datasets(sample_config)
    aligned = align_datasets(datasets, sample_config.primary_key)
    
    train_loader, val_loader, test_loader = make_dataloaders(sample_config, aligned)
    
    # All loaders should exist (even if small)
    assert train_loader is not None
    assert test_loader is not None
    
    # Test that we can iterate through a batch
    batch = next(iter(train_loader))
    features, targets = batch
    
    assert isinstance(features, list)
    assert len(features) == 2  # Two datasets
    assert features[0].dim() == 2  # [batch_size, features]
    assert features[1].dim() == 2  # [batch_size, features]
    assert targets.dim() == 1  # [batch_size]

def test_alignment_with_no_common_keys():
    """Test alignment when datasets have no common keys."""
    df1 = pd.DataFrame({'id': [1, 2], 'f1': [0.1, 0.2]})
    df2 = pd.DataFrame({'id': [3, 4], 'f2': [0.3, 0.4]})
    
    datasets = {'table1': df1, 'table2': df2}
    
    with pytest.raises(ValueError, match="No common primary key values"):
        align_datasets(datasets, 'id')

def test_split_consistency():
    """Test that splits are consistent and sum to total."""
    df = pd.DataFrame({
        'id': list(range(100)),
        'feature': list(range(100)),
        'label': [i % 2 for i in range(100)]
    })
    
    from nexusflow.data.ingestion import split_df
    
    train, val, test = split_df(df, test_size=0.2, val_size=0.2, randomize=True)
    
    # Lengths should sum to original
    assert len(train) + len(val) + len(test) == 100
    
    # No overlapping indices
    train_ids = set(train['id'])
    val_ids = set(val['id'])
    test_ids = set(test['id'])
    
    assert len(train_ids & val_ids) == 0
    assert len(train_ids & test_ids) == 0
    assert len(val_ids & test_ids) == 0