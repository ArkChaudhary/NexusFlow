"""Enhanced tests for the Trainer class with real data integration."""
import pytest
import torch
import pandas as pd
import tempfile
import os
from pathlib import Path

from nexusflow.trainer.trainer import Trainer
from nexusflow.config import ConfigModel

@pytest.fixture
def real_data_config():
    """Configuration for testing with real CSV data."""
    return ConfigModel(
        project_name="real_data_test",
        primary_key="id",
        target={"target_table": "customers.csv", "target_column": "churn"},
        architecture={"global_embed_dim": 16, "refinement_iterations": 1},
        datasets=[
            {"name": "customers.csv", "transformer_type": "standard", "complexity": "small"},
            {"name": "transactions.csv", "transformer_type": "standard", "complexity": "small"}
        ],
        training={
            "use_synthetic": False,
            "batch_size": 4,
            "epochs": 1,
            "optimizer": {"lr": 1e-3},
            "split_config": {
                "test_size": 0.2,
                "validation_size": 0.2,
                "randomize": True
            }
        },
        mlops={"logging_provider": "stdout"}
    )

@pytest.fixture
def synthetic_data_config():
    """Configuration for testing with synthetic data."""
    return ConfigModel(
        project_name="synthetic_test",
        primary_key="id",
        target={"target_table": "table_a.csv", "target_column": "label"},
        architecture={"global_embed_dim": 8, "refinement_iterations": 1},
        datasets=[
            {"name": "table_a.csv"},
            {"name": "table_b.csv"}
        ],
        training={
            "use_synthetic": True,
            "synthetic": {"n_samples": 32, "feature_dim": 3},
            "batch_size": 8,
            "epochs": 1
        },
        mlops={"logging_provider": "stdout"}
    )

@pytest.fixture
def sample_data_files(tmp_path):
    """Create sample data files for testing."""
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    
    # Customer data
    customers = pd.DataFrame({
        'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'age': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        'income': [30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000],
        'churn': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })
    
    # Transaction data
    transactions = pd.DataFrame({
        'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'total_spent': [1000, 2000, 1500, 2500, 1200, 1800, 2200, 3000, 800, 1600],
        'num_transactions': [10, 20, 15, 25, 12, 18, 22, 30, 8, 16]
    })
    
    customers.to_csv(datasets_dir / "customers.csv", index=False)
    transactions.to_csv(datasets_dir / "transactions.csv", index=False)
    
    # Change working directory for relative path resolution
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    
    yield tmp_path, datasets_dir
    
    os.chdir(old_cwd)

def test_trainer_real_data_initialization(real_data_config, sample_data_files):
    """Test trainer initialization with real data."""
    tmp_path, datasets_dir = sample_data_files
    
    trainer = Trainer(real_data_config, work_dir=str(tmp_path))
    
    # Check that input dimensions were inferred correctly
    # customers.csv: 2 features (age, income) - excluding id and churn
    # transactions.csv: 2 features (total_spent, num_transactions) - excluding id
    assert trainer.input_dims == [2, 2]
    
    # Check that datasets were loaded and aligned
    assert trainer.datasets is not None
    assert len(trainer.datasets) == 2
    assert "customers.csv" in trainer.datasets
    assert "transactions.csv" in trainer.datasets
    
    # All rows should align since all IDs match
    assert len(trainer.datasets["customers.csv"]) == 10
    assert len(trainer.datasets["transactions.csv"]) == 10

def test_trainer_synthetic_data_initialization(synthetic_data_config):
    """Test trainer initialization with synthetic data."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = Trainer(synthetic_data_config, work_dir=tmp_dir)
        
        # Check synthetic dimensions
        assert trainer.input_dims == [3, 3]  # feature_dim from config
        
        # Datasets should be None for synthetic mode
        assert trainer.datasets is None

def test_trainer_sanity_check_real_data(real_data_config, sample_data_files):
    """Test sanity check with real data."""
    tmp_path, datasets_dir = sample_data_files
    
    trainer = Trainer(real_data_config, work_dir=str(tmp_path))
    
    # Should run without errors
    trainer.sanity_check()

def test_trainer_sanity_check_synthetic_data(synthetic_data_config):
    """Test sanity check with synthetic data."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = Trainer(synthetic_data_config, work_dir=tmp_dir)
        trainer.sanity_check()

def test_trainer_training_real_data(real_data_config, sample_data_files):
    """Test full training loop with real data."""
    tmp_path, datasets_dir = sample_data_files
    
    trainer = Trainer(real_data_config, work_dir=str(tmp_path))
    trainer.train()
    
    # Check that checkpoint was saved
    checkpoint_files = list(Path(tmp_path).glob("model_epoch_*.pt"))
    assert len(checkpoint_files) >= 1
    
    # Verify checkpoint contains expected keys
    checkpoint = torch.load(checkpoint_files[0], map_location='cpu')
    expected_keys = ['epoch', 'model_state', 'optimizer_state', 'config', 'input_dims']
    for key in expected_keys:
        assert key in checkpoint
    
    assert checkpoint['input_dims'] == [2, 2]

def test_trainer_training_synthetic_data(synthetic_data_config):
    """Test full training loop with synthetic data."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = Trainer(synthetic_data_config, work_dir=tmp_dir)
        trainer.train()
        
        # Check that checkpoint was saved
        checkpoint_files = list(Path(tmp_dir).glob("model_epoch_*.pt"))
        assert len(checkpoint_files) >= 1

def test_trainer_evaluation(real_data_config, sample_data_files):
    """Test model evaluation functionality."""
    tmp_path, datasets_dir = sample_data_files
    
    trainer = Trainer(real_data_config, work_dir=str(tmp_path))
    trainer.train()
    
    # Test evaluation
    metrics = trainer.evaluate()
    assert isinstance(metrics, dict)
    assert 'test_loss' in metrics
    assert 'num_test_batches' in metrics
    assert isinstance(metrics['test_loss'], float)
    assert isinstance(metrics['num_test_batches'], int)

def test_trainer_data_loader_setup(real_data_config, sample_data_files):
    """Test that data loaders are set up correctly."""
    tmp_path, datasets_dir = sample_data_files
    
    trainer = Trainer(real_data_config, work_dir=str(tmp_path))
    trainer._setup_dataloaders()
    
    assert trainer.train_loader is not None
    assert trainer.test_loader is not None
    # val_loader might be None if validation set is too small
    
    # Test that we can get a batch
    batch = next(iter(trainer.train_loader))
    features, targets = batch
    
    assert isinstance(features, list)
    assert len(features) == 2  # Two datasets
    assert features[0].dim() == 2  # [batch_size, features]
    assert features[1].dim() == 2
    assert targets.dim() == 1

def test_trainer_handles_misaligned_data():
    """Test trainer behavior with misaligned datasets."""
    config = ConfigModel(
        project_name="misaligned_test",
        primary_key="id",
        target={"target_table": "table_a.csv", "target_column": "label"},
        architecture={"global_embed_dim": 8, "refinement_iterations": 1},
        datasets=[
            {"name": "table_a.csv"},
            {"name": "table_b.csv"}
        ],
        training={"use_synthetic": False, "batch_size": 4, "epochs": 1},
        mlops={"logging_provider": "stdout"}
    )
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create datasets directory
        datasets_dir = Path(tmp_dir) / "datasets"
        datasets_dir.mkdir()
        
        # Create misaligned data
        df_a = pd.DataFrame({
            'id': [1, 2, 3],
            'feature_a': [0.1, 0.2, 0.3],
            'label': [0, 1, 0]
        })
        
        df_b = pd.DataFrame({
            'id': [4, 5, 6],  # No overlapping IDs
            'feature_b': [0.4, 0.5, 0.6]
        })
        
        df_a.to_csv(datasets_dir / "table_a.csv", index=False)
        df_b.to_csv(datasets_dir / "table_b.csv", index=False)
        
        old_cwd = os.getcwd()
        os.chdir(tmp_dir)
        
        try:
            # Should raise an error due to no common keys
            with pytest.raises(ValueError, match="No common primary key values"):
                trainer = Trainer(config, work_dir=tmp_dir)
        finally:
            os.chdir(old_cwd)

def test_trainer_missing_target_column():
    """Test trainer behavior when target column is missing."""
    config = ConfigModel(
        project_name="missing_target_test",
        primary_key="id", 
        target={"target_table": "table_a.csv", "target_column": "missing_label"},
        architecture={"global_embed_dim": 8, "refinement_iterations": 1},
        datasets=[{"name": "table_a.csv"}],
        training={"use_synthetic": False, "batch_size": 4, "epochs": 1},
        mlops={"logging_provider": "stdout"}
    )
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        datasets_dir = Path(tmp_dir) / "datasets"
        datasets_dir.mkdir()
        
        # Create data without the expected target column
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'feature': [0.1, 0.2, 0.3],
            'label': [0, 1, 0]  # target expects 'missing_label' not 'label'
        })
        df.to_csv(datasets_dir / "table_a.csv", index=False)
        
        old_cwd = os.getcwd()
        os.chdir(tmp_dir)
        
        try:
            with pytest.raises(KeyError, match="missing_label"):
                trainer = Trainer(config, work_dir=tmp_dir)
                trainer.train()
        finally:
            os.chdir(old_cwd)

def test_trainer_checkpoint_loading():
    """Test that saved checkpoints contain all necessary information."""
    config = ConfigModel(
        project_name="checkpoint_test",
        primary_key="id",
        target={"target_table": "table_a.csv", "target_column": "label"},
        architecture={"global_embed_dim": 4, "refinement_iterations": 1},
        datasets=[{"name": "table_a.csv"}],
        training={
            "use_synthetic": True,
            "synthetic": {"n_samples": 16, "feature_dim": 2},
            "batch_size": 8,
            "epochs": 1
        },
        mlops={"logging_provider": "stdout"}
    )
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = Trainer(config, work_dir=tmp_dir)
        trainer.train()
        
        # Find the checkpoint
        checkpoint_file = list(Path(tmp_dir).glob("model_epoch_*.pt"))[0]
        
        # Load and verify checkpoint
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        
        # Should contain all metadata needed to reconstruct the model
        assert 'model_state' in checkpoint
        assert 'input_dims' in checkpoint
        assert 'config' in checkpoint
        assert checkpoint['input_dims'] == [2]  # One dataset with 2 features