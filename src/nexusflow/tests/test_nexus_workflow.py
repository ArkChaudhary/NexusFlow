"""Integration tests for the complete NexusFlow workflow."""
import pytest
import pandas as pd
import os
import tempfile
from pathlib import Path
import yaml

from nexusflow.config import ConfigModel, load_config_from_file
from nexusflow.trainer.trainer import Trainer

@pytest.fixture
def workflow_setup(tmp_path):
    """Set up a complete test environment with config and data files."""
    # Create project structure
    datasets_dir = tmp_path / "datasets" 
    configs_dir = tmp_path / "configs"
    models_dir = tmp_path / "models"
    results_dir = tmp_path / "results"
    
    for dir_path in [datasets_dir, configs_dir, models_dir, results_dir]:
        dir_path.mkdir()
    
    # Create sample data files
    df_a = pd.DataFrame({
        'id': [1, 2, 3, 4, 5, 6, 7, 8],
        'feature_a1': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        'feature_a2': [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8],
        'label': [0, 1, 0, 1, 1, 0, 1, 0]
    })
    
    df_b = pd.DataFrame({
        'id': [1, 2, 3, 4, 5, 6, 7, 8],  # All IDs align
        'feature_b1': [2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8],
        'feature_b2': [3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8]
    })
    
    # Save CSV files
    df_a.to_csv(datasets_dir / "table_a.csv", index=False)
    df_b.to_csv(datasets_dir / "table_b.csv", index=False)
    
    # Create config file
    config_data = {
        'project_name': 'workflow_test',
        'primary_key': 'id',
        'target': {
            'target_table': 'table_a.csv',
            'target_column': 'label'
        },
        'architecture': {
            'refinement_iterations': 1,
            'global_embed_dim': 16
        },
        'datasets': [
            {'name': 'table_a.csv', 'transformer_type': 'standard', 'complexity': 'small'},
            {'name': 'table_b.csv', 'transformer_type': 'standard', 'complexity': 'small'}
        ],
        'training': {
            'use_synthetic': False,  # Use real data
            'batch_size': 4,
            'epochs': 2,
            'optimizer': {'name': 'adam', 'lr': 1e-3},
            'split_config': {
                'test_size': 0.25,
                'validation_size': 0.25,
                'randomize': True
            }
        },
        'mlops': {
            'logging_provider': 'stdout',
            'experiment_name': 'workflow_test_run'
        }
    }
    
    config_path = configs_dir / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f)
    
    # Change working directory to tmp_path
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    
    yield tmp_path, config_path, df_a, df_b
    
    # Restore working directory
    os.chdir(old_cwd)

def test_complete_workflow(workflow_setup):
    """Test the complete training workflow from config to saved model."""
    tmp_path, config_path, df_a, df_b = workflow_setup
    
    # Load configuration
    config = load_config_from_file(str(config_path))
    assert config.project_name == 'workflow_test'
    
    # Initialize trainer
    trainer = Trainer(config, work_dir=tmp_path)
    
    # Run sanity check
    trainer.sanity_check()  # Should not raise any exceptions
    
    # Verify model was initialized with correct dimensions
    expected_dims = [2, 2]  # 2 features from each table (excluding id and label)
    assert trainer.input_dims == expected_dims
    
    # Run training
    trainer.train()
    
    # Verify checkpoint files were created
    model_files = list((tmp_path / "models").glob("*.pt"))
    if not model_files:
        # Check in work_dir if models/ doesn't exist
        model_files = list(tmp_path.glob("model_epoch_*.pt"))
    
    assert len(model_files) >= 1, f"No model checkpoints found in {tmp_path}"
    
    # Verify we can load a checkpoint
    import torch
    checkpoint = torch.load(model_files[0], map_location='cpu')
    assert 'model_state' in checkpoint
    assert 'epoch' in checkpoint
    assert 'input_dims' in checkpoint
    assert checkpoint['input_dims'] == expected_dims

def test_trainer_with_synthetic_data():
    """Test trainer with synthetic data fallback."""
    config_data = {
        'project_name': 'synthetic_test',
        'primary_key': 'id',
        'target': {'target_table': 'table_a.csv', 'target_column': 'label'},
        'architecture': {'refinement_iterations': 1, 'global_embed_dim': 8},
        'datasets': [
            {'name': 'table_a.csv'},
            {'name': 'table_b.csv'}
        ],
        'training': {
            'use_synthetic': True,
            'synthetic': {'n_samples': 32, 'feature_dim': 4},
            'batch_size': 8,
            'epochs': 1
        },
        'mlops': {'logging_provider': 'stdout'}
    }
    
    config = ConfigModel(**config_data)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = Trainer(config, work_dir=tmp_dir)
        
        # Verify synthetic dimensions
        assert trainer.input_dims == [4, 4]
        
        # Run training
        trainer.train()
        
        # Verify checkpoint was saved
        checkpoint_files = list(Path(tmp_dir).glob("model_epoch_*.pt"))
        assert len(checkpoint_files) >= 1

def test_data_validation_errors():
    """Test that appropriate errors are raised for invalid data."""
    config_data = {
        'project_name': 'error_test',
        'primary_key': 'missing_key',  # This key won't exist
        'target': {'target_table': 'table_a.csv', 'target_column': 'label'},
        'architecture': {'refinement_iterations': 1, 'global_embed_dim': 8},
        'datasets': [{'name': 'table_a.csv'}],
        'training': {'use_synthetic': False, 'batch_size': 4, 'epochs': 1},
        'mlops': {'logging_provider': 'stdout'}
    }
    
    config = ConfigModel(**config_data)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a CSV without the expected primary key
        datasets_dir = Path(tmp_dir) / "datasets"
        datasets_dir.mkdir()
        
        df = pd.DataFrame({
            'id': [1, 2, 3],  # primary key is 'id' but config expects 'missing_key'
            'feature': [0.1, 0.2, 0.3],
            'label': [0, 1, 0]
        })
        df.to_csv(datasets_dir / "table_a.csv", index=False)
        
        # Change to tmp_dir for relative path resolution
        old_cwd = os.getcwd()
        os.chdir(tmp_dir)
        
        try:
            # This should raise an error due to missing primary key
            with pytest.raises(KeyError, match="missing_key"):
                trainer = Trainer(config, work_dir=tmp_dir)
        finally:
            os.chdir(old_cwd)

def test_trainer_evaluation():
    """Test model evaluation functionality."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create simple config
        config_data = {
            'project_name': 'eval_test',
            'primary_key': 'id',
            'target': {'target_table': 'table_a.csv', 'target_column': 'label'},
            'architecture': {'refinement_iterations': 1, 'global_embed_dim': 8},
            'datasets': [{'name': 'table_a.csv'}],
            'training': {
                'use_synthetic': True,
                'synthetic': {'n_samples': 64, 'feature_dim': 3},
                'batch_size': 16,
                'epochs': 1
            },
            'mlops': {'logging_provider': 'stdout'}
        }
        
        config = ConfigModel(**config_data)
        trainer = Trainer(config, work_dir=tmp_dir)
        
        # Train briefly
        trainer.train()
        
        # Test evaluation (will be empty for synthetic data without test set)
        metrics = trainer.evaluate()
        assert isinstance(metrics, dict)