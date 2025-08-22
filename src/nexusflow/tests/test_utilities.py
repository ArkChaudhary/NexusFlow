"""Test fixtures and utilities for NexusFlow tests."""
import pytest
import pandas as pd
import tempfile
import os
from pathlib import Path
import yaml

from nexusflow.config import ConfigModel

@pytest.fixture
def sample_aligned_csvs(tmp_path):
    """Create sample CSV files that align properly for testing."""
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    
    # Table A - customers
    df_a = pd.DataFrame({
        'customer_id': [100, 101, 102, 103, 104],
        'age': [25, 30, 35, 40, 45],
        'income': [50000, 60000, 70000, 80000, 90000],
        'churn': [0, 1, 0, 1, 0]
    })
    
    # Table B - transactions (aligned with customer_id)
    df_b = pd.DataFrame({
        'customer_id': [100, 101, 102, 103, 104],
        'total_spent': [1000, 1500, 2000, 2500, 3000],
        'num_purchases': [5, 8, 12, 15, 20],
        'avg_order_value': [200, 187.5, 166.7, 166.7, 150]
    })
    
    # Save CSV files
    df_a.to_csv(datasets_dir / "customers.csv", index=False)
    df_b.to_csv(datasets_dir / "transactions.csv", index=False)
    
    return tmp_path, datasets_dir, df_a, df_b

@pytest.fixture  
def sample_config_file(tmp_path):
    """Create a sample configuration file for testing."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    
    config_data = {
        'project_name': 'test_project',
        'primary_key': 'customer_id',
        'target': {
            'target_table': 'customers.csv',
            'target_column': 'churn'
        },
        'architecture': {
            'refinement_iterations': 1,
            'global_embed_dim': 32
        },
        'datasets': [
            {
                'name': 'customers.csv',
                'transformer_type': 'standard',
                'complexity': 'small',
                'context_weight': 1.0
            },
            {
                'name': 'transactions.csv', 
                'transformer_type': 'standard',
                'complexity': 'small',
                'context_weight': 1.0
            }
        ],
        'training': {
            'use_synthetic': False,
            'batch_size': 4,
            'epochs': 1,
            'optimizer': {'name': 'adam', 'lr': 1e-3},
            'split_config': {
                'test_size': 0.2,
                'validation_size': 0.2,
                'randomize': True
            }
        },
        'mlops': {
            'logging_provider': 'stdout',
            'experiment_name': 'test_experiment'
        }
    }
    
    config_path = config_dir / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f)
    
    return config_path, config_data

@pytest.fixture
def minimal_config():
    """Create a minimal configuration for basic testing."""
    return ConfigModel(
        project_name="minimal_test",
        primary_key="id",
        target={"target_table": "data.csv", "target_column": "target"},
        architecture={"global_embed_dim": 8, "refinement_iterations": 1},
        datasets=[{"name": "data.csv"}],
        training={"batch_size": 2, "epochs": 1},
        mlops={"logging_provider": "stdout"}
    )

@pytest.fixture
def working_directory_context(tmp_path):
    """Context manager that changes to a temporary working directory."""
    class WorkingDirContext:
        def __init__(self, path):
            self.path = path
            self.old_cwd = None
            
        def __enter__(self):
            self.old_cwd = os.getcwd()
            os.chdir(self.path)
            return self.path
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.old_cwd:
                os.chdir(self.old_cwd)
    
    return WorkingDirContext(tmp_path)

def create_synthetic_dataset(n_samples=100, n_features=5, n_classes=2, random_state=42):
    """Utility function to create synthetic tabular data for testing."""
    import numpy as np
    
    np.random.seed(random_state)
    
    # Generate features
    features = np.random.randn(n_samples, n_features)
    
    # Generate targets based on a simple linear combination
    weights = np.random.randn(n_features)
    logits = features @ weights
    
    if n_classes == 2:
        targets = (logits > 0).astype(int)
    else:
        # For multi-class, use quantiles
        targets = np.digitize(logits, bins=np.linspace(logits.min(), logits.max(), n_classes-1))
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(features, columns=feature_names)
    df['id'] = range(n_samples)
    df['target'] = targets
    
    return df

def assert_model_checkpoint_valid(checkpoint_path):
    """Utility function to validate a saved model checkpoint."""
    import torch
    
    assert Path(checkpoint_path).exists(), f"Checkpoint file not found: {checkpoint_path}"
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Check required keys
    required_keys = ['model_state', 'epoch', 'input_dims']
    for key in required_keys:
        assert key in checkpoint, f"Missing key in checkpoint: {key}"
    
    # Validate model state dict
    model_state = checkpoint['model_state']
    assert isinstance(model_state, dict), "model_state should be a dictionary"
    assert len(model_state) > 0, "model_state should not be empty"
    
    # Validate input dimensions
    input_dims = checkpoint['input_dims']
    assert isinstance(input_dims, list), "input_dims should be a list"
    assert all(isinstance(dim, int) and dim > 0 for dim in input_dims), \
           "All input dimensions should be positive integers"
    
    return checkpoint