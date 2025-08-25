# tests/integration/test_train_multi.py
import pytest
import tempfile
from pathlib import Path
import pandas as pd
from nexusflow.config import ConfigModel
from nexusflow.trainer.trainer import Trainer
from loguru import logger

def test_train_with_multiple_tables():
    """Test training with N=3 synthetic tables where one is noise."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        # Create config with 3 datasets
        config_dict = {
            'project_name': 'multi_test',
            'primary_key': 'id',
            'target': {'target_table': 'table_a.csv', 'target_column': 'label'},
            'architecture': {'refinement_iterations': 2, 'global_embed_dim': 32},
            'datasets': [
                {'name': 'table_a.csv', 'transformer_type': 'standard', 'complexity': 'small'},
                {'name': 'table_b.csv', 'transformer_type': 'standard', 'complexity': 'small'},
                {'name': 'table_noise.csv', 'transformer_type': 'standard', 'complexity': 'small'}
            ],
            'training': {
                'use_synthetic': True,
                'synthetic': {'n_samples': 64, 'feature_dim': 4},
                'batch_size': 8, 
                'epochs': 3
            },
            'mlops': {
                'logging_provider': 'stdout',  # Avoid file logging in tests
                'experiment_name': 'test_run'
            }
        }
        
        config = ConfigModel.model_validate(config_dict)
        trainer = Trainer(config, work_dir=str(tmp_path))
        
        try:
            # Should complete without error
            trainer.train()
            
            # Check model artifact was created
            assert (tmp_path / f"{config.project_name}.nxf").exists()
        finally:
            # Clean up any loguru handlers that might be holding file handles
            logger.remove()
            # Close MLOps logger
            trainer.mlops_logger.finish()