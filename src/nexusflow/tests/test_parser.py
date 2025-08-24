"""Unit tests for NexusFlow configuration parser."""
import pytest
import tempfile
import yaml
from pathlib import Path
from pydantic import ValidationError

from nexusflow.config import ConfigModel, load_config_from_file, DatasetConfig, TrainingConfig


class TestConfigModel:
    """Test the Pydantic configuration model validation."""
    
    def test_valid_config(self):
        """Test that a valid configuration passes validation."""
        valid_config = {
            'project_name': 'test_project',
            'primary_key': 'id',
            'target': {
                'target_table': 'table_a.csv',
                'target_column': 'label'
            },
            'architecture': {
                'global_embed_dim': 64,
                'refinement_iterations': 2
            },
            'datasets': [
                {
                    'name': 'table_a.csv',
                    'transformer_type': 'standard',
                    'complexity': 'small'
                },
                {
                    'name': 'table_b.csv',
                    'transformer_type': 'standard', 
                    'complexity': 'medium'
                }
            ]
        }
        
        config = ConfigModel.model_validate(valid_config)
        assert config.project_name == 'test_project'
        assert config.primary_key == 'id'
        assert len(config.datasets) == 2
        assert config.datasets[0].name == 'table_a.csv'
        assert config.architecture['global_embed_dim'] == 64

    def test_missing_required_fields(self):
        """Test that missing required fields raise ValidationError."""
        # Missing project_name
        invalid_config = {
            'primary_key': 'id',
            'target': {'target_table': 'table_a.csv', 'target_column': 'label'},
            'architecture': {'global_embed_dim': 64}
        }
        
        with pytest.raises(ValidationError) as exc_info:
            ConfigModel.model_validate(invalid_config)
        assert 'project_name' in str(exc_info.value)

    def test_missing_primary_key(self):
        """Test that missing primary_key raises ValidationError."""
        invalid_config = {
            'project_name': 'test',
            'target': {'target_table': 'table_a.csv', 'target_column': 'label'},
            'architecture': {'global_embed_dim': 64}
        }
        
        with pytest.raises(ValidationError) as exc_info:
            ConfigModel.model_validate(invalid_config)
        assert 'primary_key' in str(exc_info.value)

    def test_missing_target_config(self):
        """Test that missing target configuration raises ValidationError."""
        invalid_config = {
            'project_name': 'test',
            'primary_key': 'id',
            'architecture': {'global_embed_dim': 64}
        }
        
        with pytest.raises(ValidationError) as exc_info:
            ConfigModel.model_validate(invalid_config)
        assert 'target' in str(exc_info.value)

    def test_invalid_architecture_config(self):
        """Test that invalid architecture values are handled properly."""
        invalid_config = {
            'project_name': 'test',
            'primary_key': 'id',
            'target': {'target_table': 'table_a.csv', 'target_column': 'label'},
            'architecture': {'global_embed_dim': -1},  # Invalid negative value
            'datasets': [{'name': 'table_a.csv'}]  # Added required datasets
        }
        
        # This should still validate as architecture is just a dict
        config = ConfigModel.model_validate(invalid_config)
        assert config.architecture['global_embed_dim'] == -1

    def test_synthetic_data_config_validation(self):
        """Test synthetic data configuration validation."""
        # Valid synthetic config
        valid_synthetic = {
            'project_name': 'test',
            'primary_key': 'id',
            'target': {'target_table': 'table_a.csv', 'target_column': 'label'},
            'architecture': {'global_embed_dim': 64},
            'training': {
                'use_synthetic': True,
                'synthetic': {
                    'n_samples': 256,
                    'feature_dim': 5
                }
            }
        }
        
        config = ConfigModel.model_validate(valid_synthetic)
        assert config.training.use_synthetic is True
        assert config.training.synthetic.n_samples == 256
        assert config.training.synthetic.feature_dim == 5

    def test_synthetic_without_datasets(self):
        """Test that synthetic mode allows missing datasets."""
        synthetic_config = {
            'project_name': 'test',
            'primary_key': 'id',
            'target': {'target_table': 'table_a.csv', 'target_column': 'label'},
            'architecture': {'global_embed_dim': 64},
            'training': {'use_synthetic': True}
        }
        
        # Should validate even without datasets list
        config = ConfigModel.model_validate(synthetic_config)
        assert config.training.use_synthetic is True

    def test_non_synthetic_requires_datasets(self):
        """Test that non-synthetic mode requires datasets."""
        invalid_config = {
            'project_name': 'test',
            'primary_key': 'id',
            'target': {'target_table': 'table_a.csv', 'target_column': 'label'},
            'architecture': {'global_embed_dim': 64},
            'training': {'use_synthetic': False}
        }
        
        with pytest.raises(ValidationError) as exc_info:
            ConfigModel.model_validate(invalid_config)
        assert 'dataset must be specified' in str(exc_info.value)

    def test_dataset_config_defaults(self):
        """Test that dataset configuration uses proper defaults."""
        dataset_config = DatasetConfig(name='test.csv')
        
        assert dataset_config.transformer_type == 'standard'
        assert dataset_config.complexity == 'small'
        assert dataset_config.context_weight == 1.0

    def test_training_config_defaults(self):
        """Test that training configuration uses proper defaults."""
        training_config = TrainingConfig()
        
        assert training_config.batch_size == 32
        assert training_config.epochs == 10
        assert training_config.optimizer['name'] == 'adam'
        assert training_config.optimizer['lr'] == 1e-3
        assert training_config.use_synthetic is False


class TestConfigLoader:
    """Test the configuration file loading functionality."""
    
    def test_load_valid_config_file(self):
        """Test loading a valid YAML configuration file."""
        config_data = {
            'project_name': 'test_project',
            'primary_key': 'id',
            'target': {
                'target_table': 'table_a.csv',
                'target_column': 'label'
            },
            'architecture': {
                'global_embed_dim': 64,
                'refinement_iterations': 2
            },
            'datasets': [
                {'name': 'table_a.csv', 'transformer_type': 'standard'},
                {'name': 'table_b.csv', 'transformer_type': 'standard'}
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            config = load_config_from_file(temp_path)
            assert config.project_name == 'test_project'
            assert len(config.datasets) == 2
        finally:
            Path(temp_path).unlink()

    def test_load_nonexistent_file(self):
        """Test that loading a non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config_from_file('nonexistent.yaml')

    def test_load_invalid_yaml(self):
        """Test that loading invalid YAML raises appropriate error."""
        invalid_yaml = "project_name: test\ninvalid: yaml: content: [unclosed"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(invalid_yaml)
            temp_path = f.name
        
        try:
            with pytest.raises(yaml.YAMLError):
                load_config_from_file(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_load_config_with_validation_error(self):
        """Test that loading config with validation errors raises ValidationError."""
        invalid_config = {
            'project_name': 'test',
            # Missing required fields
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(invalid_config, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ValidationError):
                load_config_from_file(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_config_with_mlops_settings(self):
        """Test configuration with different MLOps providers."""
        config_data = {
            'project_name': 'test_project',
            'primary_key': 'id',
            'target': {'target_table': 'table_a.csv', 'target_column': 'label'},
            'architecture': {'global_embed_dim': 64},
            'datasets': [{'name': 'table_a.csv'}],
            'mlops': {
                'logging_provider': 'wandb',
                'experiment_name': 'test_experiment'
            }
        }
        
        config = ConfigModel.model_validate(config_data)
        assert config.mlops.logging_provider == 'wandb'
        assert config.mlops.experiment_name == 'test_experiment'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])