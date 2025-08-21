import pytest
from nexusflow.config import load_config_from_file
import yaml

def test_load_valid_config(tmp_path):
    cfg = {
        'project_name': 't1',
        'primary_key': 'id',
        'target': {'target_table': 'a.csv', 'target_column': 'label'},
        'architecture': {'refinement_iterations': 1, 'global_embed_dim': 32},
        'datasets': [{'name': 'a.csv'}],
        'training': {'batch_size': 4, 'epochs': 1},
        'mlops': {'logging_provider': 'stdout'}
    }
    p = tmp_path / 'c.yaml'
    p.write_text(yaml.safe_dump(cfg))
    loaded = load_config_from_file(str(p))
    assert loaded.project_name == 't1'
    assert len(loaded.datasets) == 1

def test_invalid_config_missing_dataset(tmp_path):
    cfg = {
        'project_name': 't1',
        'primary_key': 'id',
        'target': {'target_table': 'a.csv', 'target_column': 'label'},
        'architecture': {'refinement_iterations': 1, 'global_embed_dim': 32},
        'datasets': [],
    }
    p = tmp_path / 'c2.yaml'
    p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(Exception):
        load_config_from_file(str(p))
