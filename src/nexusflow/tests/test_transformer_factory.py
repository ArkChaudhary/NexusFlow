import pytest
import torch
from nexusflow.model.transformer_factory import TransformerFactory
from nexusflow.config import DatasetConfig

def test_transformer_factory_standard():
    config = DatasetConfig(name="test.csv", transformer_type="standard", complexity="medium")
    encoder = TransformerFactory.create_encoder(config, input_dim=10, embed_dim=64)
    
    x = torch.randn(5, 10)
    output = encoder(x)
    assert output.shape == (5, 64)

def test_transformer_factory_complexity_scaling():
    small_config = DatasetConfig(name="test.csv", transformer_type="standard", complexity="small") 
    large_config = DatasetConfig(name="test.csv", transformer_type="standard", complexity="large")
    
    small_encoder = TransformerFactory.create_encoder(small_config, 10, 64)
    large_encoder = TransformerFactory.create_encoder(large_config, 10, 64)
    
    # Large should have more parameters
    small_params = sum(p.numel() for p in small_encoder.parameters())
    large_params = sum(p.numel() for p in large_encoder.parameters())
    assert large_params > small_params
