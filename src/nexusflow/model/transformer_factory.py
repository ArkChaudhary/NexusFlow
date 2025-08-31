"""Enhanced factory for creating specialized transformers with advanced tabular architectures."""
from typing import Dict, Any
import torch.nn as nn
import torch
from nexusflow.model.nexus_former import StandardTabularEncoder, FTTransformerEncoder, TabNetEncoder
from nexusflow.config import DatasetConfig

class TransformerFactory:
    """Enhanced factory for creating specialized transformers with advanced architectures."""
    
    @staticmethod
    def create_encoder(dataset_config: DatasetConfig, input_dim: int, embed_dim: int = 64, 
                    use_moe: bool = False, num_experts: int = 4, use_flash_attn: bool = True) -> nn.Module:
        """Create appropriate encoder based on dataset configuration with advanced features."""
        transformer_type = dataset_config.transformer_type.lower()
        complexity = dataset_config.complexity.lower()
        
        # Adjust architecture based on complexity
        if complexity == 'small':
            num_heads, num_layers = 2, 1
        elif complexity == 'medium':
            num_heads, num_layers = 4, 2
        else:  # large
            num_heads, num_layers = 8, 3
        
        # Create encoder based on type
        if transformer_type == 'standard':
            return StandardTabularEncoder(
                input_dim=input_dim,
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                use_moe=use_moe,
                num_experts=num_experts,
                use_flash_attn=use_flash_attn
            )
        elif transformer_type == 'ft_transformer':
            return FTTransformerEncoder(
                input_dim=input_dim,
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                use_moe=use_moe,
                num_experts=num_experts
            )
        elif transformer_type == 'tabnet':
            return TabNetEncoder(
                input_dim=input_dim,
                embed_dim=embed_dim,
                num_steps=num_layers  # Map layers to TabNet steps
            )
        elif transformer_type == 'text':
            return TextEncoder(input_dim, embed_dim, num_heads, num_layers, use_moe, num_experts, use_flash_attn)
        elif transformer_type == 'timeseries':
            return TimeSeriesEncoder(input_dim, embed_dim, num_heads, num_layers, use_moe, num_experts, use_flash_attn)
        else:
            raise ValueError(f"Unknown transformer type: {transformer_type}")

class TextEncoder(nn.Module):
    """Enhanced encoder for text data with advanced features."""
    
    def __init__(self, input_dim: int, embed_dim: int = 64, num_heads: int = 4, num_layers: int = 2,
                 use_moe: bool = False, num_experts: int = 4, use_flash_attn: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        # Use enhanced StandardTabularEncoder for now
        self.encoder = StandardTabularEncoder(
            input_dim, embed_dim, num_heads, num_layers,
            use_moe=use_moe, num_experts=num_experts, use_flash_attn=use_flash_attn
        )
    
    def forward(self, x):
        return self.encoder(x)

class TimeSeriesEncoder(nn.Module):
    """Enhanced encoder for time-series data with temporal modeling."""
    
    def __init__(self, input_dim: int, embed_dim: int = 64, num_heads: int = 4, num_layers: int = 2,
                 use_moe: bool = False, num_experts: int = 4, use_flash_attn: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        # Enhanced with temporal position encoding
        self.temporal_encoding = nn.Parameter(torch.randn(1, input_dim, embed_dim) * 0.02)
        
        # Use enhanced StandardTabularEncoder with modifications for time-series
        self.encoder = StandardTabularEncoder(
            input_dim, embed_dim, num_heads, num_layers,
            use_moe=use_moe, num_experts=num_experts, use_flash_attn=use_flash_attn
        )
    
    def forward(self, x):
        # Add temporal encoding for time-series patterns
        if x.dim() == 2:  # [batch, features]
            batch_size = x.size(0)
            # Expand temporal encoding
            x_expanded = x.unsqueeze(1)  # [batch, 1, features]
            x_expanded = x_expanded + self.temporal_encoding[:, :1, :x.size(1)]
            x = x_expanded.squeeze(1)  # Back to [batch, features]
        
        return self.encoder(x)