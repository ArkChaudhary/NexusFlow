import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from typing import Sequence, List
import math

class ContextualEncoder(nn.Module):
    """Abstract base class for all contextual encoders."""
    
    def __init__(self, input_dim: int, embed_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the encoder."""
        raise NotImplementedError("Subclasses must implement forward method")

class StandardTabularEncoder(ContextualEncoder):
    """Small transformer-based encoder for standard tabular data."""
    
    def __init__(self, input_dim: int, embed_dim: int = 64, num_heads: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__(input_dim, embed_dim)
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, embed_dim)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Project input to embedding dimension
        x = self.input_projection(x)  # [batch, embed_dim]
        
        # Add positional encoding and expand for transformer
        x = x.unsqueeze(1) + self.pos_embedding  # [batch, 1, embed_dim]
        
        # Pass through transformer
        x = self.transformer(x)  # [batch, 1, embed_dim]
        
        # Remove sequence dimension and normalize
        x = x.squeeze(1)  # [batch, embed_dim]
        x = self.layer_norm(x)
        
        logger.debug(f"StandardTabularEncoder output: shape={x.shape} mean={x.mean().item():.4f}")
        return x

class CrossContextualAttention(nn.Module):
    """Multi-head cross-attention mechanism for communication between encoders."""
    
    def __init__(self, embed_dim: int = 64, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        if self.head_dim * num_heads != embed_dim:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        
        # Query, Key, Value projections
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Layer norm for residual connection
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, query_repr: torch.Tensor, context_reprs: List[torch.Tensor]) -> torch.Tensor:
        """Perform cross-contextual attention."""
        batch_size = query_repr.size(0)
        
        if not context_reprs:
            return self.layer_norm(query_repr)
        
        # Stack all context representations
        context_stack = torch.stack(context_reprs, dim=1)  # [batch, num_contexts, embed_dim]
        num_contexts = context_stack.size(1)
        
        # Generate queries, keys, values
        queries = self.query_proj(query_repr).unsqueeze(1)  # [batch, 1, embed_dim]
        keys = self.key_proj(context_stack)  # [batch, num_contexts, embed_dim]
        values = self.value_proj(context_stack)  # [batch, num_contexts, embed_dim]
        
        # Reshape for multi-head attention
        queries = queries.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, num_contexts, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, num_contexts, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, values)  # [batch, num_heads, 1, head_dim]
        
        # Reshape and project
        attended = attended.transpose(1, 2).contiguous().view(batch_size, 1, self.embed_dim)
        attended = attended.squeeze(1)  # [batch, embed_dim]
        
        # Output projection
        output = self.out_proj(attended)
        
        # Residual connection and layer norm
        output = self.layer_norm(query_repr + output)
        
        logger.debug(f"CrossContextualAttention output: shape={output.shape}")
        
        return output

class NexusFormer(nn.Module):
    """Enhanced NexusFormer with iterative cross-contextual refinement loops."""
    
    def __init__(self, input_dims: Sequence[int], embed_dim: int = 64, refinement_iterations: int = 3, 
                 encoder_type: str = 'standard', num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        if not isinstance(input_dims, (list, tuple)) or len(input_dims) == 0:
            raise ValueError("NexusFormer requires a non-empty sequence of input dimensions.")
        if any(int(d) <= 0 for d in input_dims):
            raise ValueError(f"All input dimensions must be positive integers, got: {input_dims}")
            
        self.input_dims = [int(d) for d in input_dims]
        self.embed_dim = embed_dim
        self.refinement_iterations = refinement_iterations
        self.num_encoders = len(input_dims)
        
        # Initialize contextual encoders
        self.encoders = nn.ModuleList()
        for i, input_dim in enumerate(self.input_dims):
            if encoder_type == 'standard':
                encoder = StandardTabularEncoder(input_dim, embed_dim, num_heads, dropout=dropout)
            else:
                raise ValueError(f"Unsupported encoder type: {encoder_type}")
            self.encoders.append(encoder)
        
        # Cross-contextual attention modules
        self.cross_attentions = nn.ModuleList([
            CrossContextualAttention(embed_dim, num_heads, dropout) 
            for _ in range(self.num_encoders)
        ])
        
        # Final fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(self.num_encoders * embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )
        
        logger.info(f"NexusFormer initialized: {self.num_encoders} encoders, {refinement_iterations} iterations")
    
    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass with iterative refinement (recycling)."""
        if len(inputs) != len(self.encoders):
            raise ValueError(f"Expected {len(self.encoders)} inputs, got {len(inputs)}")
        
        batch_size = inputs[0].size(0)
        
        # Validate input shapes and batch consistency
        for idx, (x, expected_dim) in enumerate(zip(inputs, self.input_dims)):
            if x.dim() != 2 or x.size(-1) != expected_dim:
                raise ValueError(f"Input {idx} has shape {tuple(x.shape)}, expected [batch, {expected_dim}]")
            if x.size(0) != batch_size:
                raise ValueError(f"Batch size mismatch at input {idx}: {x.size(0)} vs {batch_size}")
        
        # Initial encoding phase
        representations = []
        for i, (encoder, x) in enumerate(zip(self.encoders, inputs)):
            initial_repr = encoder(x)
            representations.append(initial_repr)
            logger.debug(f"Initial encoding {i}: shape={initial_repr.shape}")
        
        # Iterative refinement loop (recycling)
        for iteration in range(self.refinement_iterations):
            logger.debug(f"Refinement iteration {iteration + 1}/{self.refinement_iterations}")
            
            new_representations = []
            
            # Update each encoder's representation using cross-attention
            for i, cross_attention in enumerate(self.cross_attentions):
                # Get context from all OTHER encoders
                context_reprs = [representations[j] for j in range(self.num_encoders) if j != i]
                
                # Update this encoder's representation
                updated_repr = cross_attention(representations[i], context_reprs)
                new_representations.append(updated_repr)
                
                # Log attention statistics
                attention_change = torch.norm(updated_repr - representations[i], p=2, dim=-1).mean()
                logger.debug(f"Encoder {i} attention change: {attention_change.item():.6f}")
            
            representations = new_representations
        
        # Final fusion
        concatenated = torch.cat(representations, dim=-1)  # [batch, num_encoders * embed_dim]
        logger.debug(f"Final concatenated shape: {concatenated.shape}")
        
        output = self.fusion(concatenated).squeeze(-1)  # [batch]
        
        logger.debug(f"NexusFormer final output: shape={output.shape}")
        
        return output