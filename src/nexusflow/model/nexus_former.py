import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from typing import Sequence, List, Optional
import math

class MoELayer(nn.Module):
    """Mixture of Experts layer with dynamic routing."""
    
    def __init__(self, input_dim: int, num_experts: int = 4, expert_dim: int = None, top_k: int = 2, dropout: float = 0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.expert_dim = expert_dim or input_dim * 2
        
        # Gating network
        self.gate = nn.Linear(input_dim, num_experts)
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, self.expert_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.expert_dim, input_dim)
            ) for _ in range(num_experts)
        ])
        
        # Load balancing
        self.load_balance_loss_weight = 0.01
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, input_dim = x.shape
        
        # Compute gating scores
        gate_logits = self.gate(x)  # [batch, num_experts]
        gate_scores = F.softmax(gate_logits, dim=-1)
        
        # Select top-k experts
        top_k_scores, top_k_indices = torch.topk(gate_scores, self.top_k, dim=-1)
        top_k_scores = F.softmax(top_k_scores, dim=-1)
        
        # Apply experts
        output = torch.zeros_like(x)
        
        for i in range(self.top_k):
            expert_indices = top_k_indices[:, i]
            expert_scores = top_k_scores[:, i].unsqueeze(-1)
            
            # Batch process each expert
            for expert_idx in range(self.num_experts):
                mask = expert_indices == expert_idx
                if mask.any():
                    expert_input = x[mask]
                    expert_output = self.experts[expert_idx](expert_input)
                    output[mask] += expert_scores[mask] * expert_output
        
        return output

class FlashAttention(nn.Module):
    """Efficient attention implementation with tiling optimization."""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, block_size: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.block_size = block_size
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        if self.head_dim * num_heads != embed_dim:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv_proj(x).chunk(3, dim=-1)
        q, k, v = [tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) 
                   for tensor in qkv]
        
        # For small sequences, use standard attention
        if seq_len <= self.block_size:
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            out = torch.matmul(attn_weights, v)
        else:
            # Tiled attention for memory efficiency
            out = self._tiled_attention(q, k, v)
        
        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        return self.out_proj(out)
    
    def _tiled_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Tiled attention computation for memory efficiency."""
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        # Simple tiling implementation
        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        outputs = []
        
        for i in range(num_blocks):
            start_i = i * self.block_size
            end_i = min((i + 1) * self.block_size, seq_len)
            
            q_block = q[:, :, start_i:end_i, :]
            block_outputs = []
            
            for j in range(num_blocks):
                start_j = j * self.block_size
                end_j = min((j + 1) * self.block_size, seq_len)
                
                k_block = k[:, :, start_j:end_j, :]
                v_block = v[:, :, start_j:end_j, :]
                
                # Compute attention for this block pair
                scores = torch.matmul(q_block, k_block.transpose(-2, -1)) * self.scale
                attn_weights = F.softmax(scores, dim=-1)
                attn_weights = self.dropout(attn_weights)
                block_out = torch.matmul(attn_weights, v_block)
                
                block_outputs.append(block_out)
            
            # Combine outputs for this query block
            block_output = torch.cat(block_outputs, dim=-2)
            outputs.append(block_output)
        
        return torch.cat(outputs, dim=-2)

class FTTransformerEncoder(nn.Module):
    """Feature Tokenizer Transformer for tabular data."""
    
    def __init__(self, input_dim: int, embed_dim: int = 64, num_heads: int = 4, num_layers: int = 2, 
                 dropout: float = 0.1, use_moe: bool = False, num_experts: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.use_moe = use_moe
        
        # Feature tokenization - each feature becomes a token
        self.feature_embeddings = nn.ModuleList([
            nn.Linear(1, embed_dim) for _ in range(input_dim)
        ])
        
        # CLS token for aggregation
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # Positional embeddings
        self.pos_embeddings = nn.Parameter(torch.randn(1, input_dim + 1, embed_dim) * 0.02)
        
        # Transformer layers with optional MoE
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                'attention': FlashAttention(embed_dim, num_heads, dropout),
                'norm1': nn.LayerNorm(embed_dim),
                'norm2': nn.LayerNorm(embed_dim),
                'dropout': nn.Dropout(dropout)
            })
            
            if use_moe:
                layer['ffn'] = MoELayer(embed_dim, num_experts, dropout=dropout)
            else:
                layer['ffn'] = nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(embed_dim * 2, embed_dim)
                )
            
            self.layers.append(layer)
        
        self.final_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Tokenize features
        feature_tokens = []
        for i in range(self.input_dim):
            feature_val = x[:, i:i+1]  # [batch, 1]
            token = self.feature_embeddings[i](feature_val)  # [batch, embed_dim]
            feature_tokens.append(token.unsqueeze(1))  # [batch, 1, embed_dim]
        
        # Stack feature tokens
        tokens = torch.cat(feature_tokens, dim=1)  # [batch, input_dim, embed_dim]
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)  # [batch, input_dim+1, embed_dim]
        
        # Add positional embeddings
        tokens = tokens + self.pos_embeddings
        
        # Apply transformer layers
        for layer in self.layers:
            # Self-attention with residual
            attn_out = layer['attention'](tokens)
            tokens = layer['norm1'](tokens + layer['dropout'](attn_out))
            
            # Feed-forward with residual
            ffn_out = layer['ffn'](tokens)
            tokens = layer['norm2'](tokens + layer['dropout'](ffn_out))
        
        # Extract CLS token representation
        cls_output = tokens[:, 0, :]  # [batch, embed_dim]
        output = self.final_norm(cls_output)
        
        #logger.debug(f"FTTransformerEncoder output: shape={output.shape}")
        return output

class TabNetEncoder(nn.Module):
    """TabNet encoder with sequential attention mechanism."""
    
    def __init__(self, input_dim: int, embed_dim: int = 64, num_steps: int = 3, 
                 gamma: float = 1.3, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_steps = num_steps
        self.gamma = gamma
        
        # Feature transformer
        self.initial_bn = nn.BatchNorm1d(input_dim)
        self.feature_transformer = nn.Linear(input_dim, embed_dim)
        
        # Attentive transformer for feature selection
        self.attentive_transformer = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.BatchNorm1d(embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, input_dim)
            ) for _ in range(num_steps)
        ])
        
        # Decision steps
        self.decision_steps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for _ in range(num_steps)
        ])
        
        self.final_projection = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Initial processing
        x_norm = self.initial_bn(x)
        features = self.feature_transformer(x_norm)
        
        # Initialize mask and decision aggregator
        mask = torch.ones(batch_size, self.input_dim, device=x.device)
        decision_out = torch.zeros(batch_size, self.embed_dim, device=x.device)
        
        # Sequential decision steps
        for step in range(self.num_steps):
            # Attentive feature selection
            att_out = self.attentive_transformer[step](features)
            att_weights = F.softmax(att_out * mask, dim=-1)
            
            # Update mask for next step (sparsity enforcement)
            mask = mask * (self.gamma - att_weights)
            
            # Weighted feature aggregation
            selected_features = torch.sum(att_weights.unsqueeze(-1) * features.unsqueeze(1), dim=1)
            
            # Decision step
            step_out = self.decision_steps[step](selected_features)
            decision_out += step_out
        
        # Final projection
        output = self.final_projection(decision_out)
        
        #logger.debug(f"TabNetEncoder output: shape={output.shape}")
        return output

class ContextualEncoder(nn.Module):
    """Enhanced abstract base class for all contextual encoders."""
    
    def __init__(self, input_dim: int, embed_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the encoder."""
        raise NotImplementedError("Subclasses must implement forward method")

class StandardTabularEncoder(ContextualEncoder):
    """Enhanced transformer-based encoder with FlashAttention and optional MoE."""
    
    def __init__(self, input_dim: int, embed_dim: int = 64, num_heads: int = 4, num_layers: int = 2, 
                 dropout: float = 0.1, use_moe: bool = False, num_experts: int = 4, use_flash_attn: bool = True):
        super().__init__(input_dim, embed_dim)
        self.use_moe = use_moe
        self.use_flash_attn = use_flash_attn
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, embed_dim)
        
        # Build transformer layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                'norm1': nn.LayerNorm(embed_dim),
                'norm2': nn.LayerNorm(embed_dim),
                'dropout': nn.Dropout(dropout)
            })
            
            # Attention mechanism
            if use_flash_attn:
                layer['attention'] = FlashAttention(embed_dim, num_heads, dropout)
            else:
                # Fallback to standard attention
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=embed_dim * 2,
                    dropout=dropout,
                    activation='relu',
                    batch_first=True
                )
                layer['attention'] = encoder_layer
            
            # Feed-forward network
            if use_moe:
                layer['ffn'] = MoELayer(embed_dim, num_experts, dropout=dropout)
            else:
                layer['ffn'] = nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(embed_dim * 2, embed_dim)
                )
            
            self.layers.append(layer)
        
        # Final layer norm
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Project input to embedding dimension
        x = self.input_projection(x)  # [batch, embed_dim]
        
        # Add positional encoding and expand for transformer
        x = x.unsqueeze(1) + self.pos_embedding  # [batch, 1, embed_dim]
        
        # Apply transformer layers
        for layer in self.layers:
            if self.use_flash_attn:
                # Manual residual connection for FlashAttention
                attn_out = layer['attention'](x)
                x = layer['norm1'](x + layer['dropout'](attn_out))
                
                ffn_out = layer['ffn'](x.squeeze(1)).unsqueeze(1)
                x = layer['norm2'](x + layer['dropout'](ffn_out))
            else:
                # Standard transformer layer handles residuals
                x = layer['attention'](x)
        
        # Remove sequence dimension and normalize
        x = x.squeeze(1)  # [batch, embed_dim]
        x = self.layer_norm(x)
        
        #logger.debug(f"StandardTabularEncoder output: shape={x.shape} mean={x.mean().item():.4f}")
        return x

class CrossContextualAttention(nn.Module):
    """Enhanced multi-head cross-attention with FlashAttention and efficiency optimizations."""
    
    def __init__(self, embed_dim: int = 64, num_heads: int = 4, dropout: float = 0.1, 
                 top_k: int = None, use_flash_attn: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.top_k = top_k
        self.use_flash_attn = use_flash_attn
        
        if self.head_dim * num_heads != embed_dim:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        
        # Attention mechanism
        if use_flash_attn:
            self.flash_attention = FlashAttention(embed_dim, num_heads, dropout)
        else:
            # Query, Key, Value projections
            self.query_proj = nn.Linear(embed_dim, embed_dim)
            self.key_proj = nn.Linear(embed_dim, embed_dim)
            self.value_proj = nn.Linear(embed_dim, embed_dim)
            self.out_proj = nn.Linear(embed_dim, embed_dim)
            self.dropout = nn.Dropout(dropout)
        
        # Layer norm for residual connection
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Context importance weights
        self.context_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
        
    def forward(self, query_repr: torch.Tensor, context_reprs: List[torch.Tensor]) -> torch.Tensor:
        """Enhanced cross-contextual attention with adaptive context weighting."""
        batch_size = query_repr.size(0)
        
        if not context_reprs:
            return self.layer_norm(query_repr)
        
        # Apply top-k selection with improved similarity metric
        if self.top_k is not None and len(context_reprs) > self.top_k:
            similarities = []
            for ctx in context_reprs:
                # Combined cosine similarity and L2 distance
                cos_sim = F.cosine_similarity(query_repr, ctx, dim=-1).mean()
                l2_dist = torch.norm(query_repr - ctx, dim=-1).mean()
                combined_sim = cos_sim - 0.1 * l2_dist  # Weighted combination
                similarities.append(combined_sim)
            
            _, top_indices = torch.topk(torch.stack(similarities), self.top_k)
            context_reprs = [context_reprs[i] for i in top_indices.tolist()]
            
            #logger.debug(f"Selected top-{self.top_k} contexts from {len(similarities)} candidates")
        
        # Stack contexts for efficient processing
        context_stack = torch.stack(context_reprs, dim=1)  # [batch, num_contexts, embed_dim]
        
        if self.use_flash_attn:
            # Prepare input for FlashAttention: [query, context1, context2, ...]
            combined_input = torch.cat([query_repr.unsqueeze(1), context_stack], dim=1)
            
            # Apply FlashAttention
            attended_output = self.flash_attention(combined_input)
            
            # Extract query output (first token)
            output = attended_output[:, 0, :]  # [batch, embed_dim]
        else:
            # Standard cross-attention implementation
            num_contexts = context_stack.size(1)
            
            # Generate queries, keys, values
            queries = self.query_proj(query_repr).unsqueeze(1)  # [batch, 1, embed_dim]
            keys = self.key_proj(context_stack)  # [batch, num_contexts, embed_dim]
            values = self.value_proj(context_stack)  # [batch, num_contexts, embed_dim]
            
            # Reshape for multi-head attention
            queries = queries.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
            keys = keys.view(batch_size, num_contexts, self.num_heads, self.head_dim).transpose(1, 2)
            values = values.view(batch_size, num_contexts, self.num_heads, self.head_dim).transpose(1, 2)
            
            # Compute attention
            scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attention_weights = F.softmax(scores, dim=-1)
            attention_weights = self.dropout(attention_weights)
            
            attended = torch.matmul(attention_weights, values)  # [batch, num_heads, 1, head_dim]
            attended = attended.transpose(1, 2).contiguous().view(batch_size, 1, self.embed_dim)
            output = self.out_proj(attended.squeeze(1))
        
        # Adaptive context gating
        gate_input = torch.cat([query_repr, output], dim=-1)
        gate_weight = self.context_gate(gate_input)
        output = gate_weight * output + (1 - gate_weight) * query_repr
        
        # Residual connection and layer norm
        output = self.layer_norm(query_repr + output)
        
        #logger.debug(f"CrossContextualAttention output: shape={output.shape}")
        return output

class NexusFormer(nn.Module):
    """Enhanced NexusFormer with advanced tabular architectures and efficiency optimizations."""
    
    def __init__(self, input_dims: Sequence[int], embed_dim: int = 64, refinement_iterations: int = 3, 
                 encoder_type: str = 'standard', num_heads: int = 4, dropout: float = 0.1,
                 use_moe: bool = False, num_experts: int = 4, use_flash_attn: bool = True):
        super().__init__()
        
        if not isinstance(input_dims, (list, tuple)) or len(input_dims) == 0:
            raise ValueError("NexusFormer requires a non-empty sequence of input dimensions.")
        if any(int(d) <= 0 for d in input_dims):
            raise ValueError(f"All input dimensions must be positive integers, got: {input_dims}")
            
        self.input_dims = [int(d) for d in input_dims]
        self.embed_dim = embed_dim
        self.refinement_iterations = refinement_iterations
        self.num_encoders = len(input_dims)
        self.encoder_type = encoder_type
        self.use_moe = use_moe
        self.use_flash_attn = use_flash_attn
        
        # Initialize advanced encoders
        self.encoders = nn.ModuleList()
        for i, input_dim in enumerate(self.input_dims):
            if encoder_type == 'standard':
                encoder = StandardTabularEncoder(
                    input_dim, embed_dim, num_heads, 
                    dropout=dropout, use_moe=use_moe, 
                    num_experts=num_experts, use_flash_attn=use_flash_attn
                )
            elif encoder_type == 'ft_transformer':
                encoder = FTTransformerEncoder(
                    input_dim, embed_dim, num_heads, 
                    dropout=dropout, use_moe=use_moe, num_experts=num_experts
                )
            elif encoder_type == 'tabnet':
                encoder = TabNetEncoder(input_dim, embed_dim, dropout=dropout)
            else:
                raise ValueError(f"Unsupported encoder type: {encoder_type}")
            
            self.encoders.append(encoder)
        
        # Enhanced cross-contextual attention modules
        self.cross_attentions = nn.ModuleList([
            CrossContextualAttention(
                embed_dim, num_heads, dropout, 
                use_flash_attn=use_flash_attn
            ) for _ in range(self.num_encoders)
        ])
        
        # Adaptive fusion with attention pooling
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.fusion_norm = nn.LayerNorm(embed_dim)
        
        # Final prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, 1)
        )
        
        logger.info(f"Enhanced NexusFormer initialized: {self.num_encoders} {encoder_type} encoders, "
                   f"{refinement_iterations} iterations, MoE={use_moe}, FlashAttn={use_flash_attn}")
    
    def forward(self, inputs: List[torch.Tensor], key_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Enhanced forward pass with adaptive fusion and key features support."""
        if len(inputs) != len(self.encoders):
            raise ValueError(f"Expected {len(self.encoders)} inputs, got {len(inputs)}")
        
        batch_size = inputs[0].size(0)
        
        # Validate input shapes and batch consistency
        for idx, (x, expected_dim) in enumerate(zip(inputs, self.input_dims)):
            if x.dim() != 2 or x.size(-1) != expected_dim:
                raise ValueError(f"Input {idx} has shape {tuple(x.shape)}, expected [batch, {expected_dim}]")
            if x.size(0) != batch_size:
                raise ValueError(f"Batch size mismatch at input {idx}: {x.size(0)} vs {batch_size}")
        
        # Store key features for potential aggregation needs (Phase 4 support)
        if key_features is not None:
            self._current_key_features = key_features
            # For now, we don't use key_features in the forward pass
            # but store them for potential future use in prediction aggregation
        
        # Initial encoding phase
        representations = []
        for i, (encoder, x) in enumerate(zip(self.encoders, inputs)):
            initial_repr = encoder(x)
            representations.append(initial_repr)
            #logger.debug(f"Initial encoding {i}: shape={initial_repr.shape} mean={initial_repr.mean():.4f}")
        
        # Iterative refinement loop with convergence tracking
        for iteration in range(self.refinement_iterations):
            #logger.debug(f"Refinement iteration {iteration + 1}/{self.refinement_iterations}")
            
            new_representations = []
            total_change = 0.0
            
            # Update each encoder's representation using cross-attention
            for i, cross_attention in enumerate(self.cross_attentions):
                # Get context from all OTHER encoders
                context_reprs = [representations[j] for j in range(self.num_encoders) if j != i]
                
                # Update this encoder's representation
                updated_repr = cross_attention(representations[i], context_reprs)
                new_representations.append(updated_repr)
                
                # Track convergence
                change = torch.norm(updated_repr - representations[i], p=2, dim=-1).mean()
                total_change += change.item()
                #logger.debug(f"Encoder {i} refinement change: {change.item():.6f}")
            
            representations = new_representations
            
            # Early stopping if convergence achieved
            avg_change = total_change / self.num_encoders
            if avg_change < 1e-6:
                #logger.debug(f"Convergence achieved at iteration {iteration + 1}")
                break
        
        # Adaptive fusion using attention pooling
        if self.num_encoders > 1:
            # Stack representations for attention pooling
            stacked_reprs = torch.stack(representations, dim=1)  # [batch, num_encoders, embed_dim]
            
            # Self-attention for adaptive weighting
            attended_reprs, attn_weights = self.fusion_attention(
                stacked_reprs, stacked_reprs, stacked_reprs
            )
            
            # Residual connection and normalization
            fused_repr = self.fusion_norm(stacked_reprs + attended_reprs)
            
            # Global pooling (mean)
            final_repr = fused_repr.mean(dim=1)  # [batch, embed_dim]
        else:
            final_repr = representations[0]
        
        # Final prediction
        output = self.prediction_head(final_repr).squeeze(-1)  # [batch]
        
        #logger.debug(f"NexusFormer final output: shape={output.shape} mean={output.mean():.4f}")
        
        return output