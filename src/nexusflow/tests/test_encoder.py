"""Unit tests for NexusFlow encoder modules."""
import pytest
import torch
import torch.nn as nn
import numpy as np

from nexusflow.model.nexus_former import (
    ContextualEncoder, StandardTabularEncoder, CrossContextualAttention, NexusFormer, FlashAttention
)


class TestContextualEncoder:
    """Test the abstract ContextualEncoder base class."""
    
    def test_abstract_encoder_init(self):
        """Test that ContextualEncoder can be initialized."""
        encoder = ContextualEncoder(input_dim=10, embed_dim=64)
        assert encoder.input_dim == 10
        assert encoder.embed_dim == 64
    
    def test_abstract_encoder_forward_not_implemented(self):
        """Test that abstract encoder forward raises NotImplementedError."""
        encoder = ContextualEncoder(input_dim=10, embed_dim=64)
        x = torch.randn(2, 10)
        
        with pytest.raises(NotImplementedError):
            encoder.forward(x)


class TestStandardTabularEncoder:
    """Test the StandardTabularEncoder implementation."""
    
    def test_encoder_initialization(self):
        """Test encoder initialization with different parameters."""
        encoder = StandardTabularEncoder(input_dim=10, embed_dim=64, num_heads=4, num_layers=2)
        
        assert encoder.input_dim == 10
        assert encoder.embed_dim == 64
        assert isinstance(encoder.input_projection, nn.Linear)
        assert isinstance(encoder.layers, nn.ModuleList)
        assert isinstance(encoder.layer_norm, nn.LayerNorm)
        
        # Check input projection dimensions
        assert encoder.input_projection.in_features == 10
        assert encoder.input_projection.out_features == 64

    def test_encoder_forward_pass_shape(self):
        """Test that encoder forward pass returns correct shape."""
        input_dim, embed_dim = 10, 64
        batch_size = 4
        
        encoder = StandardTabularEncoder(input_dim=input_dim, embed_dim=embed_dim)
        x = torch.randn(batch_size, input_dim)
        
        output = encoder(x)
        
        assert output.shape == (batch_size, embed_dim)
        assert output.dtype == torch.float32

    def test_encoder_forward_pass_different_batch_sizes(self):
        """Test encoder with different batch sizes."""
        input_dim, embed_dim = 5, 32
        encoder = StandardTabularEncoder(input_dim=input_dim, embed_dim=embed_dim)
        
        for batch_size in [1, 3, 8, 16]:
            x = torch.randn(batch_size, input_dim)
            output = encoder(x)
            assert output.shape == (batch_size, embed_dim)

    def test_encoder_deterministic_output(self):
        """Test that encoder produces deterministic output for same input."""
        torch.manual_seed(42)
        encoder = StandardTabularEncoder(input_dim=5, embed_dim=16)
        x = torch.randn(2, 5)
        
        # Put in eval mode for deterministic behavior
        encoder.eval()
        
        with torch.no_grad():
            output1 = encoder(x)
            output2 = encoder(x)
        
        torch.testing.assert_close(output1, output2)

    def test_encoder_gradient_flow(self):
        """Test that gradients flow through encoder."""
        encoder = StandardTabularEncoder(input_dim=5, embed_dim=16)
        x = torch.randn(2, 5, requires_grad=True)
        
        output = encoder(x)
        loss = output.sum()
        loss.backward()
        
        # Check that input gradients exist
        assert x.grad is not None
        assert x.grad.shape == x.shape
        
        # Check that encoder parameters have gradients
        for param in encoder.parameters():
            assert param.grad is not None

    def test_encoder_with_different_architectures(self):
        """Test encoder with different architectural parameters."""
        configs = [
            {'num_heads': 2, 'num_layers': 1},
            {'num_heads': 8, 'num_layers': 3}, 
            {'num_heads': 4, 'num_layers': 2, 'dropout': 0.2}
        ]
        
        for config in configs:
            encoder = StandardTabularEncoder(input_dim=12, embed_dim=64, **config)
            x = torch.randn(3, 12)
            output = encoder(x)
            assert output.shape == (3, 64)


class TestCrossContextualAttention:
    """Test the CrossContextualAttention module."""
    
    def test_attention_initialization(self):
        """Test CrossContextualAttention initialization."""
        embed_dim, num_heads = 64, 4
        flashattention = CrossContextualAttention(embed_dim=embed_dim, num_heads=num_heads)
        attention = CrossContextualAttention(embed_dim=embed_dim, num_heads=num_heads, use_flash_attn=False)
        
        assert attention.embed_dim == embed_dim
        assert attention.num_heads == num_heads
        assert attention.head_dim == embed_dim // num_heads
        
        # Check linear layers
        assert isinstance(attention.query_proj, nn.Linear)
        assert isinstance(flashattention.flash_attention, FlashAttention)
        assert isinstance(attention.key_proj, nn.Linear)
        assert isinstance(attention.value_proj, nn.Linear)
        assert isinstance(attention.out_proj, nn.Linear)

    def test_attention_invalid_embed_dim(self):
        """Test that invalid embed_dim raises ValueError."""
        with pytest.raises(ValueError, match="embed_dim.*must be divisible by num_heads"):
            CrossContextualAttention(embed_dim=65, num_heads=4)  # 65 not divisible by 4

    def test_attention_forward_with_context(self):
        """Test attention forward pass with context representations."""
        embed_dim = 64
        batch_size = 3
        
        attention = CrossContextualAttention(embed_dim=embed_dim, num_heads=4)
        
        # Create query and context representations
        query_repr = torch.randn(batch_size, embed_dim)
        context_reprs = [
            torch.randn(batch_size, embed_dim),
            torch.randn(batch_size, embed_dim)
        ]
        
        output = attention(query_repr, context_reprs)
        
        assert output.shape == (batch_size, embed_dim)
        assert output.dtype == torch.float32

    def test_attention_forward_no_context(self):
        """Test attention forward pass with empty context (should return normalized query)."""
        embed_dim = 64
        batch_size = 2
        
        attention = CrossContextualAttention(embed_dim=embed_dim, num_heads=4)
        query_repr = torch.randn(batch_size, embed_dim)
        
        output = attention(query_repr, [])
        
        assert output.shape == (batch_size, embed_dim)
        # Output should be layer-normalized version of query
        expected_output = attention.layer_norm(query_repr)
        torch.testing.assert_close(output, expected_output)

    def test_attention_gradient_flow(self):
        """Test that gradients flow through attention mechanism."""
        attention = CrossContextualAttention(embed_dim=32, num_heads=4)
        
        query = torch.randn(2, 32, requires_grad=True)
        context = [torch.randn(2, 32, requires_grad=True)]
        
        output = attention(query, context)
        loss = output.sum()
        loss.backward()
        
        assert query.grad is not None
        assert context[0].grad is not None

    def test_attention_with_single_context(self):
        """Test attention with single context representation."""
        attention = CrossContextualAttention(embed_dim=32, num_heads=2)
        
        query = torch.randn(4, 32)
        context = [torch.randn(4, 32)]
        
        output = attention(query, context)
        assert output.shape == (4, 32)

    def test_attention_with_multiple_contexts(self):
        """Test attention with multiple context representations."""
        attention = CrossContextualAttention(embed_dim=48, num_heads=3)
        
        query = torch.randn(2, 48)
        contexts = [torch.randn(2, 48) for _ in range(5)]
        
        output = attention(query, contexts)
        assert output.shape == (2, 48)


class TestNexusFormer:
    """Test the complete NexusFormer model."""
    
    def test_nexus_former_initialization(self):
        """Test NexusFormer initialization."""
        input_dims = [10, 15, 8]
        embed_dim = 64
        refinement_iterations = 3
        
        model = NexusFormer(
            input_dims=input_dims,
            embed_dim=embed_dim,
            refinement_iterations=refinement_iterations
        )
        
        assert model.input_dims == input_dims
        assert model.embed_dim == embed_dim
        assert model.refinement_iterations == refinement_iterations
        assert model.num_encoders == 3
        assert len(model.encoders) == 3
        assert len(model.cross_attentions) == 3

    def test_nexus_former_invalid_input_dims(self):
        """Test that invalid input dimensions raise ValueError."""
        # Empty input dims
        with pytest.raises(ValueError, match="non-empty sequence"):
            NexusFormer(input_dims=[])
        
        # Non-positive dimensions
        with pytest.raises(ValueError, match="positive integers"):
            NexusFormer(input_dims=[10, -5, 8])
        
        # Non-list input
        with pytest.raises(ValueError, match="non-empty sequence"):
            NexusFormer(input_dims=None)

    def test_nexus_former_forward_pass(self):
        """Test NexusFormer forward pass."""
        input_dims = [5, 8, 3]
        batch_size = 4
        
        model = NexusFormer(input_dims=input_dims, embed_dim=32, refinement_iterations=2)
        
        # Create input tensors
        inputs = [torch.randn(batch_size, dim) for dim in input_dims]
        
        output = model(inputs)
        
        assert output.shape == (batch_size,)
        assert output.dtype == torch.float32

    def test_nexus_former_forward_invalid_inputs(self):
        """Test NexusFormer forward pass with invalid inputs."""
        model = NexusFormer(input_dims=[5, 8], embed_dim=32)
        
        # Wrong number of inputs
        with pytest.raises(ValueError, match="Expected 2 inputs, got 1"):
            model([torch.randn(2, 5)])
        
        # Wrong input dimensions
        with pytest.raises(ValueError, match="expected.*5"):
            inputs = [torch.randn(2, 3), torch.randn(2, 8)]  # First input has wrong dim
            model(inputs)
        
        # Inconsistent batch sizes
        with pytest.raises(ValueError, match="Batch size mismatch"):
            inputs = [torch.randn(2, 5), torch.randn(3, 8)]  # Different batch sizes
            model(inputs)

    def test_nexus_former_single_encoder(self):
        """Test NexusFormer with single encoder."""
        model = NexusFormer(input_dims=[10], embed_dim=32, refinement_iterations=1)
        
        inputs = [torch.randn(3, 10)]
        output = model(inputs)
        
        assert output.shape == (3,)

    def test_nexus_former_refinement_iterations(self):
        """Test NexusFormer with different refinement iteration counts."""
        input_dims = [6, 4]
        
        for iterations in [0, 1, 3, 5]:
            model = NexusFormer(
                input_dims=input_dims,
                embed_dim=32,
                refinement_iterations=iterations
            )
            
            inputs = [torch.randn(2, dim) for dim in input_dims]
            output = model(inputs)
            
            assert output.shape == (2,)

    def test_nexus_former_gradient_flow(self):
        """Test gradient flow through NexusFormer."""
        model = NexusFormer(input_dims=[4, 6], embed_dim=16, refinement_iterations=1)
        
        inputs = [torch.randn(2, 4, requires_grad=True), torch.randn(2, 6, requires_grad=True)]
        
        output = model(inputs)
        loss = output.sum()
        loss.backward()
        
        # Check input gradients
        for inp in inputs:
            assert inp.grad is not None
        
        # Check model parameter gradients
        for param in model.parameters():
            assert param.grad is not None

    def test_nexus_former_deterministic_eval(self):
        """Test that NexusFormer produces deterministic output in eval mode."""
        torch.manual_seed(42)
        model = NexusFormer(input_dims=[3, 5], embed_dim=16, refinement_iterations=2)
        
        inputs = [torch.randn(2, 3), torch.randn(2, 5)]
        
        model.eval()
        with torch.no_grad():
            output1 = model(inputs)
            output2 = model(inputs)
        
        torch.testing.assert_close(output1, output2)

    def test_nexus_former_training_mode(self):
        """Test NexusFormer in training vs eval mode."""
        model = NexusFormer(input_dims=[4], embed_dim=16, refinement_iterations=1)
        inputs = [torch.randn(2, 4)]
        
        # Training mode
        model.train()
        output_train = model(inputs)
        
        # Eval mode
        model.eval() 
        with torch.no_grad():
            output_eval = model(inputs)
        
        # Outputs might be different due to dropout, but shapes should match
        assert output_train.shape == output_eval.shape

    def test_nexus_former_large_model(self):
        """Test NexusFormer with larger configuration."""
        input_dims = [20, 30, 15, 25]
        model = NexusFormer(
            input_dims=input_dims,
            embed_dim=128,
            refinement_iterations=4,
            num_heads=8
        )
        
        inputs = [torch.randn(8, dim) for dim in input_dims]
        output = model(inputs)
        
        assert output.shape == (8,)
        
        # Check that model has reasonable number of parameters
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 1000  # Should have substantial number of parameters

    def test_nexus_former_different_encoder_types(self):
        """Test NexusFormer initialization with different encoder types."""
        # Standard encoder (default)
        model = NexusFormer(input_dims=[5, 8], encoder_type='standard')
        assert len(model.encoders) == 2
        
        # Invalid encoder type should raise error
        with pytest.raises(ValueError, match="Unsupported encoder type"):
            NexusFormer(input_dims=[5], encoder_type='invalid')

    def test_nexus_former_parameter_count(self):
        """Test that NexusFormer parameter count scales reasonably."""
        base_model = NexusFormer(input_dims=[10], embed_dim=32, refinement_iterations=1)
        base_params = sum(p.numel() for p in base_model.parameters())
        
        # Larger model should have more parameters
        large_model = NexusFormer(input_dims=[10, 10], embed_dim=64, refinement_iterations=2)
        large_params = sum(p.numel() for p in large_model.parameters())
        
        assert large_params > base_params


if __name__ == '__main__':
    pytest.main([__file__, '-v'])