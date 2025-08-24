"""Unit tests for encoder components."""
import pytest
import torch
from nexusflow.model.nexus_former import SimpleEncoder, NexusFormer


class TestSimpleEncoder:
    """Test cases for SimpleEncoder."""
    
    def test_forward_pass_shape(self):
        """Test that SimpleEncoder returns expected output shape."""
        input_dim = 10
        embed_dim = 64
        batch_size = 4
        
        encoder = SimpleEncoder(input_dim=input_dim, embed_dim=embed_dim)
        input_tensor = torch.randn(batch_size, input_dim)
        
        output = encoder(input_tensor)
        
        assert output.shape == (batch_size, embed_dim), f"Expected shape ({batch_size}, {embed_dim}), got {output.shape}"
        assert output.dtype == torch.float32
    
    def test_different_embed_dims(self):
        """Test SimpleEncoder with different embedding dimensions."""
        input_dim = 5
        batch_size = 2
        
        for embed_dim in [32, 128, 256]:
            encoder = SimpleEncoder(input_dim=input_dim, embed_dim=embed_dim)
            input_tensor = torch.randn(batch_size, input_dim)
            
            output = encoder(input_tensor)
            assert output.shape == (batch_size, embed_dim)
    
    def test_gradient_flow(self):
        """Test that gradients flow through the encoder."""
        encoder = SimpleEncoder(input_dim=3, embed_dim=8)
        input_tensor = torch.randn(2, 3, requires_grad=True)
        
        output = encoder(input_tensor)
        loss = output.sum()
        loss.backward()
        
        # Check that input gradients exist
        assert input_tensor.grad is not None
        assert input_tensor.grad.shape == input_tensor.shape


class TestNexusFormer:
    """Test cases for NexusFormer model."""
    
    def test_single_input_forward_pass(self):
        """Test NexusFormer with single input."""
        input_dims = [10]
        embed_dim = 32
        batch_size = 3
        
        model = NexusFormer(input_dims=input_dims, embed_dim=embed_dim)
        inputs = [torch.randn(batch_size, input_dims[0])]
        
        output = model(inputs)
        
        assert output.shape == (batch_size,), f"Expected shape ({batch_size},), got {output.shape}"
        assert output.dtype == torch.float32
    
    def test_multi_input_forward_pass(self):
        """Test NexusFormer with multiple inputs."""
        input_dims = [5, 8, 12]
        embed_dim = 64
        batch_size = 4
        
        model = NexusFormer(input_dims=input_dims, embed_dim=embed_dim)
        inputs = [torch.randn(batch_size, dim) for dim in input_dims]
        
        output = model(inputs)
        
        assert output.shape == (batch_size,), f"Expected shape ({batch_size},), got {output.shape}"
        assert output.dtype == torch.float32
    
    def test_invalid_input_dimensions_raises_error(self):
        """Test that NexusFormer raises error for invalid input dimensions."""
        with pytest.raises(ValueError, match="non-empty sequence"):
            NexusFormer(input_dims=[])
        
        with pytest.raises(ValueError, match="positive integers"):
            NexusFormer(input_dims=[5, 0, 3])
        
        with pytest.raises(ValueError, match="positive integers"):
            NexusFormer(input_dims=[-1, 5])
    
    def test_mismatched_input_count_raises_error(self):
        """Test that providing wrong number of inputs raises error."""
        model = NexusFormer(input_dims=[5, 8])
        
        # Too few inputs
        with pytest.raises(ValueError, match="expected 2 inputs, got 1"):
            model([torch.randn(2, 5)])
        
        # Too many inputs
        with pytest.raises(ValueError, match="expected 2 inputs, got 3"):
            model([torch.randn(2, 5), torch.randn(2, 8), torch.randn(2, 10)])
    
    def test_mismatched_feature_dimensions_raises_error(self):
        """Test that wrong feature dimensions raise error."""
        model = NexusFormer(input_dims=[5, 8])
        
        with pytest.raises(ValueError, match="expected \\[batch, 5\\]"):
            model([torch.randn(2, 3), torch.randn(2, 8)])  # Wrong first dim
        
        with pytest.raises(ValueError, match="expected \\[batch, 8\\]"):
            model([torch.randn(2, 5), torch.randn(2, 10)])  # Wrong second dim
    
    def test_mismatched_batch_sizes_raises_error(self):
        """Test that mismatched batch sizes raise error."""
        model = NexusFormer(input_dims=[5, 8])
        
        with pytest.raises(ValueError, match="same batch size"):
            model([torch.randn(2, 5), torch.randn(3, 8)])  # Different batch sizes
    
    def test_gradient_flow_multi_input(self):
        """Test gradient flow through multi-input NexusFormer."""
        model = NexusFormer(input_dims=[3, 4], embed_dim=16)
        inputs = [
            torch.randn(2, 3, requires_grad=True),
            torch.randn(2, 4, requires_grad=True)
        ]
        
        output = model(inputs)
        loss = output.sum()
        loss.backward()
        
        # Check that gradients exist for all inputs
        for inp in inputs:
            assert inp.grad is not None
            assert inp.grad.shape == inp.shape
    
    def test_different_embed_dims(self):
        """Test NexusFormer with different embedding dimensions."""
        input_dims = [6, 9]
        batch_size = 3
        
        for embed_dim in [16, 32, 128]:
            model = NexusFormer(input_dims=input_dims, embed_dim=embed_dim)
            inputs = [torch.randn(batch_size, dim) for dim in input_dims]
            
            output = model(inputs)
            assert output.shape == (batch_size,)