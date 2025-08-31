"""Post-training optimization module for NexusFlow models."""
import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic
from torch.nn.utils import prune
from loguru import logger
from pathlib import Path
import time
from typing import Dict, Any, Optional, Union


def _extract_pytorch_model(model_input) -> nn.Module:
    """Extract PyTorch model from either raw model or NexusFlowModelArtifact."""
    if hasattr(model_input, 'model'):
        # It's a NexusFlowModelArtifact
        return model_input.model
    elif isinstance(model_input, nn.Module):
        # It's already a PyTorch model
        return model_input
    else:
        raise ValueError(f"Expected nn.Module or NexusFlowModelArtifact, got {type(model_input)}")


def quantize_model(model_input, qconfig_spec: Optional[Dict] = None):
    """
    Apply dynamic quantization to reduce model size and improve inference speed.
    
    Args:
        model_input: PyTorch model or NexusFlowModelArtifact to quantize
        qconfig_spec: Optional quantization configuration specification
        
    Returns:
        Quantized model (same type as input)
    """
    logger.info("🔧 Starting dynamic quantization...")
    
    # Extract PyTorch model
    pytorch_model = _extract_pytorch_model(model_input)
    
    # Calculate original model size
    original_size = sum(p.numel() * 4 for p in pytorch_model.parameters()) / (1024 * 1024)  # MB
    logger.info(f"   Original model size: {original_size:.2f} MB")
    
    # Define layers to quantize (typically Linear layers)
    qconfig_spec = qconfig_spec or {nn.Linear}
    
    # Apply dynamic quantization
    start_time = time.time()
    quantized_pytorch_model = quantize_dynamic(
        pytorch_model, 
        qconfig_spec, 
        dtype=torch.qint8
    )
    quantization_time = time.time() - start_time
    
    # Calculate quantized model size
    quantized_size = sum(
        param.numel() * (1 if param.dtype == torch.qint8 else 4) 
        for param in quantized_pytorch_model.parameters()
    ) / (1024 * 1024)
    
    size_reduction = ((original_size - quantized_size) / original_size) * 100
    
    logger.info(f"✅ Dynamic quantization complete:")
    logger.info(f"   Quantized model size: {quantized_size:.2f} MB")
    logger.info(f"   Size reduction: {size_reduction:.1f}%")
    logger.info(f"   Quantization time: {quantization_time:.2f}s")
    logger.info(f"   Target layers: {len([m for m in pytorch_model.modules() if type(m) in qconfig_spec])} Linear layers")
    
    # Return the same type as input
    if hasattr(model_input, 'model'):
        # Update the artifact's model
        model_input.model = quantized_pytorch_model
        return model_input
    else:
        return quantized_pytorch_model


def prune_model(model_input, amount: float = 0.2):
    """
    Apply global unstructured pruning to reduce model parameters.
    
    Args:
        model_input: PyTorch model or NexusFlowModelArtifact to prune
        amount: Fraction of parameters to prune (0.0 to 1.0)
        
    Returns:
        Pruned model (same type as input)
    """
    logger.info(f"✂️  Starting global unstructured pruning (amount={amount:.1%})...")
    
    # Extract PyTorch model
    pytorch_model = _extract_pytorch_model(model_input)
    
    # Calculate original statistics
    original_params = sum(p.numel() for p in pytorch_model.parameters())
    original_size = original_params * 4 / (1024 * 1024)  # MB
    
    logger.info(f"   Original parameters: {original_params:,}")
    logger.info(f"   Original model size: {original_size:.2f} MB")
    
    # Collect all Linear layers for pruning
    parameters_to_prune = []
    linear_layers = []
    
    for name, module in pytorch_model.named_modules():
        if isinstance(module, nn.Linear):
            parameters_to_prune.append((module, 'weight'))
            linear_layers.append(name)
    
    logger.info(f"   Found {len(parameters_to_prune)} Linear layers to prune")
    
    if not parameters_to_prune:
        logger.warning("   No Linear layers found - skipping pruning")
        return model_input
    
    # Apply global unstructured pruning
    start_time = time.time()
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    
    # Permanently remove pruned weights
    for module, param_name in parameters_to_prune:
        prune.remove(module, param_name)
    
    pruning_time = time.time() - start_time
    
    # Calculate post-pruning statistics
    pruned_params = sum(p.numel() for p in pytorch_model.parameters())
    pruned_size = pruned_params * 4 / (1024 * 1024)
    
    actual_sparsity = (original_params - pruned_params) / original_params
    size_reduction = ((original_size - pruned_size) / original_size) * 100
    
    logger.info(f"✅ Global unstructured pruning complete:")
    logger.info(f"   Pruned parameters: {pruned_params:,}")
    logger.info(f"   Actual sparsity: {actual_sparsity:.1%}")
    logger.info(f"   Size reduction: {size_reduction:.1f}%")
    logger.info(f"   New model size: {pruned_size:.2f} MB")
    logger.info(f"   Pruning time: {pruning_time:.2f}s")
    logger.info(f"   Pruned layers: {len(linear_layers)}")
    
    # Return the same type as input
    if hasattr(model_input, 'model'):
        # The model was modified in-place, just return the artifact
        return model_input
    else:
        return pytorch_model


def optimize_model(model_input, method: str, **kwargs) -> tuple:
    """
    Apply optimization method to model and return optimization metadata.
    
    Args:
        model_input: PyTorch model or NexusFlowModelArtifact to optimize
        method: Optimization method ('quantization' or 'pruning')
        **kwargs: Additional arguments for optimization methods
        
    Returns:
        Tuple of (optimized_model, optimization_metadata)
    """
    logger.info(f"🚀 Starting model optimization with method: {method}")
    
    # Extract PyTorch model for statistics
    pytorch_model = _extract_pytorch_model(model_input)
    
    original_params = sum(p.numel() for p in pytorch_model.parameters())
    original_size = original_params * 4 / (1024 * 1024)
    
    start_time = time.time()
    
    if method.lower() == 'quantization':
        optimized_model = quantize_model(model_input, **kwargs)
        optimization_type = 'dynamic_quantization'
    elif method.lower() == 'pruning':
        amount = kwargs.get('amount', 0.2)
        optimized_model = prune_model(model_input, amount)
        optimization_type = 'global_unstructured_pruning'
    else:
        raise ValueError(f"Unknown optimization method: {method}. Supported: 'quantization', 'pruning'")
    
    optimization_time = time.time() - start_time
    
    # Calculate final statistics from the optimized model
    optimized_pytorch_model = _extract_pytorch_model(optimized_model)
    final_params = sum(p.numel() for p in optimized_pytorch_model.parameters())
    final_size = final_params * 4 / (1024 * 1024)
    
    # Create optimization metadata
    metadata = {
        'method': optimization_type,
        'optimization_time': optimization_time,
        'original_parameters': original_params,
        'optimized_parameters': final_params,
        'original_size_mb': original_size,
        'optimized_size_mb': final_size,
        'parameter_reduction': (original_params - final_params) / original_params,
        'size_reduction': (original_size - final_size) / original_size,
        'kwargs': kwargs
    }
    
    # Update model artifact metadata if applicable
    if hasattr(optimized_model, 'meta'):
        optimized_model.meta['optimization'] = metadata
    
    logger.info(f"🎯 Model optimization summary:")
    logger.info(f"   Method: {optimization_type}")
    logger.info(f"   Parameter reduction: {metadata['parameter_reduction']:.1%}")
    logger.info(f"   Size reduction: {metadata['size_reduction']:.1%}")
    logger.info(f"   Total time: {optimization_time:.2f}s")
    
    return optimized_model, metadata