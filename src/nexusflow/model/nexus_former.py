import torch
import torch.nn as nn
from loguru import logger
from typing import Sequence

class SimpleEncoder(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x):
        out = self.net(x)
        logger.debug(f"SimpleEncoder output shape: {out.shape}")
        return out

class NexusFormer(nn.Module):
    def __init__(self, input_dims: Sequence[int], embed_dim: int = 64):
        super().__init__()
        if not isinstance(input_dims, (list, tuple)) or len(input_dims) == 0:
            raise ValueError("NexusFormer requires a non-empty sequence of input dimensions.")
        if any(int(d) <= 0 for d in input_dims):
            raise ValueError(f"All input dimensions must be positive integers, got: {input_dims}")
        self.input_dims = [int(d) for d in input_dims]
        self.encoders = nn.ModuleList([SimpleEncoder(d, embed_dim) for d in self.input_dims])
        self.fusion = nn.Sequential(
            nn.Linear(len(input_dims) * embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, inputs):
        # inputs: Sequence[Tensor] with shapes [batch, D_i]
        if len(inputs) != len(self.encoders):
            raise ValueError(f"NexusFormer expected {len(self.encoders)} inputs, got {len(inputs)}.")
        reps = []
        batch = None
        for idx, (enc, x, expected) in enumerate(zip(self.encoders, inputs, self.input_dims)):
            if x.dim() != 2 or x.size(-1) != expected:
                raise ValueError(
                    f"Input {idx} has shape {tuple(x.shape)}, expected [batch, {expected}]."
                )
            if batch is None:
                batch = x.size(0)
            elif x.size(0) != batch:
                raise ValueError(
                    f"All inputs must share the same batch size; "
                    f"mismatch at input {idx}: {x.size(0)} vs {batch}."
                )
            reps.append(enc(x))
        concat = torch.cat(reps, dim=-1)
        logger.debug(f"NexusFormer concatenated representation shape: {concat.shape}")
        out = self.fusion(concat)
        logger.debug(f"NexusFormer output shape: {out.shape}")
        return out.squeeze(-1)
