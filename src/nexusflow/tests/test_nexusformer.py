import torch
from nexusflow.model.nexus_former import NexusFormer

def test_nexusformer_forward_shapes():
    model = NexusFormer([4, 6], embed_dim=8)
    x1 = torch.randn(3, 4)
    x2 = torch.randn(3, 6)
    out = model([x1, x2])
    assert out.shape == (3,)  # regression output