import torch
from nexusflow.api.model_api import ModelAPI
from nexusflow.model.nexus_former import NexusFormer

def test_model_save_and_load(tmp_path):
    model = NexusFormer([3, 3], embed_dim=4)
    api = ModelAPI(model)
    save_path = tmp_path / "model.pt"
    api.save(save_path)

    loaded_api = ModelAPI.load(save_path, lambda: NexusFormer([3, 3], embed_dim=4))
    x1 = torch.randn(2, 3)
    x2 = torch.randn(2, 3)
    preds = loaded_api.predict([x1, x2])
    assert preds.shape == (2,)
