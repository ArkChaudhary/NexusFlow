import torch
from pathlib import Path
from loguru import logger

class ModelAPI:
    def __init__(self, model, preprocess_meta=None):
        self.model = model
        self.preprocess_meta = preprocess_meta or {}

    def save(self, path: str):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'model_state': self.model.state_dict(), 'meta': self.preprocess_meta}, p)
        logger.info(f"Model saved to: {p}")

    @staticmethod
    def load(path: str, model_constructor):
        p = Path(path)
        if not p.exists():
            logger.error(f"Model file not found: {p}")
            raise FileNotFoundError(p)
        ckpt = torch.load(p, map_location='cpu')
        model = model_constructor()
        model.load_state_dict(ckpt['model_state'])
        logger.info(f"Model loaded from: {p}")
        return ModelAPI(model, preprocess_meta=ckpt.get('meta', {}))

    def predict(self, inputs):
        self.model.eval()
        with torch.no_grad():
            out = self.model(inputs)
        return out
