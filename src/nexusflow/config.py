"""Configuration loader and Pydantic schema for NexusFlow."""
from pydantic import BaseModel, Field, ValidationError, model_validator
from typing import List, Dict, Any, Optional
import yaml
from loguru import logger
import os

class DatasetConfig(BaseModel):
    name: str
    transformer_type: str = 'standard'
    complexity: str = 'small'
    context_weight: float = 1.0

class SyntheticDataConfig(BaseModel):
    """Configuration for synthetic data generation."""
    n_samples: int = Field(default=256, description="Number of synthetic samples to generate")
    feature_dim: int = Field(default=5, description="Number of features per dataset")

class TrainingConfig(BaseModel):
    batch_size: int = 32
    epochs: int = 10
    optimizer: Dict[str, Any] = Field(default_factory=lambda: {'name': 'adam', 'lr': 1e-3})
    split_config: Dict[str, Any] = Field(default_factory=lambda: {'test_size': 0.15, 'validation_size': 0.15, 'randomize': True})

    use_synthetic: bool = Field(default=False, description="Whether to use synthetic data instead of real data")
    synthetic: SyntheticDataConfig | None = Field(default=None, description="Synthetic data generation settings")

    @model_validator(mode='after')
    def _ensure_synthetic_when_enabled(self):
        if self.use_synthetic and self.synthetic is None:
            self.synthetic = SyntheticDataConfig()
        return self

class MLOpsConfig(BaseModel):
    logging_provider: str = 'stdout'
    experiment_name: str = 'nexus_run'

class ConfigModel(BaseModel):
    project_name: str
    primary_key: str
    target: Dict[str, Any]
    architecture: Dict[str, Any]
    datasets: Optional[List[DatasetConfig]] = None
    training: TrainingConfig = TrainingConfig()
    mlops: MLOpsConfig = MLOpsConfig()

    @model_validator(mode='after')
    def _require_data_or_synthetic(self):
        if not self.training.use_synthetic and not (self.datasets and len(self.datasets) >= 1):
            raise ValueError('At least one dataset must be specified when use_synthetic is False')
        return self

def load_config_from_file(path: str) -> ConfigModel:
    if not os.path.exists(path):
        logger.error(f"Config file not found: {path}")
        raise FileNotFoundError(path)
    with open(path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    try:
        cfg = ConfigModel.model_validate(raw)
    except ValidationError as e:
        logger.error("Configuration validation failed: {}".format(e))
        raise
    logger.info(f"Config parsed: project={cfg.project_name}, datasets={[d.name for d in cfg.datasets]}")
    return cfg