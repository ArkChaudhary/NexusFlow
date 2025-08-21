"""Configuration loader and Pydantic schema for NexusFlow."""
from pydantic import BaseModel, Field, ValidationError, field_validator
from typing import List, Dict, Any
import yaml
from loguru import logger
import os

class DatasetConfig(BaseModel):
    name: str
    transformer_type: str = 'standard'
    complexity: str = 'small'
    context_weight: float = 1.0

class TrainingConfig(BaseModel):
    batch_size: int = 32
    epochs: int = 10
    optimizer: Dict[str, Any] = Field(default_factory=lambda: {'name': 'adam', 'lr': 1e-3})
    split_config: Dict[str, Any] = Field(default_factory=lambda: {'test_size': 0.15, 'validation_size': 0.15, 'randomize': True})

class MLOpsConfig(BaseModel):
    logging_provider: str = 'stdout'
    experiment_name: str = 'nexus_run'

class ConfigModel(BaseModel):
    project_name: str
    primary_key: str
    target: Dict[str, Any]
    architecture: Dict[str, Any]
    datasets: List[DatasetConfig]
    training: TrainingConfig = TrainingConfig()
    mlops: MLOpsConfig = MLOpsConfig()

    @field_validator('datasets')
    def at_least_one_dataset(cls, v):
        if not v or len(v) < 1:
            raise ValueError('At least one dataset must be specified')
        return v

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
