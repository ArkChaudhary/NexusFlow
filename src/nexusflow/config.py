"""Enhanced configuration loader with preprocessing support for NexusFlow Phase 2."""
from pydantic import BaseModel, Field, ValidationError, model_validator
from typing import List, Dict, Any, Optional, Literal
import yaml
from loguru import logger
import os

class DatasetConfig(BaseModel):
    name: str
    transformer_type: Literal['standard', 'ft_transformer', 'tabnet', 'text', 'timeseries'] = 'standard'
    complexity: Literal['small', 'medium', 'large'] = 'small'
    context_weight: float = 1.0
    categorical_columns: Optional[List[str]] = None
    numerical_columns: Optional[List[str]] = None

class SyntheticDataConfig(BaseModel):
    """Configuration for synthetic data generation."""
    n_samples: int = Field(default=256, description="Number of synthetic samples to generate")
    feature_dim: int = Field(default=5, description="Number of features per dataset")

class OptimizerConfig(BaseModel):
    name: str = 'adam'
    lr: float = 0.001
    weight_decay: float = 0.0001

class SplitConfig(BaseModel):
    test_size: float = 0.2
    validation_size: float = 0.2
    randomize: bool = True

class TrainingConfig(BaseModel):
    batch_size: int = 32
    epochs: int = 10
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    split_config: SplitConfig = Field(default_factory=SplitConfig)
    
    # Synthetic data options
    use_synthetic: bool = Field(default=False, description="Whether to use synthetic data instead of real data")
    synthetic: Optional[SyntheticDataConfig] = Field(default=None, description="Synthetic data generation settings")
    
    # Training features
    early_stopping: bool = Field(default=False, description="Enable early stopping")
    patience: int = Field(default=5, description="Early stopping patience")
    gradient_clipping: float = Field(default=1.0, description="Gradient clipping threshold")
    
    # Preprocessing features
    use_advanced_preprocessing: bool = Field(default=True, description="Enable advanced preprocessing pipeline")
    auto_detect_types: bool = Field(default=True, description="Auto-detect categorical/numerical columns")

    @model_validator(mode='after')
    def _ensure_synthetic_when_enabled(self):
        if self.use_synthetic and self.synthetic is None:
            self.synthetic = SyntheticDataConfig()
        return self

class ArchitectureConfig(BaseModel):
    global_embed_dim: int = Field(default=128, description="Global embedding dimension")
    refinement_iterations: int = Field(default=4, description="Number of refinement iterations")
    
    # MoE configuration
    use_moe: bool = Field(default=False, description="Enable Mixture of Experts")
    num_experts: int = Field(default=6, description="Number of expert networks")
    
    # Attention configuration  
    use_flash_attn: bool = Field(default=True, description="Enable FlashAttention optimization")
    top_k_contexts: Optional[int] = Field(default=None, description="Limit cross-attention to top-k contexts")

class MLOpsConfig(BaseModel):
    logging_provider: Literal['stdout', 'wandb', 'mlflow'] = 'stdout'
    experiment_name: str = 'nexus_run'
    log_attention_patterns: bool = Field(default=False, description="Log attention heatmaps for visualization")

class ConfigModel(BaseModel):
    project_name: str
    primary_key: str
    target: Dict[str, Any]
    architecture: ArchitectureConfig = Field(default_factory=ArchitectureConfig)
    datasets: Optional[List[DatasetConfig]] = None
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    mlops: MLOpsConfig = Field(default_factory=MLOpsConfig)

    @model_validator(mode='after')
    def _require_data_or_synthetic(self):
        if not self.training.use_synthetic and not (self.datasets and len(self.datasets) >= 1):
            raise ValueError('At least one dataset must be specified when use_synthetic is False')
        return self
    
    @model_validator(mode='after')
    def _validate_moe_config(self):
        """Validate MoE configuration parameters."""
        if self.architecture.use_moe:
            if self.architecture.num_experts < 2:
                raise ValueError('num_experts must be >= 2 when MoE is enabled')
        return self
    
    @model_validator(mode='after')
    def _validate_transformer_types(self):
        """Validate transformer types for datasets."""
        if self.datasets:
            valid_types = {'standard', 'ft_transformer', 'tabnet', 'text', 'timeseries'}
            for dataset in self.datasets:
                if dataset.transformer_type not in valid_types:
                    raise ValueError(f"Invalid transformer_type: {dataset.transformer_type}. "
                                   f"Must be one of {valid_types}")
        return self
    
    @model_validator(mode='after')
    def _validate_preprocessing_config(self):
        """Validate preprocessing configuration."""
        if self.datasets and self.training.use_advanced_preprocessing:
            for dataset in self.datasets:
                # If manual column specification is provided, validate it
                if dataset.categorical_columns is not None and dataset.numerical_columns is not None:
                    overlap = set(dataset.categorical_columns) & set(dataset.numerical_columns)
                    if overlap:
                        raise ValueError(f"Columns cannot be both categorical and numerical: {overlap}")
        return self

def load_config_from_file(path: str) -> ConfigModel:
    """Load and validate configuration from YAML file with preprocessing support."""
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
    
    # Enhanced logging with preprocessing features
    advanced_features = []
    if cfg.architecture.use_moe:
        advanced_features.append(f"MoE({cfg.architecture.num_experts} experts)")
    if cfg.architecture.use_flash_attn:
        advanced_features.append("FlashAttention")
    if cfg.architecture.top_k_contexts:
        advanced_features.append(f"TopK({cfg.architecture.top_k_contexts})")
    if cfg.training.use_advanced_preprocessing:
        advanced_features.append("Advanced Preprocessing")
    
    features_str = ", ".join(advanced_features) if advanced_features else "None"
    
    dataset_types = [d.transformer_type for d in cfg.datasets] if cfg.datasets else ["synthetic"]
    
    logger.info(f"Enhanced config parsed: project={cfg.project_name}")
    logger.info(f"  Datasets: {[d.name for d in cfg.datasets] if cfg.datasets else ['synthetic']}")
    logger.info(f"  Transformer types: {set(dataset_types)}")
    logger.info(f"  Advanced features: {features_str}")
    logger.info(f"  MLOps provider: {cfg.mlops.logging_provider}")
    
    # Log preprocessing configuration
    if cfg.datasets and cfg.training.use_advanced_preprocessing:
        logger.info("  Preprocessing configuration:")
        for dataset in cfg.datasets:
            if dataset.categorical_columns or dataset.numerical_columns:
                logger.info(f"    {dataset.name}: categorical={len(dataset.categorical_columns or [])}, "
                           f"numerical={len(dataset.numerical_columns or [])}")
            else:
                logger.info(f"    {dataset.name}: auto-detect columns")
    
    return cfg