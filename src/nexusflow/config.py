"""Enhanced configuration loader with relational data support for NexusFlow."""
from pydantic import BaseModel, Field, ValidationError, model_validator
from typing import List, Dict, Any, Optional, Literal, Union
import yaml
from loguru import logger
import os

class ForeignKeyConfig(BaseModel):
    """Configuration for foreign key relationships."""
    columns: Union[str, List[str]]  # Column(s) in this table
    references_table: str           # Table being referenced
    references_columns: Union[str, List[str]]  # Column(s) in referenced table

class DatasetConfig(BaseModel):
    name: str
    transformer_type: Literal['standard', 'ft_transformer', 'tabnet', 'text', 'timeseries'] = 'standard'
    complexity: Literal['small', 'medium', 'large'] = 'small'
    context_weight: float = 1.0
    categorical_columns: Optional[List[str]] = None
    numerical_columns: Optional[List[str]] = None
    
    # Relational data support
    primary_key: Union[str, List[str]]  # Can be composite key
    foreign_keys: Optional[List[ForeignKeyConfig]] = None

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
    # Remove top-level primary_key - now defined per dataset
    target: Dict[str, Any]
    architecture: ArchitectureConfig = Field(default_factory=ArchitectureConfig)
    datasets: Optional[List[DatasetConfig]] = None
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    mlops: MLOpsConfig = Field(default_factory=MLOpsConfig)
    
    # Backward compatibility property
    @property
    def primary_key(self) -> str:
        """Backward compatibility: return target table's primary key."""
        if self.datasets:
            target_table = self.target.get('target_table')
            if target_table:
                for dataset in self.datasets:
                    if dataset.name == target_table:
                        pk = dataset.primary_key
                        return pk[0] if isinstance(pk, list) else pk
        return "id"  # fallback

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
    def _validate_relational_config(self):
        """Validate relational data configuration."""
        if self.datasets and not self.training.use_synthetic:
            # Ensure target table exists
            target_table = self.target.get('target_table')
            if target_table:
                table_names = [d.name for d in self.datasets]
                if target_table not in table_names:
                    raise ValueError(f"target_table '{target_table}' not found in datasets: {table_names}")
            
            # Validate foreign key references
            for dataset in self.datasets:
                if dataset.foreign_keys:
                    for fk in dataset.foreign_keys:
                        # Check if referenced table exists
                        ref_table = fk.references_table
                        if ref_table not in [d.name for d in self.datasets]:
                            raise ValueError(f"Foreign key in '{dataset.name}' references unknown table: '{ref_table}'")
                        
                        # Validate column consistency
                        fk_cols = fk.columns if isinstance(fk.columns, list) else [fk.columns]
                        ref_cols = fk.references_columns if isinstance(fk.references_columns, list) else [fk.references_columns]
                        
                        if len(fk_cols) != len(ref_cols):
                            raise ValueError(f"Foreign key column count mismatch in '{dataset.name}': "
                                           f"{len(fk_cols)} columns reference {len(ref_cols)} columns")
        
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

    @model_validator(mode='after')
    def _validate_relational_integrity(self):
        """Enhanced validation for relational data integrity."""
        if self.datasets and not self.training.use_synthetic:
            table_names = {d.name for d in self.datasets}
            
            # Validate target table exists
            target_table = self.target.get('target_table')
            if target_table and target_table not in table_names:
                raise ValueError(f"target_table '{target_table}' not found in datasets: {table_names}")
            
            # Enhanced foreign key validation
            for dataset in self.datasets:
                if dataset.foreign_keys:
                    for fk in dataset.foreign_keys:
                        # Check referenced table exists
                        if fk.references_table not in table_names:
                            raise ValueError(
                                f"Foreign key in '{dataset.name}' references unknown table: '{fk.references_table}'"
                            )
                        
                        # Validate column count consistency
                        fk_cols = fk.columns if isinstance(fk.columns, list) else [fk.columns]
                        ref_cols = fk.references_columns if isinstance(fk.references_columns, list) else [fk.references_columns]
                        
                        if len(fk_cols) != len(ref_cols):
                            raise ValueError(
                                f"Foreign key column count mismatch in '{dataset.name}': "
                                f"{len(fk_cols)} columns reference {len(ref_cols)} columns"
                            )
                        
                        # NEW: Validate that referenced columns match referenced table's primary key
                        ref_dataset = next(d for d in self.datasets if d.name == fk.references_table)
                        ref_pk = ref_dataset.primary_key if isinstance(ref_dataset.primary_key, list) else [ref_dataset.primary_key]
                        
                        if set(ref_cols) != set(ref_pk):
                            logger.warning(
                                f"Foreign key in '{dataset.name}' references non-primary key columns in '{fk.references_table}': "
                                f"{ref_cols} (primary key: {ref_pk})"
                            )
        
        return self

    @model_validator(mode='after') 
    def _validate_join_graph_integrity(self):
        """Validate that join graph forms a connected component."""
        if self.datasets and len(self.datasets) > 1 and not self.training.use_synthetic:
            # Build adjacency graph
            graph = {}
            for dataset in self.datasets:
                graph[dataset.name] = set()
                if dataset.foreign_keys:
                    for fk in dataset.foreign_keys:
                        graph[dataset.name].add(fk.references_table)
                        # Add reverse edge for undirected connectivity check
                        if fk.references_table not in graph:
                            graph[fk.references_table] = set()
                        graph[fk.references_table].add(dataset.name)
            
            # Check connectivity using DFS
            target_table = self.target.get('target_table')
            start_node = target_table if target_table else list(graph.keys())[0]
            
            visited = set()
            stack = [start_node]
            
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                stack.extend(graph.get(node, []))
            
            all_tables = {d.name for d in self.datasets}
            unreachable = all_tables - visited
            
            if unreachable:
                logger.warning(
                    f"Tables not connected to main join graph: {unreachable}. "
                    "These tables will be processed independently."
                )
        
        return self

def load_config_from_file(path: str) -> ConfigModel:
    """Load and validate configuration from YAML file with relational data support."""
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
    
    # Enhanced logging with relational and preprocessing features
    advanced_features = []
    if cfg.architecture.use_moe:
        advanced_features.append(f"MoE({cfg.architecture.num_experts} experts)")
    if cfg.architecture.use_flash_attn:
        advanced_features.append("FlashAttention")
    if cfg.architecture.top_k_contexts:
        advanced_features.append(f"TopK({cfg.architecture.top_k_contexts})")
    if cfg.training.use_advanced_preprocessing:
        advanced_features.append("Advanced Preprocessing")
    
    # Check for relational features
    if cfg.datasets:
        has_foreign_keys = any(d.foreign_keys for d in cfg.datasets)
        has_composite_keys = any(isinstance(d.primary_key, list) for d in cfg.datasets)
        if has_foreign_keys or has_composite_keys:
            advanced_features.append("Relational Data")
    
    features_str = ", ".join(advanced_features) if advanced_features else "None"
    
    dataset_types = [d.transformer_type for d in cfg.datasets] if cfg.datasets else ["synthetic"]
    
    logger.info(f"Enhanced config parsed: project={cfg.project_name}")
    logger.info(f"  Datasets: {[d.name for d in cfg.datasets] if cfg.datasets else ['synthetic']}")
    logger.info(f"  Transformer types: {set(dataset_types)}")
    logger.info(f"  Advanced features: {features_str}")
    logger.info(f"  MLOps provider: {cfg.mlops.logging_provider}")
    
    # Log relational configuration
    if cfg.datasets:
        target_table = cfg.target.get('target_table')
        if target_table:
            logger.info(f"  Target table: {target_table}")
        
        # Log foreign key relationships
        total_relationships = 0
        for dataset in cfg.datasets:
            if dataset.foreign_keys:
                total_relationships += len(dataset.foreign_keys)
                logger.info(f"    {dataset.name}: {len(dataset.foreign_keys)} foreign key(s)")
        
        if total_relationships > 0:
            logger.info(f"  Total relational relationships: {total_relationships}")
    
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