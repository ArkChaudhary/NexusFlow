"""Enhanced trainer with Phase 2 preprocessing pipeline integration."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from loguru import logger
from pathlib import Path
import json
import pandas as pd
from typing import Dict, Optional, Any
import time

from nexusflow.config import ConfigModel
from nexusflow.model.nexus_former import NexusFormer
from nexusflow.data.ingestion import load_and_preprocess_datasets, make_dataloaders, flatten_relational_data, load_table, align_datasets
from nexusflow.data.preprocessor import TabularPreprocessor
from nexusflow.api.model_api import ModelAPI

# Import existing MLOpsLogger from the original trainer
class MLOpsLogger:
    """Enhanced MLOps logger with advanced metrics tracking."""
    
    def __init__(self, provider: str = "stdout", experiment_name: str = "nexus_run"):
        self.provider = provider.lower()
        self.experiment_name = experiment_name
        self.metrics_log = []
        
        if self.provider == "wandb":
            try:
                import wandb
                wandb.init(project="nexusflow", name=experiment_name)
                self.wandb = wandb
                logger.info("Initialized Weights & Biases logging")
            except ImportError:
                logger.warning("wandb not installed, falling back to stdout")
                self.provider = "stdout"
                self.wandb = None
        elif self.provider == "mlflow":
            try:
                import mlflow
                mlflow.start_run(run_name=experiment_name)
                self.mlflow = mlflow
                logger.info("Initialized MLflow logging")
            except ImportError:
                logger.warning("mlflow not installed, falling back to stdout")
                self.provider = "stdout"
                self.mlflow = None
        else:
            self.provider = "stdout"
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics with enhanced formatting."""
        metric_str = " ".join([f"{k}={v:.6f}" for k, v in metrics.items()])
        logger.info(f"📊 Metrics (step {step}): {metric_str}")
        
        log_entry = {"step": step, "metrics": metrics, "timestamp": time.time()}
        self.metrics_log.append(log_entry)
        
        if self.provider == "wandb" and hasattr(self, 'wandb') and self.wandb:
            self.wandb.log(metrics, step=step)
        elif self.provider == "mlflow" and hasattr(self, 'mlflow') and self.mlflow:
            for key, value in metrics.items():
                self.mlflow.log_metric(key, value, step=step)
    
    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters with enhanced organization."""
        logger.info(f"🔧 Configuration: {len(params)} parameters")
        for key, value in params.items():
            logger.debug(f"  {key}: {value}")
        
        if self.provider == "wandb" and hasattr(self, 'wandb') and self.wandb:
            self.wandb.config.update(params)
        elif self.provider == "mlflow" and hasattr(self, 'mlflow') and self.mlflow:
            for key, value in params.items():
                self.mlflow.log_param(key, value)
    
    def log_architecture_stats(self, model: nn.Module, config: ConfigModel):
        """Log detailed architecture statistics."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        arch_stats = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),
            'use_moe': config.architecture.use_moe,  # Direct attribute access
            'use_flash_attn': config.architecture.use_flash_attn,  # Direct attribute access
            'use_advanced_preprocessing': config.training.use_advanced_preprocessing,
            'num_experts': config.architecture.num_experts if config.architecture.use_moe else 0
        }
        
        logger.info(f"🏗️  Architecture Stats: {trainable_params:,} trainable params, "
                    f"{arch_stats['model_size_mb']:.2f}MB")
        
        self.log_params(arch_stats)
    
    def finish(self):
        """Clean up logging resources."""
        if self.provider == "wandb" and hasattr(self, 'wandb') and self.wandb:
            self.wandb.finish()
        elif self.provider == "mlflow" and hasattr(self, 'mlflow') and self.mlflow:
            self.mlflow.end_run()
    
    def save_metrics_log(self, path: str):
        """Save internal metrics log to file."""
        with open(path, 'w') as f:
            json.dump(self.metrics_log, f, indent=2)

class Trainer:
    """
    Enhanced trainer with Phase 2 preprocessing pipeline integration.
    """
    
    def __init__(self, config: ConfigModel, work_dir: str = '.'):
        self.cfg = config
        self.work_dir = Path(work_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize enhanced MLOps logging
        self.mlops_logger = MLOpsLogger(
            provider=self.cfg.mlops.logging_provider,
            experiment_name=self.cfg.mlops.experiment_name
        )
        
        # Setup file logging
        self._setup_file_logging()
        
        logger.info(f"🚀 Enhanced Phase 2 Trainer initialized (device={self.device})")

        # Initialize data-related attributes
        self.datasets = None
        self.preprocessors = {}
        self.preprocessing_metadata = None
        self.input_dims = None
        self.train_loader = None
        self.val_loader = None 
        self.test_loader = None
        
        # Enhanced training state tracking
        self.best_val_metric = float('inf')
        self.best_model_state = None
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        self.training_history = []
        
        # Load and process datasets with new preprocessing pipeline
        self._setup_enhanced_data()
        
        # Initialize model with preprocessing-aware architecture
        self._initialize_preprocessing_aware_model()
        
        # Setup optimizer
        self._setup_optimizer()
        
        # Log comprehensive model statistics
        self.mlops_logger.log_architecture_stats(self.model, self.cfg)

    def _setup_enhanced_data(self):
        """Enhanced data setup with RESTORED multi-table support."""
        training_cfg = self.cfg.training
        
        if training_cfg.use_synthetic:
            logger.info("🔬 Using synthetic data mode")
            n_datasets = len(self.cfg.datasets) if self.cfg.datasets else 2
            feature_dim = training_cfg.synthetic.feature_dim if training_cfg.synthetic else 5
            
            self.input_dims = [feature_dim] * n_datasets
            self.datasets = None
            self.preprocessing_metadata = None
            logger.info(f"   Synthetic data: {n_datasets} datasets × {feature_dim} features")
            
        else:
            logger.info("📊 Loading datasets with RESTORED multi-table support...")
            
            # RESTORED: Use multi-table preprocessing pipeline (no flattening)
            if training_cfg.use_advanced_preprocessing:
                self.datasets, self.preprocessors = load_and_preprocess_datasets(self.cfg)
                logger.info(f"✅ Multi-table preprocessing applied to {len(self.preprocessors)} dataset(s)")
                
                # Calculate input dimensions for each separate dataset
                self.input_dims = []
                for dataset_cfg in self.cfg.datasets:
                    dataset_name = dataset_cfg.name
                    if dataset_name in self.datasets:
                        df = self.datasets[dataset_name]
                        
                        # Calculate features for this specific dataset
                        excluded_cols = {self.cfg.primary_key}
                        target_col = self.cfg.target.get('target_column')
                        if target_col and target_col in df.columns:
                            excluded_cols.add(target_col)
                        
                        feature_cols = [col for col in df.columns if col not in excluded_cols]
                        self.input_dims.append(len(feature_cols))
                        
                        logger.info(f"   Dataset {dataset_name}: {len(feature_cols)} features")
                
            else:
                # Fallback to legacy preprocessing but keep multi-table structure
                logger.info("Using multi-table data with simple preprocessing")
                raw_datasets = {}
                for dataset_cfg in self.cfg.datasets:
                    path = f"datasets/{dataset_cfg.name}"
                    df = load_table(path)
                    raw_datasets[dataset_cfg.name] = df
                
                # Use align_datasets instead of flattening
                self.datasets = align_datasets(raw_datasets, self.cfg.primary_key)
                self.preprocessors = {}
                
                # Calculate dimensions for each aligned dataset
                self.input_dims = []
                for dataset_cfg in self.cfg.datasets:
                    dataset_name = dataset_cfg.name
                    if dataset_name in self.datasets:
                        df = self.datasets[dataset_name]
                        
                        excluded_cols = {self.cfg.primary_key}
                        target_col = self.cfg.target.get('target_column')
                        if target_col and target_col in df.columns:
                            excluded_cols.add(target_col)
                        
                        feature_cols = [col for col in df.columns if col not in excluded_cols]
                        self.input_dims.append(len(feature_cols))
            
            # Enhanced data quality reporting
            total_samples = len(list(self.datasets.values())[0]) if self.datasets else 0
            total_features = sum(self.input_dims)
            
            logger.info(f"📈 Multi-table data pipeline complete:")
            logger.info(f"   Samples per table: {total_samples:,}")
            logger.info(f"   Feature dimensions: {self.input_dims}")
            logger.info(f"   Total features: {total_features}")
            logger.info(f"   Separate datasets: {len(self.cfg.datasets)}")
            logger.info(f"   Preprocessing: {'advanced' if self.preprocessors else 'simple'}")

    def _initialize_preprocessing_aware_model(self):
        """Initialize model with preprocessing awareness."""
        embed_dim = self.cfg.architecture.global_embed_dim  # Direct attribute access
        refinement_iterations = self.cfg.architecture.refinement_iterations  # Direct attribute access
        
        # Determine encoder strategy based on preprocessing
        if self.cfg.datasets:
            transformer_types = {d.transformer_type for d in self.cfg.datasets}
            if len(transformer_types) == 1:
                encoder_type = list(transformer_types)[0]
            else:
                encoder_type = 'standard'
        else:
            encoder_type = 'standard'
        
        self.model = NexusFormer(
            input_dims=self.input_dims,
            embed_dim=embed_dim,
            refinement_iterations=refinement_iterations,
            encoder_type=encoder_type,
            use_moe=self.cfg.architecture.use_moe,  # Direct attribute access
            num_experts=self.cfg.architecture.num_experts,  # Direct attribute access
            use_flash_attn=self.cfg.architecture.use_flash_attn  # Direct attribute access
        ).to(self.device)
        
        # Store preprocessing information in model for inference
        if self.preprocessors:
            self.model._preprocessing_metadata = {
                'preprocessors': self.preprocessors,
                'input_dims': self.input_dims,
                'datasets_config': [d.dict() for d in self.cfg.datasets]
            }
        
        logger.info(f"🧠 Enhanced model initialized: {encoder_type} encoders")
        logger.info(f"   Advanced features: MoE={self.cfg.architecture.use_moe}, "
                    f"FlashAttn={self.cfg.architecture.use_flash_attn}")
        logger.info(f"   Preprocessing: {'integrated' if self.preprocessors else 'legacy'}")

    def _setup_optimizer(self):
        """Setup optimizer with enhanced features."""
        optim_config = self.cfg.training.optimizer
        lr = optim_config.lr  # Direct attribute access instead of .get()
        
        if optim_config.name.lower() == 'adam':
            weight_decay = optim_config.weight_decay  # Direct attribute access
            self.optim = torch.optim.Adam(
                self.model.parameters(), 
                lr=lr, 
                weight_decay=weight_decay
            )
        else:
            self.optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optim, mode='min', factor=0.5, patience=3
        )

    def _setup_file_logging(self):
        """Setup enhanced structured logging."""
        log_dir = self.work_dir / "results" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"enhanced_training_{self.cfg.mlops.experiment_name}.log"
        self.log_handler_id = logger.add(
            log_file,
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="10 MB",
            retention="7 days"
        )
        logger.info(f"📝 Enhanced logging enabled: {log_file}")
    
    def _cleanup_logging(self):
        """Clean up logging handlers to prevent file locking."""
        if hasattr(self, 'log_handler_id'):
            try:
                logger.remove(self.log_handler_id)
                logger.info("📝 Logging handler cleaned up")
            except ValueError:
                pass  # Handler already removed

    def _setup_enhanced_dataloaders(self):
        """Enhanced DataLoader setup with preprocessing integration."""
        if self.cfg.training.use_synthetic:
            # Use existing synthetic data generation logic
            self._setup_synthetic_dataloaders()
        else:
            # Use enhanced preprocessing-aware dataloaders
            logger.info("🔄 Creating enhanced dataloaders with preprocessing...")
            
            self.train_loader, self.val_loader, self.test_loader, self.preprocessing_metadata = make_dataloaders(
                self.cfg, self.datasets, self.preprocessors
            )
            
            logger.info(f"Enhanced DataLoaders created with preprocessing metadata")
            if self.preprocessing_metadata:
                logger.info(f"  Preprocessing info: {len(self.preprocessing_metadata['dataset_order'])} datasets")
                logger.info(f"  Feature dimensions: {self.preprocessing_metadata['feature_dimensions']}")

    def _setup_synthetic_dataloaders(self):
        """Setup synthetic dataloaders (existing logic)."""
        synthetic_config = self.cfg.training.synthetic
        n_samples = synthetic_config.n_samples if synthetic_config else 256
        batch_size = self.cfg.training.batch_size
        
        # Generate synthetic data
        torch.manual_seed(42)
        Xs = []
        
        for i, dim in enumerate(self.input_dims):
            base_signal = torch.randn(n_samples, 1)
            noise = torch.randn(n_samples, dim) * 0.3
            
            if i == 0:
                features = base_signal.expand(-1, dim) + noise
            elif i == len(self.input_dims) - 1:
                features = torch.randn(n_samples, dim) * 0.8
            else:
                signal_strength = 0.7 - (i * 0.2)
                features = base_signal.expand(-1, dim) * signal_strength + noise
            
            Xs.append(features)
        
        # Create target
        target_col = self.cfg.target["target_column"]
        if target_col == 'label':
            linear_combo = sum(X.mean(dim=1) * (0.8 - i * 0.2) for i, X in enumerate(Xs))
            probs = torch.sigmoid(linear_combo + torch.randn(n_samples) * 0.2)
            y = torch.bernoulli(probs).long()
        else:
            y = sum(X.mean(dim=1) * (0.8 - i * 0.2) for i, X in enumerate(Xs)) + torch.randn(n_samples) * 0.1
        
        # Split data
        train_size = int(0.7 * n_samples)
        val_size = int(0.15 * n_samples)
        
        indices = torch.randperm(n_samples)
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size+val_size] 
        test_idx = indices[train_size+val_size:]
        
        # Create datasets
        train_data = torch.utils.data.TensorDataset(*[X[train_idx] for X in Xs], y[train_idx])
        val_data = torch.utils.data.TensorDataset(*[X[val_idx] for X in Xs], y[val_idx])
        test_data = torch.utils.data.TensorDataset(*[X[test_idx] for X in Xs], y[test_idx])
        
        self.train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        
        logger.info(f"🔄 Synthetic DataLoaders: train={len(self.train_loader)} "
                   f"val={len(self.val_loader)} test={len(self.test_loader)} batches")

    def _calculate_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Enhanced loss calculation."""
        target_col = self.cfg.target["target_column"]
        
        if target_col == 'label':
            if targets.dtype == torch.long and predictions.dim() == 1:
                return torch.nn.functional.binary_cross_entropy_with_logits(
                    predictions, targets.float()
                )
            else:
                return torch.nn.functional.cross_entropy(predictions, targets)
        else:
            return torch.nn.functional.huber_loss(predictions, targets.float(), delta=1.0)

    def _calculate_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Enhanced metrics calculation."""
        target_col = self.cfg.target["target_column"]
        metrics = {}
        
        if target_col == 'label':
            if predictions.dim() == 1:
                probs = torch.sigmoid(predictions)
                preds = (probs > 0.5).long()
                metrics['confidence'] = probs.std().item()
            else:
                probs = torch.softmax(predictions, dim=1)
                preds = torch.argmax(predictions, dim=1)
                metrics['confidence'] = probs.max(dim=1)[0].mean().item()
            
            accuracy = (preds == targets).float().mean().item()
            metrics['accuracy'] = accuracy
            
            # Binary classification metrics
            if len(torch.unique(targets)) == 2:
                tp = ((preds == 1) & (targets == 1)).sum().float()
                fp = ((preds == 1) & (targets == 0)).sum().float()
                tn = ((preds == 0) & (targets == 0)).sum().float()
                fn = ((preds == 0) & (targets == 1)).sum().float()
                
                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                
                metrics.update({
                    'precision': precision.item(),
                    'recall': recall.item(), 
                    'f1_score': f1.item()
                })
        else:
            # Regression metrics
            mse = torch.nn.functional.mse_loss(predictions, targets.float()).item()
            mae = torch.nn.functional.l1_loss(predictions, targets.float()).item()
            
            ss_res = ((targets.float() - predictions) ** 2).sum()
            ss_tot = ((targets.float() - targets.float().mean()) ** 2).sum()
            r2 = 1 - (ss_res / (ss_tot + 1e-8))
            
            metrics.update({
                'mse': mse,
                'mae': mae,
                'rmse': mse ** 0.5,
                'r2_score': r2.item()
            })
        
        return metrics

    def _validate_epoch(self) -> Dict[str, float]:
        """Enhanced validation with comprehensive metrics."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                if self.cfg.training.use_synthetic:
                    *features, targets = batch
                else:
                    features, targets = batch
                
                features = [f.to(self.device) for f in features]
                targets = targets.to(self.device)
                
                predictions = self.model(features)
                loss = self._calculate_loss(predictions, targets)
                
                total_loss += loss.item()
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
        
        # Calculate comprehensive metrics
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        
        avg_loss = total_loss / len(self.val_loader)
        metrics = self._calculate_metrics(all_predictions, all_targets)
        metrics['val_loss'] = avg_loss
        
        return metrics

    def sanity_check(self):
        """Enhanced sanity check with preprocessing validation."""
        logger.info("🔍 Running comprehensive Phase 2 sanity checks...")
        
        # Test model forward pass
        batch_size = 2
        dummy_inputs = [torch.randn(batch_size, dim).to(self.device) for dim in self.input_dims]
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(dummy_inputs)
        
        logger.info(f"✅ Model forward pass: output_shape={output.shape}")
        
        # Test preprocessing pipeline
        if self.preprocessors:
            logger.info("✅ Advanced preprocessing pipeline active:")
            for dataset_name, preprocessor in self.preprocessors.items():
                feature_info = preprocessor.get_feature_info()
                logger.info(f"   {dataset_name}: {feature_info['total_features']} total features")
                logger.info(f"     Categorical: {len(feature_info['categorical_columns'])}")
                logger.info(f"     Numerical: {len(feature_info['numerical_columns'])}")
        else:
            logger.info("ℹ️  Using legacy preprocessing")
        
        # Test advanced features
        if self.cfg.advanced.use_moe:
            logger.info("🔀 MoE layers active")
        if self.cfg.advanced.use_flash_attn:
            logger.info("⚡ FlashAttention enabled")
        
        # Setup and test data loaders
        self._setup_enhanced_dataloaders()
        
        if self.train_loader:
            sample_batch = next(iter(self.train_loader))
            if self.cfg.training.use_synthetic:
                *features, targets = sample_batch
            else:
                features, targets = sample_batch
            
            logger.info(f"📦 Sample batch: features={[f.shape for f in features]} targets={targets.shape}")
            
            # Test validation
            if self.val_loader:
                val_metrics = self._validate_epoch()
                logger.info(f"📈 Initial validation: {val_metrics}")
        
        logger.info("✅ All Phase 2 enhanced sanity checks passed!")

    def train(self):
        """Enhanced training loop with Phase 2 preprocessing integration."""
        epochs = int(self.cfg.training.epochs)
        logger.info(f"🎯 Starting Phase 2 enhanced training: {epochs} epochs")

        self._setup_enhanced_dataloaders()
        
        if self.train_loader is None:
            raise RuntimeError("No training data available")

        # Training loop
        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_batches = 0
            
            for batch_idx, batch in enumerate(self.train_loader):
                if self.cfg.training.use_synthetic:
                    *features, targets = batch
                else:
                    features, targets = batch
                
                features = [f.to(self.device) for f in features]
                targets = targets.to(self.device)

                self.optim.zero_grad()
                predictions = self.model(features)
                loss = self._calculate_loss(predictions, targets)
                loss.backward()
                
                # Gradient clipping
                if self.cfg.training.gradient_clipping > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.training.gradient_clipping)
                
                self.optim.step()
                
                train_loss += loss.item()
                train_batches += 1
                
                if batch_idx % 50 == 0 and batch_idx > 0:
                    logger.debug(f"Epoch {epoch} Batch {batch_idx}: loss={loss.item():.6f}")
            
            avg_train_loss = train_loss / train_batches
            
            # Validation phase
            val_metrics = self._validate_epoch()
            
            # Learning rate scheduling
            if val_metrics and 'val_loss' in val_metrics:
                self.scheduler.step(val_metrics['val_loss'])
            
            # Combine metrics
            epoch_time = time.time() - epoch_start_time
            epoch_metrics = {
                'train_loss': avg_train_loss,
                'epoch': epoch,
                'epoch_time': epoch_time,
                'learning_rate': self.optim.param_groups[0]['lr']
            }
            epoch_metrics.update(val_metrics)
            
            # Best model tracking
            current_metric = val_metrics.get('val_loss', avg_train_loss)
            if current_metric < self.best_val_metric:
                self.best_val_metric = current_metric
                self.best_epoch = epoch
                self.best_model_state = self.model.state_dict().copy()
                self.epochs_without_improvement = 0
                logger.info(f"🌟 New best model at epoch {epoch}: {current_metric:.6f}")
            else:
                self.epochs_without_improvement += 1
            
            # Progress logging
            progress_pct = (epoch / epochs) * 100
            logger.info(f"🔄 Epoch {epoch}/{epochs} ({progress_pct:.1f}%): "
                       f"train_loss={avg_train_loss:.6f} val_loss={val_metrics.get('val_loss', 'N/A')} "
                       f"time={epoch_time:.2f}s lr={self.optim.param_groups[0]['lr']:.2e}")
            
            # Store training history
            self.training_history.append(epoch_metrics)
            
            # Log to MLOps
            self.mlops_logger.log_metrics(epoch_metrics, step=epoch)
            
            # Early stopping
            if (self.cfg.training.early_stopping and 
                self.epochs_without_improvement >= self.cfg.training.patience):
                logger.info(f"🛑 Early stopping triggered after {epoch} epochs "
                           f"(no improvement for {self.epochs_without_improvement} epochs)")
                break
        
        # Post-training finalization
        self._finalize_enhanced_training()

    def _finalize_enhanced_training(self):
        """Enhanced training finalization with preprocessing artifacts."""
        if self.best_model_state is not None:
            # Save best model
            best_model_path = self.work_dir / "best_model.pt"
            self._save_enhanced_model(best_model_path)
            
            # Create enhanced model artifact with preprocessing
            nxf_path = self.work_dir / f"{self.cfg.project_name}.nxf"
            self._create_enhanced_model_artifact(nxf_path)
            
            # Save preprocessing artifacts
            if self.preprocessors:
                self._save_preprocessing_artifacts()
            
            # Save training history
            history_path = self.work_dir / "results" / "training_history.json"
            history_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(history_path, 'w') as f:
                json.dump({
                    'training_history': self.training_history,
                    'best_epoch': self.best_epoch,
                    'best_metric': self.best_val_metric,
                    'total_epochs': len(self.training_history),
                    'preprocessing_enabled': bool(self.preprocessors),
                    'early_stopped': self.epochs_without_improvement >= self.cfg.training.patience
                }, f, indent=2)
        
        # Save metrics log
        metrics_log_path = self.work_dir / "results" / "metrics.json"
        metrics_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.mlops_logger.save_metrics_log(str(metrics_log_path))
        
        # Cleanup logging and MLOps
        self._cleanup_logging()
        self.mlops_logger.finish()
        
        logger.info(f"🎉 Phase 2 enhanced training complete! Best model at epoch {self.best_epoch} "
                   f"with metric {self.best_val_metric:.6f}")

    def _save_preprocessing_artifacts(self):
        """Save preprocessing artifacts for inference."""
        preprocess_dir = self.work_dir / "preprocessing"
        preprocess_dir.mkdir(parents=True, exist_ok=True)
        
        # Save each preprocessor
        for dataset_name, preprocessor in self.preprocessors.items():
            preprocessor_path = preprocess_dir / f"{dataset_name}_preprocessor.pkl"
            preprocessor.save(str(preprocessor_path))
            logger.info(f"💾 Saved preprocessor: {preprocessor_path}")
        
        # Save preprocessing metadata
        if self.preprocessing_metadata:
            metadata_path = preprocess_dir / "preprocessing_metadata.json"
            with open(metadata_path, 'w') as f:
                # Convert to JSON-serializable format
                serializable_metadata = {}
                for key, value in self.preprocessing_metadata.items():
                    if key == 'preprocessor_info':
                        # Skip preprocessor objects, just save the structure info
                        serializable_metadata[key] = {
                            dataset: {k: v for k, v in info.items() if k != 'preprocessor'}
                            for dataset, info in value.items()
                        }
                    else:
                        serializable_metadata[key] = value
                
                json.dump(serializable_metadata, f, indent=2)
            
            logger.info(f"💾 Saved preprocessing metadata: {metadata_path}")

    def _save_enhanced_model(self, path: Path):
        """Save enhanced model with preprocessing information."""
        if self.best_model_state is None:
            logger.warning("No best model state to save")
            return
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        enhanced_checkpoint = {
            'epoch': self.best_epoch,
            'model_state': self.best_model_state,
            'config': self.cfg.model_dump() if hasattr(self.cfg, 'model_dump') else dict(self.cfg),
            'input_dims': self.input_dims,
            'best_val_metric': self.best_val_metric,
            'training_complete': True,
            'preprocessing_enabled': bool(self.preprocessors),
            'preprocessing_metadata': self.preprocessing_metadata
        }
        
        torch.save(enhanced_checkpoint, path)
        logger.info(f"Enhanced model saved: {path}")

    def _create_enhanced_model_artifact(self, path: Path):
        """Create enhanced .nxf model artifact with relational data support."""
        if self.best_model_state is None:
            logger.warning("No trained model to create artifact from")
            return
        
        # Create enhanced model instance
        embed_dim = self.cfg.architecture.global_embed_dim
        refinement_iterations = self.cfg.architecture.refinement_iterations
        
        model = NexusFormer(
            input_dims=self.input_dims,
            embed_dim=embed_dim,
            refinement_iterations=refinement_iterations,
            encoder_type=getattr(self.model, 'encoder_type', 'standard'),
            use_moe=self.cfg.architecture.use_moe,
            num_experts=self.cfg.architecture.num_experts,
            use_flash_attn=self.cfg.architecture.use_flash_attn
        )
        model.load_state_dict(self.best_model_state)
        
        # Enhanced metadata with relational and preprocessing information
        meta = {
            'config': self.cfg.model_dump() if hasattr(self.cfg, 'model_dump') else dict(self.cfg),
            'input_dims': self.input_dims,
            'best_val_metric': self.best_val_metric,
            'best_epoch': self.best_epoch,
            'model_class': 'NexusFormer',
            'training_complete': True,
            'relational_features': {
                'relational_data_support': True,
                'original_datasets': len(self.cfg.datasets),
                'flattened_features': sum(self.input_dims),
                'join_relationships': len([fk for ds in self.cfg.datasets for fk in (ds.foreign_keys or [])]),
                'preprocessing_metadata': self.preprocessing_metadata
            },
            'phase_2_features': {
                'advanced_preprocessing': bool(self.preprocessors),
                'relational_joins': True,
                'preprocessor_datasets': list(self.preprocessors.keys()) if self.preprocessors else []
            },
            'architecture_features': {
                'encoder_type': getattr(self.model, 'encoder_type', 'standard'),
                'use_moe': self.cfg.architecture.use_moe,
                'num_experts': self.cfg.architecture.num_experts,
                'use_flash_attn': self.cfg.architecture.use_flash_attn,
                'refinement_iterations': refinement_iterations,
                'embed_dim': embed_dim
            }
        }
        
        # Include preprocessors and relational schema
        if self.preprocessors:
            meta['preprocessors'] = self.preprocessors
            meta['relational_schema'] = {
                'datasets': [ds.dict() for ds in self.cfg.datasets],
                'target_table': self.cfg.target.get('target_table'),
                'join_order': [ds.name for ds in self.cfg.datasets]
            }
        
        # Create enhanced ModelAPI instance
        model_api = ModelAPI(model, preprocess_meta=meta)
        model_api.save(str(path))
        
        logger.info(f"🎁 Relational model artifact created: {path}")
        logger.info(f"   Features: relational joins, {'advanced preprocessing' if self.preprocessors else 'simple preprocessing'}")
        logger.info(f"   Original datasets: {len(self.cfg.datasets)}, Final features: {sum(self.input_dims)}")

    def _validate_relational_data_at_inference(self, new_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Validate and process new relational data for inference.
        
        This function ensures new data follows the same relational schema
        and applies the same joins and preprocessing as during training.
        """
        logger.info("🔍 Validating relational data for inference...")
        
        # Check that all required tables are present
        required_tables = [ds.name for ds in self.cfg.datasets]
        provided_tables = list(new_data.keys())
        
        missing_tables = set(required_tables) - set(provided_tables)
        if missing_tables:
            raise ValueError(f"Missing required tables for inference: {missing_tables}")
        
        # Apply the same relational flattening as during training
        flattened_df = flatten_relational_data(new_data, self.cfg)
        
        # Apply same preprocessing if available
        if self.preprocessors:
            preprocessor = list(self.preprocessors.values())[0]  # We have one preprocessor for flattened data
            
            target_col = self.cfg.target.get('target_column')
            feature_df = flattened_df.copy()
            if target_col and target_col in feature_df.columns:
                feature_df = feature_df.drop(columns=[target_col])
            
            processed_features = preprocessor.transform(feature_df)
            
            # Reconstruct with target if present
            final_cols = preprocessor.categorical_columns + preprocessor.numerical_columns
            processed_df = processed_features[final_cols].copy()
            
            if target_col and target_col in flattened_df.columns:
                processed_df[target_col] = flattened_df[target_col]
            
            return processed_df
        
        return flattened_df

    def evaluate(self) -> Dict[str, float]:
        """Enhanced evaluation with Phase 2 preprocessing metrics."""
        if self.test_loader is None:
            logger.warning("No test data available for evaluation")
            return {}
        
        logger.info("📊 Running Phase 2 enhanced evaluation...")
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info(f"🏆 Using best model from epoch {self.best_epoch}")
        
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        batch_times = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                batch_start = time.time()
                
                if self.cfg.training.use_synthetic:
                    *features, targets = batch
                else:
                    features, targets = batch
                
                features = [f.to(self.device) for f in features]
                targets = targets.to(self.device)
                
                predictions = self.model(features)
                loss = self._calculate_loss(predictions, targets)
                
                batch_time = time.time() - batch_start
                batch_times.append(batch_time)
                
                total_loss += loss.item()
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
        
        # Calculate enhanced metrics
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        
        avg_test_loss = total_loss / len(self.test_loader)
        metrics = self._calculate_metrics(all_predictions, all_targets)
        
        # Add performance and preprocessing metrics
        metrics.update({
            'test_loss': avg_test_loss,
            'num_test_samples': len(all_targets),
            'num_test_batches': len(self.test_loader),
            'avg_inference_time': sum(batch_times) / len(batch_times),
            'total_inference_time': sum(batch_times),
            'samples_per_second': len(all_targets) / sum(batch_times),
            'preprocessing_enabled': bool(self.preprocessors)
        })
        
        # Log final evaluation
        self.mlops_logger.log_metrics(metrics, step=self.best_epoch)
        
        logger.info(f"🎯 Phase 2 enhanced evaluation complete:")
        for key, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"   {key}: {value:.6f}")
            else:
                logger.info(f"   {key}: {value}")
        
        return metrics