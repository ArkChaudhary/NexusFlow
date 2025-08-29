import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from loguru import logger
from pathlib import Path
import json
import sys
from typing import Dict, Optional, Any
import time
import os

from nexusflow.config import ConfigModel
from nexusflow.model.nexus_former import NexusFormer
from nexusflow.model.transformer_factory import TransformerFactory
from nexusflow.data.ingestion import load_datasets, align_datasets, get_feature_dimensions, make_dataloaders
from nexusflow.api.model_api import ModelAPI

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
        # Enhanced console logging with color coding for important metrics
        metric_str = " ".join([f"{k}={v:.6f}" for k, v in metrics.items()])
        logger.info(f"📊 Metrics (step {step}): {metric_str}")
        
        # Store in internal log with timestamp
        log_entry = {"step": step, "metrics": metrics, "timestamp": time.time()}
        self.metrics_log.append(log_entry)
        
        # Log to external provider
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
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'encoder_type': getattr(config, 'encoder_type', 'mixed'),
            'use_moe': config.advanced.use_moe,
            'use_flash_attn': config.advanced.use_flash_attn,
            'num_experts': config.advanced.num_experts if config.advanced.use_moe else 0
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
    
    def save_metrics_log(self, path: str):
        """Save internal metrics log to file."""
        with open(path, 'w') as f:
            json.dump(self.metrics_log, f, indent=2)

class Trainer:
    """
    Enhanced trainer with advanced architecture support and comprehensive monitoring.
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
        
        logger.info(f"🚀 Enhanced Trainer initialized (device={self.device})")

        # Initialize data-related attributes
        self.datasets = None
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
        
        # Load and process datasets
        self._setup_data()
        
        # Initialize enhanced model with advanced features
        self._initialize_enhanced_model()
        
        # Setup optimizer with gradient clipping
        self._setup_optimizer()
        
        # Log comprehensive model statistics
        self.mlops_logger.log_architecture_stats(self.model, self.cfg)

    def _initialize_enhanced_model(self):
        """Initialize model with advanced architecture features."""
        embed_dim = self.cfg.architecture.get('global_embed_dim', 64)
        refinement_iterations = self.cfg.architecture.get('refinement_iterations', 3)
        
        # Determine encoder type strategy
        if self.cfg.datasets:
            # Check if all datasets use the same transformer type
            transformer_types = {d.transformer_type for d in self.cfg.datasets}
            if len(transformer_types) == 1:
                encoder_type = list(transformer_types)[0]
            else:
                # Mixed types - use factory for each dataset individually
                encoder_type = 'mixed'
        else:
            encoder_type = 'standard'
        
        self.model = NexusFormer(
            input_dims=self.input_dims,
            embed_dim=embed_dim,
            refinement_iterations=refinement_iterations,
            encoder_type=encoder_type,
            use_moe=self.cfg.advanced.use_moe,
            num_experts=self.cfg.advanced.num_experts,
            use_flash_attn=self.cfg.advanced.use_flash_attn
        ).to(self.device)
        
        logger.info(f"🧠 Enhanced model initialized: {encoder_type} encoders")
        logger.info(f"   Advanced features: MoE={self.cfg.advanced.use_moe}, "
                   f"FlashAttn={self.cfg.advanced.use_flash_attn}")
    
    def _setup_optimizer(self):
        """Setup optimizer with enhanced features."""
        optim_config = self.cfg.training.optimizer
        lr = optim_config.get('lr', 1e-3)
        
        if optim_config.get('name', 'adam').lower() == 'adam':
            # Enhanced Adam with weight decay
            weight_decay = optim_config.get('weight_decay', 1e-4)
            self.optim = torch.optim.Adam(
                self.model.parameters(), 
                lr=lr, 
                weight_decay=weight_decay
            )
        else:
            self.optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optim, mode='min', factor=0.5, patience=3, verbose=True
        )

    def _setup_file_logging(self):
        """Setup enhanced structured logging."""
        log_dir = self.work_dir / "results" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"enhanced_training_{self.cfg.mlops.experiment_name}.log"
        logger.add(
            log_file,
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="10 MB",
            retention="7 days"
        )
        logger.info(f"📝 Enhanced logging enabled: {log_file}")

    def _log_attention_patterns(self, epoch: int):
        """Enhanced attention pattern logging with heatmap visualization."""
        if not self.cfg.mlops.log_attention_patterns or self.val_loader is None:
            return
        
        viz_dir = self.work_dir / "results" / "attention_viz"
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Enable attention logging
        for cross_attn in self.model.cross_attentions:
            cross_attn._log_attention = True
        
        # Collect attention patterns from validation batches
        attention_data = {'epoch': epoch, 'patterns': {}}
        
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                if i >= 5:  # Limit to first 5 batches
                    break
                
                if self.cfg.training.use_synthetic:
                    *features, targets = batch
                else:
                    features, targets = batch
                
                features = [f.to(self.device) for f in features]
                _ = self.model(features)  # Forward pass to collect attention
        
        # Extract and save attention data
        for i, cross_attn in enumerate(self.model.cross_attentions):
            if hasattr(cross_attn, '_attention_history'):
                attention_data['patterns'][f'encoder_{i}'] = cross_attn._attention_history
                cross_attn._attention_history = []  # Reset
        
        # Save attention patterns
        with open(viz_dir / f"attention_epoch_{epoch}.json", 'w') as f:
            json.dump(attention_data, f, indent=2, default=str)
        
        # Disable attention logging
        for cross_attn in self.model.cross_attentions:
            cross_attn._log_attention = False

    def _log_attention_heatmaps(self, epoch: int):
        """Log attention patterns for visualization."""
        viz_dir = self.work_dir / "results" / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Enable attention logging
        for cross_attn in self.model.cross_attentions:
            cross_attn._log_attention = True
        
        # Run a few validation batches to collect attention patterns
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                if i >= 3:  # Only log first few batches
                    break
                # ... process batch and collect attention ...
        
        # Save attention summaries
        attention_data = {}
        for i, cross_attn in enumerate(self.model.cross_attentions):
            if hasattr(cross_attn, '_attention_history'):
                attention_data[f'encoder_{i}'] = cross_attn._attention_history
                cross_attn._attention_history = []  # Reset
        
        import json
        with open(viz_dir / f"attention_epoch_{epoch}.json", 'w') as f:
            json.dump(attention_data, f, indent=2, default=str)
        
        # Disable attention logging
        for cross_attn in self.model.cross_attentions:
            cross_attn._log_attention = False

    def _setup_data(self):
        """Enhanced data setup with comprehensive logging."""
        training_cfg = self.cfg.training
        
        if training_cfg.use_synthetic:
            logger.info("🔬 Using synthetic data mode")
            n_datasets = len(self.cfg.datasets) if self.cfg.datasets else 2
            feature_dim = training_cfg.synthetic.feature_dim if training_cfg.synthetic else 5
            
            self.input_dims = [feature_dim] * n_datasets
            self.datasets = None
            logger.info(f"   Synthetic data: {n_datasets} datasets × {feature_dim} features")
            
        else:
            logger.info("📊 Loading and aligning real datasets...")
            raw_datasets = load_datasets(self.cfg)
            self.datasets = align_datasets(raw_datasets, self.cfg.primary_key)
            self.input_dims = get_feature_dimensions(
                self.datasets,
                self.cfg.primary_key,
                self.cfg.target['target_column']
            )
            
            # Enhanced data quality reporting
            total_samples = len(list(self.datasets.values())[0]) if self.datasets else 0
            total_features = sum(self.input_dims)
            
            logger.info(f"✅ Data alignment complete:")
            logger.info(f"   Aligned samples: {total_samples:,}")
            logger.info(f"   Datasets: {len(self.datasets)}")
            logger.info(f"   Feature dimensions: {self.input_dims}")
            logger.info(f"   Total features: {total_features}")
            
            # Data quality metrics
            for name, df in self.datasets.items():
                missing_pct = df.isnull().sum().sum() / df.size * 100
                unique_pct = df.nunique().sum() / len(df) * 100
                logger.info(f"   {name}: {missing_pct:.2f}% missing, {unique_pct:.1f}% unique ratio")

    def _setup_dataloaders(self):
        """Enhanced DataLoader setup with improved synthetic data."""
        if self.cfg.training.use_synthetic:
            synthetic_config = self.cfg.training.synthetic
            n_samples = synthetic_config.n_samples if synthetic_config else 256
            batch_size = self.cfg.training.batch_size
            
            # Generate more sophisticated synthetic data
            torch.manual_seed(42)
            Xs = []
            
            for i, dim in enumerate(self.input_dims):
                # Create correlated features with some noise
                base_signal = torch.randn(n_samples, 1)
                noise = torch.randn(n_samples, dim) * 0.3
                
                # Add dataset-specific patterns
                if i == 0:  # Primary dataset with strongest signal
                    features = base_signal.expand(-1, dim) + noise
                elif i == len(self.input_dims) - 1:  # Last dataset as noise
                    features = torch.randn(n_samples, dim) * 0.8
                else:  # Intermediate datasets with moderate signal
                    signal_strength = 0.7 - (i * 0.2)
                    features = base_signal.expand(-1, dim) * signal_strength + noise
                
                Xs.append(features)
            
            # Create target with realistic signal-to-noise ratio
            target_col = self.cfg.target["target_column"]
            if target_col == 'label':
                # Binary classification with clear decision boundary
                linear_combo = sum(X.mean(dim=1) * (0.8 - i * 0.2) for i, X in enumerate(Xs))
                probs = torch.sigmoid(linear_combo + torch.randn(n_samples) * 0.2)
                y = torch.bernoulli(probs).long()
            else:
                # Regression with multi-source signal
                y = sum(X.mean(dim=1) * (0.8 - i * 0.2) for i, X in enumerate(Xs)) + torch.randn(n_samples) * 0.1
            
            # Enhanced data splitting
            train_size = int(0.7 * n_samples)
            val_size = int(0.15 * n_samples)
            
            indices = torch.randperm(n_samples)
            train_idx = indices[:train_size]
            val_idx = indices[train_size:train_size+val_size] 
            test_idx = indices[train_size+val_size:]
            
            # Create datasets with proper tensors
            train_data = torch.utils.data.TensorDataset(*[X[train_idx] for X in Xs], y[train_idx])
            val_data = torch.utils.data.TensorDataset(*[X[val_idx] for X in Xs], y[val_idx])
            test_data = torch.utils.data.TensorDataset(*[X[test_idx] for X in Xs], y[test_idx])
            
            self.train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
            self.val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
            self.test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
            
            logger.info(f"🔄 Enhanced synthetic DataLoaders: train={len(self.train_loader)} "
                       f"val={len(self.val_loader)} test={len(self.test_loader)} batches")
        else:
            self.train_loader, self.val_loader, self.test_loader = make_dataloaders(self.cfg, self.datasets)

    def _calculate_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Enhanced loss calculation with label smoothing for classification."""
        target_col = self.cfg.target["target_column"]
        
        if target_col == 'label':
            # Binary classification with optional label smoothing
            if targets.dtype == torch.long and predictions.dim() == 1:
                return torch.nn.functional.binary_cross_entropy_with_logits(
                    predictions, targets.float()
                )
            else:
                return torch.nn.functional.cross_entropy(predictions, targets)
        else:
            # Huber loss for robust regression
            return torch.nn.functional.huber_loss(predictions, targets.float(), delta=1.0)

    def _calculate_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Enhanced metrics calculation with confidence intervals."""
        target_col = self.cfg.target["target_column"]
        metrics = {}
        
        if target_col == 'label':
            # Enhanced classification metrics
            if predictions.dim() == 1:
                probs = torch.sigmoid(predictions)
                preds = (probs > 0.5).long()
                metrics['confidence'] = probs.std().item()  # Prediction confidence
            else:
                probs = torch.softmax(predictions, dim=1)
                preds = torch.argmax(predictions, dim=1)
                metrics['confidence'] = probs.max(dim=1)[0].mean().item()
            
            accuracy = (preds == targets).float().mean().item()
            metrics['accuracy'] = accuracy
            
            # Additional classification metrics
            if len(torch.unique(targets)) == 2:  # Binary classification
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
            # Enhanced regression metrics
            mse = torch.nn.functional.mse_loss(predictions, targets.float()).item()
            mae = torch.nn.functional.l1_loss(predictions, targets.float()).item()
            
            # R-squared calculation
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
        """Enhanced sanity check with architecture validation."""
        logger.info("🔍 Running comprehensive sanity checks...")
        
        # Test model forward pass
        batch_size = 2
        dummy_inputs = [torch.randn(batch_size, dim).to(self.device) for dim in self.input_dims]
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(dummy_inputs)
        
        logger.info(f"✅ Model forward pass: output_shape={output.shape}")
        
        # Test advanced features
        if self.cfg.advanced.use_moe:
            logger.info("🔀 MoE layers active")
        if self.cfg.advanced.use_flash_attn:
            logger.info("⚡ FlashAttention enabled")
        
        # Setup data loaders and test them
        self._setup_dataloaders()
        
        # Test training batch
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
        
        logger.info("✅ All enhanced sanity checks passed!")

    def train(self):
        """Enhanced training loop with advanced monitoring and early stopping."""
        epochs = int(self.cfg.training.epochs)
        logger.info(f"🎯 Starting enhanced training: {epochs} epochs")

        self._setup_dataloaders()
        
        if self.train_loader is None:
            raise RuntimeError("No training data available")

        # Training loop with enhanced monitoring
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
                
                # Enhanced gradient clipping
                if self.cfg.training.gradient_clipping > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.training.gradient_clipping)
                
                self.optim.step()
                
                train_loss += loss.item()
                train_batches += 1
                
                # Log batch-level metrics periodically
                if batch_idx % 50 == 0 and batch_idx > 0:
                    logger.debug(f"Epoch {epoch} Batch {batch_idx}: loss={loss.item():.6f}")
            
            avg_train_loss = train_loss / train_batches
            
            # Validation phase
            val_metrics = self._validate_epoch()
            
            # Learning rate scheduling
            if val_metrics and 'val_loss' in val_metrics:
                self.scheduler.step(val_metrics['val_loss'])
            
            # Combine metrics with timing
            epoch_time = time.time() - epoch_start_time
            epoch_metrics = {
                'train_loss': avg_train_loss,
                'epoch': epoch,
                'epoch_time': epoch_time,
                'learning_rate': self.optim.param_groups[0]['lr']
            }
            epoch_metrics.update(val_metrics)
            
            # Enhanced best model tracking
            current_metric = val_metrics.get('val_loss', avg_train_loss)
            if current_metric < self.best_val_metric:
                self.best_val_metric = current_metric
                self.best_epoch = epoch
                self.best_model_state = self.model.state_dict().copy()
                self.epochs_without_improvement = 0
                logger.info(f"🌟 New best model at epoch {epoch}: {current_metric:.6f}")
            else:
                self.epochs_without_improvement += 1
            
            # Log attention patterns periodically
            if epoch % 5 == 0:
                self._log_attention_patterns(epoch)
            
            # Enhanced progress logging
            progress_pct = (epoch / epochs) * 100
            logger.info(f"🔄 Epoch {epoch}/{epochs} ({progress_pct:.1f}%): "
                       f"train_loss={avg_train_loss:.6f} val_loss={val_metrics.get('val_loss', 'N/A')} "
                       f"time={epoch_time:.2f}s lr={self.optim.param_groups[0]['lr']:.2e}")
            
            # Store training history
            self.training_history.append(epoch_metrics)
            
            # Log to MLOps
            self.mlops_logger.log_metrics(epoch_metrics, step=epoch)
            
            # Early stopping check
            if (self.cfg.training.early_stopping and 
                self.epochs_without_improvement >= self.cfg.training.patience):
                logger.info(f"🛑 Early stopping triggered after {epoch} epochs "
                           f"(no improvement for {self.epochs_without_improvement} epochs)")
                break
            
            # Save checkpoint every 10 epochs or at the end
            if epoch % 10 == 0 or epoch == epochs:
                self._save_checkpoint(self.work_dir / f"checkpoint_epoch_{epoch}.pt", epoch, epoch_metrics)
        
        # Post-training cleanup and artifact creation
        self._finalize_training()

    def _finalize_training(self):
        """Enhanced training finalization with comprehensive artifacts."""
        if self.best_model_state is not None:
            # Save best model
            best_model_path = self.work_dir / "best_model.pt"
            self._save_best_model(best_model_path)
            
            # Create enhanced model artifact
            nxf_path = self.work_dir / f"{self.cfg.project_name}.nxf"
            self._create_model_artifact(nxf_path)
            
            # Save training history
            history_path = self.work_dir / "results" / "training_history.json"
            history_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(history_path, 'w') as f:
                json.dump({
                    'training_history': self.training_history,
                    'best_epoch': self.best_epoch,
                    'best_metric': self.best_val_metric,
                    'total_epochs': len(self.training_history),
                    'early_stopped': self.epochs_without_improvement >= self.cfg.training.patience
                }, f, indent=2)
        
        # Save metrics log
        metrics_log_path = self.work_dir / "results" / "metrics.json"
        metrics_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.mlops_logger.save_metrics_log(str(metrics_log_path))
        
        # Cleanup MLOps logging
        self.mlops_logger.finish()
        
        logger.info(f"🎉 Enhanced training complete! Best model at epoch {self.best_epoch} "
                   f"with metric {self.best_val_metric:.6f}")

    def _save_checkpoint(self, path: Path, epoch: int, metrics: Dict[str, float]):
        """Enhanced checkpoint saving with architecture info."""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optim.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'config': self.cfg.model_dump() if hasattr(self.cfg, 'model_dump') else dict(self.cfg),
            'input_dims': self.input_dims,
            'metrics': metrics,
            'best_val_metric': self.best_val_metric,
            'best_epoch': self.best_epoch,
            'training_history': self.training_history,
            'architecture_info': {
                'encoder_type': self.model.encoder_type,
                'use_moe': self.model.use_moe,
                'use_flash_attn': self.model.use_flash_attn,
                'total_parameters': sum(p.numel() for p in self.model.parameters())
            }
        }
        
        torch.save(checkpoint, path)
        logger.debug(f"💾 Enhanced checkpoint saved: {path}")

    def _save_best_model(self, path: Path):
        """Save the best model state."""
        if self.best_model_state is None:
            logger.warning("No best model state to save")
            return
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        best_checkpoint = {
            'epoch': self.best_epoch,
            'model_state': self.best_model_state,
            'config': self.cfg.model_dump() if hasattr(self.cfg, 'model_dump') else dict(self.cfg),
            'input_dims': self.input_dims,
            'best_val_metric': self.best_val_metric,
            'training_complete': True
        }
        
        torch.save(best_checkpoint, path)
        logger.info(f"Best model saved: {path}")

    def _create_model_artifact(self, path: Path):
        """Create enhanced .nxf model artifact with advanced architecture metadata."""
        if self.best_model_state is None:
            logger.warning("No trained model to create artifact from")
            return
        
        # Create enhanced model instance
        embed_dim = self.cfg.architecture.get('global_embed_dim', 64)
        refinement_iterations = self.cfg.architecture.get('refinement_iterations', 3)
        
        model = NexusFormer(
            input_dims=self.input_dims,
            embed_dim=embed_dim,
            refinement_iterations=refinement_iterations,
            encoder_type=getattr(self.model, 'encoder_type', 'standard'),
            use_moe=self.cfg.advanced.use_moe,
            num_experts=self.cfg.advanced.num_experts,
            use_flash_attn=self.cfg.advanced.use_flash_attn
        )
        model.load_state_dict(self.best_model_state)
        
        # Enhanced metadata
        meta = {
            'config': self.cfg.model_dump() if hasattr(self.cfg, 'model_dump') else dict(self.cfg),
            'input_dims': self.input_dims,
            'best_val_metric': self.best_val_metric,
            'best_epoch': self.best_epoch,
            'model_class': 'NexusFormer',
            'training_complete': True,
            'architecture_features': {
                'encoder_type': getattr(self.model, 'encoder_type', 'standard'),
                'use_moe': self.cfg.advanced.use_moe,
                'num_experts': self.cfg.advanced.num_experts,
                'use_flash_attn': self.cfg.advanced.use_flash_attn,
                'refinement_iterations': refinement_iterations,
                'embed_dim': embed_dim
            },
            'performance_metrics': {
                'best_metric': self.best_val_metric,
                'total_epochs': len(self.training_history),
                'parameters': sum(p.numel() for p in model.parameters())
            }
        }
        
        # Create enhanced ModelAPI instance
        model_api = ModelAPI(model, preprocess_meta=meta)
        model_api.save(str(path))
        
        logger.info(f"🎁 Enhanced model artifact created: {path}")

    def evaluate(self) -> Dict[str, float]:
        """Enhanced evaluation with detailed performance analysis."""
        if self.test_loader is None:
            logger.warning("No test data available for evaluation")
            return {}
        
        logger.info("📊 Running comprehensive enhanced evaluation...")
        
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
        
        # Add performance metrics
        metrics.update({
            'test_loss': avg_test_loss,
            'num_test_samples': len(all_targets),
            'num_test_batches': len(self.test_loader),
            'avg_inference_time': sum(batch_times) / len(batch_times),
            'total_inference_time': sum(batch_times),
            'samples_per_second': len(all_targets) / sum(batch_times)
        })
        
        # Log final evaluation
        self.mlops_logger.log_metrics(metrics, step=self.best_epoch)
        
        logger.info(f"🎯 Enhanced evaluation complete:")
        for key, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"   {key}: {value:.6f}")
        
        return metrics