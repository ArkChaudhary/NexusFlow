import torch
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
from nexusflow.data.ingestion import load_datasets, align_datasets, get_feature_dimensions, make_dataloaders
from nexusflow.api.model_api import ModelAPI

class MLOpsLogger:
    """Handles logging to different MLOps providers."""
    
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
        """Log metrics to the configured provider."""
        # Always log to console
        metric_str = " ".join([f"{k}={v:.6f}" for k, v in metrics.items()])
        logger.info(f"Metrics (step {step}): {metric_str}")
        
        # Store in internal log
        log_entry = {"step": step, "metrics": metrics, "timestamp": time.time()}
        self.metrics_log.append(log_entry)
        
        # Log to external provider
        if self.provider == "wandb" and hasattr(self, 'wandb') and self.wandb:
            self.wandb.log(metrics, step=step)
        elif self.provider == "mlflow" and hasattr(self, 'mlflow') and self.mlflow:
            for key, value in metrics.items():
                self.mlflow.log_metric(key, value, step=step)
    
    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters."""
        logger.info(f"Logging parameters: {params}")
        
        if self.provider == "wandb" and hasattr(self, 'wandb') and self.wandb:
            self.wandb.config.update(params)
        elif self.provider == "mlflow" and hasattr(self, 'mlflow') and self.mlflow:
            for key, value in params.items():
                self.mlflow.log_param(key, value)
    
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
    Enhanced trainer with MLOps integration, best model tracking, and comprehensive logging.
    """
    
    def __init__(self, config: ConfigModel, work_dir: str = '.'):
        self.cfg = config
        self.work_dir = Path(work_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize MLOps logging
        self.mlops_logger = MLOpsLogger(
            provider=self.cfg.mlops.logging_provider,
            experiment_name=self.cfg.mlops.experiment_name
        )
        
        # Setup file logging
        self._setup_file_logging()
        
        logger.info(f"Enhanced Trainer initialized (device={self.device})")

        # Initialize data-related attributes
        self.datasets = None
        self.input_dims = None
        self.train_loader = None
        self.val_loader = None 
        self.test_loader = None
        
        # Training state tracking
        self.best_val_metric = float('inf')  # Assuming lower is better (loss)
        self.best_model_state = None
        self.best_epoch = 0
        
        # Load and process datasets to determine input dimensions
        self._setup_data()
        
        # Initialize model with enhanced architecture
        embed_dim = self.cfg.architecture.get('global_embed_dim', 64)
        refinement_iterations = self.cfg.architecture.get('refinement_iterations', 3)
        
        self.model = NexusFormer(
            input_dims=self.input_dims, 
            embed_dim=embed_dim,
            refinement_iterations=refinement_iterations
        ).to(self.device)
        
        # Initialize optimizer with config parameters
        optim_config = self.cfg.training.optimizer
        lr = optim_config.get('lr', 1e-3)
        
        if optim_config.get('name', 'adam').lower() == 'adam':
            self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        else:
            self.optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        
        # Log model parameters and hyperparameters to MLOps
        model_params = {
            'model_name': 'NexusFormer',
            'input_dims': self.input_dims,
            'embed_dim': embed_dim,
            'refinement_iterations': refinement_iterations,
            'num_parameters': sum(p.numel() for p in self.model.parameters()),
            'learning_rate': lr,
            'optimizer': optim_config.get('name', 'adam'),
            'batch_size': self.cfg.training.batch_size,
            'epochs': self.cfg.training.epochs
        }
        self.mlops_logger.log_params(model_params)
        
        logger.info(f"Model initialized: {sum(p.numel() for p in self.model.parameters())} parameters")

    def _setup_file_logging(self):
        """Setup structured logging to file."""
        log_dir = self.work_dir / "results" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Add file handler to loguru
        log_file = log_dir / f"training_{self.cfg.mlops.experiment_name}.log"
        logger.add(
            log_file,
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="10 MB",
            retention="7 days"
        )
        logger.info(f"File logging enabled: {log_file}")

    def _setup_data(self):
        """Enhanced data setup with better logging."""
        training_cfg = self.cfg.training
        
        if training_cfg.use_synthetic:
            logger.info("Using synthetic data mode")
            n_datasets = len(self.cfg.datasets) if self.cfg.datasets else 2
            feature_dim = 5  # default
            
            if training_cfg.synthetic is not None:
                feature_dim = training_cfg.synthetic.feature_dim
            
            self.input_dims = [feature_dim] * n_datasets
            self.datasets = None
            logger.info(f"Synthetic data: {n_datasets} datasets × {feature_dim} features")
            
        else:
            logger.info("Loading and aligning real datasets...")
            raw_datasets = load_datasets(self.cfg)
            self.datasets = align_datasets(raw_datasets, self.cfg.primary_key)
            self.input_dims = get_feature_dimensions(
                self.datasets,
                self.cfg.primary_key,
                self.cfg.target['target_column']
            )
            
            # Enhanced logging with data quality metrics
            total_samples = len(list(self.datasets.values())[0]) if self.datasets else 0
            total_features = sum(self.input_dims)
            
            logger.info(f"Data alignment complete:")
            logger.info(f"  Aligned samples: {total_samples}")
            logger.info(f"  Datasets: {len(self.datasets)}")
            logger.info(f"  Feature dimensions: {self.input_dims}")
            logger.info(f"  Total features: {total_features}")
            
            # Log data quality metrics
            for name, df in self.datasets.items():
                missing_pct = df.isnull().sum().sum() / df.size * 100
                logger.debug(f"  {name}: {missing_pct:.2f}% missing values")

    def _setup_dataloaders(self):
        """Create DataLoaders with enhanced synthetic data support."""
        if self.cfg.training.use_synthetic:
            synthetic_config = self.cfg.training.synthetic
            n_samples = synthetic_config.n_samples if synthetic_config else 256
            batch_size = self.cfg.training.batch_size
            
            # Generate more realistic synthetic data
            torch.manual_seed(42)  # For reproducibility
            Xs = [torch.randn(n_samples, dim) * 0.5 for dim in self.input_dims]
            
            # Create synthetic target with some correlation to features
            target_col = self.cfg.target["target_column"]
            if target_col == 'label':
                # Binary classification with some signal
                linear_combo = sum(X.mean(dim=1) for X in Xs) / len(Xs)
                probs = torch.sigmoid(linear_combo + torch.randn(n_samples) * 0.1)
                y = torch.bernoulli(probs).long()
            else:
                # Regression with signal
                y = sum(X.mean(dim=1) for X in Xs) / len(Xs) + torch.randn(n_samples) * 0.1
            
            # Split synthetic data
            train_size = int(0.7 * n_samples)
            val_size = int(0.15 * n_samples)
            
            indices = torch.randperm(n_samples)
            train_idx, val_idx, test_idx = indices[:train_size], indices[train_size:train_size+val_size], indices[train_size+val_size:]
            
            # Create datasets
            train_data = torch.utils.data.TensorDataset(*[X[train_idx] for X in Xs], y[train_idx])
            val_data = torch.utils.data.TensorDataset(*[X[val_idx] for X in Xs], y[val_idx])
            test_data = torch.utils.data.TensorDataset(*[X[test_idx] for X in Xs], y[test_idx])
            
            self.train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
            self.val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
            self.test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
            
            logger.info(f"Synthetic DataLoaders: train={len(self.train_loader)} val={len(self.val_loader)} test={len(self.test_loader)} batches")
        else:
            self.train_loader, self.val_loader, self.test_loader = make_dataloaders(self.cfg, self.datasets)

    def _calculate_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate loss based on task type."""
        target_col = self.cfg.target["target_column"]
        
        if target_col == 'label':
            # Classification task
            if targets.dtype == torch.long and predictions.dim() == 1:
                # Binary classification with logits
                return torch.nn.functional.binary_cross_entropy_with_logits(
                    predictions, targets.float()
                )
            else:
                # Multi-class (expand logits if needed)
                if predictions.dim() == 1:
                    # Convert single logit to binary classification probabilities
                    predictions = torch.stack([1-torch.sigmoid(predictions), torch.sigmoid(predictions)], dim=1)
                return torch.nn.functional.cross_entropy(predictions, targets)
        else:
            # Regression task
            return torch.nn.functional.mse_loss(predictions, targets.float())

    def _calculate_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Calculate evaluation metrics based on task type."""
        target_col = self.cfg.target["target_column"]
        metrics = {}
        
        if target_col == 'label':
            # Classification metrics
            if predictions.dim() == 1:
                # Binary classification
                probs = torch.sigmoid(predictions)
                preds = (probs > 0.5).long()
            else:
                # Multi-class
                preds = torch.argmax(predictions, dim=1)
            
            accuracy = (preds == targets).float().mean().item()
            metrics['accuracy'] = accuracy
        else:
            # Regression metrics
            mse = torch.nn.functional.mse_loss(predictions, targets.float()).item()
            mae = torch.nn.functional.l1_loss(predictions, targets.float()).item()
            metrics['mse'] = mse
            metrics['mae'] = mae
        
        return metrics

    def _validate_epoch(self) -> Dict[str, float]:
        """Run validation and return metrics."""
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
        
        # Calculate metrics
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        
        avg_loss = total_loss / len(self.val_loader)
        metrics = self._calculate_metrics(all_predictions, all_targets)
        metrics['val_loss'] = avg_loss
        
        return metrics

    def sanity_check(self):
        """Enhanced sanity check with validation data."""
        logger.info("Running comprehensive sanity checks...")
        
        # Test model forward pass
        batch_size = 2
        dummy_inputs = [torch.randn(batch_size, dim).to(self.device) for dim in self.input_dims]
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(dummy_inputs)
        
        logger.info(f"Model forward pass: output_shape={output.shape}")
        
        # Setup data loaders and test them
        self._setup_dataloaders()
        
        # Test a training batch
        if self.train_loader:
            sample_batch = next(iter(self.train_loader))
            if self.cfg.training.use_synthetic:
                *features, targets = sample_batch
            else:
                features, targets = sample_batch
            
            logger.info(f"Sample batch: features={[f.shape for f in features]} targets={targets.shape}")
            
            # Test validation if available
            if self.val_loader:
                val_metrics = self._validate_epoch()
                logger.info(f"Initial validation metrics: {val_metrics}")
        
        logger.info("All sanity checks passed!")

    def train(self):
        """Enhanced training loop with validation tracking and checkpointing."""
        epochs = int(self.cfg.training.epochs)
        logger.info(f"Starting enhanced training: {epochs} epochs")

        self._setup_dataloaders()
        
        if self.train_loader is None:
            raise RuntimeError("No training data available")

        # Training loop
        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_batches = 0
            
            for batch in self.train_loader:
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
                self.optim.step()
                
                train_loss += loss.item()
                train_batches += 1
            
            avg_train_loss = train_loss / train_batches
            
            # Validation phase
            val_metrics = self._validate_epoch()
            
            # Combine metrics
            epoch_time = time.time() - epoch_start_time
            epoch_metrics = {
                'train_loss': avg_train_loss,
                'epoch': epoch,
                'epoch_time': epoch_time
            }
            epoch_metrics.update(val_metrics)
            
            # Log metrics
            self.mlops_logger.log_metrics(epoch_metrics, step=epoch)
            
            # Check for best model (using validation loss if available)
            current_metric = val_metrics.get('val_loss', avg_train_loss)
            if current_metric < self.best_val_metric:
                self.best_val_metric = current_metric
                self.best_epoch = epoch
                self.best_model_state = self.model.state_dict().copy()
                logger.info(f"New best model at epoch {epoch}: {current_metric:.6f}")
            
            # Save regular checkpoint
            self._save_checkpoint(self.work_dir / f"model_epoch_{epoch}.pt", epoch, epoch_metrics)
            
            # Progress update
            logger.info(f"Epoch {epoch}/{epochs}: train_loss={avg_train_loss:.6f} val_loss={val_metrics.get('val_loss', 'N/A')} time={epoch_time:.2f}s")
        
        # Save best model
        if self.best_model_state is not None:
            best_model_path = self.work_dir / "best_model.pt"
            self._save_best_model(best_model_path)
            
            # Create model artifact (.nxf file)
            nxf_path = self.work_dir / f"{self.cfg.project_name}.nxf"
            self._create_model_artifact(nxf_path)
        
        # Save metrics log
        metrics_log_path = self.work_dir / "results" / "metrics.json"
        metrics_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.mlops_logger.save_metrics_log(str(metrics_log_path))
        
        # Cleanup MLOps logging
        self.mlops_logger.finish()
        
        logger.info(f"Training complete! Best model at epoch {self.best_epoch} with metric {self.best_val_metric:.6f}")

    def _save_checkpoint(self, path: Path, epoch: int, metrics: Dict[str, float]):
        """Save training checkpoint with metrics."""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optim.state_dict(),
            'config': self.cfg.model_dump() if hasattr(self.cfg, 'model_dump') else dict(self.cfg),
            'input_dims': self.input_dims,
            'metrics': metrics,
            'best_val_metric': self.best_val_metric,
            'best_epoch': self.best_epoch
        }
        
        torch.save(checkpoint, path)
        logger.debug(f"Checkpoint saved: {path}")

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
        """Create the .nxf model artifact with prediction interface."""
        if self.best_model_state is None:
            logger.warning("No trained model to create artifact from")
            return
        
        # Create the model instance
        embed_dim = self.cfg.architecture.get('global_embed_dim', 64)
        refinement_iterations = self.cfg.architecture.get('refinement_iterations', 3)
        
        model = NexusFormer(
            input_dims=self.input_dims, 
            embed_dim=embed_dim,
            refinement_iterations=refinement_iterations
        )
        model.load_state_dict(self.best_model_state)
        
        # Prepare metadata
        meta = {
            'config': self.cfg.model_dump() if hasattr(self.cfg, 'model_dump') else dict(self.cfg),
            'input_dims': self.input_dims,
            'best_val_metric': self.best_val_metric,
            'best_epoch': self.best_epoch,
            'model_class': 'NexusFormer',
            'training_complete': True
        }
        
        # Create ModelAPI instance
        model_api = ModelAPI(model, preprocess_meta=meta)
        model_api.save(str(path))
        
        logger.info(f"Model artifact created: {path}")

    def evaluate(self) -> Dict[str, float]:
        """Enhanced evaluation with comprehensive metrics."""
        if self.test_loader is None:
            logger.warning("No test data available for evaluation")
            return {}
        
        logger.info("Running comprehensive evaluation on test set...")
        
        # Load best model if available
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info(f"Using best model from epoch {self.best_epoch}")
        
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        batch_count = 0
        
        with torch.no_grad():
            for batch in self.test_loader:
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
                batch_count += 1
        
        # Calculate comprehensive metrics
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        
        avg_test_loss = total_loss / batch_count
        metrics = self._calculate_metrics(all_predictions, all_targets)
        metrics['test_loss'] = avg_test_loss
        metrics['num_test_samples'] = len(all_targets)
        metrics['num_test_batches'] = batch_count
        
        # Log final evaluation metrics
        self.mlops_logger.log_metrics(metrics, step=self.best_epoch)
        
        logger.info(f"Final evaluation metrics: {metrics}")
        return metrics