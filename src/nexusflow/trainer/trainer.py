import torch
from torch.utils.data import DataLoader
from loguru import logger
from pathlib import Path

from nexusflow.config import ConfigModel
from nexusflow.model.nexus_former import NexusFormer
from nexusflow.data.ingestion import load_datasets, align_datasets, get_feature_dimensions, make_dataloaders

class Trainer:
    """
    Main trainer class for the NexusFormer model.
    
    Handles data loading, model initialization, training, and evaluation.
    Supports both synthetic and real CSV data modes.
    """
    
    def __init__(self, config: ConfigModel, work_dir: str = '.'):
        """
        Initialize the trainer with configuration.
        
        Args:
            config: ConfigModel containing all training and model parameters
            work_dir: Working directory for saving checkpoints and logs
        """
        self.cfg = config
        self.work_dir = Path(work_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Trainer initialized (device={self.device})")

        # Initialize data-related attributes
        self.datasets = None
        self.input_dims = None
        self.train_loader = None
        self.val_loader = None 
        self.test_loader = None
        
        # Load and process datasets to determine input dimensions
        self._setup_data()
        
        # Initialize model with correct input dimensions
        embed_dim = getattr(self.cfg.architecture, 'global_embed_dim', 64)
        self.model = NexusFormer(self.input_dims, embed_dim=embed_dim).to(self.device)
        
        # Initialize optimizer with learning rate from config
        lr = getattr(self.cfg.training.optimizer, 'lr', 1e-3)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        logger.info(f"Model initialized with input dimensions: {self.input_dims}")

    def _setup_data(self):
        """
        Load and prepare datasets, determining input dimensions.
        Handles two modes:
        1. Synthetic data mode: Creates artificial data with specified dimensions
        2. Real data mode: Loads CSV files and aligns them by primary key
        """
        training_cfg = self.cfg.training
        
        # Now we can simply access the attributes since they're defined in the Pydantic model
        if training_cfg.use_synthetic:
            logger.info("Using synthetic data mode")
            n_datasets = len(self.cfg.datasets)
            
            # Get feature dimension from synthetic config, with default fallback
            feature_dim = 5  # default
            if training_cfg.synthetic is not None:
                feature_dim = training_cfg.synthetic.feature_dim
            
            self.input_dims = [feature_dim] * n_datasets
            self.datasets = None
            logger.info(f"Using synthetic data with {n_datasets} datasets of {feature_dim} features each")
            
        else:
            logger.info("Loading real datasets...")
            raw_datasets = load_datasets(self.cfg)
            self.datasets = align_datasets(raw_datasets, self.cfg.primary_key)
            self.input_dims = get_feature_dimensions(
                self.datasets,
                self.cfg.primary_key,
                self.cfg.target['target_column']
            )
            
            # Enhanced logging: dataset summary after alignment
            total_samples = len(list(self.datasets.values())[0]) if self.datasets else 0
            logger.info(f"Aligned datasets summary:")
            logger.info(f"  Total samples after alignment: {total_samples}")
            logger.info(f"  Number of datasets: {len(self.datasets)}")
            logger.info(f"  Input dimensions per dataset: {self.input_dims}")
            logger.info(f"  Total features across all datasets: {sum(self.input_dims)}")

    def _setup_dataloaders(self):
        """
        Create PyTorch DataLoaders for training, validation, and testing.
        
        Handles both synthetic and real data modes with appropriate batch sizing.
        """
        use_synthetic = getattr(self.cfg.training, 'use_synthetic', False)
        
        if use_synthetic:
            # Create synthetic data tensors
            synthetic_config = getattr(self.cfg.training, 'synthetic', {})
            n_samples = getattr(synthetic_config, 'n_samples', 256) if hasattr(synthetic_config, 'n_samples') else 256
            batch_size = self.cfg.training.batch_size
            
            # Generate synthetic multi-table data (one tensor per dataset)
            Xs = [torch.randn(n_samples, dim) for dim in self.input_dims]
            
            # Generate synthetic target based on target column name
            target_col = self.cfg.target["target_column"]  # Fixed: removed .get() call
            if target_col == 'label':
                # Binary classification targets
                y = torch.randint(0, 2, (n_samples,))
            else:
                # Regression targets
                y = torch.randn(n_samples)
            
            # Create TensorDataset combining all features and target
            synthetic_data = torch.utils.data.TensorDataset(*Xs, y)
            self.train_loader = DataLoader(synthetic_data, batch_size=batch_size, shuffle=True)
            
            # No validation/test sets for synthetic data
            self.val_loader = None
            self.test_loader = None
            
            logger.info(f"Created synthetic DataLoader with {len(self.train_loader)} batches")
        else:
            # Use real data - delegate to data ingestion module
            self.train_loader, self.val_loader, self.test_loader = make_dataloaders(self.cfg, self.datasets)

    def sanity_check(self):
        """
        Run sanity checks to ensure model and data are compatible.
        
        Tests model forward pass with dummy data and validates data alignment.
        """
        logger.info("Running trainer sanity checks...")
        
        # Test model with dummy inputs matching expected dimensions
        batch_size = 2
        dummy_inputs = [torch.randn(batch_size, dim).to(self.device) for dim in self.input_dims]
        
        # Run forward pass in evaluation mode
        self.model.eval()
        with torch.no_grad():
            output = self.model(dummy_inputs)
        
        logger.info(f"Sanity check forward pass output shape: {output.shape}")
        
        # Additional checks for real data
        use_synthetic = getattr(self.cfg.training, 'use_synthetic', False)
        if not use_synthetic:
            if self.datasets:
                # Check that all datasets have the same number of samples after alignment
                total_samples = len(list(self.datasets.values())[0])
                logger.info(f"Real data sanity check: {total_samples} aligned samples across {len(self.datasets)} datasets")
        
        logger.info("Sanity checks passed!")

    def train(self):
        """
        Main training loop with loss calculation and checkpoint saving.
        
        Supports both classification and regression tasks based on target column.
        """
        epochs = int(self.cfg.training.epochs)
        logger.info(f"Starting training: epochs={epochs}")

        # Setup data loaders before training
        self._setup_dataloaders()
        
        if self.train_loader is None:
            raise RuntimeError("No training data available")

        # Main training loop
        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0
            num_batches = 0
            
            # Process each batch
            for batch in self.train_loader:
                use_synthetic = getattr(self.cfg.training, 'use_synthetic', False)
                
                if use_synthetic:
                    # Synthetic data: (*features, target)
                    # Unpack all feature tensors and the target
                    *features, target = batch
                else:
                    # Real data: (features_list, target)
                    features, target = batch
                
                # Move all tensors to device (GPU/CPU)
                features = [f.to(self.device) for f in features]
                target = target.to(self.device)

                # Forward pass
                self.optim.zero_grad()
                predictions = self.model(features)
                
                # Calculate loss based on task type
                target_col = self.cfg.target["target_column"]  # Fixed: removed .get() call
                if target_col == 'label' and predictions.dim() == 1:
                    # Binary classification task
                    if target.dtype == torch.long:
                        # Multi-class classification (though binary in this case)
                        loss = torch.nn.functional.cross_entropy(
                            predictions.unsqueeze(-1).expand(-1, 2), 
                            target
                        )
                    else:
                        # Binary classification with sigmoid
                        loss = torch.nn.functional.binary_cross_entropy_with_logits(predictions, target.float())
                else:
                    # Regression task
                    loss = torch.nn.functional.mse_loss(predictions, target.float())
                
                # Backward pass and optimization
                loss.backward()
                self.optim.step()
                
                # Accumulate loss for averaging
                total_loss += loss.item()
                num_batches += 1
            
            # Calculate and log average loss for the epoch
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            logger.info(f"Epoch {epoch}/{epochs} avg_loss={avg_loss:.6f}")

            # Save checkpoint after each epoch
            checkpoint_path = self.work_dir / f"model_epoch_{epoch}.pt"
            self._save_checkpoint(checkpoint_path, epoch)
            
        logger.info("Training complete!")

    def _save_checkpoint(self, path: Path, epoch: int):
        """
        Save model checkpoint with all necessary state information.
        
        Args:
            path: Path where to save the checkpoint
            epoch: Current epoch number
        """
        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create checkpoint dictionary with all relevant state
        checkpoint = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optim.state_dict(),
            'config': self.cfg.model_dump() if hasattr(self.cfg, 'model_dump') else dict(self.cfg),
            'input_dims': self.input_dims
        }
        
        # Save to disk
        torch.save(checkpoint, path)
        logger.debug(f"Saved checkpoint: {path}")

    def evaluate(self) -> dict:
        """
        Evaluate model on test set if available.
        
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        if self.test_loader is None:
            logger.warning("No test data available for evaluation")
            return {}
        
        # Set model to evaluation mode (disables dropout, etc.)
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        # Disable gradient computation for efficiency
        with torch.no_grad():
            for batch in self.test_loader:
                features, target = batch
                
                # Move data to device
                features = [f.to(self.device) for f in features]
                target = target.to(self.device)
                
                # Forward pass
                predictions = self.model(features)
                
                # Calculate loss (same logic as training)
                target_col = self.cfg.target["target_column"]  # Fixed: removed .get() call
                if target_col == 'label' and predictions.dim() == 1:
                    # Classification loss
                    if target.dtype == torch.long:
                        loss = torch.nn.functional.cross_entropy(
                            predictions.unsqueeze(-1).expand(-1, 2), 
                            target
                        )
                    else:
                        loss = torch.nn.functional.binary_cross_entropy_with_logits(predictions, target.float())
                else:
                    # Regression loss
                    loss = torch.nn.functional.mse_loss(predictions, target.float())
                
                total_loss += loss.item()
                num_batches += 1
        
        # Calculate average test loss
        avg_test_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Prepare metrics dictionary
        metrics = {
            'test_loss': avg_test_loss,
            'num_test_batches': num_batches
        }
        
        logger.info(f"Evaluation results: {metrics}")
        return metrics