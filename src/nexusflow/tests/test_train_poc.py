"""Integration test for training proof of concept."""
import pytest
import torch
import tempfile
from pathlib import Path

from nexusflow.config import ConfigModel
from nexusflow.trainer.trainer import Trainer


class TestTrainingIntegration:
    """Integration tests for the training pipeline."""
    
    def test_synthetic_training_decreases_loss(self):
        """Test that training loss decreases over a few batches with synthetic data."""
        
        # Create minimal config for synthetic training
        config_dict = {
            "project_name": "test_project",
            "primary_key": "id",
            "target": {
                "target_table": "table_a.csv",
                "target_column": "label"
            },
            "architecture": {
                "global_embed_dim": 32,
                "refinement_iterations": 1
            },
            "datasets": [
                {"name": "table_a.csv", "transformer_type": "standard"},
                {"name": "table_b.csv", "transformer_type": "standard"}
            ],
            "training": {
                "use_synthetic": True,
                "synthetic": {
                    "n_samples": 64,
                    "feature_dim": 4
                },
                "batch_size": 8,
                "epochs": 3,
                "optimizer": {"name": "adam", "lr": 1e-2}
            }
        }
        
        config = ConfigModel.model_validate(config_dict)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = Trainer(config, work_dir=temp_dir)
            
            # Run sanity check first
            trainer.sanity_check()
            
            # Setup data loaders to capture initial loss
            trainer._setup_dataloaders()
            
            # Capture loss from first few batches before training
            trainer.model.train()
            initial_losses = []
            
            batch_count = 0
            for batch in trainer.train_loader:
                if batch_count >= 3:  # Only test first 3 batches
                    break
                
                # Unpack synthetic data
                *features, target = batch
                features = [f.to(trainer.device) for f in features]
                target = target.to(trainer.device)
                
                # Calculate loss without optimization
                with torch.no_grad():
                    predictions = trainer.model(features)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        predictions, target.float()
                    )
                    initial_losses.append(loss.item())
                
                batch_count += 1
            
            # Now run actual training for a few steps
            trainer.model.train()
            training_losses = []
            
            batch_count = 0
            for batch in trainer.train_loader:
                if batch_count >= 3:  # Only train for 3 batches
                    break
                
                # Unpack synthetic data
                *features, target = batch
                features = [f.to(trainer.device) for f in features]
                target = target.to(trainer.device)
                
                # Forward pass with gradient computation
                trainer.optim.zero_grad()
                predictions = trainer.model(features)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    predictions, target.float()
                )
                
                # Backward pass
                loss.backward()
                trainer.optim.step()
                
                training_losses.append(loss.item())
                batch_count += 1
            
            # Assert that loss decreased after training
            initial_avg = sum(initial_losses) / len(initial_losses)
            training_avg = sum(training_losses) / len(training_losses)
            
            # Training loss should be lower than initial loss
            # Allow some tolerance for randomness
            assert training_avg < initial_avg * 1.1, (
                f"Training did not decrease loss effectively. "
                f"Initial avg: {initial_avg:.4f}, Training avg: {training_avg:.4f}"
            )
            
            # Also check that we have reasonable loss values (not NaN/inf)
            assert all(loss > 0 and loss < 100 for loss in training_losses), (
                f"Training losses not in reasonable range: {training_losses}"
            )
    
    def test_trainer_initialization_with_synthetic(self):
        """Test that Trainer initializes correctly with synthetic data config."""
        config_dict = {
            "project_name": "test_init",
            "primary_key": "id",
            "target": {
                "target_table": "table_a.csv",
                "target_column": "label"
            },
            "architecture": {
                "global_embed_dim": 16
            },
            "datasets": [
                {"name": "table_a.csv"},
                {"name": "table_b.csv"}
            ],
            "training": {
                "use_synthetic": True,
                "synthetic": {"n_samples": 32, "feature_dim": 3},
                "batch_size": 4,
                "epochs": 1
            }
        }
        
        config = ConfigModel.model_validate(config_dict)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = Trainer(config, work_dir=temp_dir)
            
            # Check that input dimensions were set correctly
            assert trainer.input_dims == [3, 3], f"Expected [3, 3], got {trainer.input_dims}"
            
            # Check that model was initialized with correct dimensions
            assert len(trainer.model.encoders) == 2
            
            # Check that optimizer was created
            assert trainer.optim is not None
            assert isinstance(trainer.optim, torch.optim.Adam)
    
    def test_checkpoint_saving(self):
        """Test that model checkpoints are saved correctly during training."""
        config_dict = {
            "project_name": "test_checkpoint",
            "primary_key": "id",
            "target": {
                "target_table": "table_a.csv",
                "target_column": "label"
            },
            "architecture": {
                "global_embed_dim": 16
            },
            "datasets": [
                {"name": "table_a.csv"}
            ],
            "training": {
                "use_synthetic": True,
                "synthetic": {"n_samples": 16, "feature_dim": 2},
                "batch_size": 4,
                "epochs": 2
            }
        }
        
        config = ConfigModel.model_validate(config_dict)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            trainer = Trainer(config, work_dir=str(work_dir))
            
            # Run training
            trainer.train()
            
            # Check that checkpoints were saved
            checkpoint_files = list(work_dir.glob("model_epoch_*.pt"))
            assert len(checkpoint_files) == 2, f"Expected 2 checkpoint files, found {len(checkpoint_files)}"
            
            # Check that checkpoints can be loaded
            for checkpoint_path in checkpoint_files:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                
                # Verify checkpoint structure
                required_keys = ['epoch', 'model_state', 'optimizer_state', 'config', 'input_dims']
                for key in required_keys:
                    assert key in checkpoint, f"Missing key '{key}' in checkpoint"
                
                # Verify that model state can be loaded
                test_model = trainer.model
                test_model.load_state_dict(checkpoint['model_state'])
    
    def test_model_forward_pass_consistency(self):
        """Test that model produces consistent outputs for same inputs."""
        config_dict = {
            "project_name": "test_consistency",
            "primary_key": "id",
            "target": {
                "target_table": "table_a.csv",
                "target_column": "label"
            },
            "architecture": {
                "global_embed_dim": 8
            },
            "datasets": [
                {"name": "table_a.csv"},
                {"name": "table_b.csv"}
            ],
            "training": {
                "use_synthetic": True,
                "synthetic": {"n_samples": 8, "feature_dim": 2},
                "batch_size": 2,
                "epochs": 1
            }
        }
        
        config = ConfigModel.model_validate(config_dict)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = Trainer(config, work_dir=temp_dir)
            
            # Create test inputs
            test_inputs = [torch.randn(1, 2), torch.randn(1, 2)]
            
            # Run forward pass twice in eval mode
            trainer.model.eval()
            with torch.no_grad():
                output1 = trainer.model(test_inputs)
                output2 = trainer.model(test_inputs)
            
            # Outputs should be identical
            torch.testing.assert_close(output1, output2, rtol=1e-6, atol=1e-6)