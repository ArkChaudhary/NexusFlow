import torch
from torch.utils.data import DataLoader
from loguru import logger
from pathlib import Path

from nexusflow.config import ConfigModel
from nexusflow.model.nexus_former import NexusFormer
from nexusflow.data.dataset import NexusFlowDataset
from nexusflow.data.ingestion import load_table, validate_primary_key

class Trainer:
    def __init__(self, config: ConfigModel, work_dir: str = '.'):
        self.cfg = config
        self.work_dir = Path(work_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Trainer initialized (device={self.device})")

        input_dims = []
        for dcfg in self.cfg.datasets:
            df = load_table(f"datasets/{dcfg.name}")
            validate_primary_key(df, self.cfg.primary_key)
            num_features = len([c for c in df.columns if c not in [self.cfg.primary_key, self.cfg.target['target_column']]])
            input_dims.append(num_features)
        self.model = NexusFormer(input_dims, embed_dim=self.cfg.architecture.get('global_embed_dim', 64)).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.cfg.training.optimizer.get('lr', 1e-3))

    def sanity_check(self):
        logger.info("Running trainer sanity checks...")
        B = 2
        dummy_inputs = [torch.randn(B, 5).to(self.device) for _ in self.model.encoders]
        out = self.model(dummy_inputs)
        logger.info(f"Sanity check forward pass output shape: {out.shape}")

    def train(self):
        epochs = int(self.cfg.training.epochs)
        batch_size = int(self.cfg.training.batch_size)
        logger.info(f"Starting training: epochs={epochs} batch_size={batch_size}")

        if self.cfg.training.get('use_synthetic', True):
            # --- Synthetic dataset path (for fast debugging) ---
            n = self.cfg.training.get('synthetic', {}).get('n_samples', 256)
            fdim = self.cfg.training.get('synthetic', {}).get('feature_dim', 5)
            Xs = [torch.randn(n, fdim) for _ in self.model.encoders]
            y = torch.randn(n)
            ds = torch.utils.data.TensorDataset(*Xs, y)

        else:
            # --- Real dataset path ---
            from ..data.ingestion import split_df
            import pandas as pd

            # load all configured tables
            dfs = []
            for dcfg in self.cfg.datasets:
                df = load_table(f"datasets/{dcfg.name}")
                validate_primary_key(df, self.cfg.primary_key)
                dfs.append(df)

            # merge on primary_key so we align rows across tables
            merged = dfs[0]
            for df in dfs[1:]:
                merged = pd.merge(merged, df, on=self.cfg.primary_key, how="inner")

            if self.cfg.target['target_column'] not in merged.columns:
                raise KeyError(f"Target column '{self.cfg.target['target_column']}' missing in merged dataset")

            # split into train/val/test
            train_df, _, _ = split_df(
                merged,
                test_size=self.cfg.training.split_config.get("test_size", 0.15),
                val_size=self.cfg.training.split_config.get("validation_size", 0.15),
                randomize=self.cfg.training.split_config.get("randomize", True),
            )

            # Wrap with dataset
            ds = NexusFlowDataset(train_df, target_col=self.cfg.target['target_column'])

        # --- Common training loop ---
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0
            for batch in loader:
                if self.cfg.training.get('use_synthetic', True):
                    *features, target = batch
                else:
                    features, target = batch
                    features = [features]  # wrap single tensor into list

                features = [f.to(self.device) for f in features]
                target = target.to(self.device)

                self.optim.zero_grad()
                preds = self.model(features)
                loss = torch.nn.functional.mse_loss(preds, target)
                loss.backward()
                self.optim.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(loader)
            logger.info(f"Epoch {epoch}/{epochs} avg_loss={avg_loss:.6f}")

            ckpt = self.work_dir / f"model_epoch_{epoch}.pt"
            torch.save({'epoch': epoch, 'model_state': self.model.state_dict()}, ckpt)
            logger.debug(f"Saved checkpoint: {ckpt}")
        logger.info("Training complete.")

