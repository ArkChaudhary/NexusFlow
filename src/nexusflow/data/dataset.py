from torch.utils.data import Dataset
import pandas as pd
import torch

class NexusFlowDataset(Dataset):
    """
    Wraps a pandas DataFrame for use in PyTorch training loops.
    - Fails fast if target column is missing.
    - Automatically infers target dtype:
        * torch.long if target column is integer (classification).
        * torch.float32 otherwise (regression).
    """

    def __init__(self, df: pd.DataFrame, target_col: str = 'label'):
        self.df = df.copy().reset_index(drop=True)
        self.target_col = target_col

        if self.target_col not in self.df.columns:
            raise KeyError(f"Target column '{self.target_col}' missing from DataFrame")

        # Separate feature columns (exclude primary key + target if present)
        self.feature_cols = [c for c in self.df.columns if c != self.target_col]

        # Simple fill for NaNs
        self.df = self.df.fillna(0)

        # Decide target dtype once (classification vs regression)
        if pd.api.types.is_integer_dtype(self.df[self.target_col]):
            self.target_dtype = torch.long
        else:
            self.target_dtype = torch.float32

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        features = torch.tensor(row[self.feature_cols].values.astype('float32'))
        target = torch.tensor(row[self.target_col], dtype=self.target_dtype)
        return features, target
