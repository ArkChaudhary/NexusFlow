import pandas as pd
from nexusflow.data.dataset import NexusFlowDataset

def test_dataset_length_and_getitem():
    df = pd.DataFrame({
        "f1": [1.0, 2.0],
        "f2": [3.0, 4.0],
        "label": [0, 1]
    })
    ds = NexusFlowDataset(df, target_col="label")
    assert len(ds) == 2
    features, target = ds[0]
    assert features.shape[0] == 2  # 2 features
    assert target.item() in [0, 1]