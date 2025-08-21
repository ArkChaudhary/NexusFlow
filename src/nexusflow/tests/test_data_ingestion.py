import pandas as pd
from nexusflow.data.ingestion import load_table, validate_primary_key, split_df

def test_load_and_validate(tmp_path):
    df = pd.DataFrame({'id': [1,2,3], 'a': [0.1, 0.2, 0.3]})
    p = tmp_path / 't.csv'
    df.to_csv(p, index=False)
    loaded = load_table(str(p))
    assert 'id' in loaded.columns
    assert validate_primary_key(loaded, 'id') is True

def test_split_df(tmp_path):
    import numpy as np
    df = pd.DataFrame({'id': list(range(100)), 'x': list(range(100))})
    t, v, te = split_df(df, test_size=0.1, val_size=0.1, randomize=True)
    assert len(t) + len(v) + len(te) == 100
