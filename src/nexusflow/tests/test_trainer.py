import torch
from nexusflow.trainer.trainer import Trainer
from nexusflow.config import ConfigModel

def make_dummy_config():
    return ConfigModel(
        project_name="demo",
        primary_key="id",
        target={"target_table": "table_a.csv", "target_column": "label"},
        architecture={"global_embed_dim": 8, "refinement_iterations": 1},
        datasets=[{"name": "table_a.csv"}, {"name": "table_b.csv"}],
        training={"batch_size": 4, "epochs": 1, "optimizer": {"lr": 1e-3}},
        mlops={"logging_provider": "stdout"}
    )

def test_trainer_sanity_check_runs(tmp_path):
    cfg = make_dummy_config()
    trainer = Trainer(cfg, work_dir=tmp_path)
    trainer.sanity_check()  # should not raise

def test_trainer_trains_and_saves_checkpoint(tmp_path):
    cfg = make_dummy_config()
    trainer = Trainer(cfg, work_dir=tmp_path)
    trainer.train()
    files = list(tmp_path.glob("model_epoch_*.pt"))
    assert len(files) >= 1