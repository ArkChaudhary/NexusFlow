from nexusflow.project_manager import ProjectManager
import pytest
import os

def test_init_project_creates_folders(tmp_path):
    pm = ProjectManager(base_dir=tmp_path)
    project_path = pm.init_project("demo_proj")
    expected_dirs = ["configs", "datasets", "models", "notebooks", "results", "src"]
    for d in expected_dirs:
        assert (tmp_path / "demo_proj" / d).exists()

def test_init_project_already_exists(tmp_path):
    pm = ProjectManager(base_dir=tmp_path)
    pm.init_project("demo_proj")
    with pytest.raises(FileExistsError):
        pm.init_project("demo_proj")
