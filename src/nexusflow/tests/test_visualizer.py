import json
from nexusflow.viz.visualizer import save_flow_summary

def test_save_flow_summary(tmp_path):
    summary = {"iter1": {"attn": [0.5, 0.5]}}
    path = tmp_path / "flow.json"
    save_flow_summary(path, summary)
    loaded = json.loads(path.read_text())
    assert loaded["iter1"]["attn"] == [0.5, 0.5]