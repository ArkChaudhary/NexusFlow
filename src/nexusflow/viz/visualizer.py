from loguru import logger
from pathlib import Path
import json

def save_flow_summary(path: str, summary: dict):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved flow summary to: {p}")
