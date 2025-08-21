"""Command-line interface for NexusFlow PoC."""
import typer
from loguru import logger
from .project_manager import ProjectManager
from .config import load_config_from_file
from .trainer.trainer import Trainer
from pathlib import Path

app = typer.Typer(help="NexusFlow CLI (PoC)")

@app.command()
def init(
    project_name: str = typer.Argument(..., help="Project name (directory to create)")
) -> None:
    pm = ProjectManager()
@app.command()
def train(
    config_path: Path = typer.Option(
        "configs/config.yaml",
        "--config",
        "-c",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to YAML configuration file.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", "-n", help="Validate data/model setup without training."
    ),
) -> None:
    try:
        cfg = load_config_from_file(str(config_path))
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        raise typer.Exit(code=2)
    except Exception:
        logger.exception("Failed to load configuration from: {}", config_path)
        raise typer.Exit(code=1)

    logger.info("Configuration loaded; starting Trainer")
    try:
        trainer = Trainer(cfg)
    except Exception:
        logger.exception("Failed to construct Trainer from configuration")
        raise typer.Exit(code=1)

    if dry_run:
        logger.info("Dry run requested — validating data and model init only.")
        try:
            trainer.sanity_check()
        except Exception:
            logger.exception("Sanity check failed")
            raise typer.Exit(code=3)
        return

    try:
        trainer.train()
    except Exception:
        logger.exception("Training failed")
        raise typer.Exit(code=4)
def train(config_path: str = "configs/config.yaml", dry_run: bool = False):
    cfg = load_config_from_file(config_path)
    logger.info("Configuration loaded; starting Trainer")
    trainer = Trainer(cfg)
    if dry_run:
        logger.info("Dry run requested — validating data and model init only.")
        trainer.sanity_check()
        return
    trainer.train()

if __name__ == '__main__':
    app()
