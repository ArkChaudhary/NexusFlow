"""Enhanced command-line interface for NexusFlow with real data support."""
import typer
from loguru import logger
from pathlib import Path
import sys

from nexusflow.project_manager import ProjectManager
from nexusflow.config import load_config_from_file
from nexusflow.trainer.trainer import Trainer

app = typer.Typer(help="NexusFlow CLI - Multi-Transformer Framework for Tabular Data")

@app.command()
def init(
    project_name: str = typer.Argument(..., help="Project name (directory to create)"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing project directory")
) -> None:
    """Initialize a new NexusFlow project with standard directory structure."""
    try:
        pm = ProjectManager()
        project_path = pm.init_project(project_name, force=force)
        
        typer.echo(f"✅ Project initialized at: {project_path}")
        typer.echo("\nNext steps:")
        typer.echo("1. Place your CSV files in the datasets/ directory")
        typer.echo("2. Edit configs/config.yaml to match your data")
        typer.echo("3. Run: nexusflow train --config configs/config.yaml")
        
    except FileExistsError:
        typer.echo(f"❌ Project directory '{project_name}' already exists. Use --force to overwrite.", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"❌ Failed to initialize project: {e}", err=True)
        raise typer.Exit(code=1)

@app.command()
def train(
    config_path: Path = typer.Option(
        "configs/config.yaml",
        "--config", "-c",
        help="Path to YAML configuration file"
    ),
    dry_run: bool = typer.Option(
        False, 
        "--dry-run", "-n", 
        help="Validate data/model setup without training"
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level", "-l",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )
) -> None:
    """Train a NexusFlow model using the specified configuration."""
    
    # Setup logging
    logger.remove()  # Remove default handler
    logger.add(
        sys.stdout, 
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # Also log to file if results directory exists
    results_dir = Path("results")
    if results_dir.exists():
        log_file = results_dir / "logs" / "train.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="10 MB"
        )
        logger.info(f"Logging to file: {log_file}")

    # Validate config file exists
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        typer.echo(f"❌ Configuration file not found: {config_path}", err=True)
        typer.echo("💡 Tip: Run 'nexusflow init <project_name>' to create a project structure", err=True)
        raise typer.Exit(code=2)

    # Load configuration
    try:
        config = load_config_from_file(str(config_path))
        logger.info(f"Successfully loaded configuration: {config.project_name}")
    except Exception as e:
        logger.exception("Failed to load configuration")
        typer.echo(f"❌ Failed to load configuration: {e}", err=True)
        raise typer.Exit(code=1)

    # Initialize trainer
    try:
        work_dir = Path(".")
        trainer = Trainer(config, work_dir=str(work_dir))
        logger.info("Trainer initialized successfully")
    except Exception as e:
        logger.exception("Failed to initialize trainer")
        typer.echo(f"❌ Failed to initialize trainer: {e}", err=True)
        typer.echo("💡 Check that your datasets exist and have the correct primary key", err=True)
        raise typer.Exit(code=1)

    # Run dry run or full training
    if dry_run:
        logger.info("Running dry run - validating setup only")
        typer.echo("🔍 Running dry run validation...")
        
        try:
            trainer.sanity_check()
            typer.echo("✅ Dry run passed - configuration and data are valid")
            typer.echo(f"📊 Model will use input dimensions: {trainer.input_dims}")
            if not config.training.get('use_synthetic', False):
                total_samples = len(list(trainer.datasets.values())[0]) if trainer.datasets else 0
                typer.echo(f"📈 Training data: {total_samples} aligned samples across {len(config.datasets)} datasets")
        except Exception as e:
            logger.exception("Dry run validation failed")
            typer.echo(f"❌ Validation failed: {e}", err=True)
            raise typer.Exit(code=3)
    else:
        logger.info("Starting full training")
        typer.echo("🚀 Starting training...")
        
        try:
            # Run sanity check first
            trainer.sanity_check()
            
            # Run training
            trainer.train()
            
            typer.echo("✅ Training completed successfully!")
            
            # Show where model was saved
            model_files = list(Path(".").glob("model_epoch_*.pt"))
            if model_files:
                latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
                typer.echo(f"💾 Model saved to: {latest_model}")
            
            # Run evaluation if test data available
            if not config.training.get('use_synthetic', False):
                logger.info("Running evaluation on test set")
                metrics = trainer.evaluate()
                if metrics:
                    typer.echo("📊 Evaluation Results:")
                    for metric, value in metrics.items():
                        typer.echo(f"   {metric}: {value:.4f}")
                        
        except Exception as e:
            logger.exception("Training failed")
            typer.echo(f"❌ Training failed: {e}", err=True)
            raise typer.Exit(code=4)

@app.command()
def validate(
    config_path: Path = typer.Option(
        "configs/config.yaml",
        "--config", "-c",
        help="Path to YAML configuration file"
    )
) -> None:
    """Validate configuration file and check data availability."""
    
    if not config_path.exists():
        typer.echo(f"❌ Configuration file not found: {config_path}", err=True)
        raise typer.Exit(code=1)

    try:
        # Load and validate config
        config = load_config_from_file(str(config_path))
        typer.echo(f"✅ Configuration is valid")
        typer.echo(f"   Project: {config.project_name}")
        typer.echo(f"   Primary key: {config.primary_key}")
        typer.echo(f"   Target: {config.target['target_table']}:{config.target['target_column']}")
        typer.echo(f"   Datasets: {len(config.datasets)}")
        
        # Check if using synthetic data
        if config.training.get('use_synthetic', False):
            typer.echo("ℹ️  Using synthetic data - no CSV validation needed")
            return
        
        # Check dataset files exist
        missing_files = []
        for dataset in config.datasets:
            dataset_path = Path("datasets") / dataset.name
            if not dataset_path.exists():
                missing_files.append(str(dataset_path))
        
        if missing_files:
            typer.echo("❌ Missing dataset files:")
            for file in missing_files:
                typer.echo(f"   {file}")
            raise typer.Exit(code=2)
        else:
            typer.echo("✅ All dataset files found")
            
    except Exception as e:
        typer.echo(f"❌ Validation failed: {e}", err=True)
        raise typer.Exit(code=1)

@app.command()
def info():
    """Show information about NexusFlow."""
    typer.echo("🔗 NexusFlow - Multi-Transformer Framework for Tabular Data")
    typer.echo("")
    typer.echo("A collaborative intelligence framework inspired by AlphaFold 2's Evoformer")
    typer.echo("architecture, designed for learning from heterogeneous tabular datasets.")
    typer.echo("")
    typer.echo("Key Features:")
    typer.echo("• Multi-table learning without flattening")
    typer.echo("• Iterative cross-contextual attention")  
    typer.echo("• Native support for heterogeneous data modalities")
    typer.echo("• Built-in visualization and interpretability")
    typer.echo("• Developer-friendly SDK with MLOps integration")

if __name__ == '__main__':
    app()