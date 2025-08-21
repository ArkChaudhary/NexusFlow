"""ProjectManager handles creation of the standardized project layout."""
from pathlib import Path
from loguru import logger

class ProjectManager:
    def __init__(self, base_dir: str = '.'):
        self.base_dir = Path(base_dir).resolve()

    def init_project(self, project_name: str, force: bool = False) -> str:
        # Guard against absolute paths and traversal outside base_dir
        if Path(project_name).is_absolute():
            logger.error("Absolute paths are not allowed for project_name: {}", project_name)
            raise ValueError(f"Absolute paths are not allowed for project_name: {project_name}")

        project_dir = self.base_dir / project_name
        resolved_dir = project_dir.resolve()
        try:
            # Ensure the resolved path is within base_dir
            resolved_dir.relative_to(self.base_dir)
        except ValueError:
            logger.error("project_name escapes base_dir: base_dir={}, resolved={}", self.base_dir, resolved_dir)
            raise ValueError(f"project_name escapes base_dir: {resolved_dir}")

        if resolved_dir.exists() and not force:
            logger.error("Project directory already exists: {}", resolved_dir)
            raise FileExistsError(f"Project directory already exists: {resolved_dir}")

        for d in ['configs', 'datasets', 'models', 'notebooks', 'results', 'src']:
            dir_path = resolved_dir / d
            created = not dir_path.exists()
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.debug("Directory {}: {}", "created" if created else "already existed", dir_path)

        logger.info("Project initialized at: {}", resolved_dir)
        return str(resolved_dir)
