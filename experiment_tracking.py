"""Experiment tracking utilities for MRI-KAN."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from packaging import version


class ExperimentTracker:
    """Lightweight wrapper around common experiment tracking frameworks.

    The tracker tries to use Weights & Biases or MLflow if available.  If neither
    is installed, tracking calls become no-ops which keeps the training loop
    functional in minimal environments such as unit tests.
    """

    def __init__(self, project: str = "mri-kan", run_name: Optional[str] = None) -> None:
        self.project = project
        self.run_name = run_name
        self.logger = logging.getLogger(__name__)
        self._backend: Optional[str] = None
        self._run: Any = None
        self._version = version.parse("0.1.0")
        self._init_backend()

    def _init_backend(self) -> None:
        try:
            import wandb

            self._backend = "wandb"
            self._run = wandb.init(project=self.project, name=self.run_name, reinit=True)
            self.logger.info("Using Weights & Biases for experiment tracking")
            return
        except Exception:  # noqa: BLE001 - optional dependency
            pass
        try:
            import mlflow

            self._backend = "mlflow"
            mlflow.set_experiment(self.project)
            self._run = mlflow
            self.logger.info("Using MLflow for experiment tracking")
            return
        except Exception:  # noqa: BLE001 - optional dependency
            pass
        self.logger.warning("No experiment tracking backend available; running in no-op mode")

    def log_params(self, params: Dict[str, Any]) -> None:
        if self._backend == "wandb":
            self._run.config.update(params, allow_val_change=True)
        elif self._backend == "mlflow":
            for k, v in params.items():
                self._run.log_param(k, v)

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        if self._backend == "wandb":
            self._run.log(metrics, step=step)
        elif self._backend == "mlflow":
            for k, v in metrics.items():
                self._run.log_metric(k, v, step=step)

    def log_artifact(self, path: Path) -> None:
        if self._backend == "wandb":
            self._run.save(str(path))
        elif self._backend == "mlflow":
            self._run.log_artifact(str(path))

    def update_version(self, improved: bool) -> str:
        """Update semantic version based on performance improvement."""
        if improved:
            self._version = version.Version(str(self._version.major) + "." + str(self._version.minor) + "." + str(self._version.micro + 1))
        return str(self._version)

    def finish(self) -> None:
        if self._backend == "wandb" and self._run is not None:
            self._run.finish()
        self.logger.info("Experiment tracking finished")
