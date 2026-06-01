"""Training orchestrator with MLflow logging.

Runs the full training pipeline: load features, split, tune hyperparameters
via walk-forward CV, train final models, log to MLflow.

Owner: modeler subagent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from packages.shared.schemas.ml import ModelCandidate, PlayerType, ScoringRulesSnapshot


def train_model(
    feature_dir: Path,
    model_candidate: ModelCandidate,
    player_type: PlayerType,
    scoring_rules: ScoringRulesSnapshot,
    mlflow_experiment_name: str = "phase1-bakeoff",
    n_optuna_trials: int = 50,
) -> str:
    """Train a single model candidate for a single population.

    Parameters
    ----------
    feature_dir : Path
        Directory containing features_hitters.parquet / features_pitchers.parquet.
    model_candidate : ModelCandidate
        Which of M1/M2/M3 to train.
    player_type : PlayerType
        HITTER or PITCHER.
    scoring_rules : ScoringRulesSnapshot
        Frozen scoring rules to log with the model artifact.
    mlflow_experiment_name : str
        MLflow experiment name for grouping runs.
    n_optuna_trials : int
        Number of Optuna trials for hyperparameter search.

    Returns
    -------
    str
        MLflow run ID of the logged training run.
    """
    raise NotImplementedError("To be implemented by modeler subagent")
