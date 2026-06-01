"""Evaluation report generator.

Produces evaluation_report.md per validation_protocol.md section 7.4.
The single Test touch runs through this module.

Owner: evaluator subagent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

    from packages.shared.schemas.ml import ModelCandidate, PlayerType


def generate_report(
    predictions: dict[ModelCandidate, pd.DataFrame],
    actuals: pd.DataFrame,
    output_dir: Path,
    mlflow_run_ids: dict[ModelCandidate, str] | None = None,
) -> Path:
    """Generate the full evaluation report for all model candidates.

    This is the single entry point for the Test touch. It:
    1. Joins predictions with actuals.
    2. Filters to eligible populations (>=100 PA hitters, >=30 IP pitchers).
    3. Computes headline metrics for each candidate x population.
    4. Computes baselines on the same population.
    5. Applies the baseline comparison threshold (>=10% RMSE improvement).
    6. Generates calibration diagnostics (for M2 and M3).
    7. Generates slice tables.
    8. Writes evaluation_report.md and saves diagnostic plots.
    9. Applies the selection rule from model_bakeoff.md section 6.

    Parameters
    ----------
    predictions : dict[ModelCandidate, pd.DataFrame]
        Prediction DataFrames per model candidate.
        Must conform to schemas.PREDICTION_REQUIRED_COLUMNS.
    actuals : pd.DataFrame
        Actual fantasy points keyed on (player_id, game_pk).
    output_dir : Path
        Directory to write evaluation_report.md and plots.
    mlflow_run_ids : dict[ModelCandidate, str] | None
        MLflow run IDs for traceability.

    Returns
    -------
    Path
        Path to the generated evaluation_report.md.
    """
    raise NotImplementedError("To be implemented by evaluator subagent")
