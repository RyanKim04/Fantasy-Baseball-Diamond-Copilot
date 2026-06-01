"""Evaluation module for player projection.

This module owns all model evaluation: metrics computation, baseline
comparisons, calibration diagnostics, and report generation.

The evaluation code is written and committed BEFORE any modeling code,
per the critical workflow rule in CLAUDE.md. The validation protocol
is pre-registered in validation_protocol.md.

The evaluator owns the single Test touch (validation_protocol.md section 11).
No other module may read Test-set targets.
"""

from packages.ml.evaluation.baselines import (
    predict_naive_last_game,
    predict_season_to_date_mean,
    predict_trailing_7d_mean,
)
from packages.ml.evaluation.metrics import (
    compute_population_metrics,
    interval_coverage,
    mae,
    pinball_loss,
    rmse,
    sharpness,
    spearman_rho,
)
from packages.ml.evaluation.report import generate_report
from packages.ml.evaluation.schemas import validate_prediction_dataframe

__all__ = [
    "compute_population_metrics",
    "generate_report",
    "interval_coverage",
    "mae",
    "pinball_loss",
    "predict_naive_last_game",
    "predict_season_to_date_mean",
    "predict_trailing_7d_mean",
    "rmse",
    "sharpness",
    "spearman_rho",
    "validate_prediction_dataframe",
]
