"""Prediction CSV column contract for the evaluation pipeline.

The modeler produces predictions as DataFrames conforming to these schemas.
The evaluator's report.py consumes them. This file is the contract boundary.

Owner: evaluator subagent.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Required columns in prediction output CSV/DataFrame
# ---------------------------------------------------------------------------
PREDICTION_REQUIRED_COLUMNS: list[str] = [
    "player_id",
    "game_pk",
    "game_date",
    "player_type",       # "hitter" or "pitcher"
    "model_candidate",   # "m1_ridge", "m2_lgbm_quantile", "m3_cqr_aci"
    "predicted_mean",
]

PREDICTION_OPTIONAL_COLUMNS: list[str] = [
    "predicted_p10",
    "predicted_p90",
    "actual_points",     # Filled in by evaluator, not modeler
]

ALL_PREDICTION_COLUMNS: list[str] = PREDICTION_REQUIRED_COLUMNS + PREDICTION_OPTIONAL_COLUMNS


def validate_prediction_dataframe(df: "pd.DataFrame") -> list[str]:
    """Validate that a prediction DataFrame has the required columns.

    Parameters
    ----------
    df : pd.DataFrame
        Prediction output from a model.

    Returns
    -------
    list[str]
        List of validation errors. Empty if valid.
    """
    import pandas as pd  # noqa: F811

    errors: list[str] = []

    for col in PREDICTION_REQUIRED_COLUMNS:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")

    if "predicted_mean" in df.columns and df["predicted_mean"].isna().any():
        errors.append("predicted_mean contains null values")

    if "player_type" in df.columns:
        valid_types = {"hitter", "pitcher"}
        invalid = set(df["player_type"].unique()) - valid_types
        if invalid:
            errors.append(f"Invalid player_type values: {invalid}")

    return errors


# Type alias for import convenience
try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore[assignment]
