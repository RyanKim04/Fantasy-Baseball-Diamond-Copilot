"""Prediction CSV column contract for the evaluation pipeline.

The modeler produces predictions as DataFrames conforming to these schemas.
The evaluator's report.py consumes them. This file is the contract boundary.

Owner: evaluator subagent.
"""

from __future__ import annotations

import pandas as pd  # noqa: TC002 - used at runtime in validate_prediction_dataframe

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

VALID_PLAYER_TYPES: set[str] = {"hitter", "pitcher"}
VALID_MODEL_CANDIDATES: set[str] = {"m1_ridge", "m2_lgbm_quantile", "m3_cqr_aci"}


def validate_prediction_dataframe(df: pd.DataFrame) -> list[str]:
    """Validate that a prediction DataFrame has the required columns and types.

    Parameters
    ----------
    df : pd.DataFrame
        Prediction output from a model.

    Returns
    -------
    list[str]
        List of validation errors. Empty if valid.
    """
    errors: list[str] = []

    # Check required columns
    for col in PREDICTION_REQUIRED_COLUMNS:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")

    if errors:
        # If required columns are missing, skip further checks
        return errors

    # Check for null predicted_mean
    if df["predicted_mean"].isna().any():
        n_null = int(df["predicted_mean"].isna().sum())
        errors.append(f"predicted_mean contains {n_null} null values")

    # Check player_type values
    invalid_types = set(df["player_type"].unique()) - VALID_PLAYER_TYPES
    if invalid_types:
        errors.append(f"Invalid player_type values: {invalid_types}")

    # Check model_candidate values
    invalid_candidates = set(df["model_candidate"].unique()) - VALID_MODEL_CANDIDATES
    if invalid_candidates:
        errors.append(f"Invalid model_candidate values: {invalid_candidates}")

    # Check key uniqueness: (player_id, game_pk) should be unique per model_candidate
    for mc in df["model_candidate"].unique():
        mc_df = df[df["model_candidate"] == mc]
        dup_count = mc_df.duplicated(subset=["player_id", "game_pk"]).sum()
        if dup_count > 0:
            errors.append(
                f"Duplicate (player_id, game_pk) pairs in {mc}: {dup_count} duplicates"
            )

    # Check interval consistency if both p10 and p90 are present
    if "predicted_p10" in df.columns and "predicted_p90" in df.columns:
        both_present = df["predicted_p10"].notna() & df["predicted_p90"].notna()
        if both_present.any():
            p10_vals = df.loc[both_present, "predicted_p10"]
            p90_vals = df.loc[both_present, "predicted_p90"]
            inverted = p10_vals > p90_vals
            n_inverted = int(inverted.sum())
            if n_inverted > 0:
                errors.append(
                    f"predicted_p10 > predicted_p90 in {n_inverted} rows (inverted intervals)"
                )

    return errors
