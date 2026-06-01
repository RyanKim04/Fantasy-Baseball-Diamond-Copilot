"""Unified prediction API for the player projection model.

This module provides the public predict() interface that Phase 2+ services
consume. It loads a trained model from MLflow and returns predictions
conforming to the PlayerProjection schema.

Owner: modeler subagent.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from packages.shared.schemas.ml import PlayerProjection

logger = logging.getLogger(__name__)

# Cache for loaded models to avoid repeated MLflow fetches
_model_cache: dict[str, Any] = {}


def _load_model_from_mlflow(model_version: str = "Production") -> Any:
    """Load a model from the MLflow registry.

    Parameters
    ----------
    model_version : str
        MLflow model stage ("Production", "Staging") or a specific
        version number as a string.

    Returns
    -------
    Any
        The loaded model object (ProjectionModel implementation).

    Raises
    ------
    mlflow.exceptions.MlflowException
        If the model is not found in the registry.
    """
    cache_key = f"model_{model_version}"
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    import mlflow  # lazy import — not available in unit-test env

    model_uri = f"models:/player_projection/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri)

    _model_cache[cache_key] = model
    logger.info("Loaded model from MLflow: %s", model_uri)
    return model


def _find_next_game(player_id: int, target_date: date) -> tuple[date, int | None]:
    """Find the next scheduled game for a player on or after target_date.

    This is a placeholder that will integrate with the schedule data
    in Phase 2. For now, it returns target_date with no game_pk.

    Parameters
    ----------
    player_id : int
        MLBAM player ID.
    target_date : date
        Date on or after which to find the next game.

    Returns
    -------
    tuple[date, int | None]
        (game_date, game_pk). game_pk may be None if not yet known.

    Raises
    ------
    ValueError
        If no upcoming game is found.
    """
    # TODO: Phase 2 integration with games schedule table
    # For now, assume the target_date itself is a game day
    return target_date, None


def _assemble_features_for_player(
    player_id: int,
    game_date: date,
) -> pd.DataFrame:
    """Assemble features for a single player on a specific date.

    This calls the feature pipeline for a single (player, date) pair.
    It is the inference-time feature assembly path.

    Parameters
    ----------
    player_id : int
        MLBAM player ID.
    game_date : date
        Date of the game.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with feature columns.
    """
    # TODO: Integrate with feature pipeline (packages/ml/features/pipeline.py)
    # For now, raise NotImplementedError -- this will be wired in Phase 2
    # when the feature pipeline supports single-player inference.
    msg = (
        "Single-player feature assembly not yet implemented. "
        "This requires integration with the feature pipeline in Phase 2. "
        "For batch predictions, use the training pipeline directly."
    )
    raise NotImplementedError(msg)


def predict(
    player_id: int,
    target_date: date,
    model_version: str = "Production",
) -> PlayerProjection:
    """Predict fantasy points for a player on a given date.

    Loads the production model from MLflow, builds features for the player's
    next game on or after target_date, and returns the projection.

    Parameters
    ----------
    player_id : int
        MLBAM player ID.
    target_date : date
        Date on or after which to find the next scheduled game.
    model_version : str
        MLflow model stage or version. Default "Production".

    Returns
    -------
    PlayerProjection
        Contains mean, p10, p90 fantasy points plus metadata.

    Raises
    ------
    ValueError
        If no upcoming game is found for the player.
    NotImplementedError
        If single-player feature assembly is not yet wired (pre-Phase 2).
    """
    # Find the next game
    game_date, game_pk = _find_next_game(player_id, target_date)

    # Assemble features
    X = _assemble_features_for_player(player_id, game_date)

    # Load model
    model = _load_model_from_mlflow(model_version)

    # Predict
    mean_pred = float(model.predict(X)[0])

    # Get quantile predictions if available
    p10 = mean_pred  # fallback
    p90 = mean_pred  # fallback

    if hasattr(model, "predict_quantiles"):
        quantiles = model.predict_quantiles(X)
        if quantiles.size > 0:
            p10 = float(quantiles[0, 0])
            p90 = float(quantiles[0, 1])

    return PlayerProjection(
        player_id=player_id,
        game_date=game_date,
        game_pk=game_pk,
        mean=mean_pred,
        p10=p10,
        p90=p90,
        model_version=model_version,
    )


def predict_batch(
    X: pd.DataFrame,
    model: Any,
    model_candidate_name: str,
    player_type_name: str,
) -> pd.DataFrame:
    """Produce predictions for a batch of rows in the evaluation schema format.

    This is the bridge between the model's predict methods and the evaluator's
    expected DataFrame format (per packages/ml/evaluation/schemas.py).

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (rows to predict). Must also contain metadata columns
        (player_id, game_pk, game_date).
    model : Any
        A fitted ProjectionModel implementation.
    model_candidate_name : str
        Value from ModelCandidate enum (e.g., "m1_ridge").
    player_type_name : str
        Value from PlayerType enum (e.g., "hitter").

    Returns
    -------
    pd.DataFrame
        Prediction DataFrame conforming to evaluation schemas.
    """
    from packages.shared.schemas.ml import NON_FEATURE_COLUMNS

    # Extract feature-only columns for prediction
    feature_cols = [c for c in X.columns if c not in NON_FEATURE_COLUMNS]
    X_features = X[feature_cols]

    # Point predictions
    mean_preds = model.predict(X_features)

    result = pd.DataFrame({
        "player_id": X["player_id"].values,
        "game_pk": X["game_pk"].values,
        "game_date": X["game_date"].values,
        "player_type": player_type_name,
        "model_candidate": model_candidate_name,
        "predicted_mean": mean_preds,
    })

    # Quantile predictions (if model produces intervals)
    if model.produces_intervals:
        quantiles = model.predict_quantiles(X_features)
        if quantiles.size > 0:
            result["predicted_p10"] = quantiles[:, 0]
            result["predicted_p90"] = quantiles[:, 1]
    else:
        result["predicted_p10"] = np.nan
        result["predicted_p90"] = np.nan

    return result
