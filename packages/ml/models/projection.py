"""Unified prediction API for the player projection model.

This module provides the public predict() interface that Phase 2+ services
consume. It loads a trained model from MLflow and returns predictions
conforming to the PlayerProjection schema.

Owner: modeler subagent.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from packages.shared.schemas.ml import PlayerProjection


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
    """
    raise NotImplementedError("To be implemented by modeler subagent")
