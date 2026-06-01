"""Pydantic v2 schemas for ML cross-module types.

These schemas define the contracts between feature-engineer, modeler, and evaluator.
All cross-boundary data must conform to these schemas.
"""

from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class PlayerType(str, Enum):
    """Whether a model/feature set is for hitters or pitchers."""

    HITTER = "hitter"
    PITCHER = "pitcher"


class ModelCandidate(str, Enum):
    """The three candidates from the model bake-off plan."""

    M1_RIDGE = "m1_ridge"
    M2_LGBM_QUANTILE = "m2_lgbm_quantile"
    M3_CQR_ACI = "m3_cqr_aci"


class SplitName(str, Enum):
    """Temporal data split names."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    LIVE = "live"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NON_FEATURE_COLUMNS: set[str] = {
    "player_id",
    "game_pk",
    "game_date",
    "season_year",
    "player_type",
    "fantasy_points",
}


# ---------------------------------------------------------------------------
# Feature pipeline output schema
# ---------------------------------------------------------------------------
class FeatureRow(BaseModel):
    """Schema for a single row in the assembled feature table.

    Every feature builder must produce columns that are mergeable into this
    shape. The actual feature columns vary (they are dynamic), but these
    index/metadata columns are fixed.
    """

    player_id: int
    game_pk: int
    game_date: date
    season_year: int
    player_type: PlayerType
    fantasy_points: float | None = Field(
        default=None,
        description="Target variable: per-game fantasy points. None for inference rows.",
    )


# ---------------------------------------------------------------------------
# Prediction output schema (contract between modeler and evaluator)
# ---------------------------------------------------------------------------
class PredictionRow(BaseModel):
    """Schema for a single prediction output row.

    This is the contract that the modeler's predict() must produce and that
    the evaluator's report.py consumes. Keyed on (player_id, game_pk).
    """

    player_id: int
    game_pk: int
    game_date: date
    player_type: PlayerType
    model_candidate: ModelCandidate
    predicted_mean: float
    predicted_p10: float | None = None
    predicted_p90: float | None = None
    actual_points: float | None = Field(
        default=None,
        description="Filled in by evaluator at evaluation time, not by modeler.",
    )


# ---------------------------------------------------------------------------
# Single-player prediction API response
# ---------------------------------------------------------------------------
class PlayerProjection(BaseModel):
    """Response schema for the predict() API.

    Input: player_id + date.
    Output: mean, p10, p90 fantasy points for the next game on or after that date.
    """

    player_id: int
    game_date: date
    game_pk: int | None = None
    mean: float
    p10: float
    p90: float
    model_version: str


# ---------------------------------------------------------------------------
# Walk-forward CV fold definition
# ---------------------------------------------------------------------------
class WalkForwardFold(BaseModel):
    """Definition of a single walk-forward CV fold."""

    fold_number: int = Field(ge=1, le=5)
    train_start: date
    train_end: date
    val_start: date
    val_end: date


# ---------------------------------------------------------------------------
# Scoring rules snapshot
# ---------------------------------------------------------------------------
class ScoringRuleEntry(BaseModel):
    """A single stat category and its fantasy point value."""

    stat_category: str
    points_value: float
    is_negative: bool = False


class ScoringRulesSnapshot(BaseModel):
    """Frozen scoring rules used for a training run.

    Stored alongside the model artifact in MLflow. Both train and test targets
    must be computed using the same snapshot (validation_protocol.md section 10.7).
    """

    league_id: int
    rules: list[ScoringRuleEntry]
    snapshot_date: date
    source: str = "yahoo"


# ---------------------------------------------------------------------------
# Evaluation summary (evaluator -> architect)
# ---------------------------------------------------------------------------
class PopulationMetrics(BaseModel):
    """Headline metrics for one population (hitters or pitchers)."""

    player_type: PlayerType
    model_candidate: ModelCandidate
    n_predictions: int
    rmse: float
    mae: float
    spearman_rho: float
    coverage_80: float | None = None
    sharpness_mean_width: float | None = None
    pinball_10: float | None = None
    pinball_50: float | None = None
    pinball_90: float | None = None


class BaselineComparison(BaseModel):
    """RMSE comparison of a model candidate against the three baselines."""

    player_type: PlayerType
    model_candidate: ModelCandidate
    model_rmse: float
    baseline_naive_last_game_rmse: float
    baseline_trailing_7d_rmse: float
    baseline_season_to_date_rmse: float
    best_baseline_rmse: float
    improvement_pct: float = Field(
        description="(1 - model_rmse / best_baseline_rmse) * 100. Must be >= 10.0 to pass.",
    )
    passes_threshold: bool
