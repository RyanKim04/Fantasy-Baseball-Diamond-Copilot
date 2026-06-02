"""Training orchestrator with MLflow logging.

Runs the full training pipeline: load features, split, tune hyperparameters
via walk-forward CV, train final models, log to MLflow.

Owner: modeler subagent.
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import optuna
import pandas as pd

from packages.ml.models.cqr_aci import CQRACIModel, tune_aci_gamma
from packages.ml.evaluation.metrics import pinball_loss
from packages.ml.models.lgbm_quantile import (
    DEFAULT_PARAMS,
    QUANTILE_LEVELS,
    LGBMQuantileModel,
)
from packages.ml.models.ridge import ALPHA_GRID, RidgeProjectionModel
from packages.ml.training.splits import (
    HEADLINE_FOLD_NUMBERS,
    get_calibration_split,
    get_walk_forward_folds,
    split_by_date,
)
from packages.shared.schemas.ml import ModelCandidate, PlayerType, ScoringRulesSnapshot, SplitName

logger = logging.getLogger(__name__)

# Columns that are metadata/index, not features
from packages.shared.schemas.ml import NON_FEATURE_COLUMNS  # noqa: E402


def _get_git_sha() -> str:
    """Get the current git SHA for logging."""
    try:
        result = subprocess.run(  # noqa: S603, S607
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _compute_hash(content: str) -> str:
    """Compute SHA-256 hash of a string (for feature-set / scoring-rules hashing)."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _compute_feature_set_hash(columns: list[str]) -> str:
    """Compute a hash of the feature column names."""
    return _compute_hash(",".join(sorted(columns)))


def _compute_scoring_rules_hash(scoring_rules: ScoringRulesSnapshot) -> str:
    """Compute a hash of the scoring rules."""
    return _compute_hash(scoring_rules.model_dump_json())


def _compute_protocol_hash() -> str:
    """Compute a hash of the validation protocol document."""
    protocol_path = Path(__file__).parent.parent / "evaluation" / "validation_protocol.md"
    if protocol_path.exists():
        return _compute_hash(protocol_path.read_text())
    return "protocol-not-found"


def _load_features(feature_dir: Path, player_type: PlayerType) -> pd.DataFrame:
    """Load the feature parquet file for the given population.

    Parameters
    ----------
    feature_dir : Path
        Directory containing the parquet files.
    player_type : PlayerType
        HITTER or PITCHER.

    Returns
    -------
    pd.DataFrame
        Feature table with metadata columns and feature columns.
    """
    suffix = "hitters" if player_type == PlayerType.HITTER else "pitchers"
    path = feature_dir / f"features_{suffix}.parquet"

    if not path.exists():
        msg = f"Feature file not found: {path}"
        raise FileNotFoundError(msg)

    df = pd.read_parquet(path)
    logger.info("Loaded %d rows from %s", len(df), path)
    return df


def _extract_features_and_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate feature columns from metadata and target.

    Parameters
    ----------
    df : pd.DataFrame
        Full feature table with metadata columns.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        (X, y) where X has only feature columns and y is fantasy_points.
    """
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    X = df[feature_cols]
    y = df["fantasy_points"]
    return X, y


def _tune_ridge(
    df_train_val: pd.DataFrame,
    date_column: str = "game_date",
) -> float:
    """Tune Ridge alpha via walk-forward CV.

    Uses the 5 walk-forward folds and selects the alpha that minimizes
    mean RMSE across headline folds (2-5).

    Parameters
    ----------
    df_train_val : pd.DataFrame
        Combined Train + Validation data.
    date_column : str
        Name of the date column.

    Returns
    -------
    float
        Best alpha value.
    """
    folds = get_walk_forward_folds(df_train_val, date_column)

    best_alpha = 1.0
    best_rmse = float("inf")

    for alpha in ALPHA_GRID:
        fold_rmses: list[float] = []

        for fold_idx, (train_fold, val_fold) in enumerate(folds):
            fold_num = fold_idx + 1
            if fold_num not in HEADLINE_FOLD_NUMBERS:
                continue

            X_train, y_train = _extract_features_and_target(train_fold)
            X_val, y_val = _extract_features_and_target(val_fold)

            if len(X_val) == 0:
                continue

            model = RidgeProjectionModel(alpha=alpha)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            rmse = float(np.sqrt(np.mean((np.asarray(y_val) - preds) ** 2)))
            fold_rmses.append(rmse)

        if fold_rmses:
            mean_rmse = float(np.mean(fold_rmses))
            if mean_rmse < best_rmse:
                best_rmse = mean_rmse
                best_alpha = float(alpha)

    logger.info("Ridge tuning: best_alpha=%.4f, best_cv_rmse=%.4f", best_alpha, best_rmse)
    return best_alpha


def _tune_lgbm(
    df_train_val: pd.DataFrame,
    n_trials: int = 50,
    date_column: str = "game_date",
) -> dict[str, Any]:
    """Tune LightGBM hyperparameters via Optuna with walk-forward CV.

    Minimizes mean Pinball loss across headline folds (2-5), summed
    across all three quantile levels.

    Parameters
    ----------
    df_train_val : pd.DataFrame
        Combined Train + Validation data.
    n_trials : int
        Number of Optuna trials.
    date_column : str
        Name of the date column.

    Returns
    -------
    dict[str, Any]
        Best hyperparameters.
    """
    folds = get_walk_forward_folds(df_train_val, date_column)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "num_leaves": trial.suggest_categorical("num_leaves", [31, 63, 127]),
            "min_data_in_leaf": trial.suggest_categorical("min_data_in_leaf", [20, 50, 100]),
            "learning_rate": trial.suggest_categorical("learning_rate", [0.01, 0.03, 0.05]),
            "feature_fraction": trial.suggest_categorical("feature_fraction", [0.7, 0.85, 1.0]),
            "bagging_fraction": trial.suggest_categorical("bagging_fraction", [0.7, 0.85, 1.0]),
            "bagging_freq": 1,
            "n_estimators": 2000,
            "verbose": -1,
        }

        fold_losses: list[float] = []

        for fold_idx, (train_fold, val_fold) in enumerate(folds):
            fold_num = fold_idx + 1
            if fold_num not in HEADLINE_FOLD_NUMBERS:
                continue

            X_train, y_train = _extract_features_and_target(train_fold)
            X_val, y_val = _extract_features_and_target(val_fold)

            if len(X_val) == 0:
                continue

            model = LGBMQuantileModel(params=params)
            model.fit(X_train, y_train, X_val, y_val)

            # Compute summed pinball loss across all three quantile levels
            y_val_arr = np.asarray(y_val, dtype=np.float64)
            all_preds = model.predict_all_quantiles(X_val)
            total_pinball = sum(
                pinball_loss(y_val_arr, all_preds[alpha], alpha)
                for alpha in QUANTILE_LEVELS
            )
            fold_losses.append(total_pinball)

        return float(np.mean(fold_losses)) if fold_losses else float("inf")

    # Suppress Optuna logs during tuning
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="minimize",
        study_name="lgbm_quantile_tuning",
    )
    study.optimize(objective, n_trials=n_trials)

    best_params = {
        **DEFAULT_PARAMS,
        **study.best_params,
        "bagging_freq": 1,
        "n_estimators": 2000,
        "verbose": -1,
    }

    logger.info(
        "LGBM tuning: best_trial=%d, best_value=%.4f, best_params=%s",
        study.best_trial.number,
        study.best_value,
        study.best_params,
    )

    return best_params


def train_model(
    feature_dir: Path,
    model_candidate: ModelCandidate,
    player_type: PlayerType,
    scoring_rules: ScoringRulesSnapshot,
    mlflow_experiment_name: str = "phase1-bakeoff",
    n_optuna_trials: int = 50,
    m2_best_params: dict[str, Any] | None = None,
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
    m2_best_params : dict[str, Any] | None
        M2's tuned hyperparameters, passed to M3 so it reuses M2's
        boosters without re-tuning (model_bakeoff.md section 3).
        Ignored for M1 and M2 candidates.

    Returns
    -------
    str
        MLflow run ID of the logged training run.
    """
    # Load features
    df = _load_features(feature_dir, player_type)

    # Split data
    splits = split_by_date(df)
    df_train = splits[SplitName.TRAIN]
    df_val = splits[SplitName.VALIDATION]
    # NOTE: Test split is never touched here (validation_protocol.md section 11)

    df_train_val = pd.concat([df_train, df_val], ignore_index=True)

    # Feature columns
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    feature_set_hash = _compute_feature_set_hash(feature_cols)
    scoring_rules_hash = _compute_scoring_rules_hash(scoring_rules)
    protocol_hash = _compute_protocol_hash()
    git_sha = _get_git_sha()

    # Set up MLflow
    mlflow.set_experiment(mlflow_experiment_name)

    run_name = f"{model_candidate.value}_{player_type.value}"

    with mlflow.start_run(run_name=run_name) as run:
        # Log common metadata
        mlflow.log_params({
            "model_candidate": model_candidate.value,
            "player_type": player_type.value,
            "feature_set_hash": feature_set_hash,
            "scoring_rules_hash": scoring_rules_hash,
            "protocol_hash": protocol_hash,
            "git_sha": git_sha,
            "n_features": len(feature_cols),
            "n_train_rows": len(df_train),
            "n_val_rows": len(df_val),
        })

        # Log scoring rules as JSON artifact
        scoring_json = scoring_rules.model_dump_json(indent=2)
        mlflow.log_text(scoring_json, "scoring_rules.json")

        if model_candidate == ModelCandidate.M1_RIDGE:
            _train_m1(df_train, df_val, df_train_val)

        elif model_candidate == ModelCandidate.M2_LGBM_QUANTILE:
            _train_m2(df_train, df_val, df_train_val, n_optuna_trials)

        elif model_candidate == ModelCandidate.M3_CQR_ACI:
            _train_m3(df_train, df_val, df_train_val, m2_best_params=m2_best_params)

        else:
            msg = f"Unknown model candidate: {model_candidate}"
            raise ValueError(msg)

        run_id = run.info.run_id
        logger.info("Training complete. MLflow run_id=%s", run_id)

    return run_id


def _train_m1(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_train_val: pd.DataFrame,
) -> RidgeProjectionModel:
    """Train M1 Ridge model with walk-forward CV for alpha tuning.

    Returns the final fitted model.
    """
    # Tune alpha
    best_alpha = _tune_ridge(df_train_val)

    # Train final model on full Train set
    X_train, y_train = _extract_features_and_target(df_train)
    X_val, y_val = _extract_features_and_target(df_val)

    model = RidgeProjectionModel(alpha=best_alpha)
    model.fit(X_train, y_train)

    # Log model params
    mlflow.log_params(model.get_params())

    # Compute validation metrics
    val_preds = model.predict(X_val)
    y_val_arr = np.asarray(y_val, dtype=np.float64)

    val_rmse = float(np.sqrt(np.mean((y_val_arr - val_preds) ** 2)))
    val_mae = float(np.mean(np.abs(y_val_arr - val_preds)))

    mlflow.log_metrics({
        "val_rmse": val_rmse,
        "val_mae": val_mae,
    })

    # Log fold-level metrics
    folds = get_walk_forward_folds(df_train_val)
    for fold_idx, (train_fold, val_fold) in enumerate(folds):
        fold_num = fold_idx + 1
        X_t, y_t = _extract_features_and_target(train_fold)
        X_v, y_v = _extract_features_and_target(val_fold)

        if len(X_v) == 0:
            continue

        fold_model = RidgeProjectionModel(alpha=best_alpha)
        fold_model.fit(X_t, y_t)
        fold_preds = fold_model.predict(X_v)
        fold_rmse = float(np.sqrt(np.mean((np.asarray(y_v) - fold_preds) ** 2)))
        mlflow.log_metric(f"fold_{fold_num}_rmse", fold_rmse)

    logger.info("M1 Ridge: val_rmse=%.4f, val_mae=%.4f", val_rmse, val_mae)

    return model


def _train_m2(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_train_val: pd.DataFrame,
    n_optuna_trials: int,
) -> LGBMQuantileModel:
    """Train M2 LightGBM quantile model with Optuna tuning.

    Returns the final fitted model.
    """
    # Tune hyperparameters
    best_params = _tune_lgbm(df_train_val, n_trials=n_optuna_trials)

    # Log best params
    for k, v in best_params.items():
        if k != "verbose":
            mlflow.log_param(f"lgbm_{k}", v)

    # Train final model on full Train set with Val for early stopping
    X_train, y_train = _extract_features_and_target(df_train)
    X_val, y_val = _extract_features_and_target(df_val)

    model = LGBMQuantileModel(params=best_params)
    model.fit(X_train, y_train, X_val, y_val)

    # Compute validation metrics
    val_preds = model.predict(X_val)
    y_val_arr = np.asarray(y_val, dtype=np.float64)

    val_rmse = float(np.sqrt(np.mean((y_val_arr - val_preds) ** 2)))
    val_mae = float(np.mean(np.abs(y_val_arr - val_preds)))

    # Quantile-specific metrics
    val_quantiles = model.predict_quantiles(X_val)
    q_lo, q_hi = val_quantiles[:, 0], val_quantiles[:, 1]
    coverage = float(np.mean((y_val_arr >= q_lo) & (y_val_arr <= q_hi)))
    sharpness = float(np.mean(q_hi - q_lo))

    # Pinball losses
    all_q = model.predict_all_quantiles(X_val)
    pb_10 = pinball_loss(y_val_arr, all_q[0.1], 0.1)
    pb_50 = pinball_loss(y_val_arr, all_q[0.5], 0.5)
    pb_90 = pinball_loss(y_val_arr, all_q[0.9], 0.9)

    mlflow.log_metrics({
        "val_rmse": val_rmse,
        "val_mae": val_mae,
        "val_coverage_80": coverage,
        "val_sharpness": sharpness,
        "val_pinball_10": pb_10,
        "val_pinball_50": pb_50,
        "val_pinball_90": pb_90,
    })

    # Log fold-level metrics
    folds = get_walk_forward_folds(df_train_val)
    for fold_idx, (train_fold, val_fold) in enumerate(folds):
        fold_num = fold_idx + 1
        X_t, y_t = _extract_features_and_target(train_fold)
        X_v, y_v = _extract_features_and_target(val_fold)

        if len(X_v) == 0:
            continue

        fold_model = LGBMQuantileModel(params=best_params)
        fold_model.fit(X_t, y_t, X_v, y_v)
        fold_preds = fold_model.predict(X_v)
        fold_rmse = float(np.sqrt(np.mean((np.asarray(y_v) - fold_preds) ** 2)))
        mlflow.log_metric(f"fold_{fold_num}_rmse", fold_rmse)

    logger.info(
        "M2 LGBM: val_rmse=%.4f, val_mae=%.4f, coverage=%.4f, sharpness=%.4f",
        val_rmse,
        val_mae,
        coverage,
        sharpness,
    )

    return model


def _train_m3(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_train_val: pd.DataFrame,
    m2_best_params: dict[str, Any] | None = None,
) -> CQRACIModel:
    """Train M3 CQR+ACI model. Reuses M2's boosters per model_bakeoff.md section 3.

    M3 reuses M2's tuned hyperparameters (no separate search) but trains
    the base boosters on the fit portion of Train only, keeping the
    calibration set disjoint per validation_protocol.md section 4.

    Parameters
    ----------
    df_train : pd.DataFrame
        Train split.
    df_val : pd.DataFrame
        Validation split.
    df_train_val : pd.DataFrame
        Combined Train + Validation data.
    m2_best_params : dict[str, Any] | None
        M2's tuned hyperparameters. If None (standalone mode), uses
        DEFAULT_PARAMS as fallback.

    Returns the final fitted model.
    """
    # Reuse M2's tuned params; fall back to defaults if M2 was not trained
    best_params = m2_best_params if m2_best_params is not None else {**DEFAULT_PARAMS}
    logger.info(
        "M3 using %s LGBM params",
        "M2's tuned" if m2_best_params is not None else "default (M2 not trained)",
    )

    # Log tuned params
    for k, v in best_params.items():
        if k != "verbose":
            mlflow.log_param(f"lgbm_{k}", v)

    # Split train into fit and calibration portions for CQR
    df_fit, df_cal = get_calibration_split(df_train)

    # Further split df_fit: last 10% chronologically becomes early-stopping
    # holdout so that the calibration set stays untouched (H4 fix).
    df_fit_dates = df_fit["game_date"]
    sorted_dates = df_fit_dates.sort_values()
    es_cutoff_idx = int(len(sorted_dates) * 0.90)
    es_cutoff_date = sorted_dates.iloc[min(es_cutoff_idx, len(sorted_dates) - 1)]

    df_fit_train = df_fit.loc[df_fit_dates <= es_cutoff_date].copy()
    df_fit_es = df_fit.loc[df_fit_dates > es_cutoff_date].copy()

    X_fit_train, y_fit_train = _extract_features_and_target(df_fit_train)
    X_fit_es, y_fit_es = _extract_features_and_target(df_fit_es)
    X_cal, y_cal = _extract_features_and_target(df_cal)
    X_val, y_val = _extract_features_and_target(df_val)

    # Use the early-stopping holdout (NOT calibration set) for early stopping
    fit_m2 = LGBMQuantileModel(params=best_params)
    if len(df_fit_es) > 0:
        fit_m2.fit(X_fit_train, y_fit_train, X_fit_es, y_fit_es)
    else:
        fit_m2.fit(X_fit_train, y_fit_train)

    # Create M3 using the fit-only M2 as base
    m3_model = CQRACIModel(base_model=fit_m2)
    m3_model.fit(X_fit_train, y_fit_train)

    # Calibrate on the calibration set
    m3_model.calibrate(X_cal, y_cal)

    mlflow.log_params({
        "cqr_adjustment": m3_model.cqr_adjustment,
        "n_calibration_rows": len(df_cal),
        "n_fit_train_rows": len(df_fit_train),
        "n_fit_es_rows": len(df_fit_es),
    })

    # Tune ACI gamma on validation data
    best_gamma = tune_aci_gamma(m3_model, X_val, y_val)
    mlflow.log_param("aci_gamma", best_gamma)

    # Compute validation metrics with the tuned gamma
    m3_model.reset_aci()
    val_preds = m3_model.predict(X_val)
    y_val_arr = np.asarray(y_val, dtype=np.float64)

    val_rmse = float(np.sqrt(np.mean((y_val_arr - val_preds) ** 2)))
    val_mae = float(np.mean(np.abs(y_val_arr - val_preds)))

    val_quantiles = m3_model.predict_quantiles(X_val)
    q_lo, q_hi = val_quantiles[:, 0], val_quantiles[:, 1]
    coverage = float(np.mean((y_val_arr >= q_lo) & (y_val_arr <= q_hi)))
    sharpness = float(np.mean(q_hi - q_lo))

    mlflow.log_metrics({
        "m3_val_rmse": val_rmse,
        "m3_val_mae": val_mae,
        "m3_val_coverage_80": coverage,
        "m3_val_sharpness": sharpness,
    })

    logger.info(
        "M3 CQR+ACI: val_rmse=%.4f, val_mae=%.4f, coverage=%.4f, sharpness=%.4f, gamma=%.3f",
        val_rmse,
        val_mae,
        coverage,
        sharpness,
        best_gamma,
    )

    return m3_model
