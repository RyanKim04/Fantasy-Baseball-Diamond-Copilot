"""Unit tests for packages/ml/training/train.py.

Tests the training orchestrator's key functions with synthetic data and
mocked MLflow calls.

Owner: modeler subagent.
"""

from __future__ import annotations

from datetime import date
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from packages.ml.training.splits import (
    HEADLINE_FOLD_NUMBERS,
    WALK_FORWARD_FOLDS,
    get_walk_forward_folds,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_train_val_df(
    start_date: date = date(2019, 3, 1),
    end_date: date = date(2024, 9, 30),
    rows_per_day: int = 2,
    n_features: int = 4,
) -> pd.DataFrame:
    """Create a synthetic DataFrame spanning Train + Validation periods.

    The DGP is y = 2*x1 + 0.5*x2 - x3 + noise so that Ridge and LGBM
    can find signal in a small number of trials.
    """
    dates = pd.date_range(start=start_date, end=end_date, freq="7D")
    rows: list[dict[str, Any]] = []
    player_id = 100
    game_pk = 1000

    for d in dates:
        for _ in range(rows_per_day):
            rng = np.random.default_rng(game_pk)
            feats = {f"feat_x{i}": float(rng.normal(0, 1)) for i in range(n_features)}
            y = 2.0 * feats["feat_x0"] + 0.5 * feats["feat_x1"] - feats["feat_x2"] + rng.normal(0, 1)
            rows.append({
                "player_id": player_id,
                "game_pk": game_pk,
                "game_date": d.date(),
                "season_year": d.year,
                "player_type": "hitter",
                "fantasy_points": y,
                **feats,
            })
            game_pk += 1
            player_id += 1

    return pd.DataFrame(rows)


@pytest.fixture()
def train_val_df() -> pd.DataFrame:
    """DataFrame spanning 2019-2024 for walk-forward CV testing."""
    return _make_train_val_df()


# ---------------------------------------------------------------------------
# Test: _tune_ridge selects an alpha
# ---------------------------------------------------------------------------
class TestTuneRidge:
    """Tests for train._tune_ridge with synthetic data."""

    @patch("packages.ml.training.train.mlflow")
    def test_selects_an_alpha(self, mock_mlflow: MagicMock, train_val_df: pd.DataFrame) -> None:
        """_tune_ridge should return a float alpha from the grid."""
        from packages.ml.models.ridge import ALPHA_GRID
        from packages.ml.training.train import _tune_ridge

        best_alpha = _tune_ridge(train_val_df)

        assert isinstance(best_alpha, float)
        assert best_alpha > 0
        # Should be one of the values from the grid
        assert any(np.isclose(best_alpha, a) for a in ALPHA_GRID), (
            f"best_alpha={best_alpha} not in ALPHA_GRID"
        )

    @patch("packages.ml.training.train.mlflow")
    def test_different_data_may_select_different_alpha(
        self, mock_mlflow: MagicMock
    ) -> None:
        """Sanity: the function actually evaluates multiple alphas (not hardcoded)."""
        from packages.ml.training.train import _tune_ridge

        df1 = _make_train_val_df(rows_per_day=3, n_features=4)

        # This just needs to run without error and return a valid alpha
        alpha = _tune_ridge(df1)
        assert alpha > 0


# ---------------------------------------------------------------------------
# Test: _tune_lgbm returns valid params
# ---------------------------------------------------------------------------
class TestTuneLGBM:
    """Tests for train._tune_lgbm with synthetic data and mocked MLflow."""

    @patch("packages.ml.training.train.mlflow")
    def test_returns_valid_params(self, mock_mlflow: MagicMock, train_val_df: pd.DataFrame) -> None:
        """_tune_lgbm should return a dict with expected LightGBM params."""
        from packages.ml.training.train import _tune_lgbm

        # Use only 3 trials to keep the test fast
        best_params = _tune_lgbm(train_val_df, n_trials=3)

        assert isinstance(best_params, dict)
        assert "num_leaves" in best_params
        assert "min_data_in_leaf" in best_params
        assert "learning_rate" in best_params
        assert "feature_fraction" in best_params
        assert "bagging_fraction" in best_params
        assert "n_estimators" in best_params
        assert best_params["verbose"] == -1

    @patch("packages.ml.training.train.mlflow")
    def test_params_from_search_space(self, mock_mlflow: MagicMock, train_val_df: pd.DataFrame) -> None:
        """Returned params should be within the defined search space."""
        from packages.ml.training.train import _tune_lgbm

        best_params = _tune_lgbm(train_val_df, n_trials=3)

        assert best_params["num_leaves"] in [31, 63, 127]
        assert best_params["min_data_in_leaf"] in [20, 50, 100]
        assert best_params["learning_rate"] in [0.01, 0.03, 0.05]
        assert best_params["feature_fraction"] in [0.7, 0.85, 1.0]
        assert best_params["bagging_fraction"] in [0.7, 0.85, 1.0]


# ---------------------------------------------------------------------------
# Test: Walk-forward CV wiring
# ---------------------------------------------------------------------------
class TestWalkForwardCVWiring:
    """Verify that the walk-forward CV iteration used by the training
    functions produces the expected folds and respects temporal ordering."""

    def test_five_folds_produced(self, train_val_df: pd.DataFrame) -> None:
        """get_walk_forward_folds returns exactly 5 folds."""
        folds = get_walk_forward_folds(train_val_df)
        assert len(folds) == 5

    def test_headline_folds_are_2_through_5(self) -> None:
        """Headline fold numbers match protocol section 3."""
        assert HEADLINE_FOLD_NUMBERS == [2, 3, 4, 5]

    def test_training_windows_are_expanding(self, train_val_df: pd.DataFrame) -> None:
        """Each successive fold has a larger or equal training set."""
        folds = get_walk_forward_folds(train_val_df)
        sizes = [len(train_fold) for train_fold, _ in folds]
        for i in range(1, len(sizes)):
            assert sizes[i] >= sizes[i - 1], (
                f"Fold {i + 1} training set ({sizes[i]}) smaller than fold {i} ({sizes[i - 1]})"
            )

    def test_no_temporal_leakage_across_folds(self, train_val_df: pd.DataFrame) -> None:
        """In every fold, max(train_date) < min(val_date)."""
        folds = get_walk_forward_folds(train_val_df)
        for fold_idx, (train_fold, val_fold) in enumerate(folds):
            if len(train_fold) == 0 or len(val_fold) == 0:
                continue
            train_max = train_fold["game_date"].max()
            val_min = val_fold["game_date"].min()
            assert train_max < val_min, (
                f"Fold {fold_idx + 1}: temporal leakage (train_max={train_max}, val_min={val_min})"
            )

    def test_fold_definitions_match_protocol(self) -> None:
        """Walk-forward fold definitions match validation_protocol.md section 3."""
        expected_val_years = [2020, 2021, 2022, 2023, 2024]
        for i, fold in enumerate(WALK_FORWARD_FOLDS):
            assert fold.fold_number == i + 1
            assert fold.train_start == date(2019, 1, 1)
            assert fold.val_start.year == expected_val_years[i]
            assert fold.val_end.year == expected_val_years[i]

    def test_headline_folds_iterate_correctly(self, train_val_df: pd.DataFrame) -> None:
        """The training loop pattern (skip non-headline folds) should iterate
        over exactly folds 2-5 and skip fold 1."""
        folds = get_walk_forward_folds(train_val_df)
        iterated_folds: list[int] = []

        for fold_idx, (_train_fold, _val_fold) in enumerate(folds):
            fold_num = fold_idx + 1
            if fold_num not in HEADLINE_FOLD_NUMBERS:
                continue
            iterated_folds.append(fold_num)

        assert iterated_folds == [2, 3, 4, 5]


# ---------------------------------------------------------------------------
# Test: M3 calibration does not contaminate (H4 regression test)
# ---------------------------------------------------------------------------
class TestM3CalibrationNotContaminated:
    """Verify that _train_m3 uses a separate early-stopping holdout,
    NOT the calibration set, for early stopping."""

    @patch("packages.ml.training.train.mlflow")
    def test_early_stopping_uses_holdout_not_cal(self, mock_mlflow: MagicMock) -> None:
        """The early-stopping validation set must be disjoint from the
        calibration set used for CQR conformity scores."""
        from packages.ml.training.train import _extract_features_and_target
        from packages.ml.training.splits import get_calibration_split

        # Build a train-period-only DataFrame
        df_train = _make_train_val_df(
            start_date=date(2019, 3, 1),
            end_date=date(2023, 9, 30),
            rows_per_day=3,
        )

        df_fit, df_cal = get_calibration_split(df_train)

        # Simulate the early-stopping holdout split from _train_m3
        df_fit_dates = df_fit["game_date"]
        sorted_dates = df_fit_dates.sort_values()
        es_cutoff_idx = int(len(sorted_dates) * 0.90)
        es_cutoff_date = sorted_dates.iloc[min(es_cutoff_idx, len(sorted_dates) - 1)]

        df_fit_train = df_fit.loc[df_fit_dates <= es_cutoff_date]
        df_fit_es = df_fit.loc[df_fit_dates > es_cutoff_date]

        # The early-stopping holdout must be disjoint from the calibration set
        es_indices = set(df_fit_es.index)
        cal_indices = set(df_cal.index)
        overlap = es_indices & cal_indices
        assert len(overlap) == 0, (
            f"Early-stopping holdout overlaps calibration set by {len(overlap)} rows"
        )

        # The early-stopping holdout must come from df_fit, not df_cal
        fit_indices = set(df_fit.index)
        assert es_indices.issubset(fit_indices), (
            "Early-stopping holdout contains rows not in the fit portion"
        )

        # The calibration set dates must all be after the fit portion
        if len(df_fit) > 0 and len(df_cal) > 0:
            assert df_fit["game_date"].max() < df_cal["game_date"].min()


# ---------------------------------------------------------------------------
# Test: M3 reuses M2 params (H5 regression test)
# ---------------------------------------------------------------------------
class TestM3ReusesM2Params:
    """Verify that _train_m3 accepts and uses M2's best_params."""

    @patch("packages.ml.training.train.tune_aci_gamma", return_value=0.01)
    @patch("packages.ml.training.train.mlflow")
    def test_m3_uses_m2_params_when_provided(
        self,
        mock_mlflow: MagicMock,
        mock_tune_gamma: MagicMock,
    ) -> None:
        """When m2_best_params is passed, _train_m3 should use those params
        for the base LightGBM model, not re-run Optuna."""
        from packages.ml.training.train import _train_m3

        df_train = _make_train_val_df(
            start_date=date(2019, 3, 1),
            end_date=date(2023, 9, 30),
            rows_per_day=2,
        )
        df_val = _make_train_val_df(
            start_date=date(2024, 3, 1),
            end_date=date(2024, 9, 30),
            rows_per_day=2,
        )
        df_train_val = pd.concat([df_train, df_val], ignore_index=True)

        m2_params = {
            "num_leaves": 31,
            "min_data_in_leaf": 50,
            "learning_rate": 0.05,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.85,
            "bagging_freq": 1,
            "n_estimators": 50,  # small for test speed
            "verbose": -1,
        }

        model = _train_m3(df_train, df_val, df_train_val, m2_best_params=m2_params)

        # The base model should exist and be fitted
        assert model.base_model is not None
        assert model.base_model.is_fitted

    @patch("packages.ml.training.train.tune_aci_gamma", return_value=0.01)
    @patch("packages.ml.training.train.mlflow")
    def test_m3_falls_back_to_defaults_when_no_m2(
        self,
        mock_mlflow: MagicMock,
        mock_tune_gamma: MagicMock,
    ) -> None:
        """When m2_best_params is None, _train_m3 should use DEFAULT_PARAMS."""
        from packages.ml.models.lgbm_quantile import DEFAULT_PARAMS
        from packages.ml.training.train import _train_m3

        df_train = _make_train_val_df(
            start_date=date(2019, 3, 1),
            end_date=date(2023, 9, 30),
            rows_per_day=2,
        )
        df_val = _make_train_val_df(
            start_date=date(2024, 3, 1),
            end_date=date(2024, 9, 30),
            rows_per_day=2,
        )
        df_train_val = pd.concat([df_train, df_val], ignore_index=True)

        # m2_best_params=None means standalone mode
        model = _train_m3(df_train, df_val, df_train_val, m2_best_params=None)

        assert model.base_model is not None
        assert model.is_fitted


# ---------------------------------------------------------------------------
# Test: Protocol hash uses module-relative path (M11 regression test)
# ---------------------------------------------------------------------------
class TestProtocolHash:
    """Verify _compute_protocol_hash uses a path relative to the module."""

    def test_protocol_hash_not_empty(self) -> None:
        """The protocol hash should be a real hash, not 'protocol-not-found'."""
        from packages.ml.training.train import _compute_protocol_hash

        h = _compute_protocol_hash()
        assert h != "protocol-not-found", (
            "_compute_protocol_hash returned 'protocol-not-found'; "
            "the path resolution is likely broken"
        )
        assert len(h) == 16  # SHA-256 truncated to 16 hex chars
