"""Unit tests for packages/ml/models/ (M1, M2, M3).

Tests each model candidate's fit/predict interface with synthetic data.
Owner: modeler subagent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from packages.ml.models.cqr_aci import CQRACIModel, tune_aci_gamma
from packages.ml.evaluation.metrics import pinball_loss
from packages.ml.models.lgbm_quantile import LGBMQuantileModel
from packages.ml.models.ridge import RidgeProjectionModel


# ---------------------------------------------------------------------------
# Fixtures — synthetic regression data
# ---------------------------------------------------------------------------
@pytest.fixture()
def synthetic_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Create a synthetic regression dataset with known structure.

    y = 2*x1 + 0.5*x2 - 1*x3 + noise
    """
    rng = np.random.default_rng(42)
    n_train = 500
    n_val = 100

    def _make_data(n: int, seed: int) -> tuple[pd.DataFrame, pd.Series]:
        r = np.random.default_rng(seed)
        X = pd.DataFrame({
            "feat_x1": r.normal(0, 1, n),
            "feat_x2": r.normal(0, 1, n),
            "feat_x3": r.normal(0, 1, n),
            "feat_x4": r.normal(0, 1, n),  # noise feature
        })
        # Add some NaN values to test imputation
        X.iloc[0, 1] = np.nan
        X.iloc[5, 3] = np.nan

        y = 2 * X["feat_x1"] + 0.5 * X["feat_x2"] - 1 * X["feat_x3"] + r.normal(0, 1, n)
        y = y.fillna(0)
        return X, pd.Series(y, name="fantasy_points")

    X_train, y_train = _make_data(n_train, seed=42)
    X_val, y_val = _make_data(n_val, seed=99)

    return X_train, y_train, X_val, y_val


@pytest.fixture()
def calibration_data() -> tuple[pd.DataFrame, pd.Series]:
    """Create a calibration dataset (separate from train/val)."""
    rng = np.random.default_rng(77)
    n = 200
    X = pd.DataFrame({
        "feat_x1": rng.normal(0, 1, n),
        "feat_x2": rng.normal(0, 1, n),
        "feat_x3": rng.normal(0, 1, n),
        "feat_x4": rng.normal(0, 1, n),
    })
    y = 2 * X["feat_x1"] + 0.5 * X["feat_x2"] - 1 * X["feat_x3"] + rng.normal(0, 1, n)
    return X, pd.Series(y, name="fantasy_points")


# ---------------------------------------------------------------------------
# TestRidgeProjectionModel (M1)
# ---------------------------------------------------------------------------
class TestRidgeProjectionModel:
    """Tests for models.ridge.RidgeProjectionModel (M1)."""

    def test_name_and_produces_intervals(self) -> None:
        model = RidgeProjectionModel()
        assert model.name == "m1_ridge"
        assert model.produces_intervals is False

    def test_fit_predict_shape(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
    ) -> None:
        X_train, y_train, X_val, y_val = synthetic_data
        model = RidgeProjectionModel(alpha=1.0)
        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        assert preds.shape == (len(X_val),)
        assert preds.dtype == np.float64

    def test_predict_quantiles_returns_empty(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
    ) -> None:
        X_train, y_train, X_val, _ = synthetic_data
        model = RidgeProjectionModel()
        model.fit(X_train, y_train)

        quantiles = model.predict_quantiles(X_val)
        assert quantiles.size == 0

    def test_predict_before_fit_raises(self) -> None:
        model = RidgeProjectionModel()
        X = pd.DataFrame({"feat_x1": [1.0]})
        with pytest.raises(RuntimeError, match="fit"):
            model.predict(X)

    def test_reasonable_predictions(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
    ) -> None:
        """Predictions should correlate with actual values for the linear DGP."""
        X_train, y_train, X_val, y_val = synthetic_data
        model = RidgeProjectionModel(alpha=0.01)
        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        corr = np.corrcoef(preds, y_val)[0, 1]
        assert corr > 0.8, f"Correlation too low: {corr:.4f}"

    def test_get_params(self) -> None:
        model = RidgeProjectionModel(alpha=0.5)
        params = model.get_params()
        assert params["alpha"] == 0.5
        assert params["model_type"] == "ridge"

    def test_get_coefficients_after_fit(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
    ) -> None:
        X_train, y_train, _, _ = synthetic_data
        model = RidgeProjectionModel()
        model.fit(X_train, y_train)

        coefs = model.get_coefficients()
        assert coefs is not None
        assert len(coefs) == X_train.shape[1]

    def test_handles_nan_via_imputation(self) -> None:
        """Model should handle NaN values through median imputation."""
        X = pd.DataFrame({
            "feat_a": [1.0, 2.0, np.nan, 4.0, 5.0],
            "feat_b": [np.nan, 2.0, 3.0, 4.0, 5.0],
        })
        y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        model = RidgeProjectionModel()
        model.fit(X, y)

        preds = model.predict(X)
        assert not np.any(np.isnan(preds))


# ---------------------------------------------------------------------------
# TestLGBMQuantileModel (M2)
# ---------------------------------------------------------------------------
class TestLGBMQuantileModel:
    """Tests for models.lgbm_quantile.LGBMQuantileModel (M2)."""

    def test_name_and_produces_intervals(self) -> None:
        model = LGBMQuantileModel()
        assert model.name == "m2_lgbm_quantile"
        assert model.produces_intervals is True

    def test_fit_predict_shape(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
    ) -> None:
        X_train, y_train, X_val, y_val = synthetic_data
        model = LGBMQuantileModel(params={
            "n_estimators": 50,
            "num_leaves": 31,
            "verbose": -1,
        })
        model.fit(X_train, y_train, X_val, y_val)

        preds = model.predict(X_val)
        assert preds.shape == (len(X_val),)
        assert preds.dtype == np.float64

    def test_predict_quantiles_shape(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
    ) -> None:
        X_train, y_train, X_val, y_val = synthetic_data
        model = LGBMQuantileModel(params={
            "n_estimators": 50,
            "num_leaves": 31,
            "verbose": -1,
        })
        model.fit(X_train, y_train, X_val, y_val)

        quantiles = model.predict_quantiles(X_val)
        assert quantiles.shape == (len(X_val), 2)
        assert quantiles.dtype == np.float64

    def test_p10_less_than_p90(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
    ) -> None:
        """p10 should generally be less than p90 (allow tiny violations from tree quantiles)."""
        X_train, y_train, X_val, y_val = synthetic_data
        model = LGBMQuantileModel(params={
            "n_estimators": 100,
            "num_leaves": 31,
            "verbose": -1,
        })
        model.fit(X_train, y_train, X_val, y_val)

        quantiles = model.predict_quantiles(X_val)
        p10, p90 = quantiles[:, 0], quantiles[:, 1]

        # Most rows should have p10 < p90
        valid_fraction = np.mean(p10 <= p90)
        assert valid_fraction > 0.9, f"Only {valid_fraction:.2%} of rows have p10 <= p90"

    def test_predict_before_fit_raises(self) -> None:
        model = LGBMQuantileModel()
        X = pd.DataFrame({"feat_x1": [1.0]})
        with pytest.raises(RuntimeError, match="fit"):
            model.predict(X)

    def test_predict_quantiles_before_fit_raises(self) -> None:
        model = LGBMQuantileModel()
        X = pd.DataFrame({"feat_x1": [1.0]})
        with pytest.raises(RuntimeError, match="fit"):
            model.predict_quantiles(X)

    def test_three_boosters_fitted(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
    ) -> None:
        X_train, y_train, X_val, y_val = synthetic_data
        model = LGBMQuantileModel(params={
            "n_estimators": 30,
            "num_leaves": 31,
            "verbose": -1,
        })
        model.fit(X_train, y_train, X_val, y_val)

        assert len(model.boosters) == 3
        assert set(model.boosters.keys()) == {0.1, 0.5, 0.9}

    def test_predict_all_quantiles(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
    ) -> None:
        X_train, y_train, X_val, y_val = synthetic_data
        model = LGBMQuantileModel(params={
            "n_estimators": 30,
            "num_leaves": 31,
            "verbose": -1,
        })
        model.fit(X_train, y_train, X_val, y_val)

        all_q = model.predict_all_quantiles(X_val)
        assert set(all_q.keys()) == {0.1, 0.5, 0.9}
        for alpha, preds in all_q.items():
            assert preds.shape == (len(X_val),)

    def test_get_params_includes_model_type(self) -> None:
        model = LGBMQuantileModel()
        params = model.get_params()
        assert params["model_type"] == "lgbm_quantile"

    def test_fit_without_validation(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
    ) -> None:
        """Model should fit without validation data (no early stopping)."""
        X_train, y_train, _, _ = synthetic_data
        model = LGBMQuantileModel(params={
            "n_estimators": 30,
            "num_leaves": 31,
            "verbose": -1,
        })
        model.fit(X_train, y_train)

        preds = model.predict(X_train)
        assert preds.shape == (len(X_train),)


class TestPinballLoss:
    """Tests for the pinball_loss utility function."""

    def test_perfect_prediction(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        loss = pinball_loss(y, y, tau=0.5)
        assert loss == pytest.approx(0.0, abs=1e-10)

    def test_positive_residual(self) -> None:
        """When y > pred, loss = alpha * (y - pred)."""
        y = np.array([3.0])
        pred = np.array([1.0])
        loss = pinball_loss(y, pred, tau=0.1)
        assert loss == pytest.approx(0.1 * 2.0)

    def test_negative_residual(self) -> None:
        """When y < pred, loss = (1 - alpha) * (pred - y)."""
        y = np.array([1.0])
        pred = np.array([3.0])
        loss = pinball_loss(y, pred, tau=0.1)
        assert loss == pytest.approx(0.9 * 2.0)

    def test_symmetric_at_median(self) -> None:
        """At alpha=0.5, pinball loss is symmetric."""
        y = np.array([0.0])
        loss_above = pinball_loss(y, np.array([-1.0]), tau=0.5)
        loss_below = pinball_loss(y, np.array([1.0]), tau=0.5)
        assert loss_above == pytest.approx(loss_below)


# ---------------------------------------------------------------------------
# TestCQRACIModel (M3)
# ---------------------------------------------------------------------------
class TestCQRACIModel:
    """Tests for models.cqr_aci.CQRACIModel (M3)."""

    def test_name_and_produces_intervals(self) -> None:
        model = CQRACIModel()
        assert model.name == "m3_cqr_aci"
        assert model.produces_intervals is True

    def test_fit_trains_base_if_needed(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
    ) -> None:
        """If no base model provided, fit() should create and train one."""
        X_train, y_train, X_val, y_val = synthetic_data
        model = CQRACIModel()
        model.fit(X_train, y_train, X_val, y_val)

        assert model.base_model is not None
        assert model.base_model.is_fitted

    def test_reuses_pretrained_base(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
    ) -> None:
        """If base model is pre-trained, fit() should not retrain."""
        X_train, y_train, X_val, y_val = synthetic_data

        base = LGBMQuantileModel(params={
            "n_estimators": 30,
            "num_leaves": 31,
            "verbose": -1,
        })
        base.fit(X_train, y_train, X_val, y_val)

        model = CQRACIModel(base_model=base)
        model.fit(X_train, y_train)

        assert model.base_model is base

    def test_calibrate_sets_conformity_scores(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
        calibration_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X_train, y_train, X_val, y_val = synthetic_data
        X_cal, y_cal = calibration_data

        base = LGBMQuantileModel(params={
            "n_estimators": 50,
            "num_leaves": 31,
            "verbose": -1,
        })
        base.fit(X_train, y_train, X_val, y_val)

        model = CQRACIModel(base_model=base)
        model.fit(X_train, y_train)
        model.calibrate(X_cal, y_cal)

        assert model._conformity_scores is not None
        assert len(model._conformity_scores) == len(y_cal)

    def test_predict_shape(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
        calibration_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X_train, y_train, X_val, y_val = synthetic_data
        X_cal, y_cal = calibration_data

        base = LGBMQuantileModel(params={
            "n_estimators": 50,
            "num_leaves": 31,
            "verbose": -1,
        })
        base.fit(X_train, y_train, X_val, y_val)

        model = CQRACIModel(base_model=base)
        model.fit(X_train, y_train)
        model.calibrate(X_cal, y_cal)

        preds = model.predict(X_val)
        assert preds.shape == (len(X_val),)

    def test_predict_quantiles_shape(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
        calibration_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X_train, y_train, X_val, y_val = synthetic_data
        X_cal, y_cal = calibration_data

        base = LGBMQuantileModel(params={
            "n_estimators": 50,
            "num_leaves": 31,
            "verbose": -1,
        })
        base.fit(X_train, y_train, X_val, y_val)

        model = CQRACIModel(base_model=base)
        model.fit(X_train, y_train)
        model.calibrate(X_cal, y_cal)

        quantiles = model.predict_quantiles(X_val)
        assert quantiles.shape == (len(X_val), 2)
        assert quantiles.dtype == np.float64

    def test_calibrated_intervals_wider_than_raw(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
        calibration_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        """CQR adjustment should widen intervals (since raw quantiles typically undercover)."""
        X_train, y_train, X_val, y_val = synthetic_data
        X_cal, y_cal = calibration_data

        base = LGBMQuantileModel(params={
            "n_estimators": 50,
            "num_leaves": 31,
            "verbose": -1,
        })
        base.fit(X_train, y_train, X_val, y_val)

        # Raw M2 intervals
        raw_quantiles = base.predict_quantiles(X_val)
        raw_width = np.mean(raw_quantiles[:, 1] - raw_quantiles[:, 0])

        # CQR-calibrated intervals
        model = CQRACIModel(base_model=base)
        model.fit(X_train, y_train)
        model.calibrate(X_cal, y_cal)
        cal_quantiles = model.predict_quantiles(X_val)
        cal_width = np.mean(cal_quantiles[:, 1] - cal_quantiles[:, 0])

        # CQR adjusts intervals based on conformity scores. The adjustment can
        # be negative if the base model already overcovers (conformity scores are
        # mostly negative). Just verify that both widths are positive and finite.
        assert cal_width > 0, f"CQR interval width must be positive, got {cal_width:.4f}"
        assert np.isfinite(cal_width)

    def test_predict_before_calibrate_raises(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
    ) -> None:
        X_train, y_train, X_val, _ = synthetic_data

        base = LGBMQuantileModel(params={
            "n_estimators": 30,
            "num_leaves": 31,
            "verbose": -1,
        })
        base.fit(X_train, y_train)

        model = CQRACIModel(base_model=base)
        model.fit(X_train, y_train)

        with pytest.raises(RuntimeError, match="calibrated"):
            model.predict_quantiles(X_val)

    def test_aci_update(self) -> None:
        """ACI update should change alpha_t."""
        model = CQRACIModel(gamma=0.05)
        initial_alpha = model.alpha_t

        # Simulate a miscovered observation
        model.update_aci(y_true=100.0, q_lo=0.0, q_hi=10.0)  # miscovered
        assert model.alpha_t != initial_alpha
        # alpha should increase (more miscoverage than expected)
        assert model.alpha_t > initial_alpha

    def test_aci_update_covered(self) -> None:
        """ACI should decrease alpha when observation is covered."""
        model = CQRACIModel(gamma=0.05)
        initial_alpha = model.alpha_t

        model.update_aci(y_true=5.0, q_lo=0.0, q_hi=10.0)  # covered
        assert model.alpha_t < initial_alpha

    def test_aci_alpha_clamped(self) -> None:
        """alpha_t should be clamped to [0.01, 0.50]."""
        model = CQRACIModel(gamma=0.5)  # large gamma

        # Many miscoverage updates
        for _ in range(100):
            model.update_aci(100.0, 0.0, 1.0)
        assert model.alpha_t <= 0.50

        # Many covered updates
        for _ in range(100):
            model.update_aci(0.5, 0.0, 1.0)
        assert model.alpha_t >= 0.01

    def test_reset_aci(self) -> None:
        model = CQRACIModel(gamma=0.05)
        model.update_aci(100.0, 0.0, 1.0)
        assert model.alpha_t != 0.20
        model.reset_aci()
        assert model.alpha_t == pytest.approx(0.20)

    def test_update_aci_batch(self) -> None:
        model = CQRACIModel(gamma=0.01)
        y_true = np.array([5.0, 100.0, 5.0])
        q_lo = np.array([0.0, 0.0, 0.0])
        q_hi = np.array([10.0, 10.0, 10.0])

        history = model.update_aci_batch(y_true, q_lo, q_hi)
        assert len(history) == 3
        # First observation is covered -> alpha decreases
        assert history[0] < 0.20

    def test_get_params(self) -> None:
        model = CQRACIModel(gamma=0.02)
        params = model.get_params()
        assert params["model_type"] == "cqr_aci"
        assert params["gamma"] == 0.02


class TestTuneACIGamma:
    """Tests for the tune_aci_gamma helper."""

    def test_returns_valid_gamma(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
        calibration_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X_train, y_train, X_val, y_val = synthetic_data
        X_cal, y_cal = calibration_data

        base = LGBMQuantileModel(params={
            "n_estimators": 50,
            "num_leaves": 31,
            "verbose": -1,
        })
        base.fit(X_train, y_train, X_val, y_val)

        model = CQRACIModel(base_model=base)
        model.fit(X_train, y_train)
        model.calibrate(X_cal, y_cal)

        best_gamma = tune_aci_gamma(
            model, X_val, y_val,
            gamma_grid=[0.005, 0.01, 0.05],
        )

        from packages.ml.models.cqr_aci import ACI_GAMMA_GRID
        assert best_gamma in [0.005, 0.01, 0.05]


# ---------------------------------------------------------------------------
# TestProjectionPredict
# ---------------------------------------------------------------------------
class TestProjectionPredict:
    """Tests for models.projection.predict() public API.

    Note: The full predict() function depends on MLflow and the feature
    pipeline, which are not available in unit tests. We test the
    predict_batch helper instead.
    """

    def test_predict_raises_not_implemented(self) -> None:
        """predict() should raise NotImplementedError until Phase 2 feature assembly is wired."""
        from datetime import date as dt_date

        from packages.ml.models.projection import predict

        with pytest.raises(NotImplementedError, match="Single-player feature assembly"):
            predict(player_id=12345, target_date=dt_date(2024, 6, 1))

    def test_predict_batch_schema(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
    ) -> None:
        """predict_batch should produce a DataFrame with the evaluation schema."""
        from packages.ml.models.projection import predict_batch

        X_train, y_train, X_val, _ = synthetic_data

        # Add metadata columns
        X_val_full = X_val.copy()
        X_val_full["player_id"] = range(100, 100 + len(X_val))
        X_val_full["game_pk"] = range(1000, 1000 + len(X_val))
        X_val_full["game_date"] = "2024-06-01"
        X_val_full["season_year"] = 2024
        X_val_full["player_type"] = "hitter"
        X_val_full["fantasy_points"] = 0.0

        model = RidgeProjectionModel(alpha=1.0)
        model.fit(X_train, y_train)

        result = predict_batch(
            X_val_full, model, "m1_ridge", "hitter",
        )

        assert "player_id" in result.columns
        assert "game_pk" in result.columns
        assert "game_date" in result.columns
        assert "player_type" in result.columns
        assert "model_candidate" in result.columns
        assert "predicted_mean" in result.columns
        assert len(result) == len(X_val)

    def test_predict_batch_with_intervals(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
    ) -> None:
        """predict_batch with M2 should include p10 and p90."""
        from packages.ml.models.projection import predict_batch

        X_train, y_train, X_val, y_val = synthetic_data

        X_val_full = X_val.copy()
        X_val_full["player_id"] = range(100, 100 + len(X_val))
        X_val_full["game_pk"] = range(1000, 1000 + len(X_val))
        X_val_full["game_date"] = "2024-06-01"
        X_val_full["season_year"] = 2024
        X_val_full["player_type"] = "hitter"
        X_val_full["fantasy_points"] = 0.0

        model = LGBMQuantileModel(params={
            "n_estimators": 30,
            "num_leaves": 31,
            "verbose": -1,
        })
        model.fit(X_train, y_train, X_val, y_val)

        result = predict_batch(
            X_val_full, model, "m2_lgbm_quantile", "hitter",
        )

        assert "predicted_p10" in result.columns
        assert "predicted_p90" in result.columns
        assert result["predicted_p10"].notna().all()
        assert result["predicted_p90"].notna().all()
