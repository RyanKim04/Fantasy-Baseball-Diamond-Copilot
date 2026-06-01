"""Unit tests for packages/ml/evaluation/.

Tests metrics, baselines, calibration, diagnostics, schemas, and report generation.
All tests use synthetic data -- no real database dependencies.

Owner: evaluator subagent (implements tests alongside the evaluation code).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from packages.ml.evaluation.baselines import (
    predict_naive_last_game,
    predict_season_to_date_mean,
    predict_trailing_7d_mean,
)
from packages.ml.evaluation.calibration import (
    coverage_at_levels,
    coverage_by_predicted_bin,
    coverage_drift_over_time,
    pit_histogram,
    reliability_diagram,
)
from packages.ml.evaluation.diagnostics import residual_vs_predicted, width_vs_predicted
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
from packages.shared.schemas.ml import ModelCandidate, PlayerType

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def rng():
    """Fixed random number generator for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture()
def synthetic_actuals():
    """Synthetic actuals DataFrame for report tests."""
    n = 200
    rng = np.random.default_rng(123)
    dates = pd.date_range("2025-04-01", periods=50, freq="D")
    player_ids = list(range(1, 21))

    rows = []
    for i in range(n):
        pid = player_ids[i % len(player_ids)]
        gdate = dates[i % len(dates)]
        rows.append({
            "player_id": pid,
            "game_pk": 100000 + i,
            "game_date": gdate,
            "player_type": "hitter" if pid <= 10 else "pitcher",
            "fantasy_points": rng.normal(10, 5),
            "season_year": 2025,
        })
    return pd.DataFrame(rows)


def _make_predictions(
    actuals: pd.DataFrame,
    candidate: str,
    with_intervals: bool = True,
    noise_scale: float = 2.0,
) -> pd.DataFrame:
    """Helper to create a prediction DataFrame from actuals."""
    rng = np.random.default_rng(99)
    pred = actuals[["player_id", "game_pk", "game_date", "player_type"]].copy()
    pred["model_candidate"] = candidate
    pred["predicted_mean"] = actuals["fantasy_points"] + rng.normal(0, noise_scale, len(actuals))
    if with_intervals:
        pred["predicted_p10"] = pred["predicted_mean"] - 8.0
        pred["predicted_p90"] = pred["predicted_mean"] + 8.0
    return pred


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------


class TestRMSE:
    """Tests for evaluation.metrics.rmse."""

    def test_perfect_predictions(self):
        y = np.array([1.0, 2.0, 3.0])
        assert rmse(y, y) == 0.0

    def test_known_value(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 5.0])
        # MSE = (0 + 0 + 4) / 3
        expected = np.sqrt(4 / 3)
        assert abs(rmse(y_true, y_pred) - expected) < 1e-10

    def test_symmetric(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([3.0, 2.0, 1.0])
        # Errors: -2, 0, 2 -> MSE = (4 + 0 + 4)/3
        expected = np.sqrt(8 / 3)
        assert abs(rmse(y_true, y_pred) - expected) < 1e-10

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            rmse(np.array([]), np.array([]))

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            rmse(np.array([1.0, 2.0]), np.array([1.0]))

    def test_non_negative(self, rng):
        y_true = rng.normal(0, 10, 100)
        y_pred = rng.normal(0, 10, 100)
        assert rmse(y_true, y_pred) >= 0


class TestMAE:
    """Tests for evaluation.metrics.mae."""

    def test_perfect_predictions(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mae(y, y) == 0.0

    def test_known_value(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 5.0])
        # MAE = (1 + 1 + 2) / 3
        assert abs(mae(y_true, y_pred) - 4 / 3) < 1e-10

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            mae(np.array([]), np.array([]))

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            mae(np.array([1.0, 2.0]), np.array([1.0]))

    def test_non_negative(self, rng):
        y_true = rng.normal(0, 10, 100)
        y_pred = rng.normal(0, 10, 100)
        assert mae(y_true, y_pred) >= 0


class TestSpearmanRho:
    """Tests for evaluation.metrics.spearman_rho."""

    def test_perfect_positive_correlation(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert abs(spearman_rho(y, y) - 1.0) < 1e-10

    def test_perfect_negative_correlation(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        assert abs(spearman_rho(y_true, y_pred) - (-1.0)) < 1e-10

    def test_monotonic_transform_preserves_rank(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 4.0, 9.0, 16.0, 25.0])  # y^2
        assert abs(spearman_rho(y_true, y_pred) - 1.0) < 1e-10

    def test_too_few_observations_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            spearman_rho(np.array([1.0]), np.array([1.0]))

    def test_in_range(self, rng):
        y_true = rng.normal(0, 1, 50)
        y_pred = rng.normal(0, 1, 50)
        rho = spearman_rho(y_true, y_pred)
        assert -1.0 <= rho <= 1.0


class TestIntervalCoverage:
    """Tests for evaluation.metrics.interval_coverage."""

    def test_all_covered(self):
        y_true = np.array([1.0, 2.0, 3.0])
        lower = np.array([0.0, 1.0, 2.0])
        upper = np.array([2.0, 3.0, 4.0])
        assert interval_coverage(y_true, lower, upper) == 1.0

    def test_none_covered(self):
        y_true = np.array([5.0, 6.0, 7.0])
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([1.0, 1.0, 1.0])
        assert interval_coverage(y_true, lower, upper) == 0.0

    def test_partial_coverage(self):
        y_true = np.array([1.0, 5.0, 3.0, 7.0])
        lower = np.array([0.0, 0.0, 2.0, 0.0])
        upper = np.array([2.0, 3.0, 4.0, 3.0])
        # Covered: index 0, 2 -> 2/4 = 0.5
        assert interval_coverage(y_true, lower, upper) == 0.5

    def test_boundary_included(self):
        y_true = np.array([1.0, 3.0])
        lower = np.array([1.0, 2.0])
        upper = np.array([2.0, 3.0])
        # Both on boundary -> covered
        assert interval_coverage(y_true, lower, upper) == 1.0

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            interval_coverage(np.array([]), np.array([]), np.array([]))


class TestSharpness:
    """Tests for evaluation.metrics.sharpness."""

    def test_known_value(self):
        lower = np.array([0.0, 1.0, 2.0])
        upper = np.array([2.0, 4.0, 5.0])
        # Widths: 2, 3, 3 -> mean = 8/3
        assert abs(sharpness(lower, upper) - 8 / 3) < 1e-10

    def test_zero_width(self):
        lower = np.array([1.0, 2.0, 3.0])
        upper = np.array([1.0, 2.0, 3.0])
        assert sharpness(lower, upper) == 0.0

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            sharpness(np.array([]), np.array([]))


class TestPinballLoss:
    """Tests for evaluation.metrics.pinball_loss."""

    def test_perfect_at_median(self):
        y_true = np.array([5.0])
        y_pred = np.array([5.0])
        assert pinball_loss(y_true, y_pred, 0.5) == 0.0

    def test_median_symmetric(self):
        # At tau=0.5, over-predict by 2 and under-predict by 2 give same loss
        y_true = np.array([3.0, 7.0])
        y_pred = np.array([5.0, 5.0])
        # idx 0: diff = -2, tau-1 = -0.5 -> (-0.5)*(-2) = 1.0
        # idx 1: diff = 2, tau = 0.5 -> 0.5*2 = 1.0
        assert abs(pinball_loss(y_true, y_pred, 0.5) - 1.0) < 1e-10

    def test_asymmetric_at_tau_01(self):
        y_true = np.array([3.0])
        y_pred = np.array([5.0])
        # diff = -2, tau-1 = -0.9 -> (-0.9)*(-2) = 1.8
        assert abs(pinball_loss(y_true, y_pred, 0.1) - 1.8) < 1e-10

    def test_asymmetric_at_tau_09(self):
        y_true = np.array([7.0])
        y_pred = np.array([5.0])
        # diff = 2, tau = 0.9 -> 0.9*2 = 1.8
        assert abs(pinball_loss(y_true, y_pred, 0.9) - 1.8) < 1e-10

    def test_invalid_tau_raises(self):
        with pytest.raises(ValueError, match="tau"):
            pinball_loss(np.array([1.0]), np.array([1.0]), 0.0)
        with pytest.raises(ValueError, match="tau"):
            pinball_loss(np.array([1.0]), np.array([1.0]), 1.0)

    def test_non_negative(self, rng):
        y_true = rng.normal(0, 5, 100)
        y_pred = rng.normal(0, 5, 100)
        assert pinball_loss(y_true, y_pred, 0.5) >= 0

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            pinball_loss(np.array([]), np.array([]), 0.5)


class TestComputePopulationMetrics:
    """Tests for evaluation.metrics.compute_population_metrics."""

    def test_returns_correct_type(self, rng):
        n = 50
        y_true = rng.normal(10, 5, n)
        y_pred = y_true + rng.normal(0, 1, n)
        p10 = y_pred - 5.0
        p90 = y_pred + 5.0

        result = compute_population_metrics(
            y_true=y_true,
            y_pred_mean=y_pred,
            y_pred_p10=p10,
            y_pred_p90=p90,
            player_type="hitter",
            model_candidate="m1_ridge",
        )

        assert result.player_type == PlayerType.HITTER
        assert result.model_candidate == ModelCandidate.M1_RIDGE
        assert result.n_predictions == n
        assert result.rmse >= 0
        assert result.mae >= 0
        assert -1.0 <= result.spearman_rho <= 1.0
        assert result.coverage_80 is not None
        assert result.sharpness_mean_width is not None
        assert result.pinball_10 is not None
        assert result.pinball_50 is not None
        assert result.pinball_90 is not None

    def test_without_intervals(self, rng):
        n = 50
        y_true = rng.normal(10, 5, n)
        y_pred = y_true + rng.normal(0, 1, n)

        result = compute_population_metrics(
            y_true=y_true,
            y_pred_mean=y_pred,
            y_pred_p10=None,
            y_pred_p90=None,
            player_type="pitcher",
            model_candidate="m1_ridge",
        )

        assert result.coverage_80 is None
        assert result.sharpness_mean_width is None
        assert result.pinball_10 is None

    def test_sharpness_equals_width(self, rng):
        n = 50
        y_true = rng.normal(10, 5, n)
        y_pred = y_true + rng.normal(0, 1, n)
        width = 10.0
        p10 = y_pred - width / 2
        p90 = y_pred + width / 2

        result = compute_population_metrics(
            y_true=y_true,
            y_pred_mean=y_pred,
            y_pred_p10=p10,
            y_pred_p90=p90,
            player_type="hitter",
            model_candidate="m2_lgbm_quantile",
        )

        assert abs(result.sharpness_mean_width - width) < 1e-10


# ---------------------------------------------------------------------------
# Baseline tests
# ---------------------------------------------------------------------------


class TestNaiveLastGame:
    """Tests for evaluation.baselines.predict_naive_last_game."""

    def test_basic_shift(self):
        df = pd.DataFrame({
            "player_id": [1, 1, 1, 2, 2],
            "game_date": pd.to_datetime(["2025-04-01", "2025-04-02", "2025-04-03",
                                          "2025-04-01", "2025-04-02"]),
            "fantasy_points": [10.0, 15.0, 20.0, 5.0, 8.0],
        })
        result = predict_naive_last_game(df)
        # First game per player -> NaN
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[3])
        # Second game -> previous game's points
        assert result.iloc[1] == 10.0
        assert result.iloc[2] == 15.0
        assert result.iloc[4] == 5.0

    def test_preserves_index(self):
        df = pd.DataFrame(
            {
                "player_id": [1, 1, 1],
                "game_date": pd.to_datetime(["2025-04-01", "2025-04-02", "2025-04-03"]),
                "fantasy_points": [10.0, 15.0, 20.0],
            },
            index=[100, 200, 300],
        )
        result = predict_naive_last_game(df)
        assert list(result.index) == [100, 200, 300]

    def test_single_player_single_game(self):
        df = pd.DataFrame({
            "player_id": [1],
            "game_date": pd.to_datetime(["2025-04-01"]),
            "fantasy_points": [10.0],
        })
        result = predict_naive_last_game(df)
        assert pd.isna(result.iloc[0])


class TestTrailing7dMean:
    """Tests for evaluation.baselines.predict_trailing_7d_mean."""

    def test_within_window(self):
        df = pd.DataFrame({
            "player_id": [1, 1, 1, 1],
            "game_date": pd.to_datetime([
                "2025-04-01", "2025-04-03", "2025-04-05", "2025-04-07"
            ]),
            "fantasy_points": [10.0, 20.0, 30.0, 40.0],
        })
        result = predict_trailing_7d_mean(df)
        # First game -> NaN (no prior games)
        assert pd.isna(result.iloc[0])
        # Second game: only 2025-04-01 is within 7d before 04-03
        assert result.iloc[1] == 10.0
        # Third game: 04-01 and 04-03 within 7d before 04-05
        assert result.iloc[2] == 15.0  # mean(10, 20)
        # Fourth game: 04-01, 04-03, 04-05 all within 7d before 04-07
        assert abs(result.iloc[3] - 20.0) < 1e-10  # mean(10, 20, 30)

    def test_outside_window_gives_nan(self):
        df = pd.DataFrame({
            "player_id": [1, 1],
            "game_date": pd.to_datetime(["2025-04-01", "2025-04-20"]),
            "fantasy_points": [10.0, 20.0],
        })
        result = predict_trailing_7d_mean(df)
        # Second game: 04-01 is more than 7 days before 04-20 -> NaN
        assert pd.isna(result.iloc[1])

    def test_multiple_players_independent(self):
        df = pd.DataFrame({
            "player_id": [1, 2, 1, 2],
            "game_date": pd.to_datetime([
                "2025-04-01", "2025-04-01", "2025-04-03", "2025-04-03"
            ]),
            "fantasy_points": [10.0, 100.0, 20.0, 200.0],
        })
        result = predict_trailing_7d_mean(df)
        # Player 1's second game: mean of 10.0
        assert result.iloc[2] == 10.0
        # Player 2's second game: mean of 100.0
        assert result.iloc[3] == 100.0


class TestSeasonToDateMean:
    """Tests for evaluation.baselines.predict_season_to_date_mean."""

    def test_basic_expanding_mean(self):
        df = pd.DataFrame({
            "player_id": [1, 1, 1],
            "game_date": pd.to_datetime(["2025-04-01", "2025-04-05", "2025-04-10"]),
            "fantasy_points": [10.0, 20.0, 30.0],
            "season_year": [2025, 2025, 2025],
        })
        result = predict_season_to_date_mean(df)
        assert pd.isna(result.iloc[0])
        assert result.iloc[1] == 10.0  # mean of [10]
        assert result.iloc[2] == 15.0  # mean of [10, 20]

    def test_season_boundary_resets(self):
        df = pd.DataFrame({
            "player_id": [1, 1, 1, 1],
            "game_date": pd.to_datetime([
                "2024-09-01", "2024-09-15", "2025-04-01", "2025-04-05"
            ]),
            "fantasy_points": [100.0, 200.0, 10.0, 20.0],
            "season_year": [2024, 2024, 2025, 2025],
        })
        result = predict_season_to_date_mean(df)
        # First game of 2024 -> NaN
        assert pd.isna(result.iloc[0])
        # Second game of 2024 -> mean of [100]
        assert result.iloc[1] == 100.0
        # First game of 2025 -> NaN (new season)
        assert pd.isna(result.iloc[2])
        # Second game of 2025 -> mean of [10]
        assert result.iloc[3] == 10.0

    def test_multiple_players(self):
        df = pd.DataFrame({
            "player_id": [1, 2, 1, 2],
            "game_date": pd.to_datetime([
                "2025-04-01", "2025-04-01", "2025-04-05", "2025-04-05"
            ]),
            "fantasy_points": [10.0, 50.0, 20.0, 60.0],
            "season_year": [2025, 2025, 2025, 2025],
        })
        result = predict_season_to_date_mean(df)
        # Player 1 second game -> mean([10]) = 10
        assert result.iloc[2] == 10.0
        # Player 2 second game -> mean([50]) = 50
        assert result.iloc[3] == 50.0


# ---------------------------------------------------------------------------
# Calibration tests
# ---------------------------------------------------------------------------


class TestReliabilityDiagram:
    """Tests for calibration.reliability_diagram."""

    def test_returns_correct_keys(self, rng):
        y_true = rng.normal(0, 1, 100)
        y_pred = y_true + rng.normal(0, 0.1, 100)
        result = reliability_diagram(y_true, y_pred, n_bins=5)
        assert "bin_centers" in result
        assert "bin_means_pred" in result
        assert "bin_means_actual" in result
        assert len(result["bin_centers"]) == 5

    def test_perfect_predictions_on_diagonal(self, rng):
        y = rng.normal(0, 1, 100)
        result = reliability_diagram(y, y, n_bins=5)
        for pred, actual in zip(result["bin_means_pred"], result["bin_means_actual"], strict=True):
            assert abs(pred - actual) < 1e-10

    def test_saves_plot(self, rng, tmp_path):
        y_true = rng.normal(0, 1, 50)
        y_pred = y_true + rng.normal(0, 0.5, 50)
        output = tmp_path / "reliability.png"
        reliability_diagram(y_true, y_pred, output_path=output)
        assert output.exists()

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            reliability_diagram(np.array([]), np.array([]))


class TestCoverageAtLevels:
    """Tests for calibration.coverage_at_levels."""

    def test_always_includes_80(self, rng):
        n = 100
        y_true = rng.normal(0, 1, n)
        lower_10 = y_true - 5.0
        upper_90 = y_true + 5.0
        result = coverage_at_levels(y_true, lower_10, upper_90)
        assert 80 in result
        assert result[80] == 1.0  # very wide intervals -> 100% coverage

    def test_multiple_levels(self, rng):
        n = 100
        y_true = rng.normal(0, 1, n)
        lower_10 = y_true - 5.0
        upper_90 = y_true + 5.0
        lower_25 = y_true - 3.0
        upper_75 = y_true + 3.0
        result = coverage_at_levels(
            y_true, lower_10, upper_90,
            lower_25=lower_25, upper_75=upper_75,
        )
        assert 80 in result
        assert 50 in result


class TestCoverageByPredictedBin:
    """Tests for calibration.coverage_by_predicted_bin."""

    def test_returns_correct_keys(self, rng):
        n = 100
        y_true = rng.normal(0, 1, n)
        y_pred = y_true + rng.normal(0, 0.1, n)
        lower = y_pred - 5.0
        upper = y_pred + 5.0
        result = coverage_by_predicted_bin(y_true, y_pred, lower, upper, n_bins=5)
        assert "bin_centers" in result
        assert "coverage" in result
        assert "n_samples" in result
        assert len(result["bin_centers"]) == 5

    def test_full_coverage(self, rng):
        n = 100
        y_true = rng.normal(0, 1, n)
        y_pred = y_true
        lower = y_pred - 100.0
        upper = y_pred + 100.0
        result = coverage_by_predicted_bin(y_true, y_pred, lower, upper, n_bins=5)
        for cov in result["coverage"]:
            assert cov == 1.0


class TestCoverageDriftOverTime:
    """Tests for calibration.coverage_drift_over_time."""

    def test_returns_correct_bins(self, rng):
        n = 100
        y_true = rng.normal(0, 1, n)
        lower = y_true - 5.0
        upper = y_true + 5.0
        dates = pd.date_range("2025-04-01", periods=n, freq="D").values
        result = coverage_drift_over_time(y_true, lower, upper, dates, n_bins=4)
        assert "bin_labels" in result
        assert "coverage" in result
        assert "n_samples" in result
        assert len(result["bin_labels"]) == 4


class TestPITHistogram:
    """Tests for calibration.pit_histogram."""

    def test_returns_correct_keys(self, rng):
        n = 100
        y_true = rng.normal(0, 1, n)
        p_mean = y_true + rng.normal(0, 0.1, n)
        p10 = p_mean - 3.0
        p90 = p_mean + 3.0
        result = pit_histogram(y_true, p_mean, p10, p90, n_bins=10)
        assert "bin_edges" in result
        assert "counts" in result
        assert len(result["bin_edges"]) == 11  # n_bins + 1 edges
        assert len(result["counts"]) == 10

    def test_counts_sum_to_one(self, rng):
        n = 200
        y_true = rng.normal(0, 1, n)
        p_mean = y_true + rng.normal(0, 0.1, n)
        p10 = p_mean - 3.0
        p90 = p_mean + 3.0
        result = pit_histogram(y_true, p_mean, p10, p90)
        assert abs(sum(result["counts"]) - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# Diagnostics tests
# ---------------------------------------------------------------------------


class TestWidthVsPredicted:
    """Tests for diagnostics.width_vs_predicted."""

    def test_returns_correct_keys(self, rng):
        n = 100
        y_pred = rng.normal(10, 5, n)
        lower = y_pred - 3.0
        upper = y_pred + 3.0
        result = width_vs_predicted(y_pred, lower, upper, n_bins=5)
        assert "predicted_means" in result
        assert "widths" in result
        assert len(result["predicted_means"]) == 5

    def test_constant_width(self, rng):
        n = 100
        y_pred = rng.normal(10, 5, n)
        lower = y_pred - 4.0
        upper = y_pred + 4.0
        result = width_vs_predicted(y_pred, lower, upper, n_bins=5)
        for w in result["widths"]:
            assert abs(w - 8.0) < 1e-10

    def test_saves_plot(self, rng, tmp_path):
        n = 50
        y_pred = rng.normal(10, 5, n)
        lower = y_pred - 3.0
        upper = y_pred + 3.0
        output = tmp_path / "width_vs_pred.png"
        width_vs_predicted(y_pred, lower, upper, output_path=output)
        assert output.exists()

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            width_vs_predicted(np.array([]), np.array([]), np.array([]))


class TestResidualVsPredicted:
    """Tests for diagnostics.residual_vs_predicted."""

    def test_returns_correct_keys(self, rng):
        n = 50
        y_true = rng.normal(10, 5, n)
        y_pred = y_true + rng.normal(0, 1, n)
        result = residual_vs_predicted(y_true, y_pred)
        assert "predicted" in result
        assert "residuals" in result
        assert len(result["predicted"]) == n

    def test_residuals_correct(self):
        y_true = np.array([3.0, 5.0, 7.0])
        y_pred = np.array([2.0, 6.0, 7.0])
        result = residual_vs_predicted(y_true, y_pred)
        assert result["residuals"] == [1.0, -1.0, 0.0]

    def test_saves_plot(self, rng, tmp_path):
        n = 50
        y_true = rng.normal(10, 5, n)
        y_pred = y_true + rng.normal(0, 1, n)
        output = tmp_path / "residual.png"
        residual_vs_predicted(y_true, y_pred, output_path=output)
        assert output.exists()


# ---------------------------------------------------------------------------
# Schema validation tests
# ---------------------------------------------------------------------------


class TestValidatePredictionDataframe:
    """Tests for evaluation.schemas.validate_prediction_dataframe."""

    def test_valid_dataframe_no_errors(self):
        df = pd.DataFrame({
            "player_id": [1, 2],
            "game_pk": [100, 200],
            "game_date": ["2025-04-01", "2025-04-01"],
            "player_type": ["hitter", "pitcher"],
            "model_candidate": ["m1_ridge", "m1_ridge"],
            "predicted_mean": [10.0, 15.0],
        })
        errors = validate_prediction_dataframe(df)
        assert errors == []

    def test_valid_with_intervals(self):
        df = pd.DataFrame({
            "player_id": [1],
            "game_pk": [100],
            "game_date": ["2025-04-01"],
            "player_type": ["hitter"],
            "model_candidate": ["m2_lgbm_quantile"],
            "predicted_mean": [10.0],
            "predicted_p10": [5.0],
            "predicted_p90": [15.0],
        })
        errors = validate_prediction_dataframe(df)
        assert errors == []

    def test_missing_required_column(self):
        df = pd.DataFrame({
            "player_id": [1],
            "game_pk": [100],
            "game_date": ["2025-04-01"],
            "player_type": ["hitter"],
            "predicted_mean": [10.0],
            # Missing model_candidate
        })
        errors = validate_prediction_dataframe(df)
        assert any("model_candidate" in e for e in errors)

    def test_null_predicted_mean(self):
        df = pd.DataFrame({
            "player_id": [1, 2],
            "game_pk": [100, 200],
            "game_date": ["2025-04-01", "2025-04-02"],
            "player_type": ["hitter", "hitter"],
            "model_candidate": ["m1_ridge", "m1_ridge"],
            "predicted_mean": [10.0, None],
        })
        errors = validate_prediction_dataframe(df)
        assert any("null" in e for e in errors)

    def test_invalid_player_type(self):
        df = pd.DataFrame({
            "player_id": [1],
            "game_pk": [100],
            "game_date": ["2025-04-01"],
            "player_type": ["catcher"],
            "model_candidate": ["m1_ridge"],
            "predicted_mean": [10.0],
        })
        errors = validate_prediction_dataframe(df)
        assert any("player_type" in e for e in errors)

    def test_invalid_model_candidate(self):
        df = pd.DataFrame({
            "player_id": [1],
            "game_pk": [100],
            "game_date": ["2025-04-01"],
            "player_type": ["hitter"],
            "model_candidate": ["xgboost"],
            "predicted_mean": [10.0],
        })
        errors = validate_prediction_dataframe(df)
        assert any("model_candidate" in e for e in errors)

    def test_duplicate_keys(self):
        df = pd.DataFrame({
            "player_id": [1, 1],
            "game_pk": [100, 100],
            "game_date": ["2025-04-01", "2025-04-01"],
            "player_type": ["hitter", "hitter"],
            "model_candidate": ["m1_ridge", "m1_ridge"],
            "predicted_mean": [10.0, 12.0],
        })
        errors = validate_prediction_dataframe(df)
        assert any("Duplicate" in e for e in errors)

    def test_inverted_intervals(self):
        df = pd.DataFrame({
            "player_id": [1],
            "game_pk": [100],
            "game_date": ["2025-04-01"],
            "player_type": ["hitter"],
            "model_candidate": ["m2_lgbm_quantile"],
            "predicted_mean": [10.0],
            "predicted_p10": [15.0],  # > p90
            "predicted_p90": [5.0],
        })
        errors = validate_prediction_dataframe(df)
        assert any("inverted" in e for e in errors)

    def test_end_to_end_with_synthetic_data(self, rng):
        """Verify validate_prediction_dataframe works end-to-end per Step 1.6."""
        n = 100
        df = pd.DataFrame({
            "player_id": rng.integers(1, 20, n),
            "game_pk": np.arange(100000, 100000 + n),
            "game_date": pd.date_range("2025-04-01", periods=n, freq="D"),
            "player_type": ["hitter" if i < 50 else "pitcher" for i in range(n)],
            "model_candidate": ["m2_lgbm_quantile"] * n,
            "predicted_mean": rng.normal(10, 5, n),
            "predicted_p10": rng.normal(5, 3, n),
            "predicted_p90": rng.normal(15, 3, n),
        })
        # Fix any inverted intervals
        mask = df["predicted_p10"] > df["predicted_p90"]
        df.loc[mask, ["predicted_p10", "predicted_p90"]] = df.loc[
            mask, ["predicted_p90", "predicted_p10"]
        ].values
        errors = validate_prediction_dataframe(df)
        assert errors == [], f"Unexpected validation errors: {errors}"


# ---------------------------------------------------------------------------
# Report tests
# ---------------------------------------------------------------------------


class TestGenerateReport:
    """Tests for evaluation.report.generate_report."""

    def test_generates_report_file(self, synthetic_actuals, tmp_path):
        """generate_report produces an evaluation_report.md file."""
        preds = {
            ModelCandidate.M1_RIDGE: _make_predictions(
                synthetic_actuals, "m1_ridge", with_intervals=False,
            ),
        }
        report_path = generate_report(
            predictions=preds,
            actuals=synthetic_actuals,
            output_dir=tmp_path,
        )
        assert report_path.exists()
        assert report_path.name == "evaluation_report.md"

    def test_report_contains_headline_metrics(self, synthetic_actuals, tmp_path):
        """Report includes RMSE and MAE in the headline table."""
        preds = {
            ModelCandidate.M1_RIDGE: _make_predictions(
                synthetic_actuals, "m1_ridge", with_intervals=False,
            ),
        }
        report_path = generate_report(
            predictions=preds,
            actuals=synthetic_actuals,
            output_dir=tmp_path,
        )
        content = report_path.read_text()
        assert "RMSE" in content
        assert "MAE" in content
        assert "Spearman" in content

    def test_report_with_intervals(self, synthetic_actuals, tmp_path):
        """Report includes coverage metrics when intervals are provided."""
        preds = {
            ModelCandidate.M2_LGBM_QUANTILE: _make_predictions(
                synthetic_actuals, "m2_lgbm_quantile", with_intervals=True,
            ),
        }
        report_path = generate_report(
            predictions=preds,
            actuals=synthetic_actuals,
            output_dir=tmp_path,
        )
        content = report_path.read_text()
        assert "Coverage" in content

    def test_report_contains_baseline_comparisons(self, synthetic_actuals, tmp_path):
        """Report includes baseline comparison table."""
        preds = {
            ModelCandidate.M1_RIDGE: _make_predictions(
                synthetic_actuals, "m1_ridge", with_intervals=False,
            ),
        }
        report_path = generate_report(
            predictions=preds,
            actuals=synthetic_actuals,
            output_dir=tmp_path,
        )
        content = report_path.read_text()
        assert "Baseline" in content
        assert "Improvement" in content

    def test_report_multiple_candidates(self, synthetic_actuals, tmp_path):
        """Report handles multiple model candidates side by side."""
        preds = {
            ModelCandidate.M1_RIDGE: _make_predictions(
                synthetic_actuals, "m1_ridge", with_intervals=False,
            ),
            ModelCandidate.M2_LGBM_QUANTILE: _make_predictions(
                synthetic_actuals, "m2_lgbm_quantile", with_intervals=True,
            ),
        }
        report_path = generate_report(
            predictions=preds,
            actuals=synthetic_actuals,
            output_dir=tmp_path,
        )
        content = report_path.read_text()
        assert "m1_ridge" in content
        assert "m2_lgbm_quantile" in content

    def test_invalid_predictions_raises(self, synthetic_actuals, tmp_path):
        """generate_report raises if predictions fail schema validation."""
        bad_preds = synthetic_actuals[["player_id", "game_pk"]].copy()
        bad_preds["predicted_mean"] = 10.0
        # Missing required columns
        with pytest.raises(ValueError, match="failed validation"):
            generate_report(
                predictions={ModelCandidate.M1_RIDGE: bad_preds},
                actuals=synthetic_actuals,
                output_dir=tmp_path,
            )

    def test_report_includes_mlflow_run_ids(self, synthetic_actuals, tmp_path):
        """Report includes MLflow run IDs when provided."""
        preds = {
            ModelCandidate.M1_RIDGE: _make_predictions(
                synthetic_actuals, "m1_ridge", with_intervals=False,
            ),
        }
        report_path = generate_report(
            predictions=preds,
            actuals=synthetic_actuals,
            output_dir=tmp_path,
            mlflow_run_ids={ModelCandidate.M1_RIDGE: "abc123"},
        )
        content = report_path.read_text()
        assert "abc123" in content

    def test_report_selection_rule(self, synthetic_actuals, tmp_path):
        """Report applies and documents the selection rule."""
        preds = {
            ModelCandidate.M1_RIDGE: _make_predictions(
                synthetic_actuals, "m1_ridge", with_intervals=False,
            ),
        }
        report_path = generate_report(
            predictions=preds,
            actuals=synthetic_actuals,
            output_dir=tmp_path,
        )
        content = report_path.read_text()
        assert "Selection Rule" in content


# ---------------------------------------------------------------------------
# Slices tests
# ---------------------------------------------------------------------------


class TestSlices:
    """Tests for evaluation.slices module."""

    def test_filter_eligible_hitters(self):
        from packages.ml.evaluation.slices import filter_eligible_hitters

        season_stats = pd.DataFrame({
            "player_id": [1, 2, 3],
            "pa": [150, 50, 200],
        })
        df = pd.DataFrame({
            "player_id": [1, 1, 2, 2, 3, 3],
            "predicted_mean": [10, 12, 8, 9, 15, 16],
        })
        result = filter_eligible_hitters(df, season_stats=season_stats)
        # Only players 1 and 3 have >= 100 PA
        assert set(result["player_id"].unique()) == {1, 3}
        assert len(result) == 4

    def test_filter_eligible_pitchers(self):
        from packages.ml.evaluation.slices import filter_eligible_pitchers

        season_stats = pd.DataFrame({
            "player_id": [10, 11, 12],
            "ip": [40, 20, 100],
        })
        df = pd.DataFrame({
            "player_id": [10, 11, 12],
            "predicted_mean": [5, 6, 7],
        })
        result = filter_eligible_pitchers(df, season_stats=season_stats)
        assert set(result["player_id"].unique()) == {10, 12}

    def test_slice_by_position(self):
        from packages.ml.evaluation.slices import slice_by_position

        df = pd.DataFrame({
            "player_id": [1, 2, 3, 4],
            "position": ["1B", "SS", "1B", "OF"],
        })
        slices = slice_by_position(df)
        assert "1B" in slices
        assert "SS" in slices
        assert "OF" in slices
        assert len(slices["1B"]) == 2

    def test_slice_by_position_no_col(self):
        from packages.ml.evaluation.slices import slice_by_position

        df = pd.DataFrame({"player_id": [1, 2]})
        slices = slice_by_position(df)
        assert "all" in slices

    def test_slice_by_experience(self):
        from packages.ml.evaluation.slices import slice_by_experience

        df = pd.DataFrame({
            "player_id": [1, 2, 3],
            "mlb_debut_year": [2025, 2020, 2023],
            "season_year": [2025, 2025, 2025],
        })
        slices = slice_by_experience(df)
        assert "rookie" in slices
        assert len(slices["rookie"]) == 1  # player 1
        assert "veteran" in slices
        assert len(slices["veteran"]) == 1  # player 2

    def test_slice_by_playing_time_tertile(self):
        from packages.ml.evaluation.slices import slice_by_playing_time_tertile

        # 6 players with varying game counts (via row count)
        df = pd.DataFrame({
            "player_id": [1] * 10 + [2] * 20 + [3] * 30 + [4] * 40 + [5] * 50 + [6] * 60,
        })
        slices = slice_by_playing_time_tertile(df)
        assert len(slices) > 0
        total_rows = sum(len(s) for s in slices.values())
        assert total_rows == len(df)
