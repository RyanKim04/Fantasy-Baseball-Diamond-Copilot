"""Unit tests for packages/ml/evaluation/.

Tests metrics, baselines, diagnostics, and report generation.
Owner: evaluator subagent (implements tests alongside the evaluation code).
"""

from __future__ import annotations


# --- Metrics tests ---


class TestRMSE:
    """Tests for evaluation.metrics.rmse."""

    pass  # To be implemented by evaluator subagent


class TestMAE:
    """Tests for evaluation.metrics.mae."""

    pass  # To be implemented by evaluator subagent


class TestSpearmanRho:
    """Tests for evaluation.metrics.spearman_rho."""

    pass  # To be implemented by evaluator subagent


class TestIntervalCoverage:
    """Tests for evaluation.metrics.interval_coverage."""

    pass  # To be implemented by evaluator subagent


class TestSharpness:
    """Tests for evaluation.metrics.sharpness."""

    pass  # To be implemented by evaluator subagent


class TestPinballLoss:
    """Tests for evaluation.metrics.pinball_loss."""

    pass  # To be implemented by evaluator subagent


class TestComputePopulationMetrics:
    """Tests for evaluation.metrics.compute_population_metrics."""

    pass  # To be implemented by evaluator subagent


# --- Baseline tests ---


class TestNaiveLastGame:
    """Tests for evaluation.baselines.predict_naive_last_game."""

    pass  # To be implemented by evaluator subagent


class TestTrailing7dMean:
    """Tests for evaluation.baselines.predict_trailing_7d_mean."""

    pass  # To be implemented by evaluator subagent


class TestSeasonToDateMean:
    """Tests for evaluation.baselines.predict_season_to_date_mean."""

    pass  # To be implemented by evaluator subagent


# --- Report tests ---


class TestGenerateReport:
    """Tests for evaluation.report.generate_report."""

    pass  # To be implemented by evaluator subagent


# --- Schema validation tests ---


class TestValidatePredictionDataframe:
    """Tests for evaluation.schemas.validate_prediction_dataframe."""

    pass  # To be implemented by evaluator subagent
