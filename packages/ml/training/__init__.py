"""Training module for player projection.

This module owns the training orchestration: temporal data splitting,
walk-forward cross-validation, hyperparameter tuning via Optuna,
and MLflow experiment logging.

The training module is the only code that touches MLflow for logging.
It consumes feature tables (read-only) and model classes, and produces
trained model artifacts.
"""
