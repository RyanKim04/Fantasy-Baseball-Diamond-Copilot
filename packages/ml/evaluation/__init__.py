"""Evaluation module for player projection.

This module owns all model evaluation: metrics computation, baseline
comparisons, calibration diagnostics, and report generation.

The evaluation code is written and committed BEFORE any modeling code,
per the critical workflow rule in CLAUDE.md. The validation protocol
is pre-registered in validation_protocol.md.

The evaluator owns the single Test touch (validation_protocol.md section 11).
No other module may read Test-set targets.
"""
