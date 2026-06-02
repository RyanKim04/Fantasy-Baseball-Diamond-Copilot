"""Evaluation report generator.

Produces evaluation_report.md per validation_protocol.md section 7.4.
The single Test touch runs through this module.

Owner: evaluator subagent.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from packages.ml.evaluation.baselines import (
    predict_naive_last_game,
    predict_season_to_date_mean,
    predict_trailing_7d_mean,
)
from packages.ml.evaluation.calibration import (
    coverage_at_levels,
    coverage_drift_over_time,
    pit_histogram,
    reliability_diagram,
)
from packages.ml.evaluation.diagnostics import residual_vs_predicted, width_vs_predicted
from packages.ml.evaluation.metrics import compute_population_metrics, rmse
from packages.ml.evaluation.schemas import validate_prediction_dataframe
from packages.ml.evaluation.slices import (
    filter_eligible_hitters,
    filter_eligible_pitchers,
    slice_by_experience,
    slice_by_playing_time_tertile,
    slice_by_position,
)
from packages.shared.schemas.ml import (
    BaselineComparison,
    ModelCandidate,
    PlayerType,
    PopulationMetrics,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASELINE_IMPROVEMENT_THRESHOLD: float = 10.0  # percent
COVERAGE_ACCEPTABLE_RANGE: tuple[float, float] = (0.75, 0.85)


def generate_report(
    predictions: dict[ModelCandidate, pd.DataFrame],
    actuals: pd.DataFrame,
    output_dir: Path,
    mlflow_run_ids: dict[ModelCandidate, str] | None = None,
    report_timestamp: datetime | None = None,
    season_stats: pd.DataFrame | None = None,
) -> Path:
    """Generate the full evaluation report for all model candidates.

    This is the single entry point for the Test touch. It:
    1. Validates prediction DataFrames.
    2. Joins predictions with actuals.
    3. Computes headline metrics for each candidate x population.
    4. Computes baselines on the same population.
    5. Applies the baseline comparison threshold (>=10% RMSE improvement).
    6. Generates calibration diagnostics (for models with intervals).
    7. Writes evaluation_report.md and saves diagnostic plots.

    Parameters
    ----------
    predictions : dict[ModelCandidate, pd.DataFrame]
        Prediction DataFrames per model candidate.
        Must conform to schemas.PREDICTION_REQUIRED_COLUMNS.
    actuals : pd.DataFrame
        Actual fantasy points. Must have columns: player_id, game_pk,
        game_date, player_type, fantasy_points. Optionally: season_year.
    output_dir : Path
        Directory to write evaluation_report.md and plots.
    mlflow_run_ids : dict[ModelCandidate, str] | None
        MLflow run IDs for traceability.
    report_timestamp : datetime | None
        Fixed timestamp for deterministic reports (protocol section 7.4).
        If None, ``datetime.now()`` is used.
    season_stats : pd.DataFrame | None
        Per-player season totals for eligibility filtering (protocol
        section 6). Must contain ``player_id``, ``pa`` (for hitters),
        ``ip`` (for pitchers). If None, eligibility filtering is skipped
        (all players included).

    Returns
    -------
    Path
        Path to the generated evaluation_report.md.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: Validate prediction DataFrames ----
    for candidate, pred_df in predictions.items():
        errors = validate_prediction_dataframe(pred_df)
        if errors:
            raise ValueError(
                f"Prediction DataFrame for {candidate.value} failed validation: {errors}"
            )

    # ---- Step 2: Join predictions with actuals ----
    joined: dict[ModelCandidate, pd.DataFrame] = {}
    for candidate, pred_df in predictions.items():
        merged = pred_df.merge(
            actuals[["player_id", "game_pk", "game_date", "player_type", "fantasy_points"]],
            on=["player_id", "game_pk"],
            how="inner",
            suffixes=("", "_actual"),
        )
        # Use actual player_type if there's a conflict
        if "player_type_actual" in merged.columns:
            merged["player_type"] = merged["player_type_actual"]
            merged = merged.drop(columns=["player_type_actual"])
        # Use actual game_date if there's a conflict
        if "game_date_actual" in merged.columns:
            merged["game_date"] = merged["game_date_actual"]
            merged = merged.drop(columns=["game_date_actual"])
        # Rename to actual_points for consistency
        merged["actual_points"] = merged["fantasy_points"]
        joined[candidate] = merged

    # ---- Step 3: Apply eligibility filtering (protocol section 6) ----
    # Build eligible populations per candidate x player_type.
    # Headline metrics use eligible-only; full-pop metrics are also reported.
    eligible_joined: dict[ModelCandidate, pd.DataFrame] = {}
    for candidate, df in joined.items():
        hitters = df[df["player_type"] == PlayerType.HITTER.value]
        pitchers = df[df["player_type"] == PlayerType.PITCHER.value]

        eligible_hitters = filter_eligible_hitters(hitters, season_stats=season_stats)
        eligible_pitchers = filter_eligible_pitchers(pitchers, season_stats=season_stats)
        eligible_joined[candidate] = pd.concat(
            [eligible_hitters, eligible_pitchers], ignore_index=True
        )

    # ---- Step 4: Compute headline metrics on ELIGIBLE population ----
    headline_metrics: list[PopulationMetrics] = []
    for candidate, df in eligible_joined.items():
        for pt in [PlayerType.HITTER, PlayerType.PITCHER]:
            pop = df[df["player_type"] == pt.value]
            if len(pop) == 0:
                continue
            headline_metrics.append(
                _compute_metrics_for_pop(pop, pt, candidate)
            )

    # ---- Step 4b: Compute full-population metrics for reference ----
    full_pop_metrics: list[PopulationMetrics] = []
    for candidate, df in joined.items():
        for pt in [PlayerType.HITTER, PlayerType.PITCHER]:
            pop = df[df["player_type"] == pt.value]
            if len(pop) == 0:
                continue
            full_pop_metrics.append(
                _compute_metrics_for_pop(pop, pt, candidate)
            )

    # ---- Step 5: Multi-level coverage (protocol section 9 item 2) ----
    # {(candidate, player_type): {nominal_level: empirical_coverage}}
    multi_level_coverage: dict[tuple[str, str], dict[int, float]] = {}
    for candidate, df in eligible_joined.items():
        has_intervals = (
            "predicted_p10" in df.columns
            and "predicted_p90" in df.columns
            and df["predicted_p10"].notna().any()
        )
        if not has_intervals:
            continue
        for pt in [PlayerType.HITTER, PlayerType.PITCHER]:
            pop = df[df["player_type"] == pt.value]
            if len(pop) == 0 or not pop["predicted_p10"].notna().any():
                continue
            coverage_dict = coverage_at_levels(
                y_true=pop["actual_points"].values,
                lower_10=pop["predicted_p10"].values,
                upper_90=pop["predicted_p90"].values,
                # Additional quantile columns are optional; pass if present
                lower_25=pop["predicted_p25"].values if "predicted_p25" in pop.columns and pop["predicted_p25"].notna().any() else None,
                upper_75=pop["predicted_p75"].values if "predicted_p75" in pop.columns and pop["predicted_p75"].notna().any() else None,
                lower_15=pop["predicted_p15"].values if "predicted_p15" in pop.columns and pop["predicted_p15"].notna().any() else None,
                upper_85=pop["predicted_p85"].values if "predicted_p85" in pop.columns and pop["predicted_p85"].notna().any() else None,
                lower_05=pop["predicted_p05"].values if "predicted_p05" in pop.columns and pop["predicted_p05"].notna().any() else None,
                upper_95=pop["predicted_p95"].values if "predicted_p95" in pop.columns and pop["predicted_p95"].notna().any() else None,
            )
            multi_level_coverage[(candidate.value, pt.value)] = coverage_dict

    # ---- Step 6: Compute baselines on eligible population ----
    all_baseline_comparisons: list[BaselineComparison] = []
    for candidate, df in eligible_joined.items():
        for pt in [PlayerType.HITTER, PlayerType.PITCHER]:
            pop = df[df["player_type"] == pt.value].copy()
            if len(pop) == 0:
                continue

            # Ensure season_year exists for season-to-date baseline
            if "season_year" not in pop.columns:
                pop["season_year"] = pd.to_datetime(pop["game_date"]).dt.year

            # Compute baseline predictions
            naive_preds = predict_naive_last_game(pop, target_col="actual_points")
            trailing_preds = predict_trailing_7d_mean(pop, target_col="actual_points")
            stdate_preds = predict_season_to_date_mean(pop, target_col="actual_points")

            # Drop NaN rows (baselines have NaN for first games)
            valid_mask = naive_preds.notna() & trailing_preds.notna() & stdate_preds.notna()
            if valid_mask.sum() == 0:
                continue

            actuals_valid = pop.loc[valid_mask, "actual_points"].values
            model_preds_valid = pop.loc[valid_mask, "predicted_mean"].values

            model_rmse_val = rmse(actuals_valid, model_preds_valid)
            naive_rmse = rmse(actuals_valid, naive_preds[valid_mask].values)
            trailing_rmse = rmse(actuals_valid, trailing_preds[valid_mask].values)
            stdate_rmse = rmse(actuals_valid, stdate_preds[valid_mask].values)

            best_baseline = min(naive_rmse, trailing_rmse, stdate_rmse)
            improvement = (1 - model_rmse_val / best_baseline) * 100 if best_baseline > 0 else 0.0

            comparison = BaselineComparison(
                player_type=pt,
                model_candidate=candidate,
                model_rmse=model_rmse_val,
                baseline_naive_last_game_rmse=naive_rmse,
                baseline_trailing_7d_rmse=trailing_rmse,
                baseline_season_to_date_rmse=stdate_rmse,
                best_baseline_rmse=best_baseline,
                improvement_pct=improvement,
                passes_threshold=improvement >= BASELINE_IMPROVEMENT_THRESHOLD,
            )
            all_baseline_comparisons.append(comparison)

    # ---- Step 7: Slice analysis (protocol sections 6 and 7.3) ----
    # slice_results: list of (candidate, player_type, slice_name, PopulationMetrics)
    slice_results: list[tuple[str, str, str, PopulationMetrics]] = []
    slice_generators = [
        ("position", slice_by_position),
        ("experience", slice_by_experience),
        ("playing_time", slice_by_playing_time_tertile),
    ]
    for candidate, df in eligible_joined.items():
        for pt in [PlayerType.HITTER, PlayerType.PITCHER]:
            pop = df[df["player_type"] == pt.value]
            if len(pop) == 0:
                continue
            for slice_name, slicer in slice_generators:
                slices = slicer(pop)
                for label, slice_df in slices.items():
                    if len(slice_df) < 2:
                        continue
                    metrics = _compute_metrics_for_pop(slice_df, pt, candidate)
                    slice_results.append(
                        (candidate.value, pt.value, f"{slice_name}/{label}", metrics)
                    )

    # ---- Step 8: Generate calibration diagnostics ----
    for candidate, df in eligible_joined.items():
        has_intervals = (
            "predicted_p10" in df.columns
            and "predicted_p90" in df.columns
            and df["predicted_p10"].notna().any()
        )
        if not has_intervals:
            continue

        for pt in [PlayerType.HITTER, PlayerType.PITCHER]:
            pop = df[df["player_type"] == pt.value]
            if len(pop) == 0:
                continue

            prefix = f"{candidate.value}_{pt.value}"

            # Reliability diagram
            reliability_diagram(
                y_true=pop["actual_points"].values,
                y_pred=pop["predicted_mean"].values,
                output_path=plots_dir / f"{prefix}_reliability.png",
            )

            # PIT histogram
            if pop["predicted_p10"].notna().all() and pop["predicted_p90"].notna().all():
                pit_histogram(
                    y_true=pop["actual_points"].values,
                    predicted_mean=pop["predicted_mean"].values,
                    predicted_p10=pop["predicted_p10"].values,
                    predicted_p90=pop["predicted_p90"].values,
                    output_path=plots_dir / f"{prefix}_pit_histogram.png",
                )

            # Width vs predicted
            width_vs_predicted(
                y_pred_mean=pop["predicted_mean"].values,
                lower=pop["predicted_p10"].values,
                upper=pop["predicted_p90"].values,
                output_path=plots_dir / f"{prefix}_width_vs_predicted.png",
            )

            # Residual vs predicted
            residual_vs_predicted(
                y_true=pop["actual_points"].values,
                y_pred_mean=pop["predicted_mean"].values,
                output_path=plots_dir / f"{prefix}_residual_vs_predicted.png",
            )

            # Coverage drift over time
            if "game_date" in pop.columns:
                coverage_drift_over_time(
                    y_true=pop["actual_points"].values,
                    lower=pop["predicted_p10"].values,
                    upper=pop["predicted_p90"].values,
                    dates=pop["game_date"].values,
                )

    # ---- Step 9: Write evaluation_report.md ----
    ts = report_timestamp if report_timestamp is not None else datetime.now()
    report_path = output_dir / "evaluation_report.md"
    report_content = _format_report(
        headline_metrics=headline_metrics,
        full_pop_metrics=full_pop_metrics,
        baseline_comparisons=all_baseline_comparisons,
        multi_level_coverage=multi_level_coverage,
        slice_results=slice_results,
        mlflow_run_ids=mlflow_run_ids,
        report_timestamp=ts,
    )
    report_path.write_text(report_content, encoding="utf-8")

    return report_path


def _compute_metrics_for_pop(
    pop: pd.DataFrame,
    pt: PlayerType,
    candidate: ModelCandidate,
) -> PopulationMetrics:
    """Compute metrics for a single population slice."""
    has_p10 = "predicted_p10" in pop.columns and pop["predicted_p10"].notna().any()
    has_p90 = "predicted_p90" in pop.columns and pop["predicted_p90"].notna().any()
    p10 = pop["predicted_p10"].values if has_p10 else None
    p90 = pop["predicted_p90"].values if has_p90 else None

    return compute_population_metrics(
        y_true=pop["actual_points"].values,
        y_pred_mean=pop["predicted_mean"].values,
        y_pred_p10=p10,
        y_pred_p90=p90,
        player_type=pt.value,
        model_candidate=candidate.value,
    )


def _format_report(
    headline_metrics: list[PopulationMetrics],
    full_pop_metrics: list[PopulationMetrics],
    baseline_comparisons: list[BaselineComparison],
    multi_level_coverage: dict[tuple[str, str], dict[int, float]],
    slice_results: list[tuple[str, str, str, PopulationMetrics]],
    mlflow_run_ids: dict[ModelCandidate, str] | None = None,
    report_timestamp: datetime | None = None,
) -> str:
    """Format the evaluation report as markdown."""
    ts = report_timestamp if report_timestamp is not None else datetime.now()
    lines: list[str] = []
    lines.append("# Evaluation Report")
    lines.append("")
    lines.append(f"Generated: {ts.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # MLflow run IDs
    if mlflow_run_ids:
        lines.append("## MLflow Run IDs")
        lines.append("")
        for candidate, run_id in mlflow_run_ids.items():
            lines.append(f"- **{candidate.value}**: `{run_id}`")
        lines.append("")

    # Headline metrics table (eligible population)
    lines.append("## Headline Metrics (Eligible Population)")
    lines.append("")
    lines.append(
        "| Model | Population | N | RMSE | MAE | Spearman | "
        "Coverage 80% | Sharpness | Pinball 10 | Pinball 50 | Pinball 90 |"
    )
    lines.append("|" + "---|" * 11)

    for m in headline_metrics:
        lines.append(_format_metrics_row(m))
    lines.append("")

    # Full-population metrics (reference)
    lines.append("## Full-Population Metrics (All Players)")
    lines.append("")
    lines.append(
        "| Model | Population | N | RMSE | MAE | Spearman | "
        "Coverage 80% | Sharpness | Pinball 10 | Pinball 50 | Pinball 90 |"
    )
    lines.append("|" + "---|" * 11)

    for m in full_pop_metrics:
        lines.append(_format_metrics_row(m))
    lines.append("")

    # Multi-level coverage table (protocol section 9 item 2)
    if multi_level_coverage:
        lines.append("## Multi-Level Coverage")
        lines.append("")
        # Collect all levels across all entries
        all_levels: set[int] = set()
        for cov_dict in multi_level_coverage.values():
            all_levels.update(cov_dict.keys())
        sorted_levels = sorted(all_levels)

        header_levels = " | ".join(f"{lvl}%" for lvl in sorted_levels)
        lines.append(f"| Model | Population | {header_levels} |")
        lines.append("|" + "---|" * (2 + len(sorted_levels)))

        for (cand, pt), cov_dict in multi_level_coverage.items():
            vals = " | ".join(
                f"{cov_dict[lvl]:.3f}" if lvl in cov_dict else "N/A"
                for lvl in sorted_levels
            )
            lines.append(f"| {cand} | {pt} | {vals} |")
        lines.append("")

    # Baseline comparisons table
    lines.append("## Baseline Comparisons")
    lines.append("")
    lines.append(
        "| Model | Population | Model RMSE | Naive Last | Trailing 7d | "
        "Season-to-Date | Best Baseline | Improvement % | Pass |"
    )
    lines.append("|" + "---|" * 9)

    for bc in baseline_comparisons:
        pass_mark = "PASS" if bc.passes_threshold else "FAIL"
        lines.append(
            f"| {bc.model_candidate.value} | {bc.player_type.value} | "
            f"{bc.model_rmse:.3f} | {bc.baseline_naive_last_game_rmse:.3f} | "
            f"{bc.baseline_trailing_7d_rmse:.3f} | {bc.baseline_season_to_date_rmse:.3f} | "
            f"{bc.best_baseline_rmse:.3f} | {bc.improvement_pct:.1f}% | {pass_mark} |"
        )
    lines.append("")

    # Coverage acceptability
    lines.append("## Coverage Acceptability")
    lines.append("")
    for m in headline_metrics:
        if m.coverage_80 is not None:
            lo, hi = COVERAGE_ACCEPTABLE_RANGE
            in_range = lo <= m.coverage_80 <= hi
            status = "PASS" if in_range else "FAIL"
            lines.append(
                f"- **{m.model_candidate.value} / {m.player_type.value}**: "
                f"coverage = {m.coverage_80:.3f} [{status}] "
                f"(acceptable range: [{lo}, {hi}])"
            )
    lines.append("")

    # Slice metrics (protocol sections 6 and 7.3)
    if slice_results:
        lines.append("## Slice Metrics")
        lines.append("")
        lines.append(
            "| Model | Population | Slice | N | RMSE | MAE | Spearman | Coverage 80% |"
        )
        lines.append("|" + "---|" * 8)

        for cand, pt, slice_label, m in slice_results:
            cov = f"{m.coverage_80:.3f}" if m.coverage_80 is not None else "N/A"
            lines.append(
                f"| {cand} | {pt} | {slice_label} | {m.n_predictions} | "
                f"{m.rmse:.3f} | {m.mae:.3f} | {m.spearman_rho:.3f} | {cov} |"
            )
        lines.append("")

    # Selection rule application
    lines.append("## Selection Rule (model_bakeoff.md section 6)")
    lines.append("")
    selected = _apply_selection_rule(headline_metrics, baseline_comparisons)
    if selected is not None:
        lines.append(f"**Selected model: {selected.value}**")
    else:
        lines.append("**PHASE 1 FAILS.** No candidate satisfies the selection rule.")
    lines.append("")

    return "\n".join(lines)


def _format_metrics_row(m: PopulationMetrics) -> str:
    """Format a single row for the metrics table."""
    cov = f"{m.coverage_80:.3f}" if m.coverage_80 is not None else "N/A"
    sharp = f"{m.sharpness_mean_width:.2f}" if m.sharpness_mean_width is not None else "N/A"
    pb10 = f"{m.pinball_10:.3f}" if m.pinball_10 is not None else "N/A"
    pb50 = f"{m.pinball_50:.3f}" if m.pinball_50 is not None else "N/A"
    pb90 = f"{m.pinball_90:.3f}" if m.pinball_90 is not None else "N/A"
    return (
        f"| {m.model_candidate.value} | {m.player_type.value} | {m.n_predictions} | "
        f"{m.rmse:.3f} | {m.mae:.3f} | {m.spearman_rho:.3f} | "
        f"{cov} | {sharp} | {pb10} | {pb50} | {pb90} |"
    )


def _apply_selection_rule(
    all_metrics: list[PopulationMetrics],
    baseline_comparisons: list[BaselineComparison],
) -> ModelCandidate | None:
    """Apply the selection rule from model_bakeoff.md section 6.

    Returns the selected ModelCandidate or None if Phase 1 fails.
    """

    def _candidate_beats_baselines(candidate: ModelCandidate) -> bool:
        """Check if candidate beats baselines by >= 10% for all populations."""
        comparisons = [bc for bc in baseline_comparisons if bc.model_candidate == candidate]
        if not comparisons:
            return False
        return all(bc.passes_threshold for bc in comparisons)

    def _candidate_coverage_in_range(
        candidate: ModelCandidate,
        lo: float,
        hi: float,
    ) -> bool:
        """Check if candidate's 80% coverage is within [lo, hi] for all populations."""
        metrics = [m for m in all_metrics if m.model_candidate == candidate]
        for m in metrics:
            if m.coverage_80 is None:
                return False
            if not (lo <= m.coverage_80 <= hi):
                return False
        return True

    # Rule 1: M3 satisfies full acceptance criteria
    m3 = ModelCandidate.M3_CQR_ACI
    if _candidate_beats_baselines(m3) and _candidate_coverage_in_range(m3, 0.75, 0.85):
        return m3

    # Rule 2: M2 beats baselines and has coverage in relaxed range
    m2 = ModelCandidate.M2_LGBM_QUANTILE
    if _candidate_beats_baselines(m2) and _candidate_coverage_in_range(m2, 0.70, 0.90):
        return m2

    # Rule 3: M1 beats baselines (no coverage requirement)
    m1 = ModelCandidate.M1_RIDGE
    if _candidate_beats_baselines(m1):
        return m1

    return None
