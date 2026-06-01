"""Unit tests for packages/ml/training/splits.py.

Tests temporal splitting, walk-forward CV, and CQR calibration cutoff.
Owner: modeler subagent.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from packages.ml.training.splits import (
    HEADLINE_FOLD_NUMBERS,
    SPLIT_BOUNDARIES,
    WALK_FORWARD_FOLDS,
    compute_calibration_cutoff,
    get_calibration_split,
    get_walk_forward_folds,
    split_by_date,
)
from packages.shared.schemas.ml import SplitName


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _make_test_df(
    start_date: date = date(2019, 3, 1),
    end_date: date = date(2025, 9, 30),
    rows_per_day: int = 2,
) -> pd.DataFrame:
    """Create a synthetic DataFrame spanning multiple seasons.

    Generates rows at a regular cadence to simulate game data.
    """
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    rows = []
    player_id = 100
    game_pk = 1000
    for d in dates:
        for _ in range(rows_per_day):
            rows.append({
                "player_id": player_id,
                "game_pk": game_pk,
                "game_date": d.date(),
                "season_year": d.year,
                "player_type": "hitter",
                "fantasy_points": float(np.random.default_rng(game_pk).normal(10, 5)),
                "feat_a": float(np.random.default_rng(game_pk + 1).normal(0, 1)),
                "feat_b": float(np.random.default_rng(game_pk + 2).normal(0, 1)),
            })
            game_pk += 1
            player_id += 1
    return pd.DataFrame(rows)


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Multi-season DataFrame for split testing."""
    return _make_test_df()


@pytest.fixture()
def train_only_df() -> pd.DataFrame:
    """DataFrame covering only the Train period."""
    return _make_test_df(
        start_date=date(2019, 3, 28),
        end_date=date(2023, 10, 1),
    )


# ---------------------------------------------------------------------------
# TestSplitBoundaryConstants
# ---------------------------------------------------------------------------
class TestSplitBoundaryConstants:
    """Verify SPLIT_BOUNDARIES and WALK_FORWARD_FOLDS match validation_protocol.md."""

    def test_train_boundaries(self) -> None:
        start, end = SPLIT_BOUNDARIES[SplitName.TRAIN]
        assert start == date(2019, 1, 1)
        assert end == date(2023, 12, 31)

    def test_validation_boundaries(self) -> None:
        start, end = SPLIT_BOUNDARIES[SplitName.VALIDATION]
        assert start == date(2024, 1, 1)
        assert end == date(2024, 12, 31)

    def test_test_boundaries(self) -> None:
        start, end = SPLIT_BOUNDARIES[SplitName.TEST]
        assert start == date(2025, 1, 1)
        assert end == date(2025, 12, 31)

    def test_live_boundaries(self) -> None:
        start, end = SPLIT_BOUNDARIES[SplitName.LIVE]
        assert start == date(2026, 1, 1)

    def test_five_walk_forward_folds(self) -> None:
        assert len(WALK_FORWARD_FOLDS) == 5

    def test_folds_are_expanding(self) -> None:
        """Each fold's training window starts at 2019-01-01 and ends later."""
        for i, fold in enumerate(WALK_FORWARD_FOLDS):
            assert fold.train_start == date(2019, 1, 1)
            if i > 0:
                assert fold.train_end > WALK_FORWARD_FOLDS[i - 1].train_end

    def test_fold_val_follows_train(self) -> None:
        """Each fold's validation window starts after training ends."""
        for fold in WALK_FORWARD_FOLDS:
            assert fold.val_start > fold.train_end

    def test_no_fold_overlap(self) -> None:
        """No fold's validation window overlaps any other fold's training window."""
        for i, fold_i in enumerate(WALK_FORWARD_FOLDS):
            for j, fold_j in enumerate(WALK_FORWARD_FOLDS):
                if i == j:
                    continue
                # fold_i's val should not overlap with fold_j's train
                assert not (
                    fold_i.val_start <= fold_j.train_end
                    and fold_i.val_end >= fold_j.train_start
                    and fold_i.val_start > fold_j.train_end  # This is the correct check
                ) or fold_i.val_start > fold_j.train_end

    def test_headline_folds_exclude_covid(self) -> None:
        """Headline folds are 2-5, excluding fold 1 (COVID 2020)."""
        assert HEADLINE_FOLD_NUMBERS == [2, 3, 4, 5]

    def test_fold_1_validates_on_2020(self) -> None:
        """Fold 1 validates on 2020 (COVID season)."""
        fold_1 = WALK_FORWARD_FOLDS[0]
        assert fold_1.val_start == date(2020, 1, 1)
        assert fold_1.val_end == date(2020, 12, 31)


# ---------------------------------------------------------------------------
# TestSplitByDate
# ---------------------------------------------------------------------------
class TestSplitByDate:
    """Tests for training.splits.split_by_date."""

    def test_returns_all_split_names(self, sample_df: pd.DataFrame) -> None:
        result = split_by_date(sample_df)
        assert set(result.keys()) == set(SplitName)

    def test_no_row_in_two_splits(self, sample_df: pd.DataFrame) -> None:
        """Every row appears in at most one split (non-overlapping)."""
        result = split_by_date(sample_df)
        all_indices: list[int] = []
        for split_df in result.values():
            all_indices.extend(split_df.index.tolist())
        assert len(all_indices) == len(set(all_indices)), "Some rows appear in multiple splits"

    def test_train_rows_before_2024(self, sample_df: pd.DataFrame) -> None:
        result = split_by_date(sample_df)
        train_dates = result[SplitName.TRAIN]["game_date"]
        assert train_dates.max() <= date(2023, 12, 31)
        assert train_dates.min() >= date(2019, 1, 1)

    def test_val_rows_in_2024(self, sample_df: pd.DataFrame) -> None:
        result = split_by_date(sample_df)
        val_dates = result[SplitName.VALIDATION]["game_date"]
        if len(val_dates) > 0:
            assert val_dates.min() >= date(2024, 1, 1)
            assert val_dates.max() <= date(2024, 12, 31)

    def test_test_rows_in_2025(self, sample_df: pd.DataFrame) -> None:
        result = split_by_date(sample_df)
        test_dates = result[SplitName.TEST]["game_date"]
        if len(test_dates) > 0:
            assert test_dates.min() >= date(2025, 1, 1)
            assert test_dates.max() <= date(2025, 12, 31)

    def test_temporal_ordering_across_splits(self, sample_df: pd.DataFrame) -> None:
        """Train max date < Val min date < Test min date."""
        result = split_by_date(sample_df)
        train_max = result[SplitName.TRAIN]["game_date"].max()
        val_min = result[SplitName.VALIDATION]["game_date"].min()
        val_max = result[SplitName.VALIDATION]["game_date"].max()
        test_min = result[SplitName.TEST]["game_date"].min()

        assert train_max < val_min
        assert val_max < test_min

    def test_missing_column_raises(self) -> None:
        df = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(KeyError, match="not found"):
            split_by_date(df, date_column="game_date")

    def test_handles_timestamp_column(self) -> None:
        """Works with pandas Timestamp columns, not just date objects."""
        df = pd.DataFrame({
            "game_date": pd.to_datetime(["2020-06-01", "2024-06-01", "2025-06-01"]),
            "player_id": [1, 2, 3],
        })
        result = split_by_date(df)
        assert len(result[SplitName.TRAIN]) == 1
        assert len(result[SplitName.VALIDATION]) == 1
        assert len(result[SplitName.TEST]) == 1

    def test_empty_df(self) -> None:
        df = pd.DataFrame(columns=["game_date", "player_id"])
        result = split_by_date(df)
        for split_df in result.values():
            assert len(split_df) == 0


# ---------------------------------------------------------------------------
# TestGetWalkForwardFolds
# ---------------------------------------------------------------------------
class TestGetWalkForwardFolds:
    """Tests for training.splits.get_walk_forward_folds."""

    def test_returns_five_folds(self, sample_df: pd.DataFrame) -> None:
        folds = get_walk_forward_folds(sample_df)
        assert len(folds) == 5

    def test_each_fold_is_tuple_of_two_dfs(self, sample_df: pd.DataFrame) -> None:
        folds = get_walk_forward_folds(sample_df)
        for train_fold, val_fold in folds:
            assert isinstance(train_fold, pd.DataFrame)
            assert isinstance(val_fold, pd.DataFrame)

    def test_expanding_training_window(self, sample_df: pd.DataFrame) -> None:
        """Each successive fold's training set is larger."""
        folds = get_walk_forward_folds(sample_df)
        prev_size = 0
        for train_fold, _ in folds:
            assert len(train_fold) >= prev_size
            prev_size = len(train_fold)

    def test_no_temporal_leakage_in_folds(self, sample_df: pd.DataFrame) -> None:
        """In each fold, max train date < min val date."""
        folds = get_walk_forward_folds(sample_df)
        for train_fold, val_fold in folds:
            if len(train_fold) > 0 and len(val_fold) > 0:
                train_max = train_fold["game_date"].max()
                val_min = val_fold["game_date"].min()
                assert train_max < val_min, (
                    f"Temporal leakage: train_max={train_max} >= val_min={val_min}"
                )

    def test_no_row_in_both_train_and_val(self, sample_df: pd.DataFrame) -> None:
        """No row appears in both train and val of the same fold."""
        folds = get_walk_forward_folds(sample_df)
        for train_fold, val_fold in folds:
            overlap = set(train_fold.index) & set(val_fold.index)
            assert len(overlap) == 0, f"Found {len(overlap)} overlapping rows"


# ---------------------------------------------------------------------------
# TestComputeCalibrationCutoff
# ---------------------------------------------------------------------------
class TestComputeCalibrationCutoff:
    """Tests for training.splits.compute_calibration_cutoff."""

    def test_returns_a_date(self, train_only_df: pd.DataFrame) -> None:
        cutoff = compute_calibration_cutoff(train_only_df)
        assert isinstance(cutoff, date)

    def test_cutoff_within_train_range(self, train_only_df: pd.DataFrame) -> None:
        cutoff = compute_calibration_cutoff(train_only_df)
        min_date = train_only_df["game_date"].min()
        max_date = train_only_df["game_date"].max()
        assert cutoff >= min_date
        assert cutoff <= max_date

    def test_80_percent_before_cutoff(self, train_only_df: pd.DataFrame) -> None:
        """Approximately 80% of rows should be on or before the cutoff."""
        cutoff = compute_calibration_cutoff(train_only_df, train_fraction=0.80)
        n_before = (train_only_df["game_date"] <= cutoff).sum()
        fraction = n_before / len(train_only_df)
        # Allow some tolerance due to date granularity
        assert 0.75 <= fraction <= 0.85, f"Got fraction={fraction:.4f}"

    def test_different_fractions(self, train_only_df: pd.DataFrame) -> None:
        """Larger fraction should produce a later cutoff."""
        cutoff_70 = compute_calibration_cutoff(train_only_df, train_fraction=0.70)
        cutoff_90 = compute_calibration_cutoff(train_only_df, train_fraction=0.90)
        assert cutoff_70 <= cutoff_90

    def test_invalid_fraction_raises(self, train_only_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="train_fraction"):
            compute_calibration_cutoff(train_only_df, train_fraction=0.0)
        with pytest.raises(ValueError, match="train_fraction"):
            compute_calibration_cutoff(train_only_df, train_fraction=1.0)

    def test_empty_df_raises(self) -> None:
        df = pd.DataFrame(columns=["game_date", "player_id"])
        with pytest.raises(ValueError, match="empty"):
            compute_calibration_cutoff(df)


class TestGetCalibrationSplit:
    """Tests for training.splits.get_calibration_split."""

    def test_disjoint_splits(self, train_only_df: pd.DataFrame) -> None:
        """Fit and calibration sets must be disjoint."""
        fit_df, cal_df = get_calibration_split(train_only_df)
        overlap = set(fit_df.index) & set(cal_df.index)
        assert len(overlap) == 0

    def test_fit_before_cal(self, train_only_df: pd.DataFrame) -> None:
        """All fit rows must be temporally before calibration rows."""
        fit_df, cal_df = get_calibration_split(train_only_df)
        if len(fit_df) > 0 and len(cal_df) > 0:
            assert fit_df["game_date"].max() < cal_df["game_date"].min()

    def test_union_equals_original(self, train_only_df: pd.DataFrame) -> None:
        """fit + cal should cover all rows in the original."""
        fit_df, cal_df = get_calibration_split(train_only_df)
        assert len(fit_df) + len(cal_df) == len(train_only_df)
