"""Unit tests for packages/ml/features/.

Tests each feature builder for correctness, anti-leakage, and null handling.
Owner: feature-engineer subagent (implements tests alongside feature builders).
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers: synthetic data generators
# ---------------------------------------------------------------------------

def _make_batting_stats(
    n_games: int = 10,
    player_id: int = 1,
    start_date: date = date(2024, 6, 1),
    team_id: int = 100,
) -> pd.DataFrame:
    """Create synthetic batting stats for one player over n_games."""
    rows = []
    for i in range(n_games):
        gd = start_date + timedelta(days=i)
        rows.append({
            "player_id": player_id,
            "game_pk": 100000 + i,
            "game_date": gd,
            "season_year": gd.year,
            "team_id": team_id,
            "ab": 4,
            "h": 1 + (i % 2),
            "doubles": i % 3,
            "triples": 0,
            "hr": i % 4,
            "r": 1,
            "rbi": i % 3,
            "bb": 1,
            "hbp": 0,
            "sb": i % 2,
            "cs": 0,
            "so": 1,
            "sf": 0,
            "pa": 5,
            "batting_order": (i % 9) + 1,
        })
    return pd.DataFrame(rows)


def _make_pitching_stats(
    n_games: int = 10,
    player_id: int = 2,
    start_date: date = date(2024, 6, 1),
    team_id: int = 200,
) -> pd.DataFrame:
    """Create synthetic pitching stats for one pitcher."""
    rows = []
    for i in range(n_games):
        gd = start_date + timedelta(days=i * 5)  # Every 5 days (starter)
        rows.append({
            "player_id": player_id,
            "game_pk": 200000 + i,
            "game_date": gd,
            "season_year": gd.year,
            "team_id": team_id,
            "ip": 6.0 + (i % 3),
            "er": 2 + (i % 2),
            "h": 5 + (i % 3),
            "bb": 2,
            "so": 6 + (i % 4),
            "hr_allowed": i % 2,
            "wins": 1 if i % 3 == 0 else 0,
            "losses": 1 if i % 3 == 1 else 0,
            "saves": 0,
            "holds": 0,
            "quality_starts": 1 if (6.0 + (i % 3)) >= 6 else 0,
            "batters_faced": 24 + i,
            "pitches_thrown": 90 + i * 3,
        })
    return pd.DataFrame(rows)


def _make_games(game_pks: list[int], home_team: int = 100, away_team: int = 200) -> pd.DataFrame:
    """Create synthetic games DataFrame."""
    rows = []
    for gpk in game_pks:
        rows.append({
            "game_pk": gpk,
            "home_team_id": home_team,
            "away_team_id": away_team,
            "venue_id": 1,
            "season_year": 2024,
            "home_probable_pitcher_id": 2,
            "away_probable_pitcher_id": 3,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Target computation tests
# ---------------------------------------------------------------------------

class TestComputeTargetBatting:
    """Tests for features.scoring.compute_target_batting."""

    def test_adds_fantasy_points_column(self) -> None:
        from packages.ml.features.scoring import compute_target_batting

        stats = _make_batting_stats(n_games=3)
        result = compute_target_batting(stats)
        assert "fantasy_points" in result.columns
        assert len(result) == 3

    def test_does_not_modify_input(self) -> None:
        from packages.ml.features.scoring import compute_target_batting

        stats = _make_batting_stats(n_games=3)
        original_cols = set(stats.columns)
        compute_target_batting(stats)
        assert set(stats.columns) == original_cols

    def test_known_stat_line(self) -> None:
        """1H (single), 1HR, 1R, 1RBI, 1BB, 1SB, 1SO = known points."""
        from packages.ml.features.scoring import compute_target_batting

        stats = pd.DataFrame([{
            "player_id": 1, "game_pk": 1,
            "h": 2, "doubles": 0, "triples": 0, "hr": 1,
            "r": 1, "rbi": 1, "bb": 1, "hbp": 0,
            "sb": 1, "cs": 0, "so": 1, "sf": 0,
        }])
        result = compute_target_batting(stats)
        # H=2*1.0=2, HR=1*3.0=3, R=1, RBI=1, BB=1, SB=1*2=2, SO=-0.5
        expected = 2 + 3 + 1 + 1 + 1 + 2 - 0.5
        assert result["fantasy_points"].iloc[0] == pytest.approx(expected, rel=1e-6)

    def test_custom_rules(self) -> None:
        from packages.ml.features.scoring import compute_target_batting

        stats = pd.DataFrame([{
            "player_id": 1, "game_pk": 1,
            "h": 1, "doubles": 0, "triples": 0, "hr": 0,
            "r": 0, "rbi": 0, "bb": 0, "hbp": 0,
            "sb": 0, "cs": 0, "so": 0, "sf": 0,
        }])
        result = compute_target_batting(stats, rules={"H": 5.0})
        assert result["fantasy_points"].iloc[0] == pytest.approx(5.0)


class TestComputeTargetPitching:
    """Tests for features.scoring.compute_target_pitching."""

    def test_adds_fantasy_points_column(self) -> None:
        from packages.ml.features.scoring import compute_target_pitching

        stats = _make_pitching_stats(n_games=3)
        result = compute_target_pitching(stats)
        assert "fantasy_points" in result.columns
        assert len(result) == 3

    def test_known_stat_line(self) -> None:
        """7IP, 2ER, 5H, 2BB, 8SO, 1W, QS = known points."""
        from packages.ml.features.scoring import compute_target_pitching

        stats = pd.DataFrame([{
            "player_id": 2, "game_pk": 1,
            "ip": 7.0, "er": 2, "h": 5, "bb": 2, "so": 8,
            "wins": 1, "losses": 0, "saves": 0, "holds": 0,
            "quality_starts": 1, "hbp": 0,
        }])
        result = compute_target_pitching(stats)
        # IP=7*3=21, SO=8*1=8, W=5, ER=2*-2=-4, H=5*-0.5=-2.5,
        # BB=2*-0.5=-1, QS=3
        expected = 21 + 8 + 5 - 4 - 2.5 - 1 + 3
        assert result["fantasy_points"].iloc[0] == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# P0 feature builder tests
# ---------------------------------------------------------------------------

class TestRollingProductionBuilder:
    """Tests for features.rolling.RollingProductionBuilder."""

    def test_protocol_compliance(self) -> None:
        from packages.ml.features.rolling import RollingProductionBuilder

        builder = RollingProductionBuilder()
        assert builder.name == "rolling_production"
        assert builder.requires_history_days == 60

    def test_hitter_output_columns(self) -> None:
        from packages.ml.features.rolling import RollingProductionBuilder

        stats = _make_batting_stats(n_games=10)
        builder = RollingProductionBuilder()
        result = builder.build(stats, as_of_date=date(2024, 6, 30), player_type="hitter")

        assert "player_id" in result.columns
        assert "game_pk" in result.columns
        # Check key rolling columns exist (no fantasy_points -- Global Rule 3)
        for w in [7, 14, 30]:
            assert f"roll_h_{w}d" in result.columns
            assert f"roll_hr_{w}d" in result.columns
            assert f"roll_rbi_{w}d" in result.columns
            assert f"roll_pa_{w}d" in result.columns
            # fantasy_points must NOT be present
            assert f"roll_fantasy_points_{w}d" not in result.columns

    def test_pitcher_output_columns(self) -> None:
        from packages.ml.features.rolling import RollingProductionBuilder

        stats = _make_pitching_stats(n_games=10)
        builder = RollingProductionBuilder()
        result = builder.build(stats, as_of_date=date(2024, 9, 30), player_type="pitcher")

        for w in [14, 30, 60]:
            assert f"roll_era_{w}d" in result.columns
            assert f"roll_whip_{w}d" in result.columns
            assert f"roll_k9_{w}d" in result.columns
            # fantasy_points must NOT be present
            assert f"roll_fantasy_points_{w}d" not in result.columns

    def test_first_game_is_nan(self) -> None:
        """First game for a player should have NaN rolling features (shift=1)."""
        from packages.ml.features.rolling import RollingProductionBuilder

        stats = _make_batting_stats(n_games=5)
        builder = RollingProductionBuilder()
        result = builder.build(stats, as_of_date=date(2024, 6, 30), player_type="hitter")

        first_row = result[result["game_pk"] == 100000]
        assert first_row["roll_h_7d"].isna().all()

    def test_no_current_game_in_rolling(self) -> None:
        """The rolling value for game N must NOT include game N's stats.

        Anti-leakage: we set game N's h to 999 and verify
        the rolling feature for game N does not change.
        """
        from packages.ml.features.rolling import RollingProductionBuilder

        stats = _make_batting_stats(n_games=5)
        builder = RollingProductionBuilder()
        result_normal = builder.build(
            stats.copy(), as_of_date=date(2024, 6, 30), player_type="hitter"
        )

        # Modify game N=4 (index 4) stats dramatically
        stats_modified = stats.copy()
        stats_modified.loc[stats_modified["game_pk"] == 100004, "h"] = 999
        result_modified = builder.build(
            stats_modified, as_of_date=date(2024, 6, 30), player_type="hitter"
        )

        # The rolling value for game 100004 should be the same in both
        r1 = result_normal[result_normal["game_pk"] == 100004]["roll_h_7d"].values
        r2 = result_modified[result_modified["game_pk"] == 100004]["roll_h_7d"].values
        np.testing.assert_array_almost_equal(r1, r2)

    def test_no_target_derived_features(self) -> None:
        """Rolling builder must NOT produce any fantasy_points columns.

        Per docs/feature_plan.md Global Rule 3, target-derived features are
        prohibited. The model must learn scoring-rule weighting from raw stats.
        """
        from packages.ml.features.rolling import RollingProductionBuilder

        stats = _make_batting_stats(n_games=10)
        # Even if fantasy_points is in the input, it must not appear in output
        stats["fantasy_points"] = 10.0
        builder = RollingProductionBuilder()
        result = builder.build(stats, as_of_date=date(2024, 6, 30), player_type="hitter")
        fp_cols = [c for c in result.columns if "fantasy_points" in c]
        assert fp_cols == [], f"Target-derived columns found: {fp_cols}"

        # Same for pitchers
        pitch = _make_pitching_stats(n_games=6)
        pitch["fantasy_points"] = 20.0
        result_p = builder.build(pitch, as_of_date=date(2024, 9, 30), player_type="pitcher")
        fp_cols_p = [c for c in result_p.columns if "fantasy_points" in c]
        assert fp_cols_p == [], f"Target-derived columns found: {fp_cols_p}"

    def test_calendar_day_window_excludes_old_games(self) -> None:
        """Games older than the calendar-day window should NOT be included.

        Create a player with games spaced 5 days apart. For a 7-day window,
        only the most recent game (within 7 calendar days) should contribute.
        """
        from packages.ml.features.rolling import RollingProductionBuilder

        # Games every 5 days: June 1, 6, 11, 16, 21
        rows = []
        for i in range(5):
            gd = date(2024, 6, 1) + timedelta(days=i * 5)
            rows.append({
                "player_id": 1, "game_pk": 300000 + i, "game_date": gd,
                "ab": 4, "h": 2, "doubles": 0, "triples": 0, "hr": 1,
                "r": 1, "rbi": 1, "bb": 1, "hbp": 0, "sb": 0,
                "cs": 0, "so": 1, "sf": 0, "pa": 5,
            })
        stats = pd.DataFrame(rows)
        builder = RollingProductionBuilder()
        result = builder.build(stats, as_of_date=date(2024, 6, 30), player_type="hitter")

        # For game on June 21 (game_pk=300004):
        # shift(1) means we look at games up to June 16.
        # 7-day window from June 16 back = June 10-16.
        # Only June 16 (game_pk=300003) falls in that window.
        # June 11 (game_pk=300002) is at day index June 11, which is 5 days
        # before June 16 -> within 7D window. So games 300002 and 300003.
        # Wait -- June 11 to June 16 = 5 days, within 7D. So sum of h = 2+2 = 4.
        row = result[result["game_pk"] == 300004]
        roll_h_7d = row["roll_h_7d"].values[0]
        # Games within 7 calendar days of June 16: June 11 and June 16 -> h=2+2=4
        assert roll_h_7d == pytest.approx(4.0)

        # For a 14-day window on June 21:
        # shifted to June 16, 14-day window = June 3-16.
        # June 6 (300001), June 11 (300002), June 16 (300003) -> h=2+2+2=6
        roll_h_14d = row["roll_h_14d"].values[0]
        assert roll_h_14d == pytest.approx(6.0)

    def test_excludes_future_data(self) -> None:
        """as_of_date guard: rows after as_of_date must be excluded."""
        from packages.ml.features.rolling import RollingProductionBuilder

        stats = _make_batting_stats(n_games=10)
        builder = RollingProductionBuilder()
        result = builder.build(
            stats, as_of_date=date(2024, 6, 5), player_type="hitter"
        )
        # Only games on or before June 5 should be in output
        assert len(result) == 5  # June 1-5


class TestRestRecencyBuilder:
    """Tests for features.rest.RestRecencyBuilder."""

    def test_protocol_compliance(self) -> None:
        from packages.ml.features.rest import RestRecencyBuilder

        builder = RestRecencyBuilder()
        assert builder.name == "rest_recency"
        assert builder.requires_history_days == 14

    def test_output_columns(self) -> None:
        from packages.ml.features.rest import RestRecencyBuilder

        stats = _make_batting_stats(n_games=5)
        builder = RestRecencyBuilder()
        result = builder.build(stats, as_of_date=date(2024, 6, 30))

        expected_cols = {
            "rest_days_since_last_game", "rest_games_in_last_7d",
            "rest_games_in_last_14d", "rest_back_to_back", "rest_missing",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_consecutive_games_days_since(self) -> None:
        """Consecutive daily games should have days_since_last_game = 1."""
        from packages.ml.features.rest import RestRecencyBuilder

        stats = _make_batting_stats(n_games=3)  # June 1, 2, 3
        builder = RestRecencyBuilder()
        result = builder.build(stats, as_of_date=date(2024, 6, 30))

        # Game 2 (June 2) should have days_since = 1
        row = result[result["game_pk"] == 100001]
        assert row["rest_days_since_last_game"].values[0] == 1.0
        assert row["rest_back_to_back"].values[0] == 1

    def test_first_game_is_missing(self) -> None:
        """First game should have rest_missing = 1."""
        from packages.ml.features.rest import RestRecencyBuilder

        stats = _make_batting_stats(n_games=3)
        builder = RestRecencyBuilder()
        result = builder.build(stats, as_of_date=date(2024, 6, 30))

        row = result[result["game_pk"] == 100000]
        assert row["rest_missing"].values[0] == 1

    def test_games_in_last_7d(self) -> None:
        """With daily games, games_in_last_7d should count prior games.

        Window is (game_date - 7, game_date) open on left: for June 8,
        cutoff = June 1, so June 2-7 = 6 games strictly after June 1.
        """
        from packages.ml.features.rest import RestRecencyBuilder

        stats = _make_batting_stats(n_games=10)  # June 1-10
        builder = RestRecencyBuilder()
        result = builder.build(stats, as_of_date=date(2024, 6, 30))

        row = result[result["game_pk"] == 100007]
        assert row["rest_games_in_last_7d"].values[0] == 6


class TestContextBuilder:
    """Tests for features.context.ContextBuilder."""

    def test_protocol_compliance(self) -> None:
        from packages.ml.features.context import ContextBuilder

        builder = ContextBuilder()
        assert builder.name == "context"
        assert builder.requires_history_days == 0

    def test_home_away_detection(self) -> None:
        from packages.ml.features.context import ContextBuilder

        stats = _make_batting_stats(n_games=2, team_id=100)
        games = _make_games([100000, 100001], home_team=100, away_team=200)

        builder = ContextBuilder()
        result = builder.build(
            stats, as_of_date=date(2024, 6, 30), games=games,
        )
        # Player is on team 100, which is home team
        assert (result["ctx_is_home"] == 1).all()

    def test_away_player(self) -> None:
        from packages.ml.features.context import ContextBuilder

        stats = _make_batting_stats(n_games=2, team_id=200)
        games = _make_games([100000, 100001], home_team=100, away_team=200)

        builder = ContextBuilder()
        result = builder.build(
            stats, as_of_date=date(2024, 6, 30), games=games,
        )
        assert (result["ctx_is_home"] == 0).all()

    def test_park_factor_uses_prior_season(self) -> None:
        from packages.ml.features.context import ContextBuilder

        stats = _make_batting_stats(n_games=1, team_id=100)
        games = _make_games([100000], home_team=100, away_team=200)
        # Park factor for 2023 should be used for 2024 games
        pf = pd.DataFrame([{
            "venue_id": 1,
            "season_year": 2023,
            "park_factor_runs": 1.15,
            "park_factor_hr": 1.20,
        }])

        builder = ContextBuilder()
        result = builder.build(
            stats, as_of_date=date(2024, 6, 30),
            games=games, park_factors=pf,
        )
        assert result["ctx_park_factor_runs"].values[0] == pytest.approx(1.15)
        assert result["ctx_park_factor_hr"].values[0] == pytest.approx(1.20)

    def test_missing_park_factor_imputed(self) -> None:
        from packages.ml.features.context import ContextBuilder

        stats = _make_batting_stats(n_games=1, team_id=100)
        games = _make_games([100000])

        builder = ContextBuilder()
        result = builder.build(
            stats, as_of_date=date(2024, 6, 30), games=games,
        )
        # No park factors provided -> imputed to 1.0 with missing flag
        assert result["ctx_park_factor_runs"].values[0] == pytest.approx(1.0)
        assert result["ctx_park_factor_missing"].values[0] == 1


class TestOpponentQualityBuilder:
    """Tests for features.opponent.OpponentQualityBuilder."""

    def test_protocol_compliance(self) -> None:
        from packages.ml.features.opponent import OpponentQualityBuilder

        builder = OpponentQualityBuilder()
        assert builder.name == "opponent_quality"
        assert builder.requires_history_days == 30

    def test_hitter_opp_columns(self) -> None:
        from packages.ml.features.opponent import OpponentQualityBuilder

        stats = _make_batting_stats(n_games=3, team_id=100)
        games = _make_games([100000, 100001, 100002])
        opp_pitching = _make_pitching_stats(
            n_games=10, player_id=2, start_date=date(2024, 5, 1)
        )

        builder = OpponentQualityBuilder()
        result = builder.build(
            stats, as_of_date=date(2024, 6, 30),
            opponent_pitching=opp_pitching,
            games=games,
            player_type="hitter",
        )
        expected_cols = {"opp_era_30d", "opp_whip_30d", "opp_k9_30d", "opp_bb9_30d", "opp_missing"}
        assert expected_cols.issubset(set(result.columns))

    def test_pitcher_opp_columns(self) -> None:
        from packages.ml.features.opponent import OpponentQualityBuilder

        stats = _make_pitching_stats(n_games=3, team_id=200)
        games = _make_games([200000, 200001, 200002], home_team=100, away_team=200)
        opp_batting = _make_batting_stats(
            n_games=30, player_id=10, start_date=date(2024, 5, 1), team_id=100,
        )

        builder = OpponentQualityBuilder()
        result = builder.build(
            stats, as_of_date=date(2024, 9, 30),
            opponent_batting=opp_batting,
            games=games,
            player_type="pitcher",
        )
        expected_cols = {"opp_obp_30d", "opp_slg_30d", "opp_k_pct_30d", "opp_missing"}
        assert expected_cols.issubset(set(result.columns))

    def test_missing_opponent_imputed(self) -> None:
        from packages.ml.features.opponent import OpponentQualityBuilder

        stats = _make_batting_stats(n_games=1)
        games = _make_games([100000])

        builder = OpponentQualityBuilder()
        result = builder.build(
            stats, as_of_date=date(2024, 6, 30),
            games=games,
            player_type="hitter",
        )
        assert result["opp_missing"].values[0] == 1
        # Should be imputed to league-average defaults
        assert result["opp_era_30d"].values[0] == pytest.approx(4.50)


class TestLineupSlotBuilder:
    """Tests for features.lineup.LineupSlotBuilder."""

    def test_protocol_compliance(self) -> None:
        from packages.ml.features.lineup import LineupSlotBuilder

        builder = LineupSlotBuilder()
        assert builder.name == "lineup_slot"
        assert builder.requires_history_days == 14

    def test_uses_prior_game_slot(self) -> None:
        """Lineup slot for game N should be the batting_order from game N-1."""
        from packages.ml.features.lineup import LineupSlotBuilder

        stats = _make_batting_stats(n_games=5)
        builder = LineupSlotBuilder()
        result = builder.build(stats, as_of_date=date(2024, 6, 30))

        # Game 100001 should use game 100000's batting order
        game0_order = stats[stats["game_pk"] == 100000]["batting_order"].values[0]
        game1_slot = result[result["game_pk"] == 100001]["lineup_batting_order_slot"].values[0]
        assert game1_slot == game0_order

    def test_first_game_is_missing(self) -> None:
        from packages.ml.features.lineup import LineupSlotBuilder

        stats = _make_batting_stats(n_games=3)
        builder = LineupSlotBuilder()
        result = builder.build(stats, as_of_date=date(2024, 6, 30))

        row = result[result["game_pk"] == 100000]
        assert row["lineup_slot_missing"].values[0] == 1
        # Imputed to 5
        assert row["lineup_batting_order_slot"].values[0] == 5.0


class TestPitcherWorkloadBuilder:
    """Tests for features.workload.PitcherWorkloadBuilder."""

    def test_protocol_compliance(self) -> None:
        from packages.ml.features.workload import PitcherWorkloadBuilder

        builder = PitcherWorkloadBuilder()
        assert builder.name == "pitcher_workload"
        assert builder.requires_history_days == 45

    def test_output_columns(self) -> None:
        from packages.ml.features.workload import PitcherWorkloadBuilder

        stats = _make_pitching_stats(n_games=6)
        builder = PitcherWorkloadBuilder()
        result = builder.build(stats, as_of_date=date(2024, 9, 30))

        expected_cols = {
            "workload_mean_ip_5starts",
            "workload_mean_pitches_5starts",
            "workload_mean_bf_5starts",
            "workload_is_starter",
            "workload_missing",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_first_game_is_missing(self) -> None:
        from packages.ml.features.workload import PitcherWorkloadBuilder

        stats = _make_pitching_stats(n_games=3)
        builder = PitcherWorkloadBuilder()
        result = builder.build(stats, as_of_date=date(2024, 9, 30))

        row = result[result["game_pk"] == 200000]
        assert row["workload_missing"].values[0] == 1

    def test_starter_detection(self) -> None:
        """Pitcher with 6+ IP per start should be classified as starter."""
        from packages.ml.features.workload import PitcherWorkloadBuilder

        stats = _make_pitching_stats(n_games=6)  # IP = 6.0 + (i%3)
        builder = PitcherWorkloadBuilder()
        result = builder.build(stats, as_of_date=date(2024, 9, 30))

        # After a few starts, should be detected as starter
        last_row = result.iloc[-1]
        assert last_row["workload_is_starter"] == 1

    def test_uses_shifted_data(self) -> None:
        """Workload for game N must not include game N's data."""
        from packages.ml.features.workload import PitcherWorkloadBuilder

        stats = _make_pitching_stats(n_games=6)
        builder = PitcherWorkloadBuilder()
        result_normal = builder.build(stats.copy(), as_of_date=date(2024, 9, 30))

        # Modify last game's IP dramatically
        stats_mod = stats.copy()
        last_gpk = stats_mod["game_pk"].iloc[-1]
        stats_mod.loc[stats_mod["game_pk"] == last_gpk, "ip"] = 0.1

        result_mod = builder.build(stats_mod, as_of_date=date(2024, 9, 30))

        # The workload for the last game should be the same (shift=1)
        r1 = result_normal[result_normal["game_pk"] == last_gpk]["workload_mean_ip_5starts"].values
        r2 = result_mod[result_mod["game_pk"] == last_gpk]["workload_mean_ip_5starts"].values
        np.testing.assert_array_almost_equal(r1, r2)


class TestPriorSeasonBuilder:
    """Tests for features.prior_season.PriorSeasonBuilder."""

    def test_protocol_compliance(self) -> None:
        from packages.ml.features.prior_season import PriorSeasonBuilder

        builder = PriorSeasonBuilder()
        assert builder.name == "prior_season"
        assert builder.requires_history_days == 0

    def test_hitter_output_columns(self) -> None:
        from packages.ml.features.prior_season import PriorSeasonBuilder

        stats = _make_batting_stats(n_games=3)
        season_batting = pd.DataFrame([{
            "player_id": 1, "season_year": 2023,
            "pa": 550, "woba": 0.340, "iso": 0.180,
            "bb_pct": 0.09, "k_pct": 0.20, "sprint_speed": 28.0,
        }])

        builder = PriorSeasonBuilder()
        result = builder.build(
            stats, as_of_date=date(2024, 6, 30),
            season_batting=season_batting,
            player_type="hitter",
        )

        expected_cols = {
            "prior_pa", "prior_woba", "prior_iso",
            "prior_bb_pct", "prior_k_pct", "prior_sprint_speed",
            "prior_missing",
        }
        assert expected_cols.issubset(set(result.columns))
        assert result["prior_woba"].values[0] == pytest.approx(0.340)
        assert result["prior_missing"].values[0] == 0

    def test_rookie_gets_league_avg(self) -> None:
        """Player with no prior-season data gets league-average + missing flag."""
        from packages.ml.features.prior_season import PriorSeasonBuilder

        stats = _make_batting_stats(n_games=2)
        # No season data for this player
        season_batting = pd.DataFrame(columns=[
            "player_id", "season_year", "pa", "woba", "iso",
            "bb_pct", "k_pct", "sprint_speed",
        ])

        builder = PriorSeasonBuilder()
        result = builder.build(
            stats, as_of_date=date(2024, 6, 30),
            season_batting=season_batting,
            player_type="hitter",
        )
        assert result["prior_missing"].values[0] == 1
        # Should be league average
        assert result["prior_woba"].values[0] == pytest.approx(0.320)

    def test_pitcher_prior_season(self) -> None:
        from packages.ml.features.prior_season import PriorSeasonBuilder

        stats = _make_pitching_stats(n_games=2)
        season_pitching = pd.DataFrame([{
            "player_id": 2, "season_year": 2023,
            "ip": 180.0, "fip": 3.50, "k_pct": 0.25,
            "bb_pct": 0.06, "gb_pct": 0.45, "fastball_velo": 95.0,
        }])

        builder = PriorSeasonBuilder()
        result = builder.build(
            stats, as_of_date=date(2024, 9, 30),
            season_pitching=season_pitching,
            player_type="pitcher",
        )
        assert result["prior_fip"].values[0] == pytest.approx(3.50)
        assert result["prior_missing"].values[0] == 0

    def test_uses_prior_season_not_current(self) -> None:
        """For 2024 games, should use 2023 stats, not 2024."""
        from packages.ml.features.prior_season import PriorSeasonBuilder

        stats = _make_batting_stats(n_games=2)
        season_batting = pd.DataFrame([
            {"player_id": 1, "season_year": 2023, "pa": 500, "woba": 0.350,
             "iso": 0.18, "bb_pct": 0.09, "k_pct": 0.20, "sprint_speed": 28.0},
            {"player_id": 1, "season_year": 2024, "pa": 300, "woba": 0.280,
             "iso": 0.12, "bb_pct": 0.07, "k_pct": 0.25, "sprint_speed": 27.0},
        ])

        builder = PriorSeasonBuilder()
        result = builder.build(
            stats, as_of_date=date(2024, 6, 30),
            season_batting=season_batting,
            player_type="hitter",
        )
        # Should use 2023 data for 2024 games
        assert result["prior_woba"].values[0] == pytest.approx(0.350)


# ---------------------------------------------------------------------------
# Pipeline assembly tests
# ---------------------------------------------------------------------------

class TestAssembleFeatures:
    """Tests for features.pipeline.assemble_features."""

    def test_returns_both_player_types(self) -> None:
        from packages.shared.schemas.ml import PlayerType

        from packages.ml.features.pipeline import assemble_features

        batting = _make_batting_stats(n_games=10, team_id=100)
        pitching = _make_pitching_stats(n_games=6, team_id=200)
        games = _make_games(
            list(batting["game_pk"]) + list(pitching["game_pk"]),
            home_team=100, away_team=200,
        )
        park_factors = pd.DataFrame(columns=[
            "venue_id", "season_year", "park_factor_runs", "park_factor_hr",
        ])
        season_batting = pd.DataFrame(columns=[
            "player_id", "season_year", "pa", "woba", "iso",
            "bb_pct", "k_pct", "sprint_speed",
        ])
        season_pitching = pd.DataFrame(columns=[
            "player_id", "season_year", "ip", "fip", "k_pct",
            "bb_pct", "gb_pct", "fastball_velo",
        ])
        players = pd.DataFrame(columns=["player_id", "mlb_debut_date"])

        result = assemble_features(
            batting_stats=batting,
            pitching_stats=pitching,
            games=games,
            park_factors=park_factors,
            season_batting=season_batting,
            season_pitching=season_pitching,
            players=players,
            scoring_rules_batting={"H": 1.0, "HR": 3.0},
            scoring_rules_pitching={"IP": 3.0, "SO": 1.0},
            start_date=date(2024, 6, 1),
            end_date=date(2024, 9, 30),
        )

        assert PlayerType.HITTER in result
        assert PlayerType.PITCHER in result
        assert len(result[PlayerType.HITTER]) > 0
        assert len(result[PlayerType.PITCHER]) > 0

    def test_output_has_required_metadata(self) -> None:
        from packages.shared.schemas.ml import PlayerType

        from packages.ml.features.pipeline import assemble_features

        batting = _make_batting_stats(n_games=5, team_id=100)
        pitching = _make_pitching_stats(n_games=3, team_id=200)
        games = _make_games(
            list(batting["game_pk"]) + list(pitching["game_pk"]),
        )

        result = assemble_features(
            batting_stats=batting,
            pitching_stats=pitching,
            games=games,
            park_factors=pd.DataFrame(columns=[
                "venue_id", "season_year", "park_factor_runs", "park_factor_hr",
            ]),
            season_batting=pd.DataFrame(columns=[
                "player_id", "season_year", "pa", "woba", "iso",
                "bb_pct", "k_pct", "sprint_speed",
            ]),
            season_pitching=pd.DataFrame(columns=[
                "player_id", "season_year", "ip", "fip", "k_pct",
                "bb_pct", "gb_pct", "fastball_velo",
            ]),
            players=pd.DataFrame(columns=["player_id", "mlb_debut_date"]),
            scoring_rules_batting=None,
            scoring_rules_pitching=None,
            start_date=date(2024, 6, 1),
            end_date=date(2024, 6, 30),
        )

        hitter_df = result[PlayerType.HITTER]
        required_cols = {
            "player_id", "game_pk", "game_date", "season_year",
            "player_type", "fantasy_points",
        }
        assert required_cols.issubset(set(hitter_df.columns))
        assert (hitter_df["player_type"] == "hitter").all()


# ---------------------------------------------------------------------------
# Leakage guard tests
# ---------------------------------------------------------------------------

class TestLeakageGuards:
    """Verify no feature uses game_date or later data.

    These tests construct a DataFrame with known future data and assert that
    feature builders produce the same output whether or not the future rows
    are present.
    """

    def test_rolling_ignores_future_rows(self) -> None:
        """Rolling builder output for game G must be identical whether future
        games G+1, G+2, ... are present in the input or not."""
        from packages.ml.features.rolling import RollingProductionBuilder

        stats_full = _make_batting_stats(n_games=10)
        stats_partial = stats_full[stats_full["game_pk"] <= 100004].copy()

        builder = RollingProductionBuilder()
        # Use as_of_date = June 5 to match partial data
        result_full = builder.build(
            stats_full, as_of_date=date(2024, 6, 5), player_type="hitter",
        )
        result_partial = builder.build(
            stats_partial, as_of_date=date(2024, 6, 5), player_type="hitter",
        )

        # Both should have same rows and values
        assert len(result_full) == len(result_partial)
        common_cols = [c for c in result_full.columns if c.startswith("roll_")]
        for col in common_cols:
            np.testing.assert_array_almost_equal(
                result_full[col].values,
                result_partial[col].values,
                err_msg=f"Mismatch in {col}: future data leaked",
            )

    def test_rest_ignores_future_rows(self) -> None:
        from packages.ml.features.rest import RestRecencyBuilder

        stats_full = _make_batting_stats(n_games=10)
        stats_partial = stats_full[stats_full["game_pk"] <= 100004].copy()

        builder = RestRecencyBuilder()
        result_full = builder.build(stats_full, as_of_date=date(2024, 6, 5))
        result_partial = builder.build(stats_partial, as_of_date=date(2024, 6, 5))

        assert len(result_full) == len(result_partial)
        for col in ["rest_days_since_last_game", "rest_games_in_last_7d"]:
            np.testing.assert_array_almost_equal(
                result_full[col].values,
                result_partial[col].values,
                err_msg=f"Mismatch in {col}: future data leaked",
            )

    def test_lineup_ignores_future_rows(self) -> None:
        from packages.ml.features.lineup import LineupSlotBuilder

        stats_full = _make_batting_stats(n_games=10)
        stats_partial = stats_full[stats_full["game_pk"] <= 100004].copy()

        builder = LineupSlotBuilder()
        result_full = builder.build(stats_full, as_of_date=date(2024, 6, 5))
        result_partial = builder.build(stats_partial, as_of_date=date(2024, 6, 5))

        assert len(result_full) == len(result_partial)
        np.testing.assert_array_almost_equal(
            result_full["lineup_batting_order_slot"].values,
            result_partial["lineup_batting_order_slot"].values,
        )

    def test_workload_ignores_future_rows(self) -> None:
        from packages.ml.features.workload import PitcherWorkloadBuilder

        stats_full = _make_pitching_stats(n_games=10)
        # as_of_date that includes first 3 games (every 5 days: June 1, 6, 11)
        as_of = date(2024, 6, 11)
        stats_partial = stats_full[
            pd.to_datetime(stats_full["game_date"]).dt.date <= as_of
        ].copy()

        builder = PitcherWorkloadBuilder()
        result_full = builder.build(stats_full, as_of_date=as_of)
        result_partial = builder.build(stats_partial, as_of_date=as_of)

        assert len(result_full) == len(result_partial)
        np.testing.assert_array_almost_equal(
            result_full["workload_mean_ip_5starts"].values,
            result_partial["workload_mean_ip_5starts"].values,
        )

    def test_rolling_shift_excludes_current_game(self) -> None:
        """Verify that rolling feature for game N does NOT include game N's data.

        This is the core anti-leakage test. We create a scenario where game N
        has extreme stats and verify the feature value doesn't reflect them.
        """
        from packages.ml.features.rolling import RollingProductionBuilder

        # Create 5 games with consistent stats (daily: June 1-5)
        stats = _make_batting_stats(n_games=5)

        builder = RollingProductionBuilder()
        result = builder.build(stats, as_of_date=date(2024, 6, 30), player_type="hitter")

        # For game 3 (game_pk=100002, June 3), the rolling_7d should only
        # include games 0 and 1 (June 1, June 2) -- shifted by 1, then
        # 7-day calendar window covers all prior games within 7 days.
        game2_roll = result[result["game_pk"] == 100002]["roll_h_7d"].values[0]
        # Games 0 and 1 h values: h = 1 + (i % 2) -> 1, 2 -> sum = 3
        expected_h = stats[stats["game_pk"].isin([100000, 100001])]["h"].sum()
        assert game2_roll == pytest.approx(expected_h)

    def test_context_no_leakage(self) -> None:
        """Context builder uses only pre-game observable data."""
        from packages.ml.features.context import ContextBuilder

        stats = _make_batting_stats(n_games=2, team_id=100)
        games = _make_games([100000, 100001])

        builder = ContextBuilder()
        result = builder.build(
            stats, as_of_date=date(2024, 6, 30), games=games,
        )
        # is_home and park factors are known before game starts
        assert "ctx_is_home" in result.columns
        assert len(result) == 2
