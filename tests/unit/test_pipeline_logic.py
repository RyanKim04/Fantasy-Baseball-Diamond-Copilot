"""Unit tests for pure pipeline helper functions.

These functions are stateless and do not require DB or API access.
"""

from __future__ import annotations


class TestParseInningsPitched:
    """Tests for _parse_innings_pitched from ingest_boxscores."""

    @staticmethod
    def _parse(ip_str: str) -> float:
        from packages.pipelines.ingest_boxscores import _parse_innings_pitched

        return _parse_innings_pitched(ip_str)

    def test_whole_innings(self) -> None:
        assert self._parse("6.0") == 6.0

    def test_one_third(self) -> None:
        result = self._parse("6.1")
        assert abs(result - 6.333333333) < 0.001

    def test_two_thirds(self) -> None:
        result = self._parse("6.2")
        assert abs(result - 6.666666667) < 0.001

    def test_zero(self) -> None:
        assert self._parse("0") == 0.0

    def test_empty_string(self) -> None:
        assert self._parse("") == 0.0

    def test_nine_innings(self) -> None:
        assert self._parse("9.0") == 9.0


class TestPctToFloat:
    """Tests for _pct_to_float from seed_historical."""

    @staticmethod
    def _pct(val: object) -> float | None:
        from packages.pipelines.seed_historical import _pct_to_float

        return _pct_to_float(val)

    def test_percentage_value(self) -> None:
        result = self._pct(25.3)
        assert result is not None
        assert abs(result - 0.253) < 0.001

    def test_already_decimal(self) -> None:
        result = self._pct(0.253)
        assert result is not None
        assert abs(result - 0.253) < 0.001

    def test_none(self) -> None:
        assert self._pct(None) is None

    def test_zero(self) -> None:
        result = self._pct(0.0)
        assert result == 0.0

    def test_negative_percentage(self) -> None:
        """Negative percentages (e.g., K-BB%) should also be divided by 100."""
        result = self._pct(-5.0)
        assert result is not None
        assert abs(result - (-0.05)) < 0.001


class TestSafeFloat:
    """Tests for _safe_float from seed_historical."""

    @staticmethod
    def _sf(val: object) -> float | None:
        from packages.pipelines.seed_historical import _safe_float

        return _safe_float(val)

    def test_normal_float(self) -> None:
        assert self._sf(3.14) == 3.14

    def test_none(self) -> None:
        assert self._sf(None) is None

    def test_nan(self) -> None:

        assert self._sf(float("nan")) is None

    def test_percentage_string(self) -> None:
        result = self._sf("25.3 %")
        assert result is not None
        assert abs(result - 25.3) < 0.001

    def test_int(self) -> None:
        assert self._sf(42) == 42.0

    def test_empty_string(self) -> None:
        assert self._sf("") is None


class TestExtractYahooPlayerId:
    """Tests for _extract_yahoo_player_id from ingest_yahoo."""

    @staticmethod
    def _extract(yahoo_id: str) -> int | None:
        from packages.pipelines.ingest_yahoo import _extract_yahoo_player_id

        return _extract_yahoo_player_id(yahoo_id)

    def test_standard_format(self) -> None:
        assert self._extract("mlb.p.545361") == 545361

    def test_numeric_string(self) -> None:
        assert self._extract("545361") == 545361

    def test_garbage_returns_none(self) -> None:
        assert self._extract("garbage") is None

    def test_partial_format(self) -> None:
        assert self._extract("mlb.545361") is None or self._extract("mlb.545361") == 545361

    def test_empty_string(self) -> None:
        assert self._extract("") is None

    def test_nfl_format_still_extracts(self) -> None:
        """Even non-MLB Yahoo IDs with the right format should extract."""
        assert self._extract("nfl.p.12345") == 12345
