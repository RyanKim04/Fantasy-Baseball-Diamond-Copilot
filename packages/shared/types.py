"""Branded NewType aliases for cross-module type safety."""

from typing import NewType

# MLBAM player ID (e.g. 545361 = Mike Trout)
PlayerID = NewType("PlayerID", int)

# MLB game_pk — unique game identifier from MLB Stats API
GameID = NewType("GameID", int)

# Yahoo league key (e.g. "422.l.12345")
LeagueID = NewType("LeagueID", str)

# Yahoo team key (e.g. "422.l.12345.t.1")
TeamID = NewType("TeamID", str)
