"""
Team batting and bowling strength features derived from player_stats table.

For each team × season, computes:
  - Top-3 batsmen average (higher → better batting depth)
  - Top-3 bowlers economy (lower → better bowling attack)
  - Batting vs bowling balance score

These are causal features: a team with high batting avg will tend to win.
Unlike win-rate features (which are outcomes), these capture WHY a team is strong.
"""
import os
import sys
import sqlite3
import pandas as pd
import numpy as np
from functools import lru_cache

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import SQLITE_DB_PATH

# IPL average batting avg and economy (for normalization)
IPL_AVG_BATTING_AVG = 28.0
IPL_AVG_ECONOMY     = 8.5
IPL_AVG_SR          = 135.0


@lru_cache(maxsize=None)
def load_player_stats_cache() -> pd.DataFrame:
    """Load player stats once and cache."""
    conn = sqlite3.connect(SQLITE_DB_PATH)
    df = pd.read_sql_query(
        "SELECT season, player_name, team, role, batting_avg, batting_sr, "
        "       runs_scored, wickets, bowling_avg, economy "
        "FROM player_stats ORDER BY season, team",
        conn,
    )
    conn.close()
    return df


def get_team_batting_strength(team: str, season: int) -> float:
    """
    Average batting average of top-3 run-scorers for this team in this season.
    Normalized to 0–1 range.
    Falls back to IPL average if no data.
    """
    df = load_player_stats_cache()
    batsmen = df[
        (df["team"] == team) &
        (df["season"] == season) &
        (df["batting_avg"] > 0)
    ].nlargest(3, "batting_avg")

    if len(batsmen) == 0:
        # Try previous season
        prev = df[(df["team"] == team) & (df["season"] == season - 1) &
                  (df["batting_avg"] > 0)].nlargest(3, "batting_avg")
        if len(prev) == 0:
            return IPL_AVG_BATTING_AVG / 60.0
        avg = prev["batting_avg"].mean()
    else:
        avg = batsmen["batting_avg"].mean()

    return float(np.clip(avg / 60.0, 0, 1))  # 60 avg = top tier


def get_team_bowling_strength(team: str, season: int) -> float:
    """
    Average economy of top-3 wicket-takers for this team (lower economy = better).
    Normalized to 0–1 range where 1 = best (most economical).
    """
    df = load_player_stats_cache()
    bowlers = df[
        (df["team"] == team) &
        (df["season"] == season) &
        (df["wickets"] > 0) &
        (df["economy"] > 0)
    ].nlargest(3, "wickets")

    if len(bowlers) == 0:
        prev = df[(df["team"] == team) & (df["season"] == season - 1) &
                  (df["wickets"] > 0)].nlargest(3, "wickets")
        if len(prev) == 0:
            return (12.0 - IPL_AVG_ECONOMY) / 6.0
        econ = prev["economy"].mean()
    else:
        econ = bowlers["economy"].mean()

    # Lower economy = better. Map [12, 6] → [0, 1]
    return float(np.clip((12.0 - econ) / 6.0, 0, 1))


def get_team_allrounder_strength(team: str, season: int) -> float:
    """
    Fraction of team's players who are allrounders (bat + bowl).
    Allrounders increase team flexibility and depth.
    """
    df = load_player_stats_cache()
    team_players = df[(df["team"] == team) & (df["season"] == season)]
    if len(team_players) == 0:
        return 0.2  # IPL average ~2 allrounders in 11
    allrounders = team_players[
        (team_players["batting_avg"] > 15) &
        (team_players["wickets"] > 0)
    ]
    return min(len(allrounders) / max(len(team_players), 1), 1.0)


def get_team_strength_features(team: str, season: int) -> dict:
    """Returns all team strength features for a given team and season."""
    bat = get_team_batting_strength(team, season)
    bowl = get_team_bowling_strength(team, season)
    allround = get_team_allrounder_strength(team, season)
    return {
        "batting_strength": bat,
        "bowling_strength": bowl,
        "allround_strength": allround,
        "bat_bowl_balance": abs(bat - bowl),   # closer to 0 = balanced team
    }
