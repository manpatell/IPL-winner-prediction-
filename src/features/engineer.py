"""
Feature engineering for IPL match prediction.

Features generated per match row (team1 vs team2):
 - Toss win / decision
 - Win percentage (all-time, last 3 seasons, last 5 matches)
 - Head-to-head win rate
 - Home ground advantage
 - Title count ratio
 - Recent form streak (wins in last 5 matches)
 - Venue win rate for team
"""
import os
import sys
import sqlite3
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import (
    SQLITE_DB_PATH, PROCESSED_MATCHES_CSV, FEATURES_CSV,
    PROCESSED_DIR, FORM_WINDOW, H2H_WINDOW_SEASONS,
)


def get_all_time_win_rates(matches: pd.DataFrame) -> dict:
    """Returns {team: win_rate} over all historical matches."""
    rates = {}
    all_teams = set(matches["team1"]) | set(matches["team2"])
    for team in all_teams:
        played = ((matches["team1"] == team) | (matches["team2"] == team)).sum()
        won = (matches["winner"] == team).sum()
        rates[team] = won / played if played > 0 else 0.5
    return rates


def get_recent_form(matches: pd.DataFrame, team: str, before_idx: int, n: int = 5) -> float:
    """Win rate of `team` in its last `n` matches before row index `before_idx`."""
    past = matches.iloc[:before_idx]
    team_matches = past[(past["team1"] == team) | (past["team2"] == team)]
    recent = team_matches.tail(n)
    if len(recent) == 0:
        return 0.5
    wins = (recent["winner"] == team).sum()
    return wins / len(recent)


def get_h2h_rate(matches: pd.DataFrame, team1: str, team2: str, before_idx: int,
                 window_seasons: int = 3) -> float:
    """Head-to-head win rate of team1 vs team2 over last `window_seasons` seasons."""
    past = matches.iloc[:before_idx]
    if len(past) == 0:
        return 0.5
    recent_seasons = sorted(past["season"].unique())[-window_seasons:]
    h2h = past[
        (past["season"].isin(recent_seasons)) &
        (((past["team1"] == team1) & (past["team2"] == team2)) |
         ((past["team1"] == team2) & (past["team2"] == team1)))
    ]
    if len(h2h) == 0:
        return 0.5
    wins1 = (h2h["winner"] == team1).sum()
    return wins1 / len(h2h)


def get_venue_win_rate(matches: pd.DataFrame, team: str, venue: str, before_idx: int) -> float:
    """Win rate of `team` at a specific venue."""
    past = matches.iloc[:before_idx]
    at_venue = past[
        (past["venue"] == venue) &
        ((past["team1"] == team) | (past["team2"] == team))
    ]
    if len(at_venue) == 0:
        return 0.5
    wins = (at_venue["winner"] == team).sum()
    return wins / len(at_venue)


def is_home_ground(team: str, venue: str) -> int:
    """Returns 1 if venue is team's home ground."""
    home_map = {
        "CSK":  "MA Chidambaram Stadium",
        "MI":   "Wankhede Stadium",
        "RCB":  "M Chinnaswamy Stadium",
        "KKR":  "Eden Gardens",
        "DC":   "Feroz Shah Kotla",
        "RR":   "Sawai Mansingh Stadium",
        "SRH":  "Rajiv Gandhi Intl Stadium",
        "PBKS": "Punjab Cricket Association Stadium",
        "GT":   "Narendra Modi Stadium",
        "LSG":  "BRSABV Ekana Cricket Stadium",
    }
    return int(home_map.get(team, "") == venue)


TITLE_COUNTS = {
    "CSK": 5, "MI": 5, "KKR": 3, "SRH": 1, "RR": 1,
    "GT": 1,  "RCB": 0, "DC": 0, "PBKS": 0, "LSG": 0,
}


def get_season_form(matches: pd.DataFrame, team: str, season: int, before_idx: int) -> float:
    """Win rate within the current season up to before_idx."""
    season_matches = matches.iloc[:before_idx]
    season_matches = season_matches[season_matches["season"] == season]
    team_matches = season_matches[(season_matches["team1"] == team) | (season_matches["team2"] == team)]
    if len(team_matches) == 0:
        return 0.5
    wins = (team_matches["winner"] == team).sum()
    return wins / len(team_matches)


def build_features(matches_csv: str = PROCESSED_MATCHES_CSV) -> pd.DataFrame:
    df = pd.read_csv(matches_csv)
    df = df.reset_index(drop=True)

    # Pre-compute all-time rates (snapshot at end - used as a static prior)
    overall_rates = get_all_time_win_rates(df)

    rows = []
    for idx, row in df.iterrows():
        t1 = row["team1"]
        t2 = row["team2"]
        venue = row.get("venue", "")
        season = int(row["season"])

        f = {}

        # Basic identifiers
        f["match_id"]  = row["match_id"]
        f["season"]    = season
        f["team1"]     = t1
        f["team2"]     = t2

        # ── Toss features ───────────────────────────────────────────────────
        f["toss_won_by_team1"] = int(row.get("toss_won_by_team1", 0))
        f["toss_decision_bat"] = int(row.get("toss_decision_bat", 0))

        # ── All-time win rates ───────────────────────────────────────────────
        f["t1_alltime_wr"]  = overall_rates.get(t1, 0.5)
        f["t2_alltime_wr"]  = overall_rates.get(t2, 0.5)
        f["wr_diff"]        = f["t1_alltime_wr"] - f["t2_alltime_wr"]

        # ── Recent form (last 5 matches before this match) ───────────────────
        f["t1_recent_form"]    = get_recent_form(df, t1, idx, FORM_WINDOW)
        f["t2_recent_form"]    = get_recent_form(df, t2, idx, FORM_WINDOW)
        f["form_diff"]         = f["t1_recent_form"] - f["t2_recent_form"]

        # ── Season form up to this match ─────────────────────────────────────
        f["t1_season_form"] = get_season_form(df, t1, season, idx)
        f["t2_season_form"] = get_season_form(df, t2, season, idx)

        # ── Head-to-head ─────────────────────────────────────────────────────
        f["h2h_t1_wr"] = get_h2h_rate(df, t1, t2, idx, H2H_WINDOW_SEASONS)

        # ── Venue features ───────────────────────────────────────────────────
        f["t1_venue_wr"]    = get_venue_win_rate(df, t1, venue, idx)
        f["t2_venue_wr"]    = get_venue_win_rate(df, t2, venue, idx)
        f["venue_wr_diff"]  = f["t1_venue_wr"] - f["t2_venue_wr"]

        # ── Home ground advantage ─────────────────────────────────────────────
        f["t1_is_home"] = is_home_ground(t1, venue)
        f["t2_is_home"] = is_home_ground(t2, venue)

        # ── Title prestige ────────────────────────────────────────────────────
        f["t1_titles"] = TITLE_COUNTS.get(t1, 0)
        f["t2_titles"] = TITLE_COUNTS.get(t2, 0)
        f["title_diff"] = f["t1_titles"] - f["t2_titles"]

        # ── Target ────────────────────────────────────────────────────────────
        f["team1_won"] = int(row["team1_won"])

        rows.append(f)

    features_df = pd.DataFrame(rows)
    return features_df


def save_features(df: pd.DataFrame):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df.to_csv(FEATURES_CSV, index=False)
    print(f"Features saved: {len(df)} rows × {len(df.columns)} cols → {FEATURES_CSV}")
    feature_cols = [c for c in df.columns if c not in ("match_id", "season", "team1", "team2", "team1_won")]
    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")


def run_feature_engineering() -> pd.DataFrame:
    df = build_features()
    save_features(df)
    return df


if __name__ == "__main__":
    run_feature_engineering()
