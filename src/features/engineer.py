"""
Feature engineering for IPL match prediction.

Features generated per match row (team1 vs team2):
 - Toss win / decision
 - Win percentage: all-time (smoothed), last 3 seasons, last 5 matches
 - Head-to-head win rate (last 3 seasons)
 - Home ground advantage
 - Recent titles (last 5 seasons only — NOT all-time)
 - Season form
 - Venue win rate for each team
"""
import os
import sys
import sqlite3
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import (
    PROCESSED_MATCHES_CSV, FEATURES_CSV,
    PROCESSED_DIR, FORM_WINDOW, H2H_WINDOW_SEASONS, SQLITE_DB_PATH,
)
from src.features.venue_features import (
    get_venue_avg_score, get_venue_toss_impact, get_venue_size,
)
from src.features.team_strength import get_team_strength_features

# Titles won in the LAST 5 SEASONS only (2020–2024).
# This reflects current dynasty strength, not historical legacy.
# KKR-2024, CSK-2023+2021, GT-2022, MI-2020
RECENT_TITLES_5YR = {
    "CSK":  2,  # 2021, 2023
    "MI":   1,  # 2020
    "KKR":  1,  # 2024
    "GT":   1,  # 2022
    "SRH":  0,
    "RR":   0,
    "RCB":  0,
    "DC":   0,
    "PBKS": 0,
    "LSG":  0,
}

# All-time title counts (kept only for historical feature context,
# NO LONGER used directly as a prediction prior)
TITLE_COUNTS = {
    "CSK": 5, "MI": 5, "KKR": 3, "SRH": 1, "RR": 1,
    "GT": 1,  "RCB": 0, "DC": 0, "PBKS": 0, "LSG": 0,
}


def get_all_time_win_rates(matches: pd.DataFrame) -> dict:
    """
    Returns {team: bayesian_smoothed_win_rate} using ONLY the provided matches.
    Called with matches.iloc[:idx] during feature building to prevent data leakage:
    a 2010 match must not know about 2015 results.

    Bayesian smoothing: blend raw win rate with 0.5 prior (10 ghost matches).
    New teams (GT, LSG with few matches) get pulled toward 0.5 — fair.
    """
    rates = {}
    all_teams = set(matches["team1"]) | set(matches["team2"])
    prior_weight = 10  # 10 ghost matches at 50%
    for team in all_teams:
        played = int(((matches["team1"] == team) | (matches["team2"] == team)).sum())
        won    = int((matches["winner"] == team).sum())
        rates[team] = (won + prior_weight * 0.5) / (played + prior_weight)
    return rates


def get_last_n_seasons_wr(matches: pd.DataFrame, team: str,
                           before_season: int, n_seasons: int = 3) -> float:
    """
    Win rate over the N most recent completed seasons before `before_season`.
    For new teams (GT, LSG) with fewer than N seasons, uses all available seasons.
    Falls back to 0.5 if no data.
    Critically: this reflects CURRENT form, not 15-year legacy.
    """
    relevant = matches[(matches["season"] < before_season) &
                       ((matches["team1"] == team) | (matches["team2"] == team))]
    if len(relevant) == 0:
        return 0.5

    available_seasons = sorted(relevant["season"].unique())[-n_seasons:]
    recent = relevant[relevant["season"].isin(available_seasons)]

    if len(recent) == 0:
        return 0.5

    wins = (recent["winner"] == team).sum()
    # Bayesian smoothing: blend with 0.5 if very few matches
    prior_weight = 4
    return (wins + prior_weight * 0.5) / (len(recent) + prior_weight)


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


def get_season_form(matches: pd.DataFrame, team: str, season: int, before_idx: int) -> float:
    """Win rate within the current season up to before_idx."""
    season_matches = matches.iloc[:before_idx]
    season_matches = season_matches[season_matches["season"] == season]
    team_matches = season_matches[(season_matches["team1"] == team) | (season_matches["team2"] == team)]
    if len(team_matches) == 0:
        return 0.5
    wins = (team_matches["winner"] == team).sum()
    return wins / len(team_matches)


def load_champions_by_season() -> dict:
    """Load season champions from SQLite season_stats table."""
    if not os.path.exists(SQLITE_DB_PATH):
        return {}
    conn = sqlite3.connect(SQLITE_DB_PATH)
    try:
        rows = conn.execute(
            "SELECT season, team FROM season_stats WHERE won_title = 1"
        ).fetchall()
        return {int(season): team for season, team in rows}
    finally:
        conn.close()


def get_recent_titles(team: str, before_season: int, champions_by_season: dict,
                      window: int = 5) -> int:
    """
    Titles won by this team in the `window` seasons before `before_season`.
    Only counts recent wins — treats a 2024 title the same as 2020,
    not stacked against a 2008 title.
    """
    count = 0
    for season in range(before_season - window, before_season):
        if champions_by_season.get(season) == team:
            count += 1
    return count


def build_features(matches_csv: str = PROCESSED_MATCHES_CSV) -> pd.DataFrame:
    df = pd.read_csv(matches_csv)
    df = df.reset_index(drop=True)
    champions_by_season = load_champions_by_season()

    # Static prior (used for first 10 rows and prediction): full-dataset rates.
    # Feature rows use cumulative rates (see inner loop) to avoid leakage.
    overall_rates = get_all_time_win_rates(df)

    rows = []
    for idx, row in df.iterrows():
        t1 = row["team1"]
        t2 = row["team2"]
        venue = row.get("venue", "")
        season = int(row["season"])

        f = {}

        # Basic identifiers
        f["match_id"] = row["match_id"]
        f["season"]   = season
        f["team1"]    = t1
        f["team2"]    = t2

        # ── Toss features ────────────────────────────────────────────────────
        f["toss_won_by_team1"] = int(row.get("toss_won_by_team1", 0))
        f["toss_decision_bat"] = int(row.get("toss_decision_bat", 0))

        # ── Cumulative all-time win rates (NO LEAKAGE) ────────────────────────
        # Computed on df.iloc[:idx] — ONLY matches BEFORE this one.
        # A 2010 match will NOT know about 2015 results.
        # For idx=0 (first match ever), no data → use end-of-dataset rates as static prior.
        if idx > 10:
            past_rates = get_all_time_win_rates(df.iloc[:idx])
        else:
            past_rates = overall_rates  # early matches: use static prior
        f["t1_alltime_wr"] = past_rates.get(t1, 0.5)
        f["t2_alltime_wr"] = past_rates.get(t2, 0.5)
        f["wr_diff"]       = f["t1_alltime_wr"] - f["t2_alltime_wr"]

        # ── Last 3 seasons win rate ──────────────────────────────────────────
        # Captures CURRENT strength: CSK's 2022+2024 struggles show here;
        # GT/LSG's recent competitiveness shows here too.
        f["t1_last3yr_wr"]   = get_last_n_seasons_wr(df, t1, season, n_seasons=3)
        f["t2_last3yr_wr"]   = get_last_n_seasons_wr(df, t2, season, n_seasons=3)
        f["last3yr_wr_diff"] = f["t1_last3yr_wr"] - f["t2_last3yr_wr"]

        # ── Recent match form (last 5 matches) ──────────────────────────────
        f["t1_recent_form"] = get_recent_form(df, t1, idx, FORM_WINDOW)
        f["t2_recent_form"] = get_recent_form(df, t2, idx, FORM_WINDOW)
        f["form_diff"]      = f["t1_recent_form"] - f["t2_recent_form"]

        # ── Season form up to this match ─────────────────────────────────────
        f["t1_season_form"] = get_season_form(df, t1, season, idx)
        f["t2_season_form"] = get_season_form(df, t2, season, idx)

        # ── Head-to-head ─────────────────────────────────────────────────────
        f["h2h_t1_wr"] = get_h2h_rate(df, t1, t2, idx, H2H_WINDOW_SEASONS)

        # ── Venue features ───────────────────────────────────────────────────
        f["t1_venue_wr"]   = get_venue_win_rate(df, t1, venue, idx)
        f["t2_venue_wr"]   = get_venue_win_rate(df, t2, venue, idx)
        f["venue_wr_diff"] = f["t1_venue_wr"] - f["t2_venue_wr"]

        # ── Home ground advantage ─────────────────────────────────────────────
        f["t1_is_home"] = is_home_ground(t1, venue)
        f["t2_is_home"] = is_home_ground(t2, venue)

        # ── Recent titles (last 5 seasons) — NOT all-time ────────────────────
        f["t1_recent_titles"]   = get_recent_titles(t1, season, champions_by_season, window=5)
        f["t2_recent_titles"]   = get_recent_titles(t2, season, champions_by_season, window=5)
        f["recent_title_diff"]  = f["t1_recent_titles"] - f["t2_recent_titles"]

        # ── Venue pitch features ──────────────────────────────────────────────
        f["venue_avg_score"]    = get_venue_avg_score(venue)
        f["venue_toss_impact"]  = get_venue_toss_impact(venue)
        f["venue_size"]         = get_venue_size(venue)

        # ── Team batting/bowling strength (causal features) ───────────────────
        t1_str = get_team_strength_features(t1, season)
        t2_str = get_team_strength_features(t2, season)
        f["t1_batting_str"]   = t1_str["batting_strength"]
        f["t2_batting_str"]   = t2_str["batting_strength"]
        f["batting_str_diff"] = t1_str["batting_strength"] - t2_str["batting_strength"]
        f["t1_bowling_str"]   = t1_str["bowling_strength"]
        f["t2_bowling_str"]   = t2_str["bowling_strength"]
        f["bowling_str_diff"] = t1_str["bowling_strength"] - t2_str["bowling_strength"]

        # ── Target ────────────────────────────────────────────────────────────
        f["team1_won"] = int(row["team1_won"])

        rows.append(f)

    features_df = pd.DataFrame(rows)
    return features_df


def save_features(df: pd.DataFrame):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df.to_csv(FEATURES_CSV, index=False)
    print(f"Features saved: {len(df)} rows × {len(df.columns)} cols → {FEATURES_CSV}")
    feature_cols = [c for c in df.columns
                    if c not in ("match_id", "season", "team1", "team2", "team1_won")]
    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")


def run_feature_engineering() -> pd.DataFrame:
    df = build_features()
    save_features(df)
    return df


if __name__ == "__main__":
    run_feature_engineering()
