"""
Data preprocessing: reads raw matches from SQLite and outputs a clean
processed DataFrame saved as CSV for feature engineering.
"""
import os
import sys
import sqlite3
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import (
    SQLITE_DB_PATH, PROCESSED_MATCHES_CSV, PROCESSED_DIR,
    ACTIVE_TEAMS_2026, TEAM_ALIASES,
)

# Teams that are currently active (drop retired franchise data)
RETIRED = {"DC_OLD", "RPS", "KTK", "PW"}


def load_matches() -> pd.DataFrame:
    conn = sqlite3.connect(SQLITE_DB_PATH)
    df = pd.read_sql_query("""
        SELECT m.match_id, m.season, m.team1, m.team2,
               m.toss_winner, m.toss_decision,
               m.winner, m.win_by_runs, m.win_by_wickets,
               m.venue,
               v.city
        FROM matches m
        LEFT JOIN venues v ON m.venue = v.name
        ORDER BY m.season, m.match_id
    """, conn)
    conn.close()
    return df


def normalize_teams(df: pd.DataFrame) -> pd.DataFrame:
    """Map old franchise names to current abbreviations."""
    alias_map = {
        "DC_OLD": "SRH",   # Deccan Chargers became SRH
        "RPS":    "CSK",   # Rising Pune Supergiant - treat as neutral
    }
    for col in ["team1", "team2", "winner", "toss_winner"]:
        df[col] = df[col].replace(alias_map)
    return df


def add_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    """For each row add: team1_won (1 if team1 won, 0 if team2 won)."""
    df = df.copy()
    df["team1_won"] = (df["winner"] == df["team1"]).astype(int)
    # Drop draws / no-result
    df = df[df["winner"].notna() & (df["winner"] != "")]
    return df


def add_toss_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["toss_won_by_team1"] = (df["toss_winner"] == df["team1"]).astype(int)
    df["toss_decision_bat"] = (df["toss_decision"] == "bat").astype(int)
    return df


def add_season_order(df: pd.DataFrame) -> pd.DataFrame:
    """Add a running 'game_number' within each season for each team."""
    df = df.sort_values(["season", "match_id"]).reset_index(drop=True)
    return df


def save_processed(df: pd.DataFrame):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df.to_csv(PROCESSED_MATCHES_CSV, index=False)
    print(f"Processed {len(df)} matches → {PROCESSED_MATCHES_CSV}")
    print(f"Columns: {list(df.columns)}")
    print(df.head(3).to_string())


def run_preprocessing() -> pd.DataFrame:
    df = load_matches()
    df = normalize_teams(df)
    df = add_binary_target(df)
    df = add_toss_features(df)
    df = add_season_order(df)
    save_processed(df)
    return df


if __name__ == "__main__":
    run_preprocessing()
