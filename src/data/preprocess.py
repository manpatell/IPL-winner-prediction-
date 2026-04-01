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
    """Normalize team names and keep only active franchises for training."""
    alias_map = TEAM_ALIASES
    for col in ["team1", "team2", "winner", "toss_winner"]:
        df[col] = df[col].replace(alias_map)

    active = set(ACTIVE_TEAMS_2026)
    before = len(df)
    df = df[
        df["team1"].isin(active)
        & df["team2"].isin(active)
        & (df["winner"].isna() | df["winner"].isin(active))
    ].copy()
    print(f"  Dropped {before - len(df)} matches with retired/inactive teams")
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
    """Sort by season and match_id to ensure temporal ordering."""
    df = df.sort_values(["season", "match_id"]).reset_index(drop=True)
    return df


def mirror_matches(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fixes the 73% class imbalance by adding a mirrored copy of every match.

    For every match (team1 vs team2, winner=team1), we add the equivalent row
    with team1/team2 swapped and winner flipped. This:
      - Doubles the dataset size
      - Makes class distribution exactly 50/50
      - Eliminates position bias (the model can't learn 'team1 always wins')
      - Is statistically legitimate: (A beat B) is the same information as (B lost to A)

    The mirrored rows get negative match_ids to distinguish them from originals.
    """
    mirrored = df.copy()
    mirrored["match_id"]        = -mirrored["match_id"]
    mirrored["team1"]           = df["team2"]
    mirrored["team2"]           = df["team1"]
    mirrored["toss_winner"]     = df["toss_winner"]  # unchanged (still same team)
    mirrored["toss_won_by_team1"] = 1 - df["toss_won_by_team1"]  # flip toss flag
    mirrored["team1_won"]       = 1 - df["team1_won"]

    combined = pd.concat([df, mirrored], ignore_index=True)
    combined = combined.sort_values(["season", "match_id"]).reset_index(drop=True)

    orig_pos = df["team1_won"].mean()
    new_pos  = combined["team1_won"].mean()
    print(f"  Class balance before mirroring: {orig_pos:.2%} team1 wins")
    print(f"  Class balance  after mirroring: {new_pos:.2%} team1 wins (target: 50%)")
    return combined


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
    df = mirror_matches(df)
    save_processed(df)
    return df


if __name__ == "__main__":
    run_preprocessing()
