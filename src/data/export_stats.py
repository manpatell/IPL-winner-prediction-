"""
Exports aggregated team statistics from SQLite to CSV for external analysis.
Produces: team_stats.csv, head_to_head_matrix.csv
"""
import os
import sys
import sqlite3
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import SQLITE_DB_PATH, PROCESSED_DIR, TEAM_STATS_CSV, ACTIVE_TEAMS_2026


def export_team_stats():
    conn = sqlite3.connect(SQLITE_DB_PATH)

    # Aggregate season stats per team
    df = pd.read_sql_query("""
        SELECT
            team,
            COUNT(*) AS seasons,
            SUM(matches_played) AS total_matches,
            SUM(wins) AS total_wins,
            SUM(losses) AS total_losses,
            ROUND(CAST(SUM(wins) AS FLOAT) / SUM(matches_played), 4) AS overall_wr,
            SUM(reached_playoff) AS playoff_appearances,
            SUM(reached_final) AS final_appearances,
            SUM(won_title) AS titles
        FROM season_stats
        GROUP BY team
        ORDER BY overall_wr DESC
    """, conn)
    conn.close()

    # Filter active teams only
    df = df[df["team"].isin(ACTIVE_TEAMS_2026)].copy()

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df.to_csv(TEAM_STATS_CSV, index=False)
    print(f"Team stats exported: {TEAM_STATS_CSV}")
    print(df.to_string(index=False))
    return df


def export_h2h_matrix():
    conn = sqlite3.connect(SQLITE_DB_PATH)
    df = pd.read_sql_query("""
        SELECT team1, team2, winner FROM matches
    """, conn)
    conn.close()

    teams = sorted(ACTIVE_TEAMS_2026)
    matrix = pd.DataFrame(0.0, index=teams, columns=teams)

    for _, row in df.iterrows():
        t1, t2, w = row["team1"], row["team2"], row["winner"]
        if t1 not in teams or t2 not in teams:
            continue
        if w == t1:
            matrix.loc[t1, t2] += 1
        elif w == t2:
            matrix.loc[t2, t1] += 1

    # Convert counts to win rates
    rate_matrix = matrix.copy()
    for t1 in teams:
        for t2 in teams:
            if t1 == t2:
                rate_matrix.loc[t1, t2] = np.nan
                continue
            total = matrix.loc[t1, t2] + matrix.loc[t2, t1]
            rate_matrix.loc[t1, t2] = matrix.loc[t1, t2] / total if total > 0 else 0.5

    path = os.path.join(PROCESSED_DIR, "h2h_win_rate_matrix.csv")
    rate_matrix.to_csv(path)
    print(f"\nH2H win rate matrix saved: {path}")
    return rate_matrix


if __name__ == "__main__":
    export_team_stats()
    h2h = export_h2h_matrix()
    print("\nHead-to-Head Win Rate Matrix:")
    print(h2h.round(2).to_string())
