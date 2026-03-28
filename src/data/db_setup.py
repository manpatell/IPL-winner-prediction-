"""
SQLite database schema creation for IPL prediction.
Creates tables: teams, matches, venues, season_stats, head_to_head.
"""
import os
import sys
import sqlite3

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import SQLITE_DB_PATH, DB_DIR

CREATE_TABLES_SQL = """
-- Teams master table
CREATE TABLE IF NOT EXISTS teams (
    team_id     TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    home_venue  TEXT,
    titles      INTEGER DEFAULT 0,
    founded     INTEGER,
    active      INTEGER DEFAULT 1
);

-- Venues table
CREATE TABLE IF NOT EXISTS venues (
    venue_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    city        TEXT,
    country     TEXT DEFAULT 'India',
    capacity    INTEGER
);

-- Matches table
CREATE TABLE IF NOT EXISTS matches (
    match_id        INTEGER PRIMARY KEY,
    season          INTEGER NOT NULL,
    team1           TEXT NOT NULL,
    team2           TEXT NOT NULL,
    toss_winner     TEXT,
    toss_decision   TEXT,
    winner          TEXT,
    win_by_runs     INTEGER DEFAULT 0,
    win_by_wickets  INTEGER DEFAULT 0,
    venue           TEXT,
    is_playoff      INTEGER DEFAULT 0,
    is_final        INTEGER DEFAULT 0,
    FOREIGN KEY (team1)   REFERENCES teams(team_id),
    FOREIGN KEY (team2)   REFERENCES teams(team_id),
    FOREIGN KEY (winner)  REFERENCES teams(team_id)
);

-- Season standings summary
CREATE TABLE IF NOT EXISTS season_stats (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    season          INTEGER NOT NULL,
    team            TEXT NOT NULL,
    matches_played  INTEGER DEFAULT 0,
    wins            INTEGER DEFAULT 0,
    losses          INTEGER DEFAULT 0,
    points          INTEGER DEFAULT 0,
    nrr             REAL DEFAULT 0.0,
    reached_playoff INTEGER DEFAULT 0,
    reached_final   INTEGER DEFAULT 0,
    won_title       INTEGER DEFAULT 0,
    UNIQUE(season, team),
    FOREIGN KEY (team) REFERENCES teams(team_id)
);

-- Head-to-head records
CREATE TABLE IF NOT EXISTS head_to_head (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    team_a      TEXT NOT NULL,
    team_b      TEXT NOT NULL,
    season      INTEGER NOT NULL,
    wins_a      INTEGER DEFAULT 0,
    wins_b      INTEGER DEFAULT 0,
    UNIQUE(team_a, team_b, season),
    FOREIGN KEY (team_a) REFERENCES teams(team_id),
    FOREIGN KEY (team_b) REFERENCES teams(team_id)
);

-- Player stats per season
CREATE TABLE IF NOT EXISTS player_stats (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    season          INTEGER NOT NULL,
    player_name     TEXT NOT NULL,
    team            TEXT NOT NULL,
    role            TEXT,
    batting_avg     REAL DEFAULT 0.0,
    batting_sr      REAL DEFAULT 0.0,
    runs_scored     INTEGER DEFAULT 0,
    wickets         INTEGER DEFAULT 0,
    bowling_avg     REAL DEFAULT 0.0,
    economy         REAL DEFAULT 0.0,
    UNIQUE(season, player_name)
);
"""

INDEXES_SQL = """
CREATE INDEX IF NOT EXISTS idx_matches_season  ON matches(season);
CREATE INDEX IF NOT EXISTS idx_matches_team1   ON matches(team1);
CREATE INDEX IF NOT EXISTS idx_matches_team2   ON matches(team2);
CREATE INDEX IF NOT EXISTS idx_matches_winner  ON matches(winner);
CREATE INDEX IF NOT EXISTS idx_season_stats_season ON season_stats(season);
CREATE INDEX IF NOT EXISTS idx_season_stats_team   ON season_stats(team);
"""


def setup_database():
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(SQLITE_DB_PATH)
    try:
        conn.executescript(CREATE_TABLES_SQL)
        conn.executescript(INDEXES_SQL)
        conn.commit()
        print(f"Database initialized at: {SQLITE_DB_PATH}")
        # Verify
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row[0] for row in cursor.fetchall()]
        print(f"Tables created: {tables}")
    finally:
        conn.close()


if __name__ == "__main__":
    setup_database()
