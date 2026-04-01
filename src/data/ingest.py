"""
Ingests raw CSV/JSON data into the SQLite database.
Populates: teams, venues, matches, season_stats, head_to_head, player_stats.
"""
import os
import sys
import csv
import json
import sqlite3
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import (
    SQLITE_DB_PATH, MATCHES_CSV, TEAMS_JSON,
    TEAM_ALIASES, ACTIVE_TEAMS_2026,
)

# Known venues with capacity
VENUE_INFO = {
    "MA Chidambaram Stadium":            ("Chennai",    "India", 50000),
    "Wankhede Stadium":                  ("Mumbai",     "India", 33000),
    "M Chinnaswamy Stadium":             ("Bengaluru",  "India", 40000),
    "Eden Gardens":                      ("Kolkata",    "India", 68000),
    "Feroz Shah Kotla":                  ("Delhi",      "India", 41820),
    "Sawai Mansingh Stadium":            ("Jaipur",     "India", 30000),
    "Rajiv Gandhi Intl Stadium":         ("Hyderabad",  "India", 55000),
    "Punjab Cricket Association Stadium":("Mohali",     "India", 26950),
    "Narendra Modi Stadium":             ("Ahmedabad",  "India", 132000),
    "DY Patil Stadium":                  ("Mumbai",     "India", 55000),
    "Brabourne Stadium":                 ("Mumbai",     "India", 20000),
    "BRSABV Ekana Cricket Stadium":      ("Lucknow",    "India", 50000),
    "Maharashtra Cricket Association Stadium": ("Pune", "India", 37406),
    "Dubai Intl Stadium":                ("Dubai",      "UAE",   25000),
    "Sharjah Cricket Stadium":           ("Sharjah",    "UAE",   16000),
    "Abu Dhabi Cricket Ground":          ("Abu Dhabi",  "UAE",   20000),
    "Newlands":                          ("Cape Town",  "South Africa", 25000),
    "Kingsmead":                         ("Durban",     "South Africa", 25000),
    "New Wanderers":                     ("Johannesburg","South Africa", 34000),
    "SuperSport Park":                   ("Centurion",  "South Africa", 22000),
    "Centurion":                         ("Centurion",  "South Africa", 22000),
}

# Player stats by season (top contributors per team)
PLAYER_STATS = [
    # (season, name, team, role, bat_avg, bat_sr, runs, wkts, bowl_avg, econ)
    (2008,"MS Dhoni","CSK","WK-Bat",41.2,137.5,414,0,0,0),
    (2008,"Virender Sehwag","DC_OLD","Bat",36.1,158.2,361,0,0,0),
    (2008,"Sohail Tanvir","RR","Bowl",0,0,0,22,13.5,6.7),
    (2008,"Shane Bond","KKR","Bowl",0,0,0,15,17.8,7.1),
    (2009,"Adam Gilchrist","DC_OLD","WK-Bat",47.3,156.8,495,0,0,0),
    (2009,"Jacques Kallis","KKR","All",38.2,110.4,283,7,28.1,7.8),
    (2009,"RP Singh","DC_OLD","Bowl",0,0,0,23,16.2,7.3),
    (2010,"Suresh Raina","CSK","Bat",47.0,153.4,520,0,0,0),
    (2010,"Rohit Sharma","MI","Bat",42.1,134.2,404,0,0,0),
    (2011,"Chris Gayle","RCB","Bat",67.4,183.7,608,0,0,0),
    (2011,"Lasith Malinga","MI","Bowl",0,0,0,28,15.5,6.8),
    (2012,"Sunil Narine","KKR","All",0,0,0,24,17.3,5.5),
    (2012,"Chris Gayle","RCB","Bat",61.3,160.6,733,0,0,0),
    (2013,"Michael Hussey","CSK","Bat",56.2,139.2,733,0,0,0),
    (2013,"Virat Kohli","RCB","Bat",45.3,138.1,634,0,0,0),
    (2014,"Robin Uthappa","KKR","WK-Bat",44.4,137.4,660,0,0,0),
    (2014,"Glenn Maxwell","PBKS","Bat",44.6,187.3,552,0,0,0),
    (2015,"David Warner","SRH","Bat",49.9,155.8,562,0,0,0),
    (2015,"AB de Villiers","RCB","Bat",56.8,187.3,513,0,0,0),
    (2016,"Virat Kohli","RCB","Bat",81.1,152.4,973,0,0,0),
    (2016,"David Warner","SRH","Bat",69.3,151.8,848,0,0,0),
    (2016,"Bhuvneshwar Kumar","SRH","Bowl",0,0,0,23,18.6,7.1),
    (2017,"David Warner","SRH","Bat",61.9,142.5,641,0,0,0),
    (2017,"Ben Stokes","RPS","All",31.8,142.3,316,12,27.8,8.1),
    (2018,"Kane Williamson","SRH","Bat",52.2,131.4,735,0,0,0),
    (2018,"Sunil Narine","KKR","All",0,0,0,17,23.4,6.1),
    (2019,"David Warner","SRH","Bat",69.9,143.8,692,0,0,0),
    (2019,"Quinton de Kock","MI","WK-Bat",38.6,145.1,529,0,0,0),
    (2020,"KL Rahul","PBKS","WK-Bat",54.9,129.2,670,0,0,0),
    (2020,"Jofra Archer","RR","Bowl",0,0,0,20,17.9,6.9),
    (2021,"Ruturaj Gaikwad","CSK","Bat",45.5,136.1,635,0,0,0),
    (2021,"Harshal Patel","RCB","Bowl",0,0,0,32,14.4,7.7),
    (2022,"Jos Buttler","RR","WK-Bat",57.8,150.6,863,0,0,0),
    (2022,"Hardik Pandya","GT","All",44.1,131.4,487,8,29.3,8.1),
    (2023,"Shubman Gill","GT","Bat",59.2,157.9,890,0,0,0),
    (2023,"Faf du Plessis","RCB","Bat",40.8,138.6,730,0,0,0),
    (2024,"Virat Kohli","RCB","Bat",61.6,154.2,741,0,0,0),
    (2024,"Travis Head","SRH","Bat",58.7,191.3,567,0,0,0),
    (2024,"Mitchell Starc","KKR","Bowl",0,0,0,17,22.4,8.5),
]

# Season standings (manually curated top-4 finishers)
SEASON_STANDINGS = {
    2008: [("RR",1),("CSK",2),("DC_OLD",3),("MI",4)],
    2009: [("DC_OLD",1),("RCB",2),("CSK",3),("DC_OLD",4)],
    2010: [("CSK",1),("MI",2),("DC_OLD",3),("RR",4)],
    2011: [("CSK",1),("RCB",2),("MI",3),("RR",4)],
    2012: [("KKR",1),("CSK",2),("MI",3),("RCB",4)],
    2013: [("MI",1),("CSK",2),("RR",3),("SRH",4)],
    2014: [("KKR",1),("PBKS",2),("CSK",3),("MI",4)],
    2015: [("MI",1),("CSK",2),("RCB",3),("RR",4)],
    2016: [("SRH",1),("RCB",2),("GT",3),("KKR",4)],
    2017: [("MI",1),("RPS",2),("KKR",3),("SRH",4)],
    2018: [("CSK",1),("SRH",2),("KKR",3),("RR",4)],
    2019: [("MI",1),("CSK",2),("DC",3),("SRH",4)],
    2020: [("MI",1),("DC",2),("SRH",3),("RCB",4)],
    2021: [("CSK",1),("KKR",2),("DC",3),("RCB",4)],
    2022: [("GT",1),("RR",2),("LSG",3),("RCB",4)],
    2023: [("CSK",1),("GT",2),("MI",3),("LSG",4)],
    2024: [("KKR",1),("SRH",2),("RR",3),("RCB",4)],
}


def normalize_team(name: str) -> str:
    if name is None:
        return ""
    clean = str(name).strip()
    return TEAM_ALIASES.get(clean, clean)


def _pick_first(row: dict, keys: list, default=""):
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return default


def _parse_int(value, default=0):
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _parse_season(row: dict) -> int:
    season_val = _pick_first(row, ["season", "Season"])
    if season_val not in (None, ""):
        return _parse_int(season_val, default=0)

    date_val = _pick_first(row, ["date", "Date", "match_date"], default="")
    if not date_val:
        return 0

    digits = "".join(ch for ch in str(date_val) if ch.isdigit())
    if len(digits) >= 4:
        return int(digits[:4])
    return 0


def _is_playoff_row(row: dict) -> int:
    stage = str(_pick_first(row, ["match_type", "stage", "type"], default="")).lower()
    return int(any(tok in stage for tok in ["qualifier", "eliminator", "playoff", "final"]))


def _is_final_row(row: dict) -> int:
    stage = str(_pick_first(row, ["match_type", "stage", "type"], default="")).lower()
    return int("final" in stage)


def ingest_teams(conn, teams_json_path: str):
    with open(teams_json_path) as f:
        teams = json.load(f)
    for team_id, info in teams.items():
        conn.execute("""
            INSERT OR REPLACE INTO teams (team_id, name, home_venue, titles, founded, active)
            VALUES (?, ?, ?, ?, ?, 1)
        """, (team_id, info["name"], info["home"], info["titles"], info["founded"]))
    # Add retired franchises
    retired = [
        ("DC_OLD", "Deccan Chargers",          "Rajiv Gandhi Intl Stadium", 1, 2008),
        ("RPS",    "Rising Pune Supergiant",   "Maharashtra Cricket Association Stadium", 0, 2016),
        ("KTK",    "Kochi Tuskers Kerala",     "Jawaharlal Nehru Stadium",  0, 2011),
        ("PW",     "Pune Warriors",            "Maharashtra Cricket Association Stadium", 0, 2011),
    ]
    for row in retired:
        conn.execute("""
            INSERT OR IGNORE INTO teams (team_id, name, home_venue, titles, founded, active)
            VALUES (?, ?, ?, ?, ?, 0)
        """, row)
    conn.commit()
    print(f"Teams ingested: {len(teams) + len(retired)}")


def ingest_venues(conn):
    for name, (city, country, capacity) in VENUE_INFO.items():
        conn.execute("""
            INSERT OR IGNORE INTO venues (name, city, country, capacity)
            VALUES (?, ?, ?, ?)
        """, (name, city, country, capacity))
    conn.commit()
    print(f"Venues ingested: {len(VENUE_INFO)}")


def ingest_matches(conn, matches_csv_path: str):
    with open(matches_csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    season_team_wins = defaultdict(lambda: defaultdict(int))
    season_team_matches = defaultdict(lambda: defaultdict(int))

    inserted_rows = []

    for idx, row in enumerate(rows, start=1):
        season = _parse_season(row)
        if not season:
            continue

        match_id = _parse_int(_pick_first(row, ["id", "match_id", "ID"]), default=idx)
        t1 = normalize_team(_pick_first(row, ["team1", "Team1"]))
        t2 = normalize_team(_pick_first(row, ["team2", "Team2"]))
        winner = normalize_team(_pick_first(row, ["winner", "Winner"]))
        toss_winner = normalize_team(_pick_first(row, ["toss_winner", "Toss_Winner"]))
        toss_decision = str(_pick_first(row, ["toss_decision", "Toss_Decision"], default="")).strip().lower()
        venue = _pick_first(row, ["venue", "Venue", "ground", "Ground"], default="")
        win_by_runs = _parse_int(_pick_first(row, ["win_by_runs", "Win_By_Runs"]))
        win_by_wickets = _parse_int(_pick_first(row, ["win_by_wickets", "Win_By_Wickets"]))
        is_playoff = _is_playoff_row(row)
        is_final = _is_final_row(row)

        if not t1 or not t2:
            continue

        conn.execute("""
            INSERT OR IGNORE INTO matches
            (match_id, season, team1, team2, toss_winner, toss_decision,
             winner, win_by_runs, win_by_wickets, venue, is_playoff, is_final)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            match_id, season, t1, t2,
            toss_winner, toss_decision,
            winner, win_by_runs, win_by_wickets,
            venue, is_playoff, is_final,
        ))
        inserted_rows.append((season, match_id, t1, t2, winner, is_playoff, is_final))
        season_team_matches[season][t1] += 1
        season_team_matches[season][t2] += 1
        if winner:
            season_team_wins[season][winner] += 1

    conn.commit()

    # Populate season_stats from ingested match outcomes
    season_champions = {}
    season_finalists = defaultdict(set)
    for season, _match_id, t1, t2, winner, _is_playoff, is_final in inserted_rows:
        if is_final:
            if winner:
                season_champions[season] = winner
            season_finalists[season].update([t1, t2])

    for season, team_matches in season_team_matches.items():
        ranked = sorted(
            team_matches.keys(),
            key=lambda t: (season_team_wins[season].get(t, 0), team_matches.get(t, 0)),
            reverse=True,
        )
        top4 = set(ranked[:4])

        for team in team_matches:
            played = season_team_matches[season].get(team, 10)
            wins   = season_team_wins[season].get(team, 0)
            losses = played - wins
            champion = season_champions.get(season)
            finalists = season_finalists.get(season, set())
            conn.execute("""
                INSERT OR REPLACE INTO season_stats
                (season, team, matches_played, wins, losses, points,
                 reached_playoff, reached_final, won_title)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                season, team, played, wins, losses, wins * 2,
                1 if team in top4 else 0,
                1 if team in finalists else 0,
                1 if team == champion else 0,
            ))
    conn.commit()
    print(f"Matches ingested: {len(inserted_rows)}")


def ingest_head_to_head(conn):
    cursor = conn.execute("""
        SELECT season, team1, team2, winner FROM matches ORDER BY season
    """)
    h2h = defaultdict(lambda: defaultdict(lambda: {"wins_a": 0, "wins_b": 0}))
    for season, t1, t2, winner in cursor.fetchall():
        key = (min(t1, t2), max(t1, t2))
        if winner == t1:
            h2h[season][key]["wins_a"] += 1
        elif winner == t2:
            h2h[season][key]["wins_b"] += 1

    for season, matches in h2h.items():
        for (ta, tb), rec in matches.items():
            conn.execute("""
                INSERT OR REPLACE INTO head_to_head (team_a, team_b, season, wins_a, wins_b)
                VALUES (?, ?, ?, ?, ?)
            """, (ta, tb, season, rec["wins_a"], rec["wins_b"]))
    conn.commit()
    print(f"Head-to-head records populated for {len(h2h)} seasons")


def ingest_player_stats(conn):
    for row in PLAYER_STATS:
        (season, name, team, role, bat_avg, bat_sr, runs, wkts, bowl_avg, econ) = row
        conn.execute("""
            INSERT OR REPLACE INTO player_stats
            (season, player_name, team, role, batting_avg, batting_sr,
             runs_scored, wickets, bowling_avg, economy)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (season, name, team, role, bat_avg, bat_sr, runs, wkts, bowl_avg, econ))
    conn.commit()
    print(f"Player stats ingested: {len(PLAYER_STATS)}")


def run_ingestion():
    from config import TEAMS_JSON
    conn = sqlite3.connect(SQLITE_DB_PATH)
    try:
        ingest_teams(conn, TEAMS_JSON)
        ingest_venues(conn)
        ingest_matches(conn, MATCHES_CSV)
        ingest_head_to_head(conn)
        ingest_player_stats(conn)
        print("All data ingested successfully.")
    finally:
        conn.close()


if __name__ == "__main__":
    run_ingestion()
