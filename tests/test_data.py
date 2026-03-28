"""
Unit tests for data pipeline.
"""
import os
import sys
import unittest
import sqlite3
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestDatasetCreation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.data.create_dataset import build_all_matches
        cls.MATCHES_DATA = build_all_matches()

    def test_matches_csv_creation(self):
        self.assertGreater(len(self.MATCHES_DATA), 100, "Should have 100+ matches")
        # Each row should have 9 fields
        for row in self.MATCHES_DATA[:5]:
            self.assertEqual(len(row), 9)

    def test_teams_json_structure(self):
        from src.data.create_dataset import save_teams_json
        from config import TEAMS
        self.assertGreaterEqual(len(TEAMS), 10, "Should have 10 active teams")

    def test_season_range(self):
        seasons = {row[0] for row in self.MATCHES_DATA}
        self.assertIn(2008, seasons)
        self.assertIn(2024, seasons)
        self.assertGreaterEqual(max(seasons), 2024)

    def test_all_winners_are_valid_teams(self):
        for row in self.MATCHES_DATA:
            team1, team2, winner = row[1], row[2], row[3]
            self.assertIn(winner, [team1, team2],
                          f"Winner {winner} not in match: {team1} vs {team2}")


class TestDatabaseSetup(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp_dir, "test.db")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_tables_created(self):
        from src.data.db_setup import CREATE_TABLES_SQL, INDEXES_SQL
        conn = sqlite3.connect(self.db_path)
        conn.executescript(CREATE_TABLES_SQL)
        conn.executescript(INDEXES_SQL)
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()
        for expected in ["teams", "venues", "matches", "season_stats",
                         "head_to_head", "player_stats"]:
            self.assertIn(expected, tables)


class TestIPLWinners(unittest.TestCase):
    def test_winner_dict_coverage(self):
        from src.data.ingest import SEASON_STANDINGS
        for season in range(2008, 2025):
            self.assertIn(season, SEASON_STANDINGS,
                          f"Season {season} missing from standings")

    def test_playoff_teams_count(self):
        from src.data.ingest import SEASON_STANDINGS
        for season, standings in SEASON_STANDINGS.items():
            self.assertEqual(len(standings), 4,
                             f"Season {season} should have exactly 4 playoff teams")


if __name__ == "__main__":
    unittest.main(verbosity=2)
