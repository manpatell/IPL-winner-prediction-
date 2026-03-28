"""
Unit tests for feature engineering.
"""
import os
import sys
import unittest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.features.engineer import (
    get_all_time_win_rates, get_recent_form, get_h2h_rate,
    get_venue_win_rate, is_home_ground, TITLE_COUNTS,
)
from src.models.base_model import FEATURE_COLS


def make_sample_df():
    """Build a minimal DataFrame for testing feature functions."""
    data = {
        "match_id":       [1, 2, 3, 4, 5],
        "season":         [2022, 2022, 2022, 2023, 2023],
        "team1":          ["MI", "CSK", "KKR", "MI",  "RCB"],
        "team2":          ["CSK", "RCB", "MI",  "RCB", "CSK"],
        "winner":         ["MI",  "CSK", "MI",  "MI",  "RCB"],
        "venue":          ["Wankhede Stadium"] * 5,
        "toss_won_by_team1": [1, 0, 1, 1, 0],
        "toss_decision_bat": [1, 1, 0, 1, 0],
        "team1_won":      [1, 1, 1, 1, 1],
    }
    return pd.DataFrame(data)


class TestWinRates(unittest.TestCase):
    def test_all_time_rates_range(self):
        df = make_sample_df()
        rates = get_all_time_win_rates(df)
        for team, rate in rates.items():
            self.assertGreaterEqual(rate, 0.0)
            self.assertLessEqual(rate, 1.0)

    def test_mi_win_rate(self):
        df = make_sample_df()
        rates = get_all_time_win_rates(df)
        # MI appears 4 times and wins all 4, but Bayesian smoothing pulls toward 0.5
        # so rate should be > 0.5 but < 1.0 (correct: avoids overfitting on small sample)
        self.assertGreater(rates["MI"], 0.5)
        self.assertLess(rates["MI"], 1.0)


class TestRecentForm(unittest.TestCase):
    def test_returns_float(self):
        df = make_sample_df()
        form = get_recent_form(df, "MI", 5, n=3)
        self.assertIsInstance(form, float)

    def test_no_history_returns_half(self):
        df = make_sample_df()
        form = get_recent_form(df, "GT", 0, n=5)
        self.assertEqual(form, 0.5)

    def test_form_range(self):
        df = make_sample_df()
        for i in range(len(df)):
            for team in ["MI", "CSK", "RCB"]:
                form = get_recent_form(df, team, i, n=3)
                self.assertGreaterEqual(form, 0.0)
                self.assertLessEqual(form, 1.0)


class TestH2H(unittest.TestCase):
    def test_returns_float(self):
        df = make_sample_df()
        rate = get_h2h_rate(df, "MI", "CSK", 5, window_seasons=3)
        self.assertIsInstance(rate, float)

    def test_range(self):
        df = make_sample_df()
        rate = get_h2h_rate(df, "MI", "CSK", 5, window_seasons=3)
        self.assertGreaterEqual(rate, 0.0)
        self.assertLessEqual(rate, 1.0)


class TestHomeGround(unittest.TestCase):
    def test_csk_home(self):
        self.assertEqual(is_home_ground("CSK", "MA Chidambaram Stadium"), 1)

    def test_mi_home(self):
        self.assertEqual(is_home_ground("MI", "Wankhede Stadium"), 1)

    def test_away(self):
        self.assertEqual(is_home_ground("CSK", "Eden Gardens"), 0)

    def test_neutral(self):
        self.assertEqual(is_home_ground("GT", "Wankhede Stadium"), 0)


class TestTitleCounts(unittest.TestCase):
    def test_csk_and_mi_lead(self):
        self.assertEqual(TITLE_COUNTS["CSK"], 5)
        self.assertEqual(TITLE_COUNTS["MI"], 5)

    def test_rcb_zero(self):
        self.assertEqual(TITLE_COUNTS["RCB"], 0)


class TestFeatureColumns(unittest.TestCase):
    def test_all_feature_cols_present(self):
        df = make_sample_df()
        from src.features.engineer import build_features
        # Needs CSV - just test FEATURE_COLS list integrity
        self.assertIn("t1_alltime_wr", FEATURE_COLS)
        self.assertIn("h2h_t1_wr", FEATURE_COLS)
        self.assertIn("form_diff", FEATURE_COLS)
        self.assertIn("t1_last3yr_wr", FEATURE_COLS)
        self.assertIn("t1_recent_titles", FEATURE_COLS)
        self.assertNotIn("t1_titles", FEATURE_COLS)     # all-time titles removed
        self.assertNotIn("title_diff", FEATURE_COLS)    # biased feature removed
        self.assertEqual(len(FEATURE_COLS), 22)


if __name__ == "__main__":
    unittest.main(verbosity=2)
