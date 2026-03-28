"""
Unit tests for the prediction module.
"""
import os
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.prediction.predict_2026 import (
    bayesian_update, rank_predictions,
    SQUAD_STRENGTH_2026, PLAYOFF_RATE_3YR,
)
from config import ACTIVE_TEAMS_2026, TEAMS


class TestBayesianUpdate(unittest.TestCase):
    def _make_equal_probs(self):
        return {t: 1.0 / len(ACTIVE_TEAMS_2026) for t in ACTIVE_TEAMS_2026}

    def test_output_sums_to_one(self):
        probs = self._make_equal_probs()
        combined = bayesian_update(probs)
        self.assertAlmostEqual(sum(combined.values()), 1.0, places=5)

    def test_all_teams_present(self):
        probs = self._make_equal_probs()
        combined = bayesian_update(probs)
        for team in ACTIVE_TEAMS_2026:
            self.assertIn(team, combined)

    def test_all_probs_positive(self):
        probs = self._make_equal_probs()
        combined = bayesian_update(probs)
        for team, prob in combined.items():
            self.assertGreater(prob, 0.0, f"Probability for {team} should be > 0")

    def test_strong_model_signal_increases_probability(self):
        # Give MI a dominant model signal
        probs = {t: 0.05 for t in ACTIVE_TEAMS_2026}
        probs["MI"] = 0.55
        combined = bayesian_update(probs)
        # MI should still be top or near top
        sorted_teams = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        top3 = [t for t, _ in sorted_teams[:3]]
        self.assertIn("MI", top3)


class TestRankPredictions(unittest.TestCase):
    def _make_probs(self):
        probs = {t: 1 / len(ACTIVE_TEAMS_2026) for t in ACTIVE_TEAMS_2026}
        probs["CSK"] = 0.3
        probs["MI"] = 0.2
        probs["KKR"] = 0.15
        # Normalize
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}

    def test_rankings_length(self):
        probs = self._make_probs()
        rankings = rank_predictions(probs)
        self.assertEqual(len(rankings), len(ACTIVE_TEAMS_2026))

    def test_first_rank_is_highest(self):
        probs = self._make_probs()
        rankings = rank_predictions(probs)
        for i in range(1, len(rankings)):
            self.assertGreaterEqual(
                rankings[i-1]["win_probability"],
                rankings[i]["win_probability"],
            )

    def test_rank_1_is_csk(self):
        probs = self._make_probs()
        rankings = rank_predictions(probs)
        self.assertEqual(rankings[0]["team_id"], "CSK")

    def test_probabilities_sum_to_100(self):
        probs = self._make_probs()
        rankings = rank_predictions(probs)
        total = sum(r["win_probability"] for r in rankings)
        self.assertAlmostEqual(total, 100.0, delta=1.0)


class TestSquadStrength(unittest.TestCase):
    def test_all_active_teams_have_strength(self):
        for team in ACTIVE_TEAMS_2026:
            self.assertIn(team, SQUAD_STRENGTH_2026,
                          f"{team} missing from squad strength")

    def test_strength_values_in_range(self):
        for team, val in SQUAD_STRENGTH_2026.items():
            self.assertGreaterEqual(val, 0)
            self.assertLessEqual(val, 10)

    def test_playoff_rates_in_range(self):
        for team, rate in PLAYOFF_RATE_3YR.items():
            self.assertGreaterEqual(rate, 0.0)
            self.assertLessEqual(rate, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
