"""
Monte Carlo Playoff Simulator for IPL 2026.

Simulates 100,000 tournament iterations:
  1. Each group-stage match outcome is sampled from model win probabilities.
  2. Top 4 teams qualify for playoffs.
  3. IPL playoff format: Qualifier 1, Eliminator, Qualifier 2, Final.
  4. Champion frequency across all iterations = championship probability.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import ACTIVE_TEAMS_2026, RESULTS_DIR

N_ITERATIONS = 100_000
RANDOM_SEED = 42


# Group stage matchups (each team plays every other team twice)
def get_group_stage_schedule() -> list:
    """Returns list of (team1, team2) for all 45 unique pairings × 2 = 90 matches."""
    from itertools import combinations
    teams = ACTIVE_TEAMS_2026
    schedule = []
    for t1, t2 in combinations(teams, 2):
        schedule.append((t1, t2))
        schedule.append((t2, t1))  # home/away swap for second leg
    return schedule


def simulate_group_stage(win_prob_matrix: dict, rng: np.random.Generator) -> dict:
    """
    Simulate group stage matches and return points table.
    win_prob_matrix[t1][t2] = probability that t1 beats t2.
    """
    points = defaultdict(int)
    nrr    = defaultdict(float)  # simplified NRR as noise

    for t1, t2 in get_group_stage_schedule():
        p = win_prob_matrix.get(t1, {}).get(t2, 0.5)
        if rng.random() < p:
            points[t1] += 2
            nrr[t1]    += rng.uniform(0.1, 1.5)
            nrr[t2]    -= rng.uniform(0.1, 1.5)
        else:
            points[t2] += 2
            nrr[t2]    += rng.uniform(0.1, 1.5)
            nrr[t1]    -= rng.uniform(0.1, 1.5)

    # Rank by points then NRR
    standings = sorted(ACTIVE_TEAMS_2026,
                       key=lambda t: (points[t], nrr[t]),
                       reverse=True)
    return {t: rank + 1 for rank, t in enumerate(standings)}


def simulate_playoffs(standings: dict, win_prob_matrix: dict, rng: np.random.Generator) -> str:
    """
    IPL playoff format:
      Qualifier 1 : 1st vs 2nd  → winner goes to Final
      Eliminator  : 3rd vs 4th  → loser eliminated
      Qualifier 2 : Loser Q1 vs Winner Elim → winner goes to Final
      Final       : Winner Q1   vs Winner Q2
    """
    ranked = sorted(standings, key=standings.get)
    rank1, rank2, rank3, rank4 = ranked[0], ranked[1], ranked[2], ranked[3]

    def match(t1, t2):
        p = win_prob_matrix.get(t1, {}).get(t2, 0.5)
        return t1 if rng.random() < p else t2

    # Qualifier 1
    q1_winner = match(rank1, rank2)
    q1_loser  = rank2 if q1_winner == rank1 else rank1

    # Eliminator
    elim_winner = match(rank3, rank4)

    # Qualifier 2
    q2_winner = match(q1_loser, elim_winner)

    # Final
    champion = match(q1_winner, q2_winner)
    return champion


def build_win_prob_matrix(model, matches_df: pd.DataFrame) -> dict:
    """Build a dict[team1][team2] = P(team1 beats team2) using the trained model."""
    from src.prediction.predict_2026 import build_matchup_features
    from config import ACTIVE_TEAMS_2026

    matrix = {t: {} for t in ACTIVE_TEAMS_2026}
    for t1 in ACTIVE_TEAMS_2026:
        for t2 in ACTIVE_TEAMS_2026:
            if t1 == t2:
                matrix[t1][t2] = 0.5
                continue
            feats = build_matchup_features(t1, t2, matches_df)
            probs = model.predict_proba(feats)
            matrix[t1][t2] = float(probs[:, 1].mean())
    return matrix


def run_monte_carlo(win_prob_matrix: dict,
                    n_iterations: int = N_ITERATIONS,
                    seed: int = RANDOM_SEED) -> dict:
    """Run Monte Carlo simulation and return championship frequency per team."""
    rng = np.random.default_rng(seed)
    champion_count = defaultdict(int)

    for i in range(n_iterations):
        standings = simulate_group_stage(win_prob_matrix, rng)
        champion  = simulate_playoffs(standings, win_prob_matrix, rng)
        champion_count[champion] += 1

    # Convert to probabilities
    probs = {t: champion_count[t] / n_iterations for t in ACTIVE_TEAMS_2026}
    return probs


def predict_2026_monte_carlo() -> list:
    """Full Monte Carlo prediction pipeline. Returns ranked list."""
    from src.models.ensemble_model import EnsembleModel
    from src.prediction.predict_2026 import rank_predictions
    from config import PROCESSED_MATCHES_CSV

    matches_df = pd.read_csv(PROCESSED_MATCHES_CSV)

    print("Loading ensemble model...")
    model = EnsembleModel()
    model.load()

    print("Building win probability matrix (all matchups)...")
    win_prob_matrix = build_win_prob_matrix(model, matches_df)

    print(f"Running {N_ITERATIONS:,} Monte Carlo iterations...")
    mc_probs = run_monte_carlo(win_prob_matrix)

    # Apply Bayesian update on top of MC probabilities
    from src.prediction.predict_2026 import bayesian_update
    final_probs = bayesian_update(mc_probs)

    rankings = rank_predictions(final_probs)
    return rankings, win_prob_matrix


def save_mc_results(rankings: list, win_prob_matrix: dict):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "prediction_2026_montecarlo.json")
    with open(path, "w") as f:
        json.dump({
            "season": 2026,
            "method": f"Monte Carlo ({N_ITERATIONS:,} iterations) + Bayesian Update",
            "n_iterations": N_ITERATIONS,
            "rankings": rankings,
        }, f, indent=2)
    print(f"Monte Carlo results saved: {path}")


if __name__ == "__main__":
    rankings, _ = predict_2026_monte_carlo()
    from src.prediction.predict_2026 import print_predictions
    print("\nMonte Carlo Prediction:")
    print_predictions(rankings)
    save_mc_results(rankings, {})
