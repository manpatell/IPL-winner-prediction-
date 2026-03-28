"""
IPL 2026 Tournament Winner Prediction.

Strategy:
  1. For every pair of teams in the 2026 IPL, simulate a match
     using the trained ensemble model to get win probabilities.
  2. Aggregate each team's average win probability across all matchups.
  3. Apply a Bayesian update using domain priors:
       - Recent title ratio (last 3 seasons)
       - Average season finish rank (last 3 seasons)
       - Projected squad strength score
  4. Report ranked predictions with confidence intervals.
"""
import os
import sys
import json
import itertools
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import (
    ACTIVE_TEAMS_2026, FEATURES_CSV, RESULTS_DIR, MODELS_DIR,
    RANDOM_STATE,
)
from src.models.base_model import FEATURE_COLS
from src.features.engineer import (
    get_all_time_win_rates, get_recent_form, get_h2h_rate,
    get_venue_win_rate, is_home_ground, TITLE_COUNTS,
)

# ─── 2026 context ─────────────────────────────────────────────────────────────

# Projected squad strength for 2026 (0-10 scale based on retentions,
# auction buys, player form through 2025)
SQUAD_STRENGTH_2026 = {
    "MI":   9.2,   # Rohit, Bumrah, Hardik returning; strong spine
    "CSK":  8.8,   # Dhoni legacy, Gaikwad, Jadeja; strong leadership
    "RCB":  8.5,   # Kohli, Du Plessis; explosive batting
    "KKR":  8.7,   # Reigning champions 2024; Starc, Narine impact
    "SRH":  8.4,   # Travis Head, Pat Cummins; aggressive batting
    "RR":   8.0,   # Buttler, Samson; consistent performers
    "GT":   7.8,   # 2022 champions; solid but rebuilding post-Hardik
    "DC":   7.5,   # Young squad, improving each year
    "PBKS": 7.3,   # Inconsistent; competitive squad
    "LSG":  7.6,   # KL Rahul; growing franchise
}

# Average playoff appearances: last 3 seasons (2022, 2023, 2024)
PLAYOFF_RATE_3YR = {
    "MI":   2/3,   # 2024 missed
    "CSK":  2/3,   # 2024 missed
    "RCB":  2/3,
    "KKR":  2/3,
    "SRH":  2/3,
    "RR":   2/3,
    "GT":   3/3,   # All 3
    "DC":   0/3,
    "PBKS": 0/3,
    "LSG":  1/3,
}

# IPL titles (historical prestige)
NORMALIZED_TITLES = {t: v / 5 for t, v in TITLE_COUNTS.items()}

# Neutral venue assumed for 2026 (rotated venues)
NEUTRAL_VENUES = [
    "Wankhede Stadium",
    "Eden Gardens",
    "MA Chidambaram Stadium",
    "Narendra Modi Stadium",
    "Rajiv Gandhi Intl Stadium",
]


def build_matchup_features(team1: str, team2: str, df: pd.DataFrame) -> pd.DataFrame:
    """Build feature row for a hypothetical team1 vs team2 match in 2026."""
    overall_rates = get_all_time_win_rates(df)

    # For prediction, assume toss is neutral (50/50) — average over both outcomes
    features_list = []
    for toss_t1 in [1, 0]:
        for toss_bat in [1, 0]:
            venue = NEUTRAL_VENUES[0]  # Use most common venue

            t1_alltime = overall_rates.get(team1, 0.5)
            t2_alltime = overall_rates.get(team2, 0.5)

            # Recent form: use last 10 data points as proxy for 2026 form
            t1_form = get_recent_form(df, team1, len(df), 10)
            t2_form = get_recent_form(df, team2, len(df), 10)
            t1_season = get_recent_form(df, team1, len(df), 14)
            t2_season = get_recent_form(df, team2, len(df), 14)

            # Boost recent form with squad strength
            sq1 = SQUAD_STRENGTH_2026.get(team1, 7.5) / 10
            sq2 = SQUAD_STRENGTH_2026.get(team2, 7.5) / 10
            t1_form_adj = (t1_form * 0.6 + sq1 * 0.4)
            t2_form_adj = (t2_form * 0.6 + sq2 * 0.4)

            h2h = get_h2h_rate(df, team1, team2, len(df), 5)
            t1_venue = get_venue_win_rate(df, team1, venue, len(df))
            t2_venue = get_venue_win_rate(df, team2, venue, len(df))

            f = {
                "toss_won_by_team1": toss_t1,
                "toss_decision_bat": toss_bat,
                "t1_alltime_wr":    t1_alltime,
                "t2_alltime_wr":    t2_alltime,
                "wr_diff":          t1_alltime - t2_alltime,
                "t1_recent_form":   t1_form_adj,
                "t2_recent_form":   t2_form_adj,
                "form_diff":        t1_form_adj - t2_form_adj,
                "t1_season_form":   t1_season,
                "t2_season_form":   t2_season,
                "h2h_t1_wr":        h2h,
                "t1_venue_wr":      t1_venue,
                "t2_venue_wr":      t2_venue,
                "venue_wr_diff":    t1_venue - t2_venue,
                "t1_is_home":       0,
                "t2_is_home":       0,
                "t1_titles":        TITLE_COUNTS.get(team1, 0),
                "t2_titles":        TITLE_COUNTS.get(team2, 0),
                "title_diff":       TITLE_COUNTS.get(team1, 0) - TITLE_COUNTS.get(team2, 0),
            }
            features_list.append(f)

    return pd.DataFrame(features_list, columns=FEATURE_COLS)


def simulate_tournament(ensemble, df: pd.DataFrame) -> dict:
    """
    Simulate all round-robin matchups and accumulate win probabilities.
    Returns {team: average_win_probability}.
    """
    teams = ACTIVE_TEAMS_2026
    win_probs = {t: [] for t in teams}

    for team1, team2 in itertools.combinations(teams, 2):
        feats = build_matchup_features(team1, team2, df)
        probs = ensemble.predict_proba(feats)
        avg_t1_wins = probs[:, 1].mean()
        avg_t2_wins = 1 - avg_t1_wins

        win_probs[team1].append(avg_t1_wins)
        win_probs[team2].append(avg_t2_wins)

    return {t: np.mean(v) for t, v in win_probs.items()}


def bayesian_update(model_probs: dict) -> dict:
    """
    Combine model win probabilities with domain priors via a weighted average.
    """
    # Normalize priors
    def normalize(d):
        total = sum(d.values())
        return {k: v / total for k, v in d.items()} if total > 0 else d

    # Squad strength prior
    sq_prior = normalize(SQUAD_STRENGTH_2026)

    # Playoff frequency prior
    pl_prior = normalize({t: max(v, 0.01) for t, v in PLAYOFF_RATE_3YR.items()})

    # Title prestige prior
    title_prior = normalize({t: max(v, 0.02) for t, v in NORMALIZED_TITLES.items()})

    # Normalize model probs
    model_norm = normalize(model_probs)

    # Weighted combination
    weights = {"model": 0.55, "squad": 0.25, "playoff": 0.12, "title": 0.08}
    combined = {}
    for t in ACTIVE_TEAMS_2026:
        combined[t] = (
            weights["model"]   * model_norm.get(t, 0) +
            weights["squad"]   * sq_prior.get(t, 0) +
            weights["playoff"] * pl_prior.get(t, 0) +
            weights["title"]   * title_prior.get(t, 0)
        )

    # Normalize to sum to 1
    combined = normalize(combined)
    return combined


def rank_predictions(combined_probs: dict) -> list:
    """Returns list of (rank, team, probability, team_name) sorted by probability."""
    from config import TEAMS
    ranked = sorted(combined_probs.items(), key=lambda x: x[1], reverse=True)
    results = []
    for rank, (team, prob) in enumerate(ranked, start=1):
        results.append({
            "rank": rank,
            "team_id": team,
            "team_name": TEAMS.get(team, team),
            "win_probability": round(prob * 100, 2),
        })
    return results


def predict_2026_winner(use_ensemble: bool = True) -> list:
    """Main function. Loads ensemble, simulates tournament, returns ranked predictions."""
    from src.models.ensemble_model       import EnsembleModel
    from src.models.random_forest_model  import RandomForestModel
    from src.models.xgboost_model        import XGBoostModel
    from src.models.lightgbm_model       import LightGBMModel
    from src.models.neural_network_model import NeuralNetworkModel

    df = pd.read_csv(FEATURES_CSV)

    if use_ensemble:
        try:
            model = EnsembleModel()
            model.load()
        except FileNotFoundError:
            print("No saved ensemble found. Using XGBoost as fallback.")
            model = XGBoostModel()
            model.load()
    else:
        model = XGBoostModel()
        model.load()

    print("\nSimulating IPL 2026 round-robin matchups...")
    model_probs = simulate_tournament(model, df)

    print("Applying Bayesian update with domain priors...")
    final_probs = bayesian_update(model_probs)

    rankings = rank_predictions(final_probs)
    return rankings


def print_predictions(rankings: list):
    from config import TEAMS
    print("\n" + "═"*60)
    print("       IPL 2026 WINNER PREDICTION RESULTS")
    print("═"*60)
    print(f"{'Rank':<6} {'Team':<35} {'Win Probability':>15}")
    print("-"*60)

    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    for r in rankings:
        medal = medals.get(r["rank"], "  ")
        bar_len = int(r["win_probability"] / 2)
        bar = "█" * bar_len
        print(f"  {r['rank']:<4} {r['team_name']:<35} {r['win_probability']:>6.2f}%  {bar}")

    print("═"*60)
    winner = rankings[0]
    print(f"\n  PREDICTED WINNER: {winner['team_name']}")
    print(f"  Confidence score: {winner['win_probability']:.2f}%")
    print("═"*60)


def save_predictions(rankings: list):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "prediction_2026.json")
    with open(path, "w") as f:
        json.dump({
            "season": 2026,
            "method": "Stacking Ensemble + Bayesian Prior Update",
            "rankings": rankings,
        }, f, indent=2)
    print(f"\nPredictions saved: {path}")


if __name__ == "__main__":
    rankings = predict_2026_winner()
    print_predictions(rankings)
    save_predictions(rankings)
