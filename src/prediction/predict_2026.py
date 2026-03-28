"""
IPL 2026 Tournament Winner Prediction.

Strategy:
  1. For every pair of teams in the 2026 IPL, simulate a match
     using the trained ensemble model to get win probabilities.
  2. Average win probabilities across all toss/venue scenarios.
  3. Apply a Bayesian update using CURRENT-STRENGTH priors:
       - Squad strength (40%) — current players, retentions, auction
       - Recent form last 3 seasons (30%) — actual 2022-2024 performance
       - ML model signal (25%)
       - Playoff appearances last 3 seasons (5%) — minor factor
  4. All-time title count is NOT a prior — a 2008 title means nothing for 2026.

Key corrections vs v1:
  - CSK squad strength reduced (Dhoni semi-retired, aging core)
  - KKR and SRH elevated (reigning champ + runner-up 2024)
  - LSG and GT get FAIR treatment — recent performance evaluated on 3 seasons,
    not penalized for not existing before 2022
  - Playoff rate corrected with actual 2022-2024 standings:
      2022 playoffs: GT, RR, LSG, RCB
      2023 playoffs: CSK, GT, MI, LSG
      2024 playoffs: KKR, SRH, RR, RCB
"""
import os
import sys
import json
import itertools
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import (
    ACTIVE_TEAMS_2026, FEATURES_CSV, PROCESSED_MATCHES_CSV,
    RESULTS_DIR, MODELS_DIR, RANDOM_STATE,
)
from src.models.base_model import FEATURE_COLS
from src.features.engineer import (
    get_all_time_win_rates, get_recent_form, get_last_n_seasons_wr,
    get_h2h_rate, get_venue_win_rate, is_home_ground,
    RECENT_TITLES_5YR,
)
from src.features.venue_features import (
    get_venue_avg_score, get_venue_toss_impact, get_venue_size,
)
from src.features.team_strength import get_team_strength_features

# ─── 2026 Squad Strength (0–10 scale) ────────────────────────────────────────
# Based on: player retentions, auction buys, captain quality,
#           batting/bowling depth, form through 2024/2025.
# This is the PRIMARY factor — what the team actually looks like NOW.
SQUAD_STRENGTH_2026 = {
    "MI":   9.0,  # Bumrah (world's best bowler), Rohit, Hardik; deep batting
    "KKR":  9.0,  # Reigning 2024 champions; Narine, Russell, Phil Salt, Starc
    "SRH":  8.8,  # 2024 runners-up; Travis Head, Pat Cummins, Klaasen, A.Sharma
    "RCB":  8.5,  # Kohli still peak form (741 runs 2024); Faf, strong line-up
    "RR":   8.2,  # Buttler, Samson (captain), consistent playoff side
    "GT":   8.0,  # Shubman Gill, strong squad post-Hardik; won 2022, final 2023
    "LSG":  7.8,  # KL Rahul, Nicholas Pooran; improving every season
    "CSK":  7.5,  # Ruturaj Gaikwad leads now; Jadeja; BUT aging core, missed 2022+2024
    "DC":   7.4,  # Jake Fraser-McGurk explosive, Axar Patel; young improving side
    "PBKS": 7.2,  # Arshdeep Singh, Shashank Singh; inconsistent but improving
}

# ─── Actual Playoff Appearances 2022–2024 ─────────────────────────────────────
# 2022 top-4: GT (won), RR (runner-up), LSG, RCB
# 2023 top-4: CSK (won), GT (runner-up), MI, LSG
# 2024 top-4: KKR (won), SRH (runner-up), RR, RCB
# Source: official IPL records
PLAYOFF_RATE_3YR = {
    "GT":   2/3,  # 2022 (won), 2023 (final) ✓✓✗
    "RR":   2/3,  # 2022 (final), 2024 ✓✗✓
    "RCB":  2/3,  # 2022, 2024 ✓✗✓
    "LSG":  2/3,  # 2022, 2023 ✓✓✗
    "CSK":  1/3,  # 2023 (won) only ✗✓✗
    "MI":   1/3,  # 2023 only ✗✓✗
    "KKR":  1/3,  # 2024 (won) only ✗✗✓
    "SRH":  1/3,  # 2024 (final) only ✗✗✓
    "DC":   0/3,  # missed all three ✗✗✗
    "PBKS": 0/3,  # missed all three ✗✗✗
}

# ─── 2024 Season Rank (most recent data point) ───────────────────────────────
# Inverse rank as a score: 1st place=10, 10th=1
SEASON_2024_RANK_SCORE = {
    "KKR":  10,  # 1st / champion
    "SRH":  9,   # 2nd / runner-up
    "RR":   8,   # 3rd
    "RCB":  7,   # 4th
    "GT":   5,   # ~5th (missed playoffs)
    "MI":   5,   # ~6th (missed playoffs)
    "LSG":  4,   # ~7th
    "DC":   4,   # ~8th
    "CSK":  3,   # ~9th (poor season, missed playoffs)
    "PBKS": 3,   # ~10th
}

# Venues to average over for fair 2026 prediction
PREDICTION_VENUES = [
    "Wankhede Stadium",
    "Eden Gardens",
    "MA Chidambaram Stadium",
    "Narendra Modi Stadium",
    "Rajiv Gandhi Intl Stadium",
    "M Chinnaswamy Stadium",
    "Sawai Mansingh Stadium",
]


def build_matchup_features(team1: str, team2: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature rows for a hypothetical team1 vs team2 matchup in 2026.
    Averages over:
      - Both toss outcomes (team1 wins toss / team2 wins toss)
      - Both toss decisions (bat / field)
      - Multiple venues
    This gives a fair, unbiased prediction that accounts for toss luck.
    """
    overall_rates = get_all_time_win_rates(df)

    features_list = []
    for toss_t1 in [1, 0]:
        for toss_bat in [1, 0]:
            for venue in PREDICTION_VENUES:
                # All-time smoothed win rates
                t1_alltime = overall_rates.get(team1, 0.5)
                t2_alltime = overall_rates.get(team2, 0.5)

                # Last 3 seasons win rate — the most honest recent strength signal
                # CSK's 2022+2024 failures will show here
                # GT/LSG's 3 seasons are all used — fair treatment
                t1_last3yr = get_last_n_seasons_wr(df, team1, 2025, n_seasons=3)
                t2_last3yr = get_last_n_seasons_wr(df, team2, 2025, n_seasons=3)

                # Blend last3yr with squad strength for 2026 projection
                sq1 = SQUAD_STRENGTH_2026.get(team1, 7.5) / 10
                sq2 = SQUAD_STRENGTH_2026.get(team2, 7.5) / 10
                # 50% recent data, 50% squad projection
                t1_form_adj = t1_last3yr * 0.5 + sq1 * 0.5
                t2_form_adj = t2_last3yr * 0.5 + sq2 * 0.5

                # Recent match form (last 10 matches in database = end of 2024)
                t1_form = get_recent_form(df, team1, len(df), 10)
                t2_form = get_recent_form(df, team2, len(df), 10)

                # Season form proxy (last 14 matches)
                t1_season = get_recent_form(df, team1, len(df), 14)
                t2_season = get_recent_form(df, team2, len(df), 14)

                # H2H over last 5 seasons
                h2h = get_h2h_rate(df, team1, team2, len(df), 5)

                # Venue win rates
                t1_venue = get_venue_win_rate(df, team1, venue, len(df))
                t2_venue = get_venue_win_rate(df, team2, venue, len(df))

                # Home ground flags
                t1_home = is_home_ground(team1, venue)
                t2_home = is_home_ground(team2, venue)

                # Recent titles (last 5 seasons = 2020-2024)
                t1_rt = RECENT_TITLES_5YR.get(team1, 0)
                t2_rt = RECENT_TITLES_5YR.get(team2, 0)

                # Venue pitch features
                v_avg_score   = get_venue_avg_score(venue)
                v_toss_impact = get_venue_toss_impact(venue)
                v_size        = get_venue_size(venue)

                # Team batting/bowling strength for 2024 (most recent season)
                t1_str = get_team_strength_features(team1, 2024)
                t2_str = get_team_strength_features(team2, 2024)

                f = {
                    "toss_won_by_team1": toss_t1,
                    "toss_decision_bat": toss_bat,
                    "t1_alltime_wr":     t1_alltime,
                    "t2_alltime_wr":     t2_alltime,
                    "wr_diff":           t1_alltime - t2_alltime,
                    "t1_last3yr_wr":     t1_form_adj,
                    "t2_last3yr_wr":     t2_form_adj,
                    "last3yr_wr_diff":   t1_form_adj - t2_form_adj,
                    "t1_recent_form":    t1_form,
                    "t2_recent_form":    t2_form,
                    "form_diff":         t1_form - t2_form,
                    "t1_season_form":    t1_season,
                    "t2_season_form":    t2_season,
                    "h2h_t1_wr":         h2h,
                    "t1_venue_wr":       t1_venue,
                    "t2_venue_wr":       t2_venue,
                    "venue_wr_diff":     t1_venue - t2_venue,
                    "t1_is_home":        t1_home,
                    "t2_is_home":        t2_home,
                    "t1_recent_titles":  t1_rt,
                    "t2_recent_titles":  t2_rt,
                    "recent_title_diff": t1_rt - t2_rt,
                    "venue_avg_score":   v_avg_score,
                    "venue_toss_impact": v_toss_impact,
                    "venue_size":        v_size,
                    "t1_batting_str":    t1_str["batting_strength"],
                    "t2_batting_str":    t2_str["batting_strength"],
                    "batting_str_diff":  t1_str["batting_strength"] - t2_str["batting_strength"],
                    "t1_bowling_str":    t1_str["bowling_strength"],
                    "t2_bowling_str":    t2_str["bowling_strength"],
                    "bowling_str_diff":  t1_str["bowling_strength"] - t2_str["bowling_strength"],
                }
                features_list.append(f)

    return pd.DataFrame(features_list, columns=FEATURE_COLS)


def simulate_tournament(model, df: pd.DataFrame) -> dict:
    """
    Simulate all round-robin matchups and accumulate win probabilities.
    Returns {team: average_win_probability}.
    """
    teams = ACTIVE_TEAMS_2026
    win_probs = {t: [] for t in teams}

    for team1, team2 in itertools.combinations(teams, 2):
        feats = build_matchup_features(team1, team2, df)
        probs = model.predict_proba(feats)
        avg_t1_wins = probs[:, 1].mean()
        avg_t2_wins = 1 - avg_t1_wins
        win_probs[team1].append(avg_t1_wins)
        win_probs[team2].append(avg_t2_wins)

    return {t: np.mean(v) for t, v in win_probs.items()}


def bayesian_update(model_probs: dict) -> dict:
    """
    Combine model probabilities with CURRENT-STRENGTH domain priors.

    Weights deliberately favour recent evidence over historical legacy:
      model       25% — ensemble ML output
      squad       40% — current player quality (2026 squads)
      recent_form 30% — actual performance 2022-2024 (corrects CSK bias)
      playoff     5%  — recent playoff appearances (minor signal)

    All-time title count is NOT a prior — it would unfairly favour
    CSK (5 titles, mostly in 2010s) over GT/LSG/KKR.
    """
    def normalize(d):
        total = sum(d.values())
        return {k: v / total for k, v in d.items()} if total > 0 else d

    # Squad strength prior
    sq_prior = normalize(SQUAD_STRENGTH_2026)

    # Correct recent form prior: combines last-3yr playoff rate + 2024 rank
    recent_form_raw = {}
    for t in ACTIVE_TEAMS_2026:
        playoff_score = PLAYOFF_RATE_3YR.get(t, 0)
        rank_score    = SEASON_2024_RANK_SCORE.get(t, 5) / 10
        recent_form_raw[t] = playoff_score * 0.6 + rank_score * 0.4
    recent_prior = normalize({t: max(v, 0.01) for t, v in recent_form_raw.items()})

    # Playoff rate (minor, already partially captured in recent_form)
    pl_prior = normalize({t: max(v, 0.01) for t, v in PLAYOFF_RATE_3YR.items()})

    # Normalize model probs
    model_norm = normalize(model_probs)

    weights = {
        "squad":        0.40,  # what the team looks like NOW
        "recent_form":  0.30,  # how they actually performed 2022-2024
        "model":        0.25,  # ensemble ML signal
        "playoff":      0.05,  # minor recency signal
    }

    combined = {}
    for t in ACTIVE_TEAMS_2026:
        combined[t] = (
            weights["squad"]       * sq_prior.get(t, 0) +
            weights["recent_form"] * recent_prior.get(t, 0) +
            weights["model"]       * model_norm.get(t, 0) +
            weights["playoff"]     * pl_prior.get(t, 0)
        )

    return normalize(combined)


def rank_predictions(combined_probs: dict) -> list:
    """Returns list sorted by win probability (highest first)."""
    from config import TEAMS
    ranked = sorted(combined_probs.items(), key=lambda x: x[1], reverse=True)
    results = []
    for rank, (team, prob) in enumerate(ranked, start=1):
        results.append({
            "rank":            rank,
            "team_id":         team,
            "team_name":       TEAMS.get(team, team),
            "win_probability": round(prob * 100, 2),
        })
    return results


def predict_2026_winner(use_ensemble: bool = True) -> list:
    """Main function. Loads ensemble, simulates tournament, returns ranked predictions."""
    from src.models.ensemble_model  import EnsembleModel
    from src.models.xgboost_model   import XGBoostModel

    matches_df = pd.read_csv(PROCESSED_MATCHES_CSV)
    df         = pd.read_csv(FEATURES_CSV)

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

    print("\nSimulating IPL 2026 round-robin matchups (toss + venue averaged)...")
    model_probs = simulate_tournament(model, matches_df)

    print("Applying current-strength Bayesian update (squad 40%, form 30%, model 25%)...")
    final_probs = bayesian_update(model_probs)

    rankings = rank_predictions(final_probs)
    return rankings


def print_predictions(rankings: list):
    print("\n" + "═"*65)
    print("         IPL 2026 WINNER PREDICTION RESULTS")
    print("  (Squad strength 40% + Recent form 30% + Model 25%)")
    print("═"*65)
    print(f"{'Rank':<6} {'Team':<35} {'Win Probability':>15}")
    print("-"*65)
    for r in rankings:
        bar = "█" * int(r["win_probability"] / 2)
        print(f"  {r['rank']:<4} {r['team_name']:<35} {r['win_probability']:>6.2f}%  {bar}")
    print("═"*65)
    w = rankings[0]
    print(f"\n  PREDICTED WINNER: {w['team_name']}")
    print(f"  Confidence score: {w['win_probability']:.2f}%")
    print("═"*65)


def save_predictions(rankings: list):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "prediction_2026.json")
    with open(path, "w") as f:
        json.dump({
            "season":  2026,
            "method":  "Stacking Ensemble + Current-Strength Bayesian Update",
            "weights": {"squad": 0.40, "recent_form": 0.30,
                        "model": 0.25, "playoff": 0.05},
            "rankings": rankings,
        }, f, indent=2)
    print(f"\nPredictions saved: {path}")


if __name__ == "__main__":
    rankings = predict_2026_winner()
    print_predictions(rankings)
    save_predictions(rankings)
