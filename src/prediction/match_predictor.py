"""
Single match predictor: given two team names, predicts the winner
using the trained ensemble model.

Usage:
  from src.prediction.match_predictor import predict_match
  result = predict_match("MI", "CSK")
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import PROCESSED_MATCHES_CSV, FEATURES_CSV
from src.models.base_model import FEATURE_COLS
from src.prediction.predict_2026 import build_matchup_features


def predict_match(team1: str, team2: str, venue: str = None,
                  toss_winner: str = None, toss_decision: str = "field") -> dict:
    """
    Predict the winner of a single match.

    Args:
        team1: Team abbreviation (e.g. 'MI')
        team2: Team abbreviation (e.g. 'CSK')
        venue: Optional venue name (defaults to neutral venue)
        toss_winner: Optional toss winner (defaults to team1)
        toss_decision: 'bat' or 'field' (default: 'field')

    Returns:
        dict with team1_win_prob, team2_win_prob, predicted_winner
    """
    from src.models.ensemble_model import EnsembleModel

    try:
        model = EnsembleModel()
        model.load()
    except FileNotFoundError:
        from src.models.xgboost_model import XGBoostModel
        model = XGBoostModel()
        model.load()

    matches_df = pd.read_csv(PROCESSED_MATCHES_CSV)

    # Build features
    feats = build_matchup_features(team1, team2, matches_df)

    # Override toss if specified
    if toss_winner is not None:
        feats["toss_won_by_team1"] = int(toss_winner == team1)
    else:
        feats["toss_won_by_team1"] = 1  # assume team1 wins toss

    feats["toss_decision_bat"] = int(toss_decision == "bat")

    # Predict
    probs = model.predict_proba(feats)
    t1_prob = probs[:, 1].mean()
    t2_prob = 1 - t1_prob

    from config import TEAMS
    winner = team1 if t1_prob >= 0.5 else team2

    return {
        "team1": team1,
        "team1_name": TEAMS.get(team1, team1),
        "team1_win_prob": round(t1_prob * 100, 2),
        "team2": team2,
        "team2_name": TEAMS.get(team2, team2),
        "team2_win_prob": round(t2_prob * 100, 2),
        "predicted_winner": winner,
        "predicted_winner_name": TEAMS.get(winner, winner),
        "confidence": round(max(t1_prob, t2_prob) * 100, 2),
    }


def print_match_result(result: dict):
    print("\n" + "─"*50)
    print(f"  MATCH PREDICTION")
    print("─"*50)
    print(f"  {result['team1_name']:<30} {result['team1_win_prob']:>6.2f}%")
    print(f"  {result['team2_name']:<30} {result['team2_win_prob']:>6.2f}%")
    print("─"*50)
    print(f"  Winner: {result['predicted_winner_name']}  ({result['confidence']:.2f}% confidence)")
    print("─"*50)


if __name__ == "__main__":
    import sys
    t1 = sys.argv[1] if len(sys.argv) > 1 else "MI"
    t2 = sys.argv[2] if len(sys.argv) > 2 else "CSK"
    result = predict_match(t1, t2)
    print_match_result(result)
