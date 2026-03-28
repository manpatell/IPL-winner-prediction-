"""
Visualization module: plots for model performance and 2026 predictions.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import RESULTS_DIR, FEATURES_CSV

PALETTE = {
    "CSK":  "#F9CD02",
    "MI":   "#004BA0",
    "RCB":  "#EC1C24",
    "KKR":  "#3A225D",
    "DC":   "#00008B",
    "PBKS": "#ED1B24",
    "RR":   "#2D9CDB",
    "SRH":  "#F7A721",
    "LSG":  "#A2D9CE",
    "GT":   "#1B2A4A",
}


def plot_win_probability_bar(rankings: list, save_path: str = None):
    """Horizontal bar chart of win probabilities."""
    fig, ax = plt.subplots(figsize=(10, 7))
    teams = [r["team_name"] for r in rankings]
    probs = [r["win_probability"] for r in rankings]
    team_ids = [r["team_id"] for r in rankings]
    colors = [PALETTE.get(tid, "#888888") for tid in team_ids]

    bars = ax.barh(teams[::-1], probs[::-1], color=colors[::-1], edgecolor="white", height=0.6)
    ax.set_xlabel("Win Probability (%)", fontsize=13, fontweight="bold")
    ax.set_title("IPL 2026 Winner Prediction\n(Stacking Ensemble + Bayesian Update)",
                 fontsize=15, fontweight="bold", pad=15)
    ax.set_xlim(0, max(probs) * 1.25)

    for bar, prob in zip(bars, probs[::-1]):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                f"{prob:.2f}%", va="center", fontsize=11, fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "win_probability.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Chart saved: {save_path}")


def plot_model_comparison(save_path: str = None):
    """Bar chart comparing model test accuracies."""
    results_file = os.path.join(RESULTS_DIR, "model_results.json")
    if not os.path.exists(results_file):
        print("model_results.json not found, skipping model comparison plot.")
        return

    with open(results_file) as f:
        results = json.load(f)

    models = list(results.keys())
    test_accs = [results[m]["test_accuracy"] * 100 for m in models]
    cv_accs   = [results[m].get("cv_accuracy", 0) * 100 for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, cv_accs,  width, label="CV Accuracy",   color="#2196F3", alpha=0.85)
    bars2 = ax.bar(x + width/2, test_accs, width, label="Test Accuracy", color="#4CAF50", alpha=0.85)

    ax.set_xlabel("Model", fontsize=13)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_title("Model Performance Comparison", fontsize=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in models], fontsize=10)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar in bars1:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{h:.1f}%",
                    ha="center", fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{h:.1f}%",
                    ha="center", fontsize=9)

    plt.tight_layout()
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "model_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Model comparison chart saved: {save_path}")


def plot_feature_importance(importances: pd.Series, model_name: str, save_path: str = None):
    """Horizontal bar chart of feature importances."""
    fig, ax = plt.subplots(figsize=(9, 6))
    importances.sort_values().plot(kind="barh", ax=ax, color="#FF9800", edgecolor="white")
    ax.set_title(f"Feature Importance – {model_name}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance Score", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, f"feature_importance_{model_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Feature importance chart saved: {save_path}")


def plot_historical_win_rates(save_path: str = None):
    """Line chart of team win rates per season."""
    df = pd.read_csv(FEATURES_CSV)
    from config import ACTIVE_TEAMS_2026

    season_rates = {}
    for season, grp in df.groupby("season"):
        for team in ACTIVE_TEAMS_2026:
            t1 = grp[grp["team1"] == team]
            t2 = grp[grp["team2"] == team]
            played = len(t1) + len(t2)
            if played == 0:
                continue
            wins = t1["team1_won"].sum() + (1 - t2["team1_won"]).sum()
            season_rates.setdefault(team, {})[season] = wins / played

    fig, ax = plt.subplots(figsize=(14, 7))
    for team, rates in season_rates.items():
        seasons = sorted(rates.keys())
        wr = [rates[s] for s in seasons]
        ax.plot(seasons, wr, marker="o", label=team,
                color=PALETTE.get(team, "#888888"), linewidth=2, markersize=5)

    ax.set_xlabel("Season", fontsize=13)
    ax.set_ylabel("Win Rate", fontsize=13)
    ax.set_title("IPL Team Win Rates by Season (2008–2024)", fontsize=15, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "historical_win_rates.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Historical win rates chart saved: {save_path}")


def generate_all_charts():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    pred_file = os.path.join(RESULTS_DIR, "prediction_2026.json")
    if os.path.exists(pred_file):
        with open(pred_file) as f:
            data = json.load(f)
        plot_win_probability_bar(data["rankings"])

    plot_model_comparison()
    plot_historical_win_rates()
    print("\nAll charts generated in:", RESULTS_DIR)


if __name__ == "__main__":
    generate_all_charts()
