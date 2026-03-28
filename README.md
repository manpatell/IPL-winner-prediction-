# IPL 2026 Winner Prediction

A full machine learning pipeline to predict the Indian Premier League (IPL) 2026 champion — built with fairness at its core. Predictions are based on **current squad strength, recent form, venue conditions, and toss impact**, not historical trophy counts.

---

## Predicted Winner: Kolkata Knight Riders (12.18%)

| Rank | Team | Win Probability |
|------|------|----------------|
| 1 | Kolkata Knight Riders | 12.18% |
| 2 | Gujarat Titans | 11.80% |
| 3 | Rajasthan Royals | 11.49% |
| 4 | Royal Challengers Bengaluru | 11.41% |
| 5 | Sunrisers Hyderabad | 10.94% |
| 6 | Lucknow Super Giants | 10.41% |
| 7 | Mumbai Indians | 9.96% |
| 8 | Chennai Super Kings | 7.76% |
| 9 | Delhi Capitals | 7.37% |
| 10 | Punjab Kings | 6.68% |

> Prediction method: Stacking Ensemble ML + Bayesian update (Squad 40% · Recent form 30% · Model 25% · Playoff rate 5%)

---

## Why This Model Is Fair

Most IPL prediction models are biased toward CSK because of their 5 all-time titles. This project fixes that:

- **All-time title count removed** — a 2011 title means nothing for 2026
- **Recent titles only** (2020–2024): CSK=2, KKR=1, GT=1, MI=1
- **GT and LSG treated fairly** — evaluated on the same 3-season window as everyone else, not penalized for being new franchises
- **Squad strength is the primary signal** — based on current player quality, retentions, and 2024/2025 auction results

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Databases | SQLite · CSV · JSON |
| ML Models | Random Forest · XGBoost · LightGBM · Neural Network · ExtraTrees |
| Ensemble | Stacking (5 base models → Logistic Regression meta-learner) |
| Validation | Walk-forward cross-validation (2014–2023 folds) |
| Explainability | SHAP feature importance |
| Tuning | Optuna hyperparameter optimization |

---

## Model Performance

| Model | CV Accuracy | Test Accuracy | AUC |
|-------|------------|--------------|-----|
| Random Forest | 63.24% | 63.24% | 0.6689 |
| XGBoost | 64.28% | 60.29% | 0.6464 |
| LightGBM | 64.52% | 62.25% | 0.6602 |
| Neural Network | 59.81% | 56.37% | 0.6163 |
| **ExtraTrees** | **63.42%** | **64.71%** | **0.6953** |
| Ensemble | — | 57.35% | 0.6509 |

---

## Features (31 total)

| Category | Features |
|----------|----------|
| Toss | `toss_won_by_team1`, `toss_decision_bat` |
| Win rates | All-time smoothed, last-3yr (Bayesian), recent form, season form |
| Head-to-head | `h2h_t1_wr` (last 5 seasons) |
| Venue | Win rate at venue, home ground flag, avg score, toss impact, size |
| Recent titles | Last 5 seasons only (2020–2024) |
| Team strength | Batting strength, bowling strength (from player stats) |

Key accuracy improvements:
- **Dataset expanded**: 159 → 1020 matches (+ mirroring = 2040 balanced rows)
- **Class balance fixed**: 73% → 50% via match mirroring
- **Data leakage fixed**: per-row cumulative win rates (no future data)
- **Bayesian smoothing**: new teams fairly regress toward 0.5

---

## Project Structure

```
IPL-winner-prediction/
├── config.py                    # Central config (paths, teams, model params)
├── main.py                      # CLI entry point
├── data/
│   ├── raw/                     # matches.csv, teams.json
│   ├── processed/               # matches_processed.csv, features.csv
│   └── db/                      # ipl.db (SQLite — 6 tables)
├── src/
│   ├── data/
│   │   ├── create_dataset.py    # Generates 1020 matches (2008–2024)
│   │   ├── db_setup.py          # SQLite schema
│   │   ├── ingest.py            # CSV/JSON → SQLite
│   │   └── preprocess.py        # Cleaning + match mirroring (50/50 balance)
│   ├── features/
│   │   ├── engineer.py          # 31-feature engineering pipeline
│   │   ├── venue_features.py    # Pitch conditions per venue
│   │   └── team_strength.py     # Batting/bowling strength from player stats
│   ├── models/
│   │   ├── base_model.py        # FEATURE_COLS, base class
│   │   ├── random_forest_model.py
│   │   ├── xgboost_model.py
│   │   ├── lightgbm_model.py
│   │   ├── neural_network_model.py
│   │   ├── extra_trees_model.py # 5th base model
│   │   ├── ensemble_model.py    # Stacking ensemble
│   │   ├── cross_validator.py   # Walk-forward CV
│   │   ├── shap_explainer.py    # SHAP feature importance
│   │   ├── tune.py              # Optuna HPO
│   │   └── trainer.py           # Training orchestrator
│   └── prediction/
│       ├── predict_2026.py      # Main prediction logic
│       ├── playoff_simulator.py # Monte Carlo tournament simulation
│       └── visualize.py         # Charts & plots
├── tests/                       # 43 unit tests (all passing)
├── outputs/
│   ├── models/                  # Saved .pkl model files
│   └── results/                 # model_results.json, prediction_2026.json
└── requirements.txt
```

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (setup → train → predict → visualize)
python main.py --mode all

# Or step by step
python main.py --mode setup      # Generate data & engineer features
python main.py --mode train      # Train all 5 models + ensemble
python main.py --mode predict    # Predict IPL 2026 winner
python main.py --mode visualize  # Generate charts
```

---

## Databases

| Type | File | Contents |
|------|------|----------|
| SQLite | `data/db/ipl.db` | 6 tables: teams, venues, matches, season_stats, head_to_head, player_stats |
| CSV | `data/raw/matches.csv` | 1020 historical matches (2008–2024) |
| JSON | `data/raw/teams.json` | Team metadata, abbreviations |

---

## Tests

```bash
python -m pytest tests/ -v
# 43 passed
```

---

## Requirements

- Python 3.9+
- `scikit-learn`, `xgboost`, `lightgbm`, `optuna`
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `shap`, `joblib`
- SQLite (built-in)

---

## License

MIT
