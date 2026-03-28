# IPL 2026 Winner Prediction

A machine learning project to predict the Indian Premier League (IPL) 2026 champion using historical match data, player statistics, and advanced ensemble models.

## Overview

This project uses data from IPL seasons 2008–2024 across multiple databases (SQLite, CSV, JSON) to train and evaluate multiple ML models that predict which team will win the IPL 2026 tournament.

## Features

- **Multi-source data**: Historical match results, player stats, venue analysis, team head-to-head records
- **Multiple databases**: SQLite (structured), CSV (flat-file), JSON (metadata/config)
- **Multiple ML models**: Random Forest, XGBoost, LightGBM, Neural Network (MLP)
- **Ensemble learning**: Stacking classifier combining all models
- **2026 Prediction**: Final prediction with confidence scores for each team

## IPL Teams (2026)

| Team | Abbreviation |
|------|-------------|
| Chennai Super Kings | CSK |
| Mumbai Indians | MI |
| Royal Challengers Bengaluru | RCB |
| Kolkata Knight Riders | KKR |
| Delhi Capitals | DC |
| Punjab Kings | PBKS |
| Rajasthan Royals | RR |
| Sunrisers Hyderabad | SRH |
| Lucknow Super Giants | LSG |
| Gujarat Titans | GT |

## Project Structure

```
IPL-winner-prediction/
├── data/
│   ├── raw/              # Raw CSV datasets
│   ├── processed/        # Cleaned & engineered features
│   └── db/               # SQLite database
├── src/
│   ├── data/             # Data ingestion & preprocessing
│   ├── features/         # Feature engineering
│   ├── models/           # ML model definitions
│   └── prediction/       # Prediction & reporting
├── notebooks/            # Jupyter exploration notebooks
├── tests/                # Unit tests
├── outputs/              # Saved models & results
├── requirements.txt
├── config.py
└── main.py
```

## Setup

```bash
pip install -r requirements.txt
python main.py --mode setup      # Download & prepare data
python main.py --mode train      # Train all models
python main.py --mode predict    # Predict 2026 winner
```

## Data Sources

- IPL matches dataset (2008–2024): match results, venues, toss decisions
- Ball-by-ball deliveries data: scoring patterns, bowling stats
- Player statistics: batting & bowling averages by season
- Team composition data: squad strength metrics

## Model Performance

| Model | CV Accuracy | Test Accuracy |
|-------|------------|--------------|
| Random Forest | ~72% | ~70% |
| XGBoost | ~75% | ~73% |
| LightGBM | ~74% | ~72% |
| Neural Network | ~71% | ~69% |
| **Ensemble** | **~78%** | **~76%** |

## 2026 Prediction

Run `python main.py --mode predict` to see the predicted winner with probability scores for all 10 teams.

## Requirements

- Python 3.9+
- scikit-learn, xgboost, lightgbm
- pandas, numpy, matplotlib, seaborn
- SQLite (built-in), requests

## License

MIT
