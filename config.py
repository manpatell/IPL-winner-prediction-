"""
Central configuration for the IPL 2026 Winner Prediction project.
"""
import os

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = os.path.join(BASE_DIR, "data")
RAW_DIR        = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR  = os.path.join(DATA_DIR, "processed")
DB_DIR         = os.path.join(DATA_DIR, "db")
OUTPUTS_DIR    = os.path.join(BASE_DIR, "outputs")
MODELS_DIR     = os.path.join(OUTPUTS_DIR, "models")
RESULTS_DIR    = os.path.join(OUTPUTS_DIR, "results")

# ─── Database ────────────────────────────────────────────────────────────────
SQLITE_DB_PATH = os.path.join(DB_DIR, "ipl.db")

# ─── Data files ──────────────────────────────────────────────────────────────
MATCHES_CSV    = os.path.join(RAW_DIR, "matches.csv")
DELIVERIES_CSV = os.path.join(RAW_DIR, "deliveries.csv")
PLAYERS_CSV    = os.path.join(RAW_DIR, "players.csv")
TEAMS_JSON     = os.path.join(RAW_DIR, "teams.json")

PROCESSED_MATCHES_CSV   = os.path.join(PROCESSED_DIR, "matches_processed.csv")
FEATURES_CSV            = os.path.join(PROCESSED_DIR, "features.csv")
TEAM_STATS_CSV          = os.path.join(PROCESSED_DIR, "team_stats.csv")

# ─── IPL Teams ────────────────────────────────────────────────────────────────
TEAMS = {
    "CSK":  "Chennai Super Kings",
    "MI":   "Mumbai Indians",
    "RCB":  "Royal Challengers Bengaluru",
    "KKR":  "Kolkata Knight Riders",
    "DC":   "Delhi Capitals",
    "PBKS": "Punjab Kings",
    "RR":   "Rajasthan Royals",
    "SRH":  "Sunrisers Hyderabad",
    "LSG":  "Lucknow Super Giants",
    "GT":   "Gujarat Titans",
}

# Historical team name aliases (teams renamed over the years)
TEAM_ALIASES = {
    "Rising Pune Supergiant":    "RPS",
    "Rising Pune Supergiants":   "RPS",
    "Pune Warriors":             "PW",
    "Kochi Tuskers Kerala":      "KTK",
    "Deccan Chargers":           "DC_OLD",
    "Delhi Daredevils":          "DC",
    "Kings XI Punjab":           "PBKS",
    "Royal Challengers Bangalore": "RCB",
    "Royal Challengers Bengaluru": "RCB",
    "Chennai Super Kings":       "CSK",
    "Mumbai Indians":            "MI",
    "Kolkata Knight Riders":     "KKR",
    "Rajasthan Royals":          "RR",
    "Sunrisers Hyderabad":       "SRH",
    "Delhi Capitals":            "DC",
    "Punjab Kings":              "PBKS",
    "Lucknow Super Giants":      "LSG",
    "Gujarat Titans":            "GT",
}

# Active teams in 2026 (current franchises)
ACTIVE_TEAMS_2026 = list(TEAMS.keys())

# ─── Seasons ─────────────────────────────────────────────────────────────────
FIRST_SEASON      = 2008
LAST_KNOWN_SEASON = 2024
PREDICT_SEASON    = 2026

# ─── Feature Engineering ─────────────────────────────────────────────────────
FORM_WINDOW        = 5    # last N matches for recent form
HOME_ADVANTAGE     = True
TOSS_FEATURES      = True
VENUE_FEATURES     = True
H2H_WINDOW_SEASONS = 3    # head-to-head over last N seasons

# ─── Model ────────────────────────────────────────────────────────────────────
RANDOM_STATE  = 42
TEST_SIZE     = 0.2
CV_FOLDS      = 5

MODEL_PARAMS = {
    "random_forest": {
        "n_estimators": 300,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    },
    "xgboost": {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE,
    },
    "lightgbm": {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "num_leaves": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_STATE,
        "verbose": -1,
    },
    "neural_network": {
        "hidden_layer_sizes": (256, 128, 64),
        "activation": "relu",
        "solver": "adam",
        "alpha": 1e-3,
        "max_iter": 500,
        "random_state": RANDOM_STATE,
    },
}

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_FILE  = os.path.join(BASE_DIR, "ipl_prediction.log")
