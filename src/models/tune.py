"""
Optuna hyperparameter tuning for XGBoost and LightGBM.

Uses walk-forward cross-validation as the objective to avoid data leakage.
Saves best params to outputs/results/best_params.json.
"""
import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import FEATURES_CSV, RESULTS_DIR, RANDOM_STATE
from src.models.base_model import FEATURE_COLS, TARGET_COL
from src.models.cross_validator import walk_forward_split

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("Optuna not installed. Run: pip install optuna")


def _cv_score(model, df: pd.DataFrame) -> float:
    """Mean accuracy over walk-forward folds."""
    from sklearn.metrics import accuracy_score
    scores = []
    for df_train, df_val, _ in walk_forward_split(df):
        model_inst = model.__class__()
        model_inst.train(df_train)
        preds = model_inst.predict(df_val[FEATURE_COLS])
        scores.append(accuracy_score(df_val[TARGET_COL], preds))
    return float(np.mean(scores)) if scores else 0.0


def tune_xgboost(df: pd.DataFrame, n_trials: int = 60) -> dict:
    if not HAS_OPTUNA:
        return {}
    from xgboost import XGBClassifier
    from src.models.base_model import BaseIPLModel

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
            "max_depth":         trial.suggest_int("max_depth", 3, 8),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
        }

        class TunedXGB(BaseIPLModel):
            name = "xgboost_tuned"
            def _build(self):
                self.model = XGBClassifier(**params, verbosity=0,
                                           random_state=RANDOM_STATE,
                                           eval_metric="logloss")

        scores = []
        for df_train, df_val, _ in walk_forward_split(df):
            from sklearn.metrics import accuracy_score
            m = TunedXGB()
            m.train(df_train)
            preds = m.predict(df_val[FEATURE_COLS])
            scores.append(accuracy_score(df_val[TARGET_COL], preds))
        return float(np.mean(scores)) if scores else 0.0

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"XGBoost best CV accuracy: {study.best_value:.4f}")
    return study.best_params


def tune_lightgbm(df: pd.DataFrame, n_trials: int = 60) -> dict:
    if not HAS_OPTUNA:
        return {}
    from lightgbm import LGBMClassifier
    from src.models.base_model import BaseIPLModel

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
            "max_depth":         trial.suggest_int("max_depth", 3, 8),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 15, 80),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        }

        class TunedLGB(BaseIPLModel):
            name = "lightgbm_tuned"
            def _build(self):
                self.model = LGBMClassifier(**params, verbose=-1,
                                            random_state=RANDOM_STATE)

        scores = []
        for df_train, df_val, _ in walk_forward_split(df):
            from sklearn.metrics import accuracy_score
            m = TunedLGB()
            m.train(df_train)
            preds = m.predict(df_val[FEATURE_COLS])
            scores.append(accuracy_score(df_val[TARGET_COL], preds))
        return float(np.mean(scores)) if scores else 0.0

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"LightGBM best CV accuracy: {study.best_value:.4f}")
    return study.best_params


def run_tuning(n_trials: int = 50) -> dict:
    if not HAS_OPTUNA:
        print("Optuna not available. Skipping HPO.")
        return {}

    df = pd.read_csv(FEATURES_CSV)
    print(f"Tuning on {len(df)} samples with walk-forward CV...")

    best_params = {}

    print("\nTuning XGBoost...")
    best_params["xgboost"] = tune_xgboost(df, n_trials)

    print("\nTuning LightGBM...")
    best_params["lightgbm"] = tune_lightgbm(df, n_trials)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "best_params.json")
    with open(path, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"\nBest params saved: {path}")
    return best_params


def load_best_params() -> dict:
    path = os.path.join(RESULTS_DIR, "best_params.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


if __name__ == "__main__":
    run_tuning(n_trials=50)
