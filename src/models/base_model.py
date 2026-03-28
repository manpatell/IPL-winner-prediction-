"""
Base model class for all IPL prediction models.
Provides common interface: train, predict, evaluate, save, load.
"""
import os
import sys
import joblib
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report, confusion_matrix,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import MODELS_DIR, RANDOM_STATE, CV_FOLDS

FEATURE_COLS = [
    # Toss
    "toss_won_by_team1", "toss_decision_bat",
    # All-time win rate (Bayesian-smoothed, cumulative — no leakage)
    "t1_alltime_wr", "t2_alltime_wr", "wr_diff",
    # Last 3 seasons win rate — reflects CURRENT strength, not 15yr legacy
    "t1_last3yr_wr", "t2_last3yr_wr", "last3yr_wr_diff",
    # Recent match form (last 5 matches)
    "t1_recent_form", "t2_recent_form", "form_diff",
    # Current season form
    "t1_season_form", "t2_season_form",
    # Head-to-head (last 3 seasons)
    "h2h_t1_wr",
    # Venue win rates
    "t1_venue_wr", "t2_venue_wr", "venue_wr_diff",
    # Home advantage
    "t1_is_home", "t2_is_home",
    # Recent titles (last 5 seasons only)
    "t1_recent_titles", "t2_recent_titles", "recent_title_diff",
    # Venue pitch features (new)
    "venue_avg_score", "venue_toss_impact", "venue_size",
    # Team batting/bowling strength — causal features (new)
    "t1_batting_str", "t2_batting_str", "batting_str_diff",
    "t1_bowling_str", "t2_bowling_str", "bowling_str_diff",
]
TARGET_COL = "team1_won"


class BaseIPLModel(ABC):
    """Abstract base class for IPL winner prediction models."""

    name: str = "base"

    def __init__(self):
        self.model = None
        self.is_trained = False
        self._build()

    @abstractmethod
    def _build(self):
        """Instantiate self.model with configured hyperparameters."""

    def get_X_y(self, df: pd.DataFrame):
        X = df[FEATURE_COLS].copy()
        y = df[TARGET_COL].copy()
        return X, y

    def train(self, df: pd.DataFrame) -> dict:
        X, y = self.get_X_y(df)
        self.model.fit(X, y)
        self.is_trained = True
        train_acc = accuracy_score(y, self.model.predict(X))
        return {"train_accuracy": round(train_acc, 4)}

    def cross_validate(self, df: pd.DataFrame) -> dict:
        X, y = self.get_X_y(df)
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(self.model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
        return {
            "cv_mean": round(scores.mean(), 4),
            "cv_std":  round(scores.std(), 4),
            "cv_scores": [round(s, 4) for s in scores],
        }

    def evaluate(self, df: pd.DataFrame) -> dict:
        X, y = self.get_X_y(df)
        preds = self.model.predict(X)
        probs = self.predict_proba(X)[:, 1] if hasattr(self.model, "predict_proba") else None
        metrics = {
            "accuracy": round(accuracy_score(y, preds), 4),
            "report":   classification_report(y, preds, target_names=["team2_won", "team1_won"]),
        }
        if probs is not None:
            try:
                metrics["roc_auc"] = round(roc_auc_score(y, probs), 4)
            except Exception:
                pass
        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X[FEATURE_COLS]
        return self.model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X[FEATURE_COLS]
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        # Fallback for models without predict_proba
        preds = self.model.predict(X)
        probs = np.zeros((len(preds), 2))
        probs[preds == 1, 1] = 0.9
        probs[preds == 0, 0] = 0.9
        return probs

    def save(self):
        os.makedirs(MODELS_DIR, exist_ok=True)
        path = os.path.join(MODELS_DIR, f"{self.name}.pkl")
        joblib.dump(self.model, path)
        print(f"Model saved: {path}")
        return path

    def load(self):
        path = os.path.join(MODELS_DIR, f"{self.name}.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"No saved model at {path}")
        self.model = joblib.load(path)
        self.is_trained = True
        print(f"Model loaded: {path}")

    def feature_importance(self) -> pd.Series | None:
        if hasattr(self.model, "feature_importances_"):
            return pd.Series(self.model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
        if hasattr(self.model, "coef_"):
            return pd.Series(abs(self.model.coef_[0]), index=FEATURE_COLS).sort_values(ascending=False)
        return None
