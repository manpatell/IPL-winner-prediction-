"""
Stacking Ensemble model combining RF, XGBoost, LightGBM, Neural Network, and ExtraTrees.
Uses a Logistic Regression meta-learner on top of base model probability outputs.
"""
import sys
import os
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import MODELS_DIR, RANDOM_STATE, CV_FOLDS, FEATURES_CSV
from src.models.base_model import FEATURE_COLS, TARGET_COL
from src.models.random_forest_model import RandomForestModel
from src.models.xgboost_model import XGBoostModel
from src.models.lightgbm_model import LightGBMModel
from src.models.neural_network_model import NeuralNetworkModel
from src.models.extra_trees_model import ExtraTreesModel

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler


class EnsembleModel:
    """
    Stacking ensemble:
      Level 0: RF + XGBoost + LightGBM + NeuralNetwork + ExtraTrees
      Level 1: Logistic Regression meta-learner
    """
    name = "ensemble"

    def __init__(self):
        self.base_models = [
            RandomForestModel(),
            XGBoostModel(),
            LightGBMModel(),
            NeuralNetworkModel(),
            ExtraTreesModel(),
        ]
        self.meta_learner = LogisticRegression(
            C=1.0, random_state=RANDOM_STATE, max_iter=500,
        )
        self.scaler = StandardScaler()
        self.is_trained = False

    def _get_meta_features(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """Return (n_samples, n_base_models) probability matrix."""
        X = df[FEATURE_COLS]
        meta = np.zeros((len(df), len(self.base_models)))
        for i, model in enumerate(self.base_models):
            probs = model.predict_proba(X)[:, 1]
            meta[:, i] = probs
        if fit:
            meta = self.scaler.fit_transform(meta)
        else:
            meta = self.scaler.transform(meta)
        return meta

    def train(self, df: pd.DataFrame) -> dict:
        y = df[TARGET_COL].values

        # Train all base models
        print("Training base models...")
        for model in self.base_models:
            model.train(df)
            print(f"  {model.name} trained.")

        # Build meta-features and train meta-learner
        meta = self._get_meta_features(df, fit=True)
        self.meta_learner.fit(meta, y)
        self.is_trained = True

        train_acc = accuracy_score(y, self.meta_learner.predict(meta))
        print(f"Ensemble train accuracy: {train_acc:.4f}")
        return {"train_accuracy": round(train_acc, 4)}

    def cross_validate(self, df: pd.DataFrame) -> dict:
        """Out-of-fold stacking cross-validation."""
        X = df[FEATURE_COLS].values
        y = df[TARGET_COL].values
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

        oof_preds = np.zeros(len(y))
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            df_train = df.iloc[train_idx]
            df_val   = df.iloc[val_idx]
            y_val    = y[val_idx]

            # Train base models on fold
            fold_bases = [
                RandomForestModel(), XGBoostModel(),
                LightGBMModel(), NeuralNetworkModel(), ExtraTreesModel(),
            ]
            meta_train = np.zeros((len(df_train), len(fold_bases)))
            meta_val   = np.zeros((len(df_val),   len(fold_bases)))
            for i, m in enumerate(fold_bases):
                m.train(df_train)
                meta_train[:, i] = m.predict_proba(df_train[FEATURE_COLS])[:, 1]
                meta_val[:, i]   = m.predict_proba(df_val[FEATURE_COLS])[:, 1]

            sc = StandardScaler()
            meta_train = sc.fit_transform(meta_train)
            meta_val   = sc.transform(meta_val)

            lr = LogisticRegression(C=1.0, random_state=RANDOM_STATE, max_iter=500)
            lr.fit(meta_train, y[train_idx])
            oof_preds[val_idx] = lr.predict(meta_val)
            print(f"  Fold {fold+1}/{CV_FOLDS} val acc: {accuracy_score(y_val, lr.predict(meta_val)):.4f}")

        cv_acc = accuracy_score(y, oof_preds)
        print(f"Ensemble OOF CV accuracy: {cv_acc:.4f}")
        return {"cv_mean": round(cv_acc, 4), "cv_std": 0.0}

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model not trained yet.")
        meta = self._get_meta_features(X if isinstance(X, pd.DataFrame)
                                       else pd.DataFrame(X, columns=FEATURE_COLS))
        return self.meta_learner.predict_proba(meta)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def evaluate(self, df: pd.DataFrame) -> dict:
        y = df[TARGET_COL].values
        preds = self.predict(df)
        probs = self.predict_proba(df)[:, 1]
        return {
            "accuracy": round(accuracy_score(y, preds), 4),
            "roc_auc":  round(roc_auc_score(y, probs), 4),
            "report":   classification_report(y, preds,
                            target_names=["team2_won", "team1_won"]),
        }

    def save(self):
        os.makedirs(MODELS_DIR, exist_ok=True)
        data = {
            "meta_learner": self.meta_learner,
            "scaler":       self.scaler,
            "base_model_names": [m.name for m in self.base_models],
        }
        path = os.path.join(MODELS_DIR, f"{self.name}.pkl")
        joblib.dump(data, path)
        # Also save individual base models
        for m in self.base_models:
            m.save()
        print(f"Ensemble saved: {path}")
        return path

    def load(self):
        path = os.path.join(MODELS_DIR, f"{self.name}.pkl")
        data = joblib.load(path)
        self.meta_learner = data["meta_learner"]
        self.scaler       = data["scaler"]
        for m in self.base_models:
            m.load()
        self.is_trained = True
        print(f"Ensemble loaded: {path}")


if __name__ == "__main__":
    df = pd.read_csv(FEATURES_CSV)
    model = EnsembleModel()
    print("Running OOF cross-validation (takes a few minutes)...")
    cv = model.cross_validate(df)
    print(f"\nEnsemble CV accuracy: {cv['cv_mean']:.4f}")
    print("\nTraining final ensemble on full data...")
    model.train(df)
    metrics = model.evaluate(df)
    print(f"Final train accuracy: {metrics['accuracy']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    model.save()
