"""
Neural Network (MLP) model for IPL match winner prediction.
Uses sklearn's MLPClassifier with StandardScaler preprocessing.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import MODEL_PARAMS, MODELS_DIR
from src.models.base_model import BaseIPLModel, FEATURE_COLS, TARGET_COL

import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


class NeuralNetworkModel(BaseIPLModel):
    name = "neural_network"

    def _build(self):
        mlp = MLPClassifier(**MODEL_PARAMS["neural_network"])
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp",    mlp),
        ])

    def feature_importance(self):
        # MLP doesn't have built-in feature importance
        return None

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


if __name__ == "__main__":
    import pandas as pd
    from config import FEATURES_CSV
    df = pd.read_csv(FEATURES_CSV)
    model = NeuralNetworkModel()
    cv = model.cross_validate(df)
    print(f"Neural Network CV: {cv['cv_mean']:.4f} ± {cv['cv_std']:.4f}")
    model.train(df)
    metrics = model.evaluate(df)
    print(f"Train accuracy: {metrics['accuracy']:.4f}")
    model.save()
