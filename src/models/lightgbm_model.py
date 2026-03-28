"""
LightGBM model for IPL match winner prediction.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import MODEL_PARAMS
from src.models.base_model import BaseIPLModel
from lightgbm import LGBMClassifier


class LightGBMModel(BaseIPLModel):
    name = "lightgbm"

    def _build(self):
        self.model = LGBMClassifier(**MODEL_PARAMS["lightgbm"])


if __name__ == "__main__":
    import pandas as pd
    from config import FEATURES_CSV
    df = pd.read_csv(FEATURES_CSV)
    model = LightGBMModel()
    cv = model.cross_validate(df)
    print(f"LightGBM CV: {cv['cv_mean']:.4f} ± {cv['cv_std']:.4f}")
    model.train(df)
    metrics = model.evaluate(df)
    print(f"Train accuracy: {metrics['accuracy']:.4f}")
    fi = model.feature_importance()
    if fi is not None:
        print("\nTop 5 features:")
        print(fi.head())
    model.save()
