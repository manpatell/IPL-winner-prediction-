"""
Random Forest model for IPL match winner prediction.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import MODEL_PARAMS
from src.models.base_model import BaseIPLModel
from sklearn.ensemble import RandomForestClassifier


class RandomForestModel(BaseIPLModel):
    name = "random_forest"

    def _build(self):
        params = MODEL_PARAMS["random_forest"]
        self.model = RandomForestClassifier(**params)


if __name__ == "__main__":
    import pandas as pd
    from config import FEATURES_CSV
    df = pd.read_csv(FEATURES_CSV)
    model = RandomForestModel()
    cv = model.cross_validate(df)
    print(f"Random Forest CV: {cv['cv_mean']:.4f} ± {cv['cv_std']:.4f}")
    model.train(df)
    metrics = model.evaluate(df)
    print(f"Train accuracy: {metrics['accuracy']:.4f}")
    fi = model.feature_importance()
    if fi is not None:
        print("\nTop 5 features:")
        print(fi.head())
    model.save()
