"""
XGBoost model for IPL match winner prediction.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import MODEL_PARAMS
from src.models.base_model import BaseIPLModel
from xgboost import XGBClassifier


class XGBoostModel(BaseIPLModel):
    name = "xgboost"

    def _build(self):
        params = {k: v for k, v in MODEL_PARAMS["xgboost"].items()
                  if k != "use_label_encoder"}
        self.model = XGBClassifier(**params, verbosity=0)


if __name__ == "__main__":
    import pandas as pd
    from config import FEATURES_CSV
    df = pd.read_csv(FEATURES_CSV)
    model = XGBoostModel()
    cv = model.cross_validate(df)
    print(f"XGBoost CV: {cv['cv_mean']:.4f} ± {cv['cv_std']:.4f}")
    model.train(df)
    metrics = model.evaluate(df)
    print(f"Train accuracy: {metrics['accuracy']:.4f}")
    fi = model.feature_importance()
    if fi is not None:
        print("\nTop 5 features:")
        print(fi.head())
    model.save()
