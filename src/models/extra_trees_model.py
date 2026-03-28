"""
Extra Trees Classifier — 5th base model for the stacking ensemble.

ExtraTreesClassifier differs from RandomForest:
  - Uses RANDOM split thresholds (not best split), increasing diversity
  - Lower variance than RF, often better on small datasets
  - Different error patterns than RF/XGBoost → improves ensemble diversity
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.models.base_model import BaseIPLModel
from sklearn.ensemble import ExtraTreesClassifier
from config import RANDOM_STATE


class ExtraTreesModel(BaseIPLModel):
    name = "extra_trees"

    def _build(self):
        self.model = ExtraTreesClassifier(
            n_estimators=400,
            max_depth=12,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",   # handles any residual class imbalance
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )


if __name__ == "__main__":
    import pandas as pd
    from config import FEATURES_CSV
    df = pd.read_csv(FEATURES_CSV)
    model = ExtraTreesModel()
    cv = model.cross_validate(df)
    print(f"ExtraTrees CV: {cv['cv_mean']:.4f} ± {cv['cv_std']:.4f}")
    model.train(df)
    metrics = model.evaluate(df)
    print(f"Train accuracy: {metrics['accuracy']:.4f}")
    fi = model.feature_importance()
    if fi is not None:
        print("\nTop 5 features:")
        print(fi.head())
    model.save()
