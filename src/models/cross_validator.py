"""
Walk-forward time-series cross-validation for IPL models.

Standard k-fold CV is invalid for time-series: it leaks future matches into
training. Walk-forward CV trains on seasons 2008...(year-1) and validates
on (year), rolling forward one season at a time.

Folds:
  Fold 1: Train 2008-2013, Val 2014
  Fold 2: Train 2008-2014, Val 2015
  ...
  Fold 9: Train 2008-2021, Val 2022
  Fold 10: Train 2008-2022, Val 2023
  Hold-out test: 2024 (never used in CV)
"""
import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import FEATURES_CSV, FIRST_SEASON
from src.models.base_model import FEATURE_COLS, TARGET_COL

# Seasons used for cross-validation folds
CV_START_YEAR     = 2014   # first validation year (min 6 seasons of training data)
CV_END_YEAR       = 2023   # last validation year
TEST_YEAR         = 2024   # held-out test season


def walk_forward_split(df: pd.DataFrame):
    """
    Yields (df_train, df_val) tuples for each walk-forward fold.
    Each fold validates on one calendar season.
    """
    seasons = sorted(df["season"].unique())
    val_seasons = [s for s in seasons if CV_START_YEAR <= s <= CV_END_YEAR]

    for val_season in val_seasons:
        df_train = df[df["season"] < val_season]
        df_val   = df[df["season"] == val_season]
        if len(df_train) == 0 or len(df_val) == 0:
            continue
        yield df_train, df_val, val_season


def walk_forward_cv(model_class, df: pd.DataFrame) -> dict:
    """
    Run walk-forward CV for a given model class.
    Instantiates a fresh model for each fold.

    Returns dict with per-fold and aggregate metrics.
    """
    from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

    fold_metrics = []
    for df_train, df_val, val_season in walk_forward_split(df):
        model = model_class()
        model.train(df_train)

        X_val = df_val[FEATURE_COLS]
        y_val = df_val[TARGET_COL].values

        preds = model.predict(X_val)
        probs = model.predict_proba(X_val)[:, 1]

        n_classes = len(set(y_val))
        fold_metrics.append({
            "val_season":   val_season,
            "n_train":      len(df_train),
            "n_val":        len(df_val),
            "accuracy":     accuracy_score(y_val, preds),
            "log_loss":     log_loss(y_val, probs, labels=[0, 1]) if n_classes > 1 else 0.0,
            "roc_auc":      roc_auc_score(y_val, probs) if n_classes > 1 else 0.5,
        })
        print(f"  Season {val_season}: acc={fold_metrics[-1]['accuracy']:.4f} "
              f"auc={fold_metrics[-1]['roc_auc']:.4f} "
              f"(train={fold_metrics[-1]['n_train']}, val={fold_metrics[-1]['n_val']})")

    if not fold_metrics:
        return {}

    accs = [m["accuracy"]  for m in fold_metrics]
    aucs = [m["roc_auc"]   for m in fold_metrics]
    lls  = [m["log_loss"]  for m in fold_metrics]

    return {
        "fold_metrics":  fold_metrics,
        "mean_accuracy": round(np.mean(accs), 4),
        "std_accuracy":  round(np.std(accs),  4),
        "mean_auc":      round(np.mean(aucs), 4),
        "std_auc":       round(np.std(aucs),  4),
        "mean_log_loss": round(np.mean(lls),  4),
    }


def run_all_walk_forward_cv(df: pd.DataFrame) -> dict:
    """
    Run walk-forward CV for all four base models.
    Returns a dict of model_name -> cv_results.
    """
    from src.models.random_forest_model  import RandomForestModel
    from src.models.xgboost_model        import XGBoostModel
    from src.models.lightgbm_model       import LightGBMModel
    from src.models.neural_network_model import NeuralNetworkModel

    model_classes = {
        "random_forest":  RandomForestModel,
        "xgboost":        XGBoostModel,
        "lightgbm":       LightGBMModel,
        "neural_network": NeuralNetworkModel,
    }

    all_results = {}
    for name, cls in model_classes.items():
        print(f"\n{'='*55}")
        print(f"Walk-forward CV: {name.upper()}")
        print(f"{'='*55}")
        results = walk_forward_cv(cls, df)
        all_results[name] = results
        if results:
            print(f"  Mean accuracy: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
            print(f"  Mean AUC:      {results['mean_auc']:.4f} ± {results['std_auc']:.4f}")

    return all_results


def print_cv_summary(all_results: dict):
    print("\n" + "="*65)
    print(f"{'Model':<20} {'Mean Acc':>10} {'Std Acc':>10} {'Mean AUC':>10}")
    print("-"*65)
    for name, res in all_results.items():
        if res:
            print(f"{name:<20} {res['mean_accuracy']:>10.4f} "
                  f"{res['std_accuracy']:>10.4f} {res['mean_auc']:>10.4f}")
    print("="*65)


if __name__ == "__main__":
    df = pd.read_csv(FEATURES_CSV)
    print(f"Dataset: {len(df)} matches across seasons {df['season'].min()}-{df['season'].max()}")
    results = run_all_walk_forward_cv(df)
    print_cv_summary(results)
