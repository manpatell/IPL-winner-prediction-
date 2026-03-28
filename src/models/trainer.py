"""
Model training orchestrator.
Trains all individual models + ensemble, saves metrics to JSON.
"""
import os
import sys
import json
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import FEATURES_CSV, RESULTS_DIR, RANDOM_STATE, TEST_SIZE
from src.models.random_forest_model  import RandomForestModel
from src.models.xgboost_model        import XGBoostModel
from src.models.lightgbm_model       import LightGBMModel
from src.models.neural_network_model import NeuralNetworkModel
from src.models.extra_trees_model    import ExtraTreesModel
from src.models.ensemble_model       import EnsembleModel
from src.models.base_model           import FEATURE_COLS, TARGET_COL
from src.models.tune                 import load_best_params


def load_features() -> pd.DataFrame:
    if not os.path.exists(FEATURES_CSV):
        raise FileNotFoundError(f"Features file not found: {FEATURES_CSV}\n"
                                "Run: python main.py --mode setup first.")
    return pd.read_csv(FEATURES_CSV)


def _apply_tuned_params(model, best_params: dict):
    """Override model hyperparameters with Optuna-tuned values if available."""
    name_map = {"random_forest": None, "xgboost": "xgboost",
                "lightgbm": "lightgbm", "neural_network": None, "extra_trees": None}
    key = name_map.get(model.name)
    if key and key in best_params and best_params[key]:
        params = best_params[key]
        for attr, val in params.items():
            if hasattr(model.model, attr):
                setattr(model.model, attr, val)
        print(f"  Applied {len(params)} tuned params to {model.name}")


def train_all(df: pd.DataFrame) -> dict:
    results = {}

    df_train, df_test = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df[TARGET_COL]
    )

    best_params = load_best_params()
    if best_params:
        print(f"Loaded Optuna-tuned params for: {list(best_params.keys())}")

    individual_models = [
        RandomForestModel(),
        XGBoostModel(),
        LightGBMModel(),
        NeuralNetworkModel(),
        ExtraTreesModel(),
    ]

    for model in individual_models:
        print(f"\n{'='*50}")
        print(f"Training: {model.name.upper()}")
        print(f"{'='*50}")

        if best_params:
            _apply_tuned_params(model, best_params)

        cv = model.cross_validate(df_train)
        print(f"  CV accuracy: {cv['cv_mean']:.4f} ± {cv['cv_std']:.4f}")

        model.train(df_train)
        train_metrics = model.evaluate(df_train)
        test_metrics  = model.evaluate(df_test)

        print(f"  Train acc: {train_metrics['accuracy']:.4f}")
        print(f"  Test  acc: {test_metrics['accuracy']:.4f}")
        if "roc_auc" in test_metrics:
            print(f"  Test AUC : {test_metrics['roc_auc']:.4f}")
        print(test_metrics["report"])

        fi = model.feature_importance()
        if fi is not None:
            print("  Top 5 features:")
            print(fi.head().to_string())

        model.save()

        results[model.name] = {
            "cv_accuracy":   cv["cv_mean"],
            "cv_std":        cv["cv_std"],
            "train_accuracy": train_metrics["accuracy"],
            "test_accuracy":  test_metrics["accuracy"],
        }
        if "roc_auc" in test_metrics:
            results[model.name]["test_roc_auc"] = test_metrics["roc_auc"]

    # ── Ensemble ─────────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print("Training: STACKING ENSEMBLE")
    print(f"{'='*50}")

    ensemble = EnsembleModel()
    ensemble.train(df_train)
    ens_test = ensemble.evaluate(df_test)
    print(f"  Ensemble test accuracy: {ens_test['accuracy']:.4f}")
    print(f"  Ensemble test AUC:      {ens_test.get('roc_auc', 'N/A')}")
    print(ens_test["report"])
    ensemble.save()

    results["ensemble"] = {
        "train_accuracy": ensemble.evaluate(df_train)["accuracy"],
        "test_accuracy":  ens_test["accuracy"],
        "test_roc_auc":   ens_test.get("roc_auc", None),
    }

    return results


def save_results(results: dict):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "model_results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {path}")

    # Print comparison table
    print("\n" + "="*65)
    print(f"{'Model':<20} {'CV Acc':>8} {'Train':>8} {'Test':>8} {'AUC':>8}")
    print("-"*65)
    for name, m in results.items():
        cv   = f"{m.get('cv_accuracy', 0):.4f}" if m.get("cv_accuracy") else "  N/A "
        tr   = f"{m['train_accuracy']:.4f}"
        te   = f"{m['test_accuracy']:.4f}"
        auc  = f"{m['test_roc_auc']:.4f}" if m.get("test_roc_auc") else "  N/A "
        print(f"{name:<20} {cv:>8} {tr:>8} {te:>8} {auc:>8}")
    print("="*65)


def run_training():
    print("Loading feature dataset...")
    df = load_features()
    print(f"Dataset: {len(df)} matches × {len(FEATURE_COLS)} features")

    results = train_all(df)
    save_results(results)
    return results


if __name__ == "__main__":
    run_training()
