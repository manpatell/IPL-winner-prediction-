"""
SHAP-based feature importance and model explainability.
Generates summary plots for tree-based models and the ensemble.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import FEATURES_CSV, RESULTS_DIR
from src.models.base_model import FEATURE_COLS, TARGET_COL


def compute_shap_values(model_instance, df: pd.DataFrame):
    """
    Compute SHAP values for a tree-based model.
    Returns (shap_values, X) arrays.
    """
    try:
        import shap
    except ImportError:
        print("shap not installed. Run: pip install shap")
        return None, None

    X = df[FEATURE_COLS].values

    # Use TreeExplainer for RF, XGBoost, LightGBM
    model = model_instance.model
    # Unwrap pipeline if needed (NeuralNetwork wraps in sklearn Pipeline)
    if hasattr(model, "named_steps"):
        print("Neural network SHAP not supported in this version.")
        return None, None

    try:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        # For binary classification, shap_values may be a list [class0, class1]
        # or a 3D array of shape (n_samples, n_features, n_classes)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        elif hasattr(shap_values, "ndim") and shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]
        return shap_values, X
    except Exception as e:
        print(f"Could not compute SHAP values: {e}")
        return None, None


def plot_shap_summary(shap_values, X: np.ndarray, model_name: str, save_path: str = None):
    """Plot SHAP summary bar chart (mean absolute SHAP per feature)."""
    try:
        import shap
    except ImportError:
        return

    mean_abs = np.abs(shap_values).mean(axis=0)
    importance_df = pd.Series(mean_abs, index=FEATURE_COLS).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    importance_df.plot(kind="barh", ax=ax, color="#FF6B35", edgecolor="white")
    ax.set_title(f"SHAP Feature Importance – {model_name}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Mean |SHAP value|", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, f"shap_importance_{model_name}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"SHAP plot saved: {save_path}")

    # Also print top features
    top5 = importance_df.sort_values(ascending=False).head(5)
    print(f"\nTop 5 SHAP features for {model_name}:")
    for feat, val in top5.items():
        print(f"  {feat:<30} {val:.4f}")


def run_shap_analysis(df: pd.DataFrame):
    """Run SHAP analysis on RF, XGBoost, and LightGBM."""
    from src.models.random_forest_model  import RandomForestModel
    from src.models.xgboost_model        import XGBoostModel
    from src.models.lightgbm_model       import LightGBMModel

    models_to_analyze = [
        ("random_forest",  RandomForestModel),
        ("xgboost",        XGBoostModel),
        ("lightgbm",       LightGBMModel),
    ]

    for name, cls in models_to_analyze:
        print(f"\n{'='*50}")
        print(f"SHAP Analysis: {name.upper()}")
        print("="*50)
        try:
            m = cls()
            m.load()
        except FileNotFoundError:
            print(f"  No saved model found for {name}. Training first...")
            m = cls()
            m.train(df)
            m.save()

        shap_vals, X = compute_shap_values(m, df)
        if shap_vals is not None:
            plot_shap_summary(shap_vals, X, name)


if __name__ == "__main__":
    df = pd.read_csv(FEATURES_CSV)
    run_shap_analysis(df)
