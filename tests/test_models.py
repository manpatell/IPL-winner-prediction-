"""
Unit tests for ML models: train/predict/evaluate interface.
"""
import os
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.models.base_model import FEATURE_COLS, TARGET_COL


def make_feature_df(n: int = 50) -> pd.DataFrame:
    """Generate synthetic feature DataFrame for model testing."""
    np.random.seed(42)
    data = {col: np.random.uniform(0, 1, n) for col in FEATURE_COLS}
    # Integer features
    for col in ["toss_won_by_team1", "toss_decision_bat", "t1_is_home",
                "t2_is_home", "t1_titles", "t2_titles"]:
        data[col] = np.random.randint(0, 2, n)
    data["t1_titles"] = np.random.randint(0, 6, n)
    data["t2_titles"] = np.random.randint(0, 6, n)
    data["title_diff"] = data["t1_titles"] - data["t2_titles"]
    data[TARGET_COL] = np.random.randint(0, 2, n)
    return pd.DataFrame(data)


class TestRandomForest(unittest.TestCase):
    def setUp(self):
        from src.models.random_forest_model import RandomForestModel
        self.model = RandomForestModel()
        self.df = make_feature_df(100)

    def test_train_returns_metrics(self):
        result = self.model.train(self.df)
        self.assertIn("train_accuracy", result)
        self.assertGreater(result["train_accuracy"], 0.0)

    def test_predict_shape(self):
        self.model.train(self.df)
        preds = self.model.predict(self.df)
        self.assertEqual(len(preds), len(self.df))

    def test_predict_proba_shape(self):
        self.model.train(self.df)
        probs = self.model.predict_proba(self.df)
        self.assertEqual(probs.shape, (len(self.df), 2))

    def test_probabilities_sum_to_one(self):
        self.model.train(self.df)
        probs = self.model.predict_proba(self.df)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    def test_feature_importance_not_none(self):
        self.model.train(self.df)
        fi = self.model.feature_importance()
        self.assertIsNotNone(fi)
        self.assertEqual(len(fi), len(FEATURE_COLS))


class TestXGBoost(unittest.TestCase):
    def setUp(self):
        from src.models.xgboost_model import XGBoostModel
        self.model = XGBoostModel()
        self.df = make_feature_df(80)

    def test_train_and_predict(self):
        self.model.train(self.df)
        preds = self.model.predict(self.df)
        self.assertEqual(len(preds), len(self.df))
        self.assertTrue(set(preds).issubset({0, 1}))

    def test_accuracy_above_chance(self):
        self.model.train(self.df)
        metrics = self.model.evaluate(self.df)
        # On training data should beat random chance
        self.assertGreaterEqual(metrics["accuracy"], 0.5)


class TestLightGBM(unittest.TestCase):
    def setUp(self):
        from src.models.lightgbm_model import LightGBMModel
        self.model = LightGBMModel()
        self.df = make_feature_df(80)

    def test_train_and_evaluate(self):
        self.model.train(self.df)
        metrics = self.model.evaluate(self.df)
        self.assertIn("accuracy", metrics)
        self.assertIn("report", metrics)


class TestNeuralNetwork(unittest.TestCase):
    def setUp(self):
        from src.models.neural_network_model import NeuralNetworkModel
        self.model = NeuralNetworkModel()
        self.df = make_feature_df(80)

    def test_pipeline_predict(self):
        self.model.train(self.df)
        probs = self.model.predict_proba(self.df)
        self.assertEqual(probs.shape, (len(self.df), 2))

    def test_no_feature_importance(self):
        fi = self.model.feature_importance()
        self.assertIsNone(fi)


class TestBaseModelSaveLoad(unittest.TestCase):
    def test_save_creates_file(self):
        import tempfile, shutil
        from src.models.random_forest_model import RandomForestModel
        import config

        # Override MODELS_DIR temporarily
        tmp = tempfile.mkdtemp()
        orig = config.MODELS_DIR
        config.MODELS_DIR = tmp
        try:
            m = RandomForestModel()
            df = make_feature_df(50)
            m.train(df)
            path = m.save()
            self.assertTrue(os.path.exists(path))
        finally:
            config.MODELS_DIR = orig
            shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main(verbosity=2)
