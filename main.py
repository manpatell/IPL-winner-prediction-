"""
IPL 2026 Winner Prediction – Main Entry Point

Usage:
  python main.py --mode setup      # Create datasets and populate database
  python main.py --mode train      # Train all models
  python main.py --mode predict    # Predict 2026 winner
  python main.py --mode all        # Run full pipeline end-to-end
  python main.py --mode visualize  # Generate charts (needs predict to run first)
"""
import argparse
import logging
import os
import sys
import time

from config import LOG_FILE, LOG_LEVEL

# ─── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="a"),
    ],
)
logger = logging.getLogger(__name__)


def mode_setup():
    logger.info("=== SETUP: Using raw dataset and populating database ===")
    t0 = time.time()

    from config import MATCHES_CSV, TEAMS_JSON
    from src.data.db_setup       import setup_database
    from src.data.ingest         import run_ingestion
    from src.data.preprocess     import run_preprocessing
    from src.features.engineer   import run_feature_engineering

    logger.info("Step 1/5: Validating raw dataset files...")
    if not os.path.exists(MATCHES_CSV):
        raise FileNotFoundError(
            f"Raw matches file not found: {MATCHES_CSV}. "
            "Add a real IPL dataset at data/raw/matches.csv and rerun setup."
        )
    if not os.path.exists(TEAMS_JSON):
        raise FileNotFoundError(
            f"Teams file not found: {TEAMS_JSON}. "
            "Add team metadata at data/raw/teams.json and rerun setup."
        )

    logger.info("Step 2/5: Creating SQLite database schema...")
    setup_database()

    logger.info("Step 3/5: Ingesting data into SQLite...")
    run_ingestion()

    logger.info("Step 4/5: Preprocessing matches...")
    run_preprocessing()

    logger.info("Step 5/5: Engineering features...")
    run_feature_engineering()

    logger.info(f"Setup complete in {time.time()-t0:.1f}s")


def mode_train():
    logger.info("=== TRAIN: Training all models ===")
    t0 = time.time()

    from src.models.trainer import run_training
    results = run_training()

    logger.info(f"Training complete in {time.time()-t0:.1f}s")
    return results


def mode_predict():
    logger.info("=== PREDICT: IPL 2026 Winner Prediction ===")
    t0 = time.time()

    from src.prediction.predict_2026 import (
        predict_2026_winner, print_predictions, save_predictions,
    )
    rankings = predict_2026_winner()
    print_predictions(rankings)
    save_predictions(rankings)

    logger.info(f"Prediction complete in {time.time()-t0:.1f}s")
    return rankings


def mode_visualize():
    logger.info("=== VISUALIZE: Generating charts ===")
    from src.prediction.visualize import generate_all_charts
    generate_all_charts()


def mode_all():
    mode_setup()
    mode_train()
    rankings = mode_predict()
    mode_visualize()
    return rankings


def parse_args():
    parser = argparse.ArgumentParser(
        description="IPL 2026 Winner Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["setup", "train", "predict", "visualize", "all"],
        default="all",
        help="Pipeline mode to run (default: all)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logger.info(f"Starting IPL 2026 prediction pipeline | mode={args.mode}")

    if args.mode == "setup":
        mode_setup()
    elif args.mode == "train":
        mode_train()
    elif args.mode == "predict":
        mode_predict()
    elif args.mode == "visualize":
        mode_visualize()
    elif args.mode == "all":
        mode_all()
