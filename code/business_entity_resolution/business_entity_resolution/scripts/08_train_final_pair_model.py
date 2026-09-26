"""Trains the final pair model on ALL development data (not a single fold),
using the hard-negative sample weights selected in script 07. This is the
model used for calibration (09), entity-model training (10), and final test
inference (11).

Run: python scripts/08_train_final_pair_model.py
"""
import os
import sys
import joblib

sys.path.insert(0, os.getcwd())
from src.utils.seed import set_seed
from src.utils.io import load_config, load_table
from src.utils.logging_utils import get_logger
from src.pipeline.train import train_final_pair_model

logger = get_logger(__name__)


def main():
    set_seed(42)
    cfg = load_config("config.yaml")

    weighted_path = os.path.join(cfg["paths"]["features_dir"], "pairwise_features_weighted.tsv")
    feature_df = load_table(weighted_path)

    model = train_final_pair_model(feature_df, model_name=cfg["pair_model"]["primary"],
                                    sample_weight_col="sample_weight")

    out_path = os.path.join(cfg["paths"]["models_dir"], "pair_model.joblib")
    joblib.dump(model, out_path)
    logger.info("Saved final pair model -> %s", out_path)

    importance = model.feature_importance()
    logger.info("Top 10 features by importance:\n%s", importance.head(10).to_string(index=False))
    importance.to_csv(os.path.join(cfg["paths"]["models_dir"], "pair_model_feature_importance.csv"), index=False)


if __name__ == "__main__":
    main()
