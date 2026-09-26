"""Stage B: trains the pair model under 5-fold GroupKFold, producing
out-of-fold predictions for every candidate pair. These OOF predictions are
the basis for hard-negative mining (07), calibration/threshold selection
(09), and entity-model training (10) — never in-sample scores.

Run: python scripts/06_train_pair_oof.py
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.getcwd())
from src.utils.seed import set_seed
from src.utils.io import load_config, load_table, save_table
from src.utils.logging_utils import get_logger
from src.data.splits import make_folds
from src.pipeline.oof import run_oof
from src.evaluation.metrics import macro_f05
from src.evaluation.threshold import sweep_threshold

logger = get_logger(__name__)


def main():
    set_seed(42)
    cfg = load_config("config.yaml")

    feature_df = load_table(os.path.join(cfg["paths"]["features_dir"], "pairwise_features.tsv"))
    labels_df = feature_df[["source1_entity_id", "candidate_source", "candidate_record_id", "label"]].copy()

    folds_df = make_folds(feature_df, group_col=cfg["validation"]["group_column"],
                           n_splits=cfg["validation"]["n_folds"], seed=cfg["seed"])
    save_table(folds_df, os.path.join(cfg["paths"]["oof_dir"], "folds.tsv"))
    logger.info("Built %d folds over %d unique S1 entities", folds_df["fold"].nunique(), len(folds_df))

    primary_model = cfg["pair_model"]["primary"]
    oof_df = run_oof(feature_df, folds_df, model_name=primary_model)

    save_table(oof_df, os.path.join(cfg["paths"]["oof_dir"], "pair_oof_predictions.tsv"))
    logger.info("Saved OOF predictions -> %s", os.path.join(cfg["paths"]["oof_dir"], "pair_oof_predictions.tsv"))

    # Quick sanity check: sweep a global threshold on these OOF scores and
    # report the resulting macro F0.5, purely as a diagnostic at this stage
    # (the real threshold selection happens in script 09, after hard-negative
    # retraining and calibration).
    oof_scored = oof_df.rename(columns={"oof_score": "oof_score"})[
        ["source1_entity_id", "candidate_record_id", "oof_score"]
    ]
    sweep = sweep_threshold(oof_scored, labels_df, cfg["threshold"]["sweep_min"],
                             cfg["threshold"]["sweep_max"], cfg["threshold"]["sweep_step"])
    best_row = sweep.iloc[0]
    logger.info("Diagnostic OOF sweep (pre-hard-negative-mining) best: threshold=%.2f macro_f05=%.4f precision=%.4f recall=%.4f",
                best_row["threshold"], best_row["macro_f05"], best_row["precision"], best_row["recall"])
    save_table(sweep, os.path.join(cfg["paths"]["oof_dir"], "threshold_sweep_pre_hardneg.tsv"))


if __name__ == "__main__":
    main()
