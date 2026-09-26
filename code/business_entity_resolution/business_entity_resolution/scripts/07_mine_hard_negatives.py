"""Mines hard negatives from OOF predictions (never in-sample scores), then
prepares sample-weighted training data with those hard negatives up-weighted
(not duplicated as rows). The weight itself is chosen by a small OOF sweep
over `hard_negative_mining.weight_range`, evaluated on macro F0.5 — not
picked arbitrarily.

Run: python scripts/07_mine_hard_negatives.py
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
from src.utils.seed import set_seed
from src.utils.io import load_config, load_table, save_table
from src.utils.logging_utils import get_logger
from src.data.splits import make_folds
from src.mining.hard_negatives import mine_hard_negatives, apply_sample_weights
from src.pipeline.oof import run_oof
from src.evaluation.metrics import macro_f05
from src.evaluation.threshold import sweep_threshold

logger = get_logger(__name__)


def _evaluate_weight(feature_df: pd.DataFrame, folds_df: pd.DataFrame, hn_keys: set,
                      weight: float, labels_df: pd.DataFrame, cfg: dict) -> float:
    """Re-runs OOF training with this hard-negative weight and returns the
    best achievable macro F0.5 from a threshold sweep on the resulting OOF
    scores — this is what "tune experimentally" actually means here, rather
    than a single untested draw from the configured range."""
    weighted = apply_sample_weights(feature_df, hn_keys, weight=weight)
    oof = run_oof(weighted, folds_df, model_name=cfg["pair_model"]["primary"], sample_weight_col="sample_weight")
    oof_scored = oof[["source1_entity_id", "candidate_record_id", "oof_score"]]
    sweep = sweep_threshold(oof_scored, labels_df, cfg["threshold"]["sweep_min"],
                             cfg["threshold"]["sweep_max"], cfg["threshold"]["sweep_step"])
    return float(sweep.iloc[0]["macro_f05"])


def main():
    set_seed(42)
    cfg = load_config("config.yaml")

    oof_df = load_table(os.path.join(cfg["paths"]["oof_dir"], "pair_oof_predictions.tsv"))
    feature_df = load_table(os.path.join(cfg["paths"]["features_dir"], "pairwise_features.tsv"))
    folds_df = load_table(os.path.join(cfg["paths"]["oof_dir"], "folds.tsv"))
    labels_df = feature_df[["source1_entity_id", "candidate_source", "candidate_record_id", "label"]].copy()

    hn_cfg = cfg["hard_negative_mining"]
    if not hn_cfg["enabled"]:
        logger.info("Hard-negative mining disabled in config — copying features through unweighted.")
        feature_df["sample_weight"] = 1.0
        save_table(feature_df, os.path.join(cfg["paths"]["features_dir"], "pairwise_features_weighted.tsv"))
        return

    hard_negatives = mine_hard_negatives(oof_df, hn_cfg)
    if hard_negatives.empty:
        logger.warning("No hard negatives matched any of the 5 categories on this data — "
                        "this is expected on a small/easy synthetic set, but should be "
                        "revisited on real data where near-duplicate negatives are common. "
                        "Proceeding with sample_weight=1.0 for all rows.")
        feature_df["sample_weight"] = 1.0
        save_table(feature_df, os.path.join(cfg["paths"]["features_dir"], "pairwise_features_weighted.tsv"))
        return

    logger.info("Mined %d hard negatives across categories:\n%s",
                len(hard_negatives), hard_negatives["hard_negative_type"].value_counts().to_string())

    hn_keys = set((str(r["source1_entity_id"]) + "|" + str(r["candidate_record_id"]))
                  for _, r in hard_negatives.iterrows())

    weight_lo, weight_hi = hn_cfg["weight_range"]
    candidate_weights = np.linspace(weight_lo, weight_hi, num=3)  # small sweep, cheap enough to always run
    logger.info("Sweeping hard-negative sample weights %s via re-run OOF training...", list(candidate_weights))

    best_weight, best_f05 = 1.0, -1.0
    for w in candidate_weights:
        f05 = _evaluate_weight(feature_df, folds_df, hn_keys, float(w), labels_df, cfg)
        logger.info("  weight=%.2f -> OOF-sweep best macro_f05=%.4f", w, f05)
        if f05 > best_f05:
            best_weight, best_f05 = float(w), f05

    logger.info("Selected hard-negative weight=%.2f (OOF macro_f05=%.4f)", best_weight, best_f05)

    weighted_df = apply_sample_weights(feature_df, hn_keys, weight=best_weight)
    save_table(weighted_df, os.path.join(cfg["paths"]["features_dir"], "pairwise_features_weighted.tsv"))
    save_table(hard_negatives, os.path.join(cfg["paths"]["oof_dir"], "hard_negatives.tsv"))
    logger.info("Saved weighted feature table -> %s",
                os.path.join(cfg["paths"]["features_dir"], "pairwise_features_weighted.tsv"))


if __name__ == "__main__":
    main()
