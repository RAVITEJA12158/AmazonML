"""Re-runs OOF with the hard-negative-weighted features (so the OOF scores
being calibrated/thresholded reflect the actual final training recipe), then:
  1. tests Platt / isotonic calibration against raw scores on OOF, keeps
     whichever (if any) improves the OOF threshold sweep's best macro F0.5
  2. sweeps the global threshold on the (possibly calibrated) OOF scores
  3. freezes that threshold — it is NOT re-tuned on the holdout in script 11,
     only confirmed there.

Run: python scripts/09_calibrate_and_threshold.py
"""
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
from src.utils.seed import set_seed
from src.utils.io import load_config, load_table, save_table
from src.utils.logging_utils import get_logger
from src.pipeline.oof import run_oof
from src.models.calibration import get_calibrator
from src.evaluation.threshold import sweep_threshold

logger = get_logger(__name__)


def _best_f05(oof_scores: pd.Series, oof_df: pd.DataFrame, labels_df: pd.DataFrame, cfg: dict):
    scored = pd.DataFrame({
        "source1_entity_id": oof_df["source1_entity_id"],
        "candidate_record_id": oof_df["candidate_record_id"],
        "oof_score": oof_scores,
    })
    sweep = sweep_threshold(scored, labels_df, cfg["threshold"]["sweep_min"],
                             cfg["threshold"]["sweep_max"], cfg["threshold"]["sweep_step"])
    return sweep.iloc[0], sweep


def main():
    set_seed(42)
    cfg = load_config("config.yaml")

    weighted_df = load_table(os.path.join(cfg["paths"]["features_dir"], "pairwise_features_weighted.tsv"))
    folds_df = load_table(os.path.join(cfg["paths"]["oof_dir"], "folds.tsv"))
    labels_df = weighted_df[["source1_entity_id", "candidate_source", "candidate_record_id", "label"]].copy()

    oof_df = run_oof(weighted_df, folds_df, model_name=cfg["pair_model"]["primary"],
                      sample_weight_col="sample_weight")
    save_table(oof_df, os.path.join(cfg["paths"]["oof_dir"], "pair_oof_predictions_final.tsv"))

    raw_scores = oof_df["oof_score"].values
    labels = oof_df["label"].values

    best_row_raw, sweep_raw = _best_f05(raw_scores, oof_df, labels_df, cfg)
    logger.info("Raw (uncalibrated) OOF: best threshold=%.2f macro_f05=%.4f",
                best_row_raw["threshold"], best_row_raw["macro_f05"])

    candidates = {"none": (raw_scores, best_row_raw["macro_f05"], best_row_raw["threshold"])}

    for method in ("platt", "isotonic"):
        calibrator = get_calibrator(method)
        # Fit calibration on OOF scores themselves — acceptable here because
        # each fold's OOF score already came from a model that never saw
        # that row; fitting the calibration curve on the full OOF set is
        # standard practice (a stricter variant would nest this in its own
        # CV, but that's overkill relative to this project's remaining risk).
        calibrator.fit(raw_scores, labels)
        calibrated_scores = calibrator.transform(raw_scores)
        best_row_cal, _ = _best_f05(calibrated_scores, oof_df, labels_df, cfg)
        logger.info("%s calibration: best threshold=%.2f macro_f05=%.4f",
                    method, best_row_cal["threshold"], best_row_cal["macro_f05"])
        candidates[method] = (calibrated_scores, best_row_cal["macro_f05"], best_row_cal["threshold"])
        if best_row_cal["macro_f05"] > candidates["none"][1]:
            joblib.dump(calibrator, os.path.join(cfg["paths"]["models_dir"], f"calibrator_{method}.joblib"))

    best_method = max(candidates, key=lambda m: candidates[m][1])
    best_scores, best_f05, best_threshold = candidates[best_method]

    logger.info("Selected calibration method: '%s' (macro_f05=%.4f, threshold=%.2f) — "
                "kept only because it beat 'none'; if 'none' wins, no calibrator is saved/used.",
                best_method, best_f05, best_threshold)

    frozen = {
        "calibration_method": best_method,
        "threshold": float(best_threshold),
        "oof_macro_f05_at_selection": float(best_f05),
    }
    with open(os.path.join(cfg["paths"]["models_dir"], "frozen_decision_config.json"), "w", encoding="utf-8") as f:
        json.dump(frozen, f, indent=2)
    logger.info("Froze decision config -> %s : %s",
                os.path.join(cfg["paths"]["models_dir"], "frozen_decision_config.json"), frozen)


if __name__ == "__main__":
    main()
