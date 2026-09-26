"""Builds entity-context features on top of the (calibrated) OOF pair
scores, then compares the hand-rule baseline against a learned entity-level
model on macro F0.5 — keeps whichever wins. Never forces top-1 per entity;
both paths score every candidate independently.

IMPORTANT: the learned entity model is evaluated under its OWN nested
out-of-fold loop, reusing the same GroupKFold folds (by source1_entity_id)
as the pair model. Earlier this script fit EntityLightGBMModel on the full
entity_df and then scored that same entity_df — in-sample, so its reported
macro F0.5 was optimistic (it hit 1.0 on the last run, which is not a real
number). Now every entity gets scored only by a fold that never trained on
it, exactly like the pair model's own OOF discipline in pipeline/oof.py.
The model saved to entity_model.joblib for actual inference is still a
single final fit on ALL development entities (standard practice — only the
*reported* metric needs to be OOF, not the deployed model).

Run: python scripts/10_train_entity_model.py
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
from src.features.build_features import add_candidate_relative_features
from src.models.entity_model import apply_hand_rule, EntityLightGBMModel, ENTITY_FEATURE_COLUMNS
from src.evaluation.metrics import macro_f05
from src.evaluation.threshold import sweep_threshold

logger = get_logger(__name__)


def _entity_model_oof(entity_df: pd.DataFrame, folds_df: pd.DataFrame) -> np.ndarray:
    """Nested OOF for the entity-level model: for each fold, fit on every
    OTHER fold's entities and score only the held-out fold's rows. Mirrors
    src/pipeline/oof.py's run_oof, but one layer up (entity-context features
    instead of raw pair features)."""
    df = entity_df.merge(folds_df, on="source1_entity_id", how="left", suffixes=("", "_grp"))
    fold_col = "fold_grp" if "fold_grp" in df.columns else "fold"
    scores = np.full(len(df), np.nan)

    cols = [c for c in ENTITY_FEATURE_COLUMNS if c in df.columns]
    for c in ENTITY_FEATURE_COLUMNS:
        if c not in df.columns:
            df[c] = 0.0
    cols = ENTITY_FEATURE_COLUMNS

    for fold in sorted(df[fold_col].dropna().unique()):
        train_mask = df[fold_col] != fold
        valid_mask = df[fold_col] == fold
        fold_model = EntityLightGBMModel()
        fold_model.fit(df.loc[train_mask], df.loc[train_mask, "label"].values)
        scores[valid_mask.values] = fold_model.predict_proba(df.loc[valid_mask].copy())
        logger.info("  entity-model fold %s: trained on %d entities' candidates, scored %d OOF rows",
                    fold, train_mask.sum(), valid_mask.sum())

    return scores


def _prep_entity_df(oof_df: pd.DataFrame, calibrator, threshold: float) -> pd.DataFrame:
    df = oof_df.copy()
    if calibrator is not None:
        df["pair_score"] = calibrator.transform(df["oof_score"].values)
    else:
        df["pair_score"] = df["oof_score"]
    df = add_candidate_relative_features(df, score_col="pair_score")
    return df


def main():
    set_seed(42)
    cfg = load_config("config.yaml")

    oof_df = load_table(os.path.join(cfg["paths"]["oof_dir"], "pair_oof_predictions_final.tsv"))
    labels_df = oof_df[["source1_entity_id", "candidate_source", "candidate_record_id", "label"]].copy()
    folds_df = load_table(os.path.join(cfg["paths"]["oof_dir"], "folds.tsv"))[["source1_entity_id", "fold"]].drop_duplicates()

    with open(os.path.join(cfg["paths"]["models_dir"], "frozen_decision_config.json"), encoding="utf-8") as f:
        frozen = json.load(f)

    calibrator = None
    if frozen["calibration_method"] != "none":
        cal_path = os.path.join(cfg["paths"]["models_dir"], f"calibrator_{frozen['calibration_method']}.joblib")
        if os.path.exists(cal_path):
            calibrator = joblib.load(cal_path)

    entity_df = _prep_entity_df(oof_df, calibrator, frozen["threshold"])

    # --- Baseline: hand-rule ---
    entity_cfg = cfg["entity_model"]
    hand_rule_df = apply_hand_rule(entity_df, frozen["threshold"], entity_cfg)
    accepted_hr = hand_rule_df[hand_rule_df["accepted"] == 1][["source1_entity_id", "candidate_record_id"]]
    metrics_hr = macro_f05(accepted_hr, labels_df)
    logger.info("Hand-rule baseline: macro_f05=%.4f precision=%.4f recall=%.4f singleton_acc=%s",
                metrics_hr["macro_f05"], metrics_hr["precision"], metrics_hr["recall"], metrics_hr["singleton_accuracy"])

    # --- Candidate: learned entity model (only meaningful with enough data) ---
    n_pos = int(entity_df["label"].sum())
    n_folds_available = folds_df["fold"].nunique()
    metrics_learned = None
    model = None
    if entity_cfg["enabled"] and n_pos >= 10 and n_folds_available >= 2:
        for c in ENTITY_FEATURE_COLUMNS:
            if c not in entity_df.columns:
                entity_df[c] = 0.0

        logger.info("Running nested OOF for the learned entity model (%d folds)...", n_folds_available)
        entity_df["entity_score"] = _entity_model_oof(entity_df, folds_df)

        scored = pd.DataFrame({
            "source1_entity_id": entity_df["source1_entity_id"].values,
            "candidate_record_id": entity_df["candidate_record_id"].values,
            "oof_score": entity_df["entity_score"].values,
        })
        sweep = sweep_threshold(scored, labels_df, cfg["threshold"]["sweep_min"],
                                 cfg["threshold"]["sweep_max"], cfg["threshold"]["sweep_step"])
        best = sweep.iloc[0]
        metrics_learned = best.to_dict()
        logger.info("Learned entity model (genuine nested OOF): best threshold=%.2f macro_f05=%.4f "
                    "precision=%.4f recall=%.4f", best["threshold"], best["macro_f05"],
                    best["precision"], best["recall"])

        # Final deployed model: one fit on ALL development entities. This is
        # standard (the reported metric above, not this model, is what had
        # to be OOF) — it's saved only if the learned approach wins below.
        model = EntityLightGBMModel()
        model.fit(entity_df, entity_df["label"].values)
    else:
        reason = f"only {n_pos} positive examples" if n_pos < 10 else f"only {n_folds_available} fold(s) available"
        logger.info("Skipping learned entity model — %s (need >= 10 positives and >= 2 folds "
                     "for a meaningful nested-OOF fit). Hand-rule baseline will be used.", reason)

    # --- Decide winner ---
    if metrics_learned is not None and metrics_learned["macro_f05"] > metrics_hr["macro_f05"]:
        chosen = "learned"
        chosen_threshold = metrics_learned["threshold"]
        joblib.dump(model, os.path.join(cfg["paths"]["models_dir"], "entity_model.joblib"))
        logger.info("Learned entity model WINS (%.4f > %.4f) — saved.",
                     metrics_learned["macro_f05"], metrics_hr["macro_f05"])
    else:
        chosen = "hand_rule"
        chosen_threshold = frozen["threshold"]
        logger.info("Hand-rule baseline WINS (or learned model unavailable) — keeping it as the "
                     "simplest option that isn't beaten by the added complexity.")

    frozen["entity_decision_method"] = chosen
    frozen["entity_decision_threshold"] = chosen_threshold
    frozen["hand_rule_macro_f05"] = metrics_hr["macro_f05"]
    frozen["learned_macro_f05"] = metrics_learned["macro_f05"] if metrics_learned else None
    with open(os.path.join(cfg["paths"]["models_dir"], "frozen_decision_config.json"), "w", encoding="utf-8") as f:
        json.dump(frozen, f, indent=2)
    logger.info("Updated frozen decision config: %s", frozen)


if __name__ == "__main__":
    main()
