"""Runs the exact frozen pipeline (no test-time parameter tuning) end to
end: normalize -> blocking funnel -> prune -> features -> pair model ->
calibration -> entity decision -> matching_results.tsv.

By default this runs against data/raw/ (the same data used for
development) as a demonstration of the full frozen pipeline. Point
`--raw-dir` at a real held-out test directory (with the same
source1/source2/source3 file naming) to run genuine test inference.

Run: python scripts/11_inference.py [--raw-dir data/raw]
"""
import os
import sys
import json
import argparse
import joblib

sys.path.insert(0, os.getcwd())
from src.utils.seed import set_seed
from src.utils.io import load_config, save_table
from src.utils.logging_utils import get_logger
from src.data.loader import load_source1, load_source23
from src.blocking.cheap_ranker import CheapRanker
from src.blocking.semantic import SemanticRetriever
from src.pipeline.inference import (
    build_candidate_pairs, build_pairwise_features, score_with_pair_model,
    apply_entity_decision, write_matching_results,
)

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=None, help="Defaults to config.yaml paths.raw_dir")
    args = parser.parse_args()

    set_seed(42)
    cfg = load_config("config.yaml")
    raw_dir = args.raw_dir or cfg["paths"]["raw_dir"]
    logger.info("Running frozen inference pipeline against raw_dir=%s", raw_dir)

    s1_df = load_source1(raw_dir)
    s23_df = load_source23(raw_dir, cfg.get("student_data", {}).get("shared_candidates_dir"))

    cheap_ranker: CheapRanker = joblib.load(os.path.join(cfg["paths"]["models_dir"], "cheap_ranker.joblib"))
    pair_model = joblib.load(os.path.join(cfg["paths"]["models_dir"], "pair_model.joblib"))

    with open(os.path.join(cfg["paths"]["models_dir"], "frozen_decision_config.json"), encoding="utf-8") as f:
        frozen = json.load(f)

    calibrator = None
    if frozen["calibration_method"] != "none":
        cal_path = os.path.join(cfg["paths"]["models_dir"], f"calibrator_{frozen['calibration_method']}.joblib")
        if os.path.exists(cal_path):
            calibrator = joblib.load(cal_path)

    entity_model = None
    entity_cfg = dict(cfg["entity_model"])
    if frozen.get("entity_decision_method") == "learned":
        em_path = os.path.join(cfg["paths"]["models_dir"], "entity_model.joblib")
        if os.path.exists(em_path):
            entity_model = joblib.load(em_path)
            entity_cfg["method"] = "lightgbm"
    decision_threshold = frozen.get("entity_decision_threshold", frozen["threshold"])

    # --- Stage A: candidate generation (this IS part of the scored submission) ---
    semantic_retriever = SemanticRetriever(backend_name=cfg["semantic_retrieval"]["embedding_backend"])
    candidates_df, s1_norm, s23_norm = build_candidate_pairs(s1_df, s23_df, cfg, cheap_ranker, semantic_retriever)
    save_table(candidates_df, os.path.join(cfg["paths"]["output_dir"], "candidate_pairs.tsv"))
    logger.info("Wrote %d candidate rows -> output/candidate_pairs.tsv", len(candidates_df))

    if candidates_df.empty:
        logger.warning("No candidates generated at all — writing an all-singleton matching_results.tsv.")
        marker_only = write_matching_results(
            candidates_df.assign(accepted=0) if not candidates_df.empty else
            __import__("pandas").DataFrame(columns=["source1_entity_id", "accepted"]),
            s1_df["source1_entity_id"],
        )
        save_table(marker_only, os.path.join(cfg["paths"]["output_dir"], "matching_results.tsv"))
        return

    # --- Stage B: features -> pair model -> calibration -> entity decision ---
    feature_df = build_pairwise_features(candidates_df, s1_norm, s23_norm)
    scored_df = score_with_pair_model(feature_df, pair_model, calibrator)
    decided_df = apply_entity_decision(scored_df, decision_threshold, entity_cfg, entity_model)

    results_df = write_matching_results(decided_df, s1_df["source1_entity_id"])
    save_table(results_df, os.path.join(cfg["paths"]["output_dir"], "matching_results.tsv"))

    n_matched_entities = results_df[results_df["candidate_source"] != "NONE"]["source1_entity_id"].nunique()
    n_singleton = (results_df["candidate_source"] == "NONE").sum()
    logger.info("Wrote matching_results.tsv: %d entities with >=1 accepted match, %d singleton predictions, "
                "%d total rows -> %s", n_matched_entities, n_singleton, len(results_df),
                os.path.join(cfg["paths"]["output_dir"], "matching_results.tsv"))


if __name__ == "__main__":
    main()
