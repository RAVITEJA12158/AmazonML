"""Stage B input: builds the rich pairwise feature table for every row in
candidate_pairs.tsv, joins in ground-truth labels where available (label=0
for candidates with no matching labeled row — i.e. blocking-generated
negatives).

Run: python scripts/05_build_features.py
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.getcwd())
from src.utils.io import load_config, load_table, save_table
from src.utils.logging_utils import get_logger
from src.data.loader import load_source1, load_source23, load_labels
from src.pipeline.inference import build_pairwise_features
from src.normalization.build import normalize_records

logger = get_logger(__name__)


def main():
    cfg = load_config("config.yaml")
    raw_dir = cfg["paths"]["raw_dir"]

    candidates_df = load_table(os.path.join(cfg["paths"]["candidates_dir"], "candidate_pairs.tsv"))
    s1_df = load_source1(raw_dir)
    s23_df = load_source23(raw_dir, cfg.get("student_data", {}).get("shared_candidates_dir"))
    labels_df = load_labels(raw_dir)

    s1_norm = normalize_records(s1_df, id_col="source1_entity_id").rename(columns={"source1_entity_id": "record_id"})
    s23_norm = normalize_records(s23_df, id_col="record_id")

    feature_df = build_pairwise_features(candidates_df, s1_norm, s23_norm)

    label_map = labels_df.set_index(["source1_entity_id", "candidate_record_id"])["label"].to_dict()
    keys = list(zip(feature_df["source1_entity_id"], feature_df["candidate_record_id"]))
    feature_df["label"] = [label_map.get(k, 0) for k in keys]

    save_table(feature_df, os.path.join(cfg["paths"]["features_dir"], "pairwise_features.tsv"))
    logger.info("Built %d feature rows (%d positive) -> %s", len(feature_df), int(feature_df["label"].sum()),
                os.path.join(cfg["paths"]["features_dir"], "pairwise_features.tsv"))


if __name__ == "__main__":
    main()
