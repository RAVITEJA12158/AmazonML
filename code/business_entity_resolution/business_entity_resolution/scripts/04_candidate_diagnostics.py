"""Blocking diagnostics — recall ceiling AND candidate-set size, both
reported, per the candidate-size addendum.

Run: python scripts/04_candidate_diagnostics.py
"""
import os
import sys
import json
import pandas as pd

sys.path.insert(0, os.getcwd())
from src.utils.io import load_config, load_table
from src.utils.logging_utils import get_logger
from src.data.loader import load_source1, load_source23, load_labels
from src.evaluation.blocking_metrics import blocking_recall, candidate_size_stats, reduction_ratio, channel_contribution

logger = get_logger(__name__)


def main():
    cfg = load_config("config.yaml")
    raw_dir = cfg["paths"]["raw_dir"]

    candidates_df = load_table(os.path.join(cfg["paths"]["candidates_dir"], "candidate_pairs.tsv"))
    s1_df = load_source1(raw_dir)
    s23_df = load_source23(raw_dir, cfg.get("student_data", {}).get("shared_candidates_dir"))
    labels_df = load_labels(raw_dir)

    recall_stats = blocking_recall(candidates_df, labels_df)
    size_stats = candidate_size_stats(candidates_df, s1_df["source1_entity_id"])
    rr = reduction_ratio(candidates_df, len(s1_df), len(s23_df))
    channel_stats = channel_contribution(candidates_df, labels_df)

    report = {**recall_stats, **size_stats, "reduction_ratio": rr}

    logger.info("=== Blocking Diagnostics ===")
    for k, v in report.items():
        logger.info("%s: %s", k, v)
    logger.info("--- Channel contribution ---\n%s", channel_stats.to_string(index=False))

    out_path = os.path.join(cfg["paths"]["candidates_dir"], "diagnostics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    channel_stats.to_csv(os.path.join(cfg["paths"]["candidates_dir"], "channel_contribution.csv"), index=False)
    logger.info("Saved diagnostics -> %s", out_path)


if __name__ == "__main__":
    main()
