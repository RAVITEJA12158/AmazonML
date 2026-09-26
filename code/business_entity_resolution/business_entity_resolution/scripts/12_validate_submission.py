"""Validates output/matching_results.tsv before submission: every S1 entity
represented, no invalid candidate references, no duplicates, valid source
labels. Exits non-zero on any failure, per a real CI-style validator.

Run: python scripts/12_validate_submission.py [--raw-dir data/raw]
"""
import os
import sys
import argparse

sys.path.insert(0, os.getcwd())
from src.utils.io import load_config
from src.utils.logging_utils import get_logger
from src.data.loader import load_source1
from src.pipeline.submit import validate_submission

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=None)
    args = parser.parse_args()

    cfg = load_config("config.yaml")
    raw_dir = args.raw_dir or cfg["paths"]["raw_dir"]

    s1_df = load_source1(raw_dir)
    s1_ids = set(s1_df["source1_entity_id"].unique())

    results_path = os.path.join(cfg["paths"]["output_dir"], "matching_results.tsv")
    candidates_path = os.path.join(cfg["paths"]["output_dir"], "candidate_pairs.tsv")

    if not os.path.exists(results_path) or not os.path.exists(candidates_path):
        logger.error("Missing output files. Expected both %s and %s — run scripts/11_inference.py first.",
                      results_path, candidates_path)
        sys.exit(1)

    errors = validate_submission(results_path, candidates_path, s1_ids)

    if errors:
        logger.error("Submission validation FAILED with %d issue(s):", len(errors))
        for e in errors:
            logger.error("  - %s", e)
        sys.exit(1)
    else:
        logger.info("Submission validation PASSED: %s and %s look correct.", results_path, candidates_path)


if __name__ == "__main__":
    main()
