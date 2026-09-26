"""The ONLY script that opens `student_data.dataset_test_dir` — the real,
unlabeled hackathon test set. Adapts it (same column-name handling as
scripts/13a) into a temporary canonical directory, then runs the frozen,
already-trained pipeline against it to produce a submission file.

No ground truth exists for this data, so no metric is computed here — you
only get output/matching_results.tsv (renamed to submission.tsv), which is
what you actually submit.

Run this once, at the very end, after scripts/14_evaluate_heldout.py has
given you a held-out number you're satisfied with.

Run: python scripts/15_score_real_test.py
"""
import os
import sys
import subprocess

sys.path.insert(0, os.getcwd())
from src.utils.io import load_config, load_table, ensure_dir
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
import importlib.util
spec = importlib.util.spec_from_file_location("adapt_module", "scripts/13a_adapt_student_data.py")
adapt_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapt_module)


def main():
    cfg = load_config("config.yaml")
    sd_cfg = cfg["student_data"]
    dataset_test_dir = sd_cfg["dataset_test_dir"]
    if not os.path.isdir(dataset_test_dir):
        raise FileNotFoundError(
            f"student_data.dataset_test_dir does not exist on this machine: {dataset_test_dir}. "
            f"Edit config.yaml before running this script."
        )

    adapted_test_dir = "artifacts/student_data/adapted_test"
    ensure_dir(adapted_test_dir)
    logger.info("Adapting the real TEST set (read-only, no labels expected): %s", dataset_test_dir)

    s1_path = adapt_module._find_file(dataset_test_dir, "source1")
    s2_path = adapt_module._find_file(dataset_test_dir, "source2")
    s3_path = adapt_module._find_file(dataset_test_dir, "source3")
    adapt_module._adapt_source_file(s1_path, os.path.join(adapted_test_dir, "source1.tsv"), source1=True)
    adapt_module._adapt_source_file(s2_path, os.path.join(adapted_test_dir, "source2.tsv"), source1=False)
    adapt_module._adapt_source_file(s3_path, os.path.join(adapted_test_dir, "source3.tsv"), source1=False)
    logger.info("Adapted real test set -> %s", adapted_test_dir)

    logger.info("Running frozen inference pipeline against the real test set...")
    result = subprocess.run([sys.executable, "scripts/11_inference.py", "--raw-dir", adapted_test_dir])
    if result.returncode != 0:
        raise RuntimeError("scripts/11_inference.py failed against the real test set")

    default_output_dir = cfg["paths"]["output_dir"]
    src_path = os.path.join(default_output_dir, "matching_results.tsv")
    results_df = load_table(src_path)
    submission_path = os.path.join(default_output_dir, "submission.tsv")
    results_df.to_csv(submission_path, sep="\t", index=False)
    logger.info("Wrote submission file -> %s (%d rows)", submission_path, len(results_df))


if __name__ == "__main__":
    main()
