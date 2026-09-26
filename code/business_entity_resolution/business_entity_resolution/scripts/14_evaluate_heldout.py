"""Scores the FROZEN, already-trained pipeline against
artifacts/splits/student_holdout/heldout_test/ — the half of the real
student train data that scripts 03-12 never saw during blocking-ranker
fitting, feature building, pair-model training, hard-negative mining,
calibration, or entity-model training.

This script trains NOTHING. It only loads the models already saved under
artifacts/models/ and runs inference against heldout_test, then compares
matching_results against heldout_test/labels.tsv for the macro F0.5 that
actually reflects generalization.

Run this exactly once, after you're done tuning against dev/. If you find
yourself running it more than once and changing things in response, you're
tuning against your test set — go back to dev/ (GroupKFold OOF) for that.

Run: python scripts/14_evaluate_heldout.py
"""
import os
import sys
import json
import subprocess

sys.path.insert(0, os.getcwd())
from src.utils.io import load_config, load_table, ensure_dir
from src.utils.logging_utils import get_logger
from src.evaluation.metrics import macro_f05

logger = get_logger(__name__)


def main():
    cfg = load_config("config.yaml")
    sd_cfg = cfg["student_data"]
    heldout_dir = os.path.join(sd_cfg.get("split_dir", "artifacts/splits/student_holdout"), "heldout_test")

    if not os.path.exists(os.path.join(heldout_dir, "source1.tsv")):
        raise FileNotFoundError(
            f"{heldout_dir}/source1.tsv not found. Run scripts/13a_adapt_student_data.py "
            f"and scripts/13_split_student_train.py first."
        )

    models_dir = cfg["paths"]["models_dir"]
    for required in ("cheap_ranker.joblib", "pair_model.joblib", "frozen_decision_config.json"):
        if not os.path.exists(os.path.join(models_dir, required)):
            raise FileNotFoundError(
                f"{required} not found under {models_dir}. Run scripts/run_all.py "
                f"(against dev/) first — this script only evaluates an already-trained pipeline."
            )

    heldout_output_dir = "output_heldout_test"
    ensure_dir(heldout_output_dir)

    logger.info("Running frozen inference against the held-out test split: %s", heldout_dir)
    result = subprocess.run([sys.executable, "scripts/11_inference.py", "--raw-dir", heldout_dir])
    if result.returncode != 0:
        raise RuntimeError("scripts/11_inference.py failed against heldout_test")
    # 11_inference.py writes to cfg["paths"]["output_dir"] (normally
    # "output/"); copy the result out immediately so it isn't confused with
    # (or overwritten by) a dev-side inference run.
    default_output_dir = cfg["paths"]["output_dir"]
    src_path = os.path.join(default_output_dir, "matching_results.tsv")
    if not os.path.exists(src_path):
        raise RuntimeError("scripts/11_inference.py did not produce matching_results.tsv")
    results_df = load_table(src_path)
    results_df.to_csv(os.path.join(heldout_output_dir, "matching_results.tsv"), sep="\t", index=False)

    labels_df = load_table(os.path.join(heldout_dir, "labels.tsv"))
    predicted = results_df[results_df["candidate_source"] != "NONE"][
        ["source1_entity_id", "candidate_record_id"]
    ]

    metrics = macro_f05(predicted, labels_df)
    logger.info("HELD-OUT TEST macro F0.5 = %.4f (precision=%.4f recall=%.4f, n_entities=%d)",
                metrics["macro_f05"], metrics["precision"], metrics["recall"], metrics["n_entities"])

    with open(os.path.join(heldout_output_dir, "heldout_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"\nThis is the honest number — dev/ OOF metrics were used for all tuning, "
          f"this heldout_test split was touched for the first time just now.")


if __name__ == "__main__":
    main()
