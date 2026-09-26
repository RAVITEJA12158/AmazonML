"""Splits the ADAPTED student train data (artifacts/student_data/adapted/,
written by scripts/13a_adapt_student_data.py) into two partitions, grouped
by source1_entity_id so no entity's candidates leak across the split:

    artifacts/splits/student_holdout/dev/{source1.tsv,labels.tsv}
    artifacts/splits/student_holdout/heldout_test/{source1.tsv,labels.tsv}

`dev` is what you point config.yaml `paths.raw_dir` at for all normal
development (scripts 03-11, GroupKFold OOF training, threshold tuning).
`heldout_test` is scored exactly once at the end, by scripts/14, for an
honest final number — never used for any tuning decision.

source2.tsv / source3.tsv are NOT split. Both partitions match against the
exact same candidate pool (that's what `source2`/`source3` actually are —
the universe of candidate businesses, not per-entity data) so holding out
S1 queries doesn't remove any of their potential matches from view. They
stay in one place (artifacts/student_data/adapted/) and every script reads
them from there via `student_data.shared_candidates_dir` in config.yaml —
see the loader.py change. This also avoids duplicating what may be a very
large file.

The real, unlabeled hackathon test set (dataset_test_dir) is never read by
this script.

Run: python scripts/13_split_student_train.py
"""
import csv
import os
import sys
import json
import random

sys.path.insert(0, os.getcwd())
from src.utils.io import load_config, ensure_dir
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def main():
    cfg = load_config("config.yaml")
    sd_cfg = cfg["student_data"]
    adapted_dir = sd_cfg.get("adapted_dir", "artifacts/student_data/adapted")
    split_dir = sd_cfg.get("split_dir", "artifacts/splits/student_holdout")
    fraction_dev = sd_cfg.get("dev_fraction", 0.5)
    seed = cfg.get("seed", 42)

    s1_path = os.path.join(adapted_dir, "source1.tsv")
    labels_path = os.path.join(adapted_dir, "labels.tsv")
    if not os.path.exists(s1_path) or not os.path.exists(labels_path):
        raise FileNotFoundError(
            f"Adapted files not found under {adapted_dir}. Run "
            f"scripts/13a_adapt_student_data.py first."
        )

    # Pass 1: stream source1 once to get every entity id, assign to a split.
    with open(s1_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        all_ids = [row["source1_entity_id"] for row in reader]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("source1_entity_id must be unique in the adapted source1.tsv")

    rng = random.Random(seed)
    shuffled = all_ids[:]
    rng.shuffle(shuffled)
    n_dev = int(round(len(shuffled) * fraction_dev))
    dev_ids = set(shuffled[:n_dev])
    # everything not in dev_ids is heldout_test

    dev_dir = os.path.join(split_dir, "dev")
    test_dir = os.path.join(split_dir, "heldout_test")
    ensure_dir(dev_dir)
    ensure_dir(test_dir)

    # Pass 2: stream source1.tsv -> dev/source1.tsv + heldout_test/source1.tsv
    counts = {"dev": 0, "heldout_test": 0}
    with open(s1_path, "r", encoding="utf-8", newline="") as fin, \
         open(os.path.join(dev_dir, "source1.tsv"), "w", encoding="utf-8", newline="") as f_dev, \
         open(os.path.join(test_dir, "source1.tsv"), "w", encoding="utf-8", newline="") as f_test:
        reader = csv.DictReader(fin, delimiter="\t")
        w_dev = csv.DictWriter(f_dev, fieldnames=reader.fieldnames, delimiter="\t", lineterminator="\n")
        w_test = csv.DictWriter(f_test, fieldnames=reader.fieldnames, delimiter="\t", lineterminator="\n")
        w_dev.writeheader()
        w_test.writeheader()
        for row in reader:
            if row["source1_entity_id"] in dev_ids:
                w_dev.writerow(row)
                counts["dev"] += 1
            else:
                w_test.writerow(row)
                counts["heldout_test"] += 1

    # Pass 3: stream labels.tsv -> split the same way, by source1_entity_id
    label_counts = {"dev": 0, "heldout_test": 0}
    with open(labels_path, "r", encoding="utf-8", newline="") as fin, \
         open(os.path.join(dev_dir, "labels.tsv"), "w", encoding="utf-8", newline="") as f_dev, \
         open(os.path.join(test_dir, "labels.tsv"), "w", encoding="utf-8", newline="") as f_test:
        reader = csv.DictReader(fin, delimiter="\t")
        w_dev = csv.DictWriter(f_dev, fieldnames=reader.fieldnames, delimiter="\t", lineterminator="\n")
        w_test = csv.DictWriter(f_test, fieldnames=reader.fieldnames, delimiter="\t", lineterminator="\n")
        w_dev.writeheader()
        w_test.writeheader()
        for row in reader:
            if row["source1_entity_id"] in dev_ids:
                w_dev.writerow(row)
                label_counts["dev"] += 1
            else:
                w_test.writerow(row)
                label_counts["heldout_test"] += 1

    # Independent verification pass: re-read what was written and confirm
    # zero entity-id leakage across the split (never trust the writer loop
    # that just ran — re-check from disk).
    with open(os.path.join(dev_dir, "source1.tsv"), encoding="utf-8") as f:
        dev_written_ids = {r["source1_entity_id"] for r in csv.DictReader(f, delimiter="\t")}
    with open(os.path.join(test_dir, "source1.tsv"), encoding="utf-8") as f:
        test_written_ids = {r["source1_entity_id"] for r in csv.DictReader(f, delimiter="\t")}
    overlap = dev_written_ids & test_written_ids
    if overlap:
        raise RuntimeError(f"Split leakage detected: {len(overlap)} entity ids appear in both partitions")

    report = {
        "seed": seed,
        "dev_fraction": fraction_dev,
        "adapted_dir": adapted_dir,
        "split_dir": split_dir,
        "total_source1_entities": len(all_ids),
        "dev_entities": counts["dev"],
        "heldout_test_entities": counts["heldout_test"],
        "dev_label_rows": label_counts["dev"],
        "heldout_test_label_rows": label_counts["heldout_test"],
        "entity_id_overlap": len(overlap),
        "note": "source2.tsv/source3.tsv are shared (not split) — read directly from "
                f"{adapted_dir} via student_data.shared_candidates_dir in config.yaml.",
    }
    with open(os.path.join(split_dir, "split_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Split complete: %d dev entities / %d heldout_test entities (overlap=%d)",
                counts["dev"], counts["heldout_test"], len(overlap))
    logger.info("Point config.yaml paths.raw_dir at: %s", dev_dir)
    logger.info("heldout_test at %s is scored ONCE, at the end, by scripts/14_evaluate_heldout.py", test_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
