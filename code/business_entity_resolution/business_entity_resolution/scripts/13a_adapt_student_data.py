"""Adapts the REAL student_resource dataset into this pipeline's canonical
schema (src/data/schema.py). This is the only place that needs to know the
real column names — everything downstream (loader.py, normalization,
blocking, features, models) only ever sees the canonical names.

Reads ONLY from `student_data.dataset_train_dir` in config.yaml. The sibling
`dataset_test_dir` (the real, unlabeled hackathon test set) is never opened
by this script or by anything it calls — that directory is read exactly
once, by scripts/15_score_real_test.py, at the very end, for scoring only.

Real files expected in dataset_train_dir (tab-separated):
    train_source1.tsv      columns: entity_id, business_name, business_address, country
    train_source2.tsv      columns: entity_id, business_name, business_address, country
    train_source3.tsv      columns: entity_id, business_name, business_address, country
    train_ground_truth.tsv columns: source1_entity_id, matched_entity_ids
                            (matched_entity_ids is a comma-separated list of
                            entity_id values, each belonging to EITHER
                            source2 or source3 — this script resolves which,
                            by checking real ID membership, not by guessing
                            an ID-prefix convention.)

If your real files use slightly different column or file names, edit
COLUMN_ALIASES / FILE_ALIASES below rather than touching any other script.

Streams every file row-by-row (csv, not pandas.read_csv) because these
files are reported to be multi-GB — the whole point of this script is to
run once and hand everything downstream small, canonical, memory-friendly
files.

Run: python scripts/13a_adapt_student_data.py
"""
import csv
import os
import sys
import json

sys.path.insert(0, os.getcwd())
from src.utils.io import load_config, ensure_dir
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Edit these if your real filenames/column names differ slightly.
FILE_ALIASES = {
    "source1": ["train_source1.tsv", "test_source1.tsv", "source1.tsv"],
    "source2": ["train_source2.tsv", "test_source2.tsv", "source2.tsv"],
    "source3": ["train_source3.tsv", "test_source3.tsv", "source3.tsv"],
    "ground_truth": ["train_ground_truth.tsv", "ground_truth.tsv", "train_labels.tsv"],
}
COLUMN_ALIASES = {
    "entity_id": ["entity_id", "record_id", "id"],
    "business_name": ["business_name", "name"],
    "business_address": ["business_address", "address"],
    "country": ["country"],
}


def _find_file(dataset_dir: str, tag: str) -> str:
    for fname in FILE_ALIASES[tag]:
        candidate = os.path.join(dataset_dir, fname)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not find a '{tag}' file under {dataset_dir}. "
        f"Tried: {FILE_ALIASES[tag]}. If your real file is named "
        f"differently, add it to FILE_ALIASES in this script."
    )


def _resolve_column(fieldnames: list, canonical: str) -> str:
    for alias in COLUMN_ALIASES[canonical]:
        if alias in fieldnames:
            return alias
    raise KeyError(
        f"Could not find a column for '{canonical}' among {fieldnames}. "
        f"Add the real column name to COLUMN_ALIASES['{canonical}'] in this script."
    )


def _adapt_source_file(src_path: str, out_path: str, source1: bool) -> tuple[int, set]:
    """Streams one real source file -> canonical source1.tsv/source23-style
    file. Returns (row_count, set_of_ids) so ground-truth resolution below
    doesn't need a second pass over the raw file."""
    ids = set()
    ensure_dir(os.path.dirname(out_path))
    with open(src_path, "r", encoding="utf-8-sig", newline="") as fin, \
         open(out_path, "w", encoding="utf-8", newline="") as fout:
        reader = csv.DictReader(fin, delimiter="\t")
        id_col = _resolve_column(reader.fieldnames, "entity_id")
        name_col = _resolve_column(reader.fieldnames, "business_name")
        addr_col = _resolve_column(reader.fieldnames, "business_address")
        country_col = _resolve_column(reader.fieldnames, "country")

        if source1:
            out_fields = ["source1_entity_id", "name", "address", "country",
                          "state", "city", "postal_code", "street_number", "street_name",
                          "phone", "email"]
        else:
            out_fields = ["record_id", "name", "address", "country",
                          "state", "city", "postal_code", "street_number", "street_name",
                          "phone", "email"]
        writer = csv.DictWriter(fout, fieldnames=out_fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()

        n = 0
        for row in reader:
            rid = row[id_col]
            ids.add(rid)
            out_row = {
                ("source1_entity_id" if source1 else "record_id"): rid,
                "name": row[name_col],
                "address": row[addr_col],
                "country": row[country_col],
                "state": "", "city": "", "postal_code": "",
                "street_number": "", "street_name": "", "phone": "", "email": "",
            }
            writer.writerow(out_row)
            n += 1
    return n, ids


def _adapt_ground_truth(gt_path: str, out_path: str, source2_ids: set, source3_ids: set) -> dict:
    """Expands `matched_entity_ids` (comma list) into canonical
    (source1_entity_id, candidate_source, candidate_record_id, label=1) rows.
    Resolves candidate_source by real ID-set membership — never by
    guessing a naming prefix, since we don't actually know the real ID
    format ahead of time."""
    ensure_dir(os.path.dirname(out_path))
    n_rows, n_positive_pairs, n_unresolved = 0, 0, 0
    unresolved_examples = []

    with open(gt_path, "r", encoding="utf-8-sig", newline="") as fin, \
         open(out_path, "w", encoding="utf-8", newline="") as fout:
        reader = csv.DictReader(fin, delimiter="\t")
        s1_col = _resolve_column(reader.fieldnames, "entity_id") if "entity_id" in reader.fieldnames \
            else "source1_entity_id"
        matched_col = "matched_entity_ids" if "matched_entity_ids" in reader.fieldnames else None
        if matched_col is None:
            raise KeyError(
                f"Ground-truth file has columns {reader.fieldnames} — expected a "
                f"'matched_entity_ids' column. Edit this script if your real column "
                f"is named differently."
            )
        writer = csv.DictWriter(
            fout, fieldnames=["source1_entity_id", "candidate_source", "candidate_record_id", "label"],
            delimiter="\t", lineterminator="\n")
        writer.writeheader()

        for row in reader:
            n_rows += 1
            s1_id = row[s1_col]
            matched = row.get(matched_col, "")
            if not matched:
                continue
            for target_id in matched.split(","):
                target_id = target_id.strip()
                if not target_id:
                    continue
                if target_id in source2_ids:
                    src = "S2"
                elif target_id in source3_ids:
                    src = "S3"
                else:
                    n_unresolved += 1
                    if len(unresolved_examples) < 5:
                        unresolved_examples.append(target_id)
                    continue
                writer.writerow({"source1_entity_id": s1_id, "candidate_source": src,
                                  "candidate_record_id": target_id, "label": 1})
                n_positive_pairs += 1

    return {
        "ground_truth_rows_read": n_rows,
        "positive_pairs_written": n_positive_pairs,
        "unresolved_matched_ids": n_unresolved,
        "unresolved_examples": unresolved_examples,
    }


def main():
    cfg = load_config("config.yaml")
    sd_cfg = cfg.get("student_data")
    if not sd_cfg:
        raise KeyError(
            "config.yaml has no `student_data` section. Add one with "
            "`dataset_train_dir` pointing at your local .../student_resource/dataset/train "
            "directory (see config.yaml.example)."
        )
    dataset_train_dir = sd_cfg["dataset_train_dir"]
    out_dir = sd_cfg.get("adapted_dir", "artifacts/student_data/adapted")

    if not os.path.isdir(dataset_train_dir):
        raise FileNotFoundError(
            f"student_data.dataset_train_dir does not exist on this machine: "
            f"{dataset_train_dir}. Edit config.yaml's `student_data.dataset_train_dir` "
            f"to your real local path before running this script."
        )

    logger.info("Reading real student_resource TRAIN data from: %s", dataset_train_dir)
    logger.info("(dataset_test_dir is never opened by this script.)")

    s1_path = _find_file(dataset_train_dir, "source1")
    s2_path = _find_file(dataset_train_dir, "source2")
    s3_path = _find_file(dataset_train_dir, "source3")
    gt_path = _find_file(dataset_train_dir, "ground_truth")

    n1, _ids1 = _adapt_source_file(s1_path, os.path.join(out_dir, "source1.tsv"), source1=True)
    logger.info("Adapted source1: %d entities -> %s", n1, os.path.join(out_dir, "source1.tsv"))

    n2, ids2 = _adapt_source_file(s2_path, os.path.join(out_dir, "source2.tsv"), source1=False)
    logger.info("Adapted source2: %d records -> %s", n2, os.path.join(out_dir, "source2.tsv"))

    n3, ids3 = _adapt_source_file(s3_path, os.path.join(out_dir, "source3.tsv"), source1=False)
    logger.info("Adapted source3: %d records -> %s", n3, os.path.join(out_dir, "source3.tsv"))

    gt_report = _adapt_ground_truth(gt_path, os.path.join(out_dir, "labels.tsv"), ids2, ids3)
    logger.info("Adapted ground truth: %s", gt_report)

    report = {
        "dataset_train_dir": dataset_train_dir,
        "source1_entities": n1,
        "source2_records": n2,
        "source3_records": n3,
        **gt_report,
    }
    with open(os.path.join(out_dir, "adapt_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if gt_report["unresolved_matched_ids"] > 0:
        logger.warning(
            "%d matched_entity_ids in ground truth did not match any source2/source3 "
            "record id (examples: %s). These positive pairs were dropped — check "
            "COLUMN_ALIASES/ID formats if this number is large.",
            gt_report["unresolved_matched_ids"], gt_report["unresolved_examples"],
        )
    logger.info("Done. Canonical files written to %s", out_dir)


if __name__ == "__main__":
    main()
