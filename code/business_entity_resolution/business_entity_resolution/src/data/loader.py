import os
import pandas as pd
from src.data.schema import SOURCE1_COLUMNS, SOURCE23_COLUMNS, LABELS_COLUMNS
from src.utils.io import load_table


def _coerce_columns(df: pd.DataFrame, expected: list) -> pd.DataFrame:
    """Add any missing expected columns as null, keep column order stable,
    keep extra columns at the end (never silently drop information)."""
    for col in expected:
        if col not in df.columns:
            df[col] = None
    extra = [c for c in df.columns if c not in expected]
    return df[expected + extra]


def load_source1(raw_dir: str) -> pd.DataFrame:
    path = _find(raw_dir, "source1")
    df = load_table(path)
    return _coerce_columns(df, SOURCE1_COLUMNS)


def load_source23(raw_dir: str, shared_candidates_dir: str = None) -> pd.DataFrame:
    """Loads and concatenates Source 2 and Source 3 into one table with a
    `source` column, since most of the pipeline treats them uniformly except
    where source-aware features (§ features/cross_source features) apply.

    Tries raw_dir FIRST for each of source2/source3 individually, falling
    back to `shared_candidates_dir` only if not found there. This matters:
    the dev/ and heldout_test/ query splits have no source2.tsv/source3.tsv
    of their own (they intentionally share one candidate pool, so they fall
    back to shared_candidates_dir = the adapted real train data), but a
    genuinely separate raw_dir that DOES ship its own source2/source3 (e.g.
    the real unlabeled test set, adapted by scripts/15) must use its own
    files, not the training candidate pool."""
    frames = []
    for tag in ("source2", "source3"):
        path = _find(raw_dir, tag, required=False)
        if path is None and shared_candidates_dir:
            path = _find(shared_candidates_dir, tag, required=False)
        if path is None:
            continue
        df = load_table(path)
        df = _coerce_columns(df, SOURCE23_COLUMNS)
        df["source"] = "S2" if tag == "source2" else "S3"
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No source2/source3 files found under {raw_dir}")
    return pd.concat(frames, ignore_index=True)


def load_labels(raw_dir: str) -> pd.DataFrame:
    path = _find(raw_dir, "labels", required=False)
    if path is None:
        return pd.DataFrame(columns=LABELS_COLUMNS)
    df = load_table(path)
    return _coerce_columns(df, LABELS_COLUMNS)


def _find(raw_dir: str, tag: str, required: bool = True) -> str:
    for ext in (".tsv", ".csv", ".parquet"):
        candidate = os.path.join(raw_dir, tag + ext)
        if os.path.exists(candidate):
            return candidate
    if required:
        raise FileNotFoundError(f"Could not find a {tag}.(tsv|csv|parquet) file under {raw_dir}")
    return None
