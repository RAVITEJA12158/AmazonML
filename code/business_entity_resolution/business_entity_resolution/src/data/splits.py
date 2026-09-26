"""GroupKFold by source1_entity_id.

IMPORTANT (documented correction carried over from the methodology doc):
GroupKFold by itself does NOT stratify by country or singleton status — it
only guarantees that a given source1_entity_id's rows never land in more
than one fold. If you need country/singleton balance across folds, you must
build it explicitly (see `grouped_stratified_folds` below); don't assume
`GroupKFold` gives it to you for free.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def make_folds(labels_or_entities: pd.DataFrame, group_col: str = "source1_entity_id",
                n_splits: int = 5, seed: int = 42) -> pd.DataFrame:
    """Returns a DataFrame with one row per unique group and a `fold` column.
    Plain GroupKFold — no stratification guarantee. See module docstring."""
    groups = labels_or_entities[[group_col]].drop_duplicates().reset_index(drop=True)
    gkf = GroupKFold(n_splits=n_splits)
    # GroupKFold needs X and groups of matching length; a dummy X works fine.
    dummy_X = np.zeros(len(groups))
    groups["fold"] = -1
    for fold_idx, (_, valid_idx) in enumerate(gkf.split(dummy_X, groups=groups[group_col])):
        groups.loc[valid_idx, "fold"] = fold_idx
    return groups


def grouped_stratified_folds(entities: pd.DataFrame, group_col: str = "source1_entity_id",
                              strat_cols=("country", "is_singleton"),
                              n_splits: int = 5, seed: int = 42) -> pd.DataFrame:
    """Optional stronger splitter: assigns each *group* (not each row) to a
    fold using a round-robin within each stratum, so fold sizes stay
    balanced by country/singleton-status. Use this explicitly if you need
    the stratification v2's wording incorrectly implied GroupKFold gives you
    automatically.
    """
    df = entities.drop_duplicates(subset=[group_col]).copy()
    df["_strat_key"] = df[list(strat_cols)].astype(str).agg("_".join, axis=1)
    rng = np.random.RandomState(seed)
    df = df.sample(frac=1.0, random_state=rng).reset_index(drop=True)

    fold_assignment = {}
    counters = {}
    for _, row in df.iterrows():
        key = row["_strat_key"]
        counters.setdefault(key, 0)
        fold_assignment[row[group_col]] = counters[key] % n_splits
        counters[key] += 1

    df["fold"] = df[group_col].map(fold_assignment)
    return df[[group_col, "fold"]]


def split_holdout(entities: pd.DataFrame, group_col: str = "source1_entity_id",
                   holdout_frac: float = 0.15, seed: int = 42):
    """Carve out an untouched holdout split, grouped by entity, before any
    development/iteration begins. Returns (dev_entity_ids, holdout_entity_ids)."""
    ids = entities[group_col].drop_duplicates().sample(frac=1.0, random_state=seed)
    n_holdout = int(len(ids) * holdout_frac)
    holdout_ids = set(ids.iloc[:n_holdout])
    dev_ids = set(ids.iloc[n_holdout:])
    return dev_ids, holdout_ids
