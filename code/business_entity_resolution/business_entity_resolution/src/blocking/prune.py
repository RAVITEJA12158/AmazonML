"""Final size control before candidate_pairs.tsv is written. This is a
scored objective now (see the candidate-size addendum): a smaller candidate
set per Source 1 entity ranks higher, independent of the leaderboard score.

Ranks each S1 entity's candidates by a combination of signals already
computed during the funnel (no new hand-tuned weights invented here):
  - cheap_rank_score        (from the logistic-regression cheap ranker)
  - max semantic similarity (name/address/combined, whichever is highest)
  - provenance_count        (candidates multiple channels agree on are safer keeps)
then truncates to max_candidates_per_entity.
"""
import numpy as np
import pandas as pd


def _combined_prune_score(df: pd.DataFrame) -> pd.Series:
    semantic_max = df[["semantic_name_score", "semantic_address_score", "semantic_combined_score"]].max(axis=1)
    # min-max normalize cheap_rank_score and provenance_count within THIS
    # entity's candidate set only, so the combination is scale-consistent
    # regardless of absolute score ranges across entities.
    def _norm(s):
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else s * 0.0

    return (0.4 * _norm(df["cheap_rank_score"])
            + 0.4 * _norm(semantic_max)
            + 0.2 * _norm(df["provenance_count"].astype(float)))


def prune_candidates(candidates_df: pd.DataFrame, max_candidates_per_entity: int,
                      min_provenance_count: int = 1) -> pd.DataFrame:
    if candidates_df.empty:
        return candidates_df

    df = candidates_df[candidates_df["provenance_count"] >= min_provenance_count].copy()
    if df.empty:
        df = candidates_df.copy()  # never prune an entity down to zero candidates silently

    kept = []
    for entity_id, group in df.groupby("source1_entity_id"):
        group = group.copy()
        group["prune_score"] = _combined_prune_score(group)
        group = group.sort_values("prune_score", ascending=False)
        kept.append(group.head(max_candidates_per_entity))

    result = pd.concat(kept, ignore_index=True) if kept else df
    return result.drop(columns=["prune_score"], errors="ignore")
