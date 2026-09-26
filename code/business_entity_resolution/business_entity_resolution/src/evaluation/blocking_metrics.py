"""Blocking recall ceiling AND candidate-set size — both scored now, per the
candidate-size addendum. Report per-entity size stats as primary, reduction
ratio as secondary."""
import pandas as pd


def blocking_recall(candidate_pairs_df: pd.DataFrame, labels_df: pd.DataFrame) -> dict:
    true_pairs = labels_df[labels_df["label"] == 1][["source1_entity_id", "candidate_record_id"]]
    if true_pairs.empty:
        return {"recall_ceiling": None, "true_pairs_total": 0, "true_pairs_recovered": 0}

    cand_keys = set(zip(candidate_pairs_df["source1_entity_id"], candidate_pairs_df["candidate_record_id"]))
    true_keys = set(zip(true_pairs["source1_entity_id"], true_pairs["candidate_record_id"]))
    recovered = len(true_keys & cand_keys)

    return {
        "recall_ceiling": recovered / len(true_keys) if true_keys else None,
        "true_pairs_total": len(true_keys),
        "true_pairs_recovered": recovered,
    }


def candidate_size_stats(candidate_pairs_df: pd.DataFrame, all_s1_ids: pd.Series) -> dict:
    """Primary metric per the addendum — smaller candidate sets per S1
    entity are scored higher, independent of recall/F0.5."""
    counts = candidate_pairs_df.groupby("source1_entity_id").size()
    counts = counts.reindex(all_s1_ids.unique(), fill_value=0)
    return {
        "avg_candidates_per_s1_entity": float(counts.mean()),
        "median_candidates_per_s1_entity": float(counts.median()),
        "max_candidates_per_s1_entity": int(counts.max()) if len(counts) else 0,
        "total_candidate_pairs": int(len(candidate_pairs_df)),
        "n_s1_entities": int(len(counts)),
    }


def reduction_ratio(candidate_pairs_df: pd.DataFrame, n_s1: int, n_s23: int) -> float:
    all_possible = n_s1 * n_s23
    if all_possible == 0:
        return None
    return 1.0 - (len(candidate_pairs_df) / all_possible)


def channel_contribution(candidate_pairs_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    """Which channels recover true matches that other channels miss? —
    the key question for deciding whether a channel is worth its runtime."""
    true_keys = set(zip(labels_df[labels_df["label"] == 1]["source1_entity_id"],
                         labels_df[labels_df["label"] == 1]["candidate_record_id"]))
    channel_cols = [c for c in candidate_pairs_df.columns if c.startswith("retrieved_")]
    rows = []
    for col in channel_cols:
        sub = candidate_pairs_df[candidate_pairs_df[col] == True]  # noqa: E712
        sub_keys = set(zip(sub["source1_entity_id"], sub["candidate_record_id"]))
        rows.append({
            "channel": col,
            "candidate_count": len(sub),
            "true_matches_recovered": len(sub_keys & true_keys),
        })
    return pd.DataFrame(rows).sort_values("true_matches_recovered", ascending=False)
