"""Mines hard negatives from OOF predictions only — never from in-sample
scores, so the model doing the mining has never seen the rows it's mining
from. Five explicit categories, per the addendum."""
import pandas as pd


def mine_hard_negatives(oof_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """oof_df must have: label, oof_score, jaro_winkler (name similarity),
    address_similarity, plus identifying columns."""
    df = oof_df.copy()
    df["hard_negative_type"] = None

    is_neg = df["label"] == 0

    cat_a = is_neg & (df["oof_score"] > cfg["high_score_threshold"])
    df.loc[cat_a, "hard_negative_type"] = "A_high_score_fp"

    b_cfg = cfg["same_name_diff_address"]
    cat_b = is_neg & (df["jaro_winkler"] > b_cfg["name_sim_min"]) & (df["address_similarity"] < b_cfg["address_sim_max"])
    df.loc[cat_b & df["hard_negative_type"].isna(), "hard_negative_type"] = "B_same_name_diff_address"

    c_cfg = cfg["same_address_diff_name"]
    cat_c = is_neg & (df["address_similarity"] > c_cfg["address_sim_min"]) & (df["jaro_winkler"] < c_cfg["name_sim_max"])
    df.loc[cat_c & df["hard_negative_type"].isna(), "hard_negative_type"] = "C_same_address_diff_name"

    # D: near-duplicate name clusters — approximate via very high name
    # similarity with a moderate (not low, not high) address similarity,
    # which is the "looks like a variant, isn't confirmed by address" case.
    cat_d = is_neg & (df["jaro_winkler"] > 0.90) & (df["address_similarity"].between(0.3, 0.7))
    df.loc[cat_d & df["hard_negative_type"].isna(), "hard_negative_type"] = "D_near_duplicate"

    # E: same postal/city, different business — address matches but name
    # clearly doesn't and it wasn't already caught by C.
    if "postal_exact" in df.columns and "city_exact" in df.columns:
        cat_e = is_neg & ((df["postal_exact"] == 1) | (df["city_exact"] == 1)) & (df["jaro_winkler"] < 0.5)
        df.loc[cat_e & df["hard_negative_type"].isna(), "hard_negative_type"] = "E_same_location"

    hard_negatives = df[df["hard_negative_type"].notna()].copy()
    return hard_negatives


def apply_sample_weights(train_df: pd.DataFrame, hard_negative_ids: set, weight: float = 3.0) -> pd.DataFrame:
    """Sample weighting instead of blind row duplication — per the
    blueprint's explicit caution against replicating hard negatives
    hundreds of times."""
    df = train_df.copy()
    key = df["source1_entity_id"].astype(str) + "|" + df["candidate_record_id"].astype(str)
    df["sample_weight"] = 1.0
    df.loc[key.isin(hard_negative_ids), "sample_weight"] = weight
    return df
