import pandas as pd
from src.features.name import name_features, address_features
from src.features.missingness import missingness_features
from src.features.contradiction import contradiction_features
from src.features.uniqueness import uniqueness_features
from src.features.cross_source import cross_source_features


def _cross_field_features(name_feats: dict, addr_feats: dict) -> dict:
    name_sim = name_feats["jaro_winkler"]
    address_sim = addr_feats["address_similarity"]
    return {
        "name_sim_x_address_sim": name_sim * address_sim,
        "name_sim_minus_address_sim": name_sim - address_sim,
        "abs_name_sim_minus_address_sim": abs(name_sim - address_sim),
        "min_name_address_sim": min(name_sim, address_sim),
        "max_name_address_sim": max(name_sim, address_sim),
        "name_high_address_high": float(name_sim > 0.8 and address_sim > 0.8),
        "name_high_address_low": float(name_sim > 0.8 and address_sim < 0.4),
        "name_low_address_high": float(name_sim < 0.4 and address_sim > 0.8),
        "name_medium_address_medium": float(0.4 <= name_sim <= 0.8 and 0.4 <= address_sim <= 0.8),
    }


def _source_features(candidate_source: str) -> dict:
    return {
        "is_source2": float(candidate_source == "S2"),
        "is_source3": float(candidate_source == "S3"),
    }


def build_feature_row(s1_norm: dict, cand_norm: dict, cand_row: dict,
                       name_freq, token_freq, cross_source_row: dict,
                       candidate_source: str) -> dict:
    """cand_row: the row from candidate_pairs.tsv (has retrieval provenance,
    semantic scores, cheap_rank_score, reciprocal flags)."""
    name_feats = name_features(s1_norm, cand_norm)
    addr_feats = address_features(s1_norm, cand_norm)
    miss_feats = missingness_features(s1_norm, cand_norm)
    contra_feats = contradiction_features(s1_norm, cand_norm)
    uniq_feats = uniqueness_features(s1_norm, cand_norm, name_freq, token_freq)
    cross_feats = _cross_field_features(name_feats, addr_feats)
    src_feats = _source_features(candidate_source)

    row = {
        "source1_entity_id": s1_norm["record_id"],
        "candidate_source": candidate_source,
        "candidate_record_id": cand_norm["record_id"],
    }
    row.update(name_feats)
    row.update(addr_feats)
    row.update(miss_feats)
    row.update(contra_feats)
    row.update(uniq_feats)
    row.update(cross_feats)
    row.update(src_feats)
    row["source_pair"] = f"S1_{candidate_source}"

    # semantic / provenance / retrieval-derived features carried over from
    # candidate_pairs.tsv, per the candidate-provenance rule.
    row["semantic_name_cosine"] = cand_row.get("semantic_name_score", 0.0)
    row["semantic_address_cosine"] = cand_row.get("semantic_address_score", 0.0)
    row["semantic_combined_cosine"] = cand_row.get("semantic_combined_score", 0.0)
    row["cheap_rank_score"] = cand_row.get("cheap_rank_score", 0.0)
    row["provenance_count"] = cand_row.get("provenance_count", 0)
    row["is_mutual_topk"] = float(cand_row.get("retrieved_reciprocal", False))
    row["reciprocal_rank"] = cand_row.get("reciprocal_rank")
    for flag in ("retrieved_geo", "retrieved_token", "retrieved_char_ngram", "retrieved_phonetic",
                 "retrieved_rare_token", "retrieved_concat_name_address"):
        row[flag] = float(cand_row.get(flag, False))

    row["cross_source_support"] = cross_source_row.get("cross_source_support", 0.0)
    row["neighbor_match_count"] = cross_source_row.get("neighbor_match_count", 0)

    return row


def add_candidate_relative_features(feature_df: pd.DataFrame, score_col: str = "cheap_rank_score") -> pd.DataFrame:
    """best_score / second_best_score / score_gap / rank / percentile,
    computed within each S1 entity's candidate group. Called again later
    with the real pair-model score once it exists (see pipeline/oof.py).

    Implemented with groupby().transform() rather than groupby().apply(),
    deliberately: apply() with a function that returns the (possibly
    mutated) group frame has a sharp edge in pandas where the grouping
    column itself can be dropped from the result depending on version/
    group_keys behavior. transform() always returns a same-length Series
    aligned to the original index, so source1_entity_id is never at risk.
    """
    df = feature_df.copy()
    g = df.groupby("source1_entity_id")[score_col]

    df["rank"] = g.rank(ascending=False, method="first")
    df["percentile"] = g.rank(pct=True)
    df["best_score"] = g.transform("max")
    df["candidate_count"] = g.transform("size")

    def _second_best(s: pd.Series) -> float:
        sorted_vals = s.sort_values(ascending=False).values
        return float(sorted_vals[1]) if len(sorted_vals) > 1 else 0.0

    df["second_best_score"] = g.transform(_second_best)
    df["score_gap"] = df["best_score"] - df["second_best_score"]
    return df
