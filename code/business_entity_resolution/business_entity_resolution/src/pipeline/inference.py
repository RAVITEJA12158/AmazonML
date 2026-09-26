import pandas as pd
from src.normalization.build import normalize_records
from src.blocking.union import build_candidates_for_entity
from src.blocking.prune import prune_candidates
from src.blocking.rare_token import build_token_idf
from src.blocking.cheap_ranker import CheapRanker
from src.blocking.semantic import SemanticRetriever
from src.features.uniqueness import build_name_frequency_table, build_token_frequency_table
from src.features.cross_source import cross_source_features
from src.features.build_features import build_feature_row, add_candidate_relative_features
from src.models.entity_model import apply_hand_rule, ENTITY_FEATURE_COLUMNS
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def build_candidate_pairs(s1_df: pd.DataFrame, s23_df: pd.DataFrame, cfg: dict,
                           cheap_ranker: CheapRanker, semantic_retriever: SemanticRetriever) -> pd.DataFrame:
    """Full Stage A: normalize -> per-entity funnel -> union -> prune ->
    candidate_pairs.tsv. This is the code that is itself part of the
    scored submission, per the problem-statement update."""
    s1_norm = normalize_records(s1_df, id_col="source1_entity_id").rename(columns={"source1_entity_id": "record_id"})
    s23_norm = normalize_records(s23_df, id_col="record_id")

    idf = build_token_idf(s23_norm.to_dict("records"))
    s1_records = s1_norm.to_dict("records")

    all_candidates = []
    for s1_rec in s1_records:
        cand_df = build_candidates_for_entity(
            s1_rec, s23_norm, idf, cheap_ranker, semantic_retriever, cfg,
            s1_pool_for_reciprocal=s1_records,
        )
        all_candidates.append(cand_df)

    candidates_df = pd.concat(all_candidates, ignore_index=True) if all_candidates else pd.DataFrame()

    prune_cfg = cfg["candidate_pruning"]
    if prune_cfg["enabled"] and not candidates_df.empty:
        candidates_df = prune_candidates(
            candidates_df,
            max_candidates_per_entity=prune_cfg["max_candidates_per_entity"],
            min_provenance_count=prune_cfg["min_provenance_count"],
        )

    return candidates_df, s1_norm, s23_norm


def build_pairwise_features(candidates_df: pd.DataFrame, s1_norm: pd.DataFrame, s23_norm: pd.DataFrame) -> pd.DataFrame:
    """Stage B input: the rich pairwise feature table, built only for
    candidates that survived Stage A."""
    if candidates_df.empty:
        return pd.DataFrame()

    s1_lookup = s1_norm.set_index("record_id").to_dict("index")
    s23_lookup = s23_norm.set_index("record_id").to_dict("index")

    name_freq = build_name_frequency_table(s23_norm.to_dict("records"))
    token_freq = build_token_frequency_table(s23_norm.to_dict("records"))

    rows = []
    for entity_id, group in candidates_df.groupby("source1_entity_id"):
        s1_rec = {**s1_lookup[entity_id], "record_id": entity_id}
        entity_cands_norm = []
        for _, cand_row in group.iterrows():
            cand_rec = {**s23_lookup[cand_row["candidate_record_id"]], "record_id": cand_row["candidate_record_id"]}
            entity_cands_norm.append(cand_rec)

        cross_source_map = cross_source_features(entity_cands_norm)

        for _, cand_row in group.iterrows():
            cand_rec = {**s23_lookup[cand_row["candidate_record_id"]], "record_id": cand_row["candidate_record_id"]}
            row = build_feature_row(
                s1_rec, cand_rec, cand_row.to_dict(), name_freq, token_freq,
                cross_source_map.get(cand_row["candidate_record_id"], {}),
                candidate_source=cand_row["candidate_source"],
            )
            rows.append(row)

    feature_df = pd.DataFrame(rows)
    feature_df = add_candidate_relative_features(feature_df, score_col="cheap_rank_score")
    return feature_df


def score_with_pair_model(feature_df: pd.DataFrame, pair_model, calibrator=None) -> pd.DataFrame:
    from src.models.pair_model import one_hot_source_pair
    df = one_hot_source_pair(feature_df.copy())
    raw_scores = pair_model.predict_proba(df)
    df["pair_score"] = calibrator.transform(raw_scores) if calibrator is not None else raw_scores
    df = add_candidate_relative_features(df, score_col="pair_score")
    return df


def apply_entity_decision(scored_df: pd.DataFrame, threshold: float, entity_cfg: dict,
                           entity_model=None) -> pd.DataFrame:
    if entity_cfg["method"] == "lightgbm" and entity_model is not None:
        for c in ENTITY_FEATURE_COLUMNS:
            if c not in scored_df.columns:
                scored_df[c] = 0.0
        scored_df["entity_score"] = entity_model.predict_proba(scored_df)
        scored_df["accepted"] = (scored_df["entity_score"] >= threshold).astype(int)
        return scored_df

    return apply_hand_rule(scored_df, threshold, entity_cfg)


def write_matching_results(scored_df: pd.DataFrame, all_s1_ids: pd.Series) -> pd.DataFrame:
    """Every Source 1 entity gets exactly one representation in the output:
    either its accepted match rows, or — if it has none — a single explicit
    NONE marker row. Silent absence is ambiguous (forgotten entity vs.
    correctly-empty prediction); a marker row is not. This is what makes
    every S1 test entity, including singletons, actually "receive exactly
    one row" as the output contract requires, while still allowing multiple
    accepted rows for entities with more than one true match.
    """
    accepted = scored_df[scored_df.get("accepted", 0) == 1][
        ["source1_entity_id", "candidate_source", "candidate_record_id"]
    ].copy()
    if "pair_score" in scored_df.columns:
        accepted["match_probability"] = scored_df.loc[scored_df["accepted"] == 1, "pair_score"].values

    matched_ids = set(accepted["source1_entity_id"].unique())
    unmatched_ids = set(all_s1_ids.unique()) - matched_ids

    marker_rows = pd.DataFrame({
        "source1_entity_id": list(unmatched_ids),
        "candidate_source": "NONE",
        "candidate_record_id": None,
        "match_probability": None,
    })

    result = pd.concat([accepted, marker_rows], ignore_index=True)
    return result.sort_values("source1_entity_id").reset_index(drop=True)
