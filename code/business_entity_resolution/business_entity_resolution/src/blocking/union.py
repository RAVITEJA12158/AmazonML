"""Orchestrates the restructured blocking funnel:

    S1 entity
        |
        v
    geographic blocking                (src/blocking/geographic.py)
        |
        v
    lexical channels (token/char/phonetic/rare-token/concat name+address)
    run WITHIN the geo pool, unioned    (src/blocking/lexical.py, phonetic.py, rare_token.py)
        |
        v
    cheap ranker (logistic regression)  (src/blocking/cheap_ranker.py)
    scores the union, keeps top N
        |
        v
    semantic retrieval, run ONLY on
    the cheap-ranker-reduced pool       (src/blocking/semantic.py)
        |
        v
    reciprocal flags on finalists       (src/blocking/reciprocal.py)
        |
        v
    candidate table with full provenance -> src/blocking/prune.py -> candidate_pairs.tsv

This is the direct implementation of: "Qwen should not retrieve everything" —
semantic retrieval in this module never sees more than
cheap_ranker.max_pool_after_ranking candidates for any single S1 entity.
"""
import pandas as pd
from src.blocking.geographic import geo_pool
from src.blocking.lexical import token_candidates, char_ngram_candidates, concatenated_name_address_candidates
from src.blocking.phonetic import phonetic_candidates
from src.blocking.rare_token import rare_token_candidates
from src.blocking.cheap_ranker import CheapRanker
from src.blocking.semantic import SemanticRetriever
from src.blocking.reciprocal import reciprocal_flags
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def build_candidates_for_entity(s1_norm: dict, s23_norm_df: pd.DataFrame, idf: dict,
                                 cheap_ranker: CheapRanker, semantic_retriever: SemanticRetriever,
                                 cfg: dict, s1_pool_for_reciprocal: list) -> pd.DataFrame:
    geo_cfg = cfg["geographic_blocking"]
    lex_cfg = cfg["lexical_blocking"]
    rank_cfg = cfg["cheap_ranker"]
    sem_cfg = cfg["semantic_retrieval"]
    recip_cfg = cfg["reciprocal_retrieval"]

    # 1. Geographic pre-blocking — first, cheapest filter.
    pool_df = geo_pool(
        pd.Series(s1_norm), s23_norm_df,
        keys=geo_cfg["keys"], fallback_key=geo_cfg["fallback_key"],
        max_pool_per_entity=geo_cfg["max_pool_per_entity"],
    )
    if pool_df.empty:
        return _empty_candidates()

    pool_norm = pool_df.to_dict("records")

    # 2. Lexical channels, run within the geo pool, unioned.
    channel_hits = {}  # record_id -> set of channel names that retrieved it
    channel_scores = {}  # record_id -> {channel: score}

    def _register(channel_name, scored_dict):
        for rid, score in scored_dict.items():
            channel_hits.setdefault(rid, set()).add(channel_name)
            channel_scores.setdefault(rid, {})[channel_name] = score

    if lex_cfg["token"]["enabled"]:
        _register("token", token_candidates(s1_norm, pool_norm, lex_cfg["token"]["top_k"]))
    if lex_cfg["char_ngram"]["enabled"]:
        _register("char_ngram", char_ngram_candidates(s1_norm, pool_norm, lex_cfg["char_ngram"]["top_k"]))
    if lex_cfg["phonetic"]["enabled"]:
        _register("phonetic", phonetic_candidates(s1_norm, pool_norm, lex_cfg["phonetic"]["top_k"]))
    if lex_cfg["rare_token"]["enabled"]:
        _register("rare_token", rare_token_candidates(s1_norm, pool_norm, idf, lex_cfg["rare_token"]["top_k"]))
    if lex_cfg["concatenated_name_address"]["enabled"]:
        _register("concat_name_address",
                   concatenated_name_address_candidates(s1_norm, pool_norm, lex_cfg["concatenated_name_address"]["top_k"]))

    # Always include the raw geo pool too (bounded already), so an entity
    # with weak lexical signal but strong geographic co-location isn't lost
    # before the cheap ranker gets a chance to see it.
    _register("geo", {c["record_id"]: 1.0 for c in pool_norm})

    union_ids = set(channel_hits.keys())
    union_norm = [c for c in pool_norm if c["record_id"] in union_ids]
    if not union_norm:
        return _empty_candidates()

    # 3. Cheap ranker — replaces hand-written weights, decides survival to
    #    the semantic-retrieval stage.
    ranked = cheap_ranker.score_pairs(s1_norm, union_norm)
    ranked["record_id"] = [c["record_id"] for c in union_norm]
    ranked = ranked.sort_values("cheap_rank_score", ascending=False)
    keep_ids = set(ranked["record_id"].head(rank_cfg["max_pool_after_ranking"]))
    reduced_pool_norm = [c for c in union_norm if c["record_id"] in keep_ids]

    # 4. Semantic retrieval — ONLY on the reduced pool.
    semantic_results = {}
    if sem_cfg["enabled"]:
        semantic_results = semantic_retriever.retrieve(
            s1_norm, reduced_pool_norm,
            top_k_name=sem_cfg["top_k_name"], top_k_address=sem_cfg["top_k_address"],
            top_k_combined=sem_cfg["top_k_combined"],
        )
        for rid in semantic_results:
            channel_hits.setdefault(rid, set())
            for flag in ("retrieved_semantic_name", "retrieved_semantic_address", "retrieved_semantic_combined"):
                if semantic_results[rid].get(flag):
                    channel_hits[rid].add(flag.replace("retrieved_", ""))

    # Final finalist set = reduced pool (cheap-ranker survivors), since
    # semantic retrieval adds scores/flags on top rather than narrowing
    # further by itself — final size control happens in prune.py.
    finalist_ids = keep_ids
    finalist_norm = reduced_pool_norm

    # 5. Reciprocal flags (approximate, cheap) on finalists only.
    recip_results = {}
    if recip_cfg["enabled"] and s1_pool_for_reciprocal:
        recip_results = reciprocal_flags(
            s1_norm, s1_pool_for_reciprocal, finalist_norm,
            top_k=recip_cfg["top_k"], backend=semantic_retriever._backend_for("reciprocal"),
        )

    # 6. Assemble the candidate table with full provenance.
    rows = []
    rank_score_map = dict(zip(ranked["record_id"], ranked["cheap_rank_score"]))
    for c in finalist_norm:
        rid = c["record_id"]
        channels = channel_hits.get(rid, set())
        sem = semantic_results.get(rid, {})
        recip = recip_results.get(rid, {"is_mutual_topk": False, "reciprocal_rank": None})
        rows.append({
            "source1_entity_id": s1_norm["record_id"],
            "candidate_source": c.get("source"),
            "candidate_record_id": rid,
            "retrieved_geo": "geo" in channels,
            "retrieved_token": "token" in channels,
            "retrieved_char_ngram": "char_ngram" in channels,
            "retrieved_phonetic": "phonetic" in channels,
            "retrieved_rare_token": "rare_token" in channels,
            "retrieved_concat_name_address": "concat_name_address" in channels,
            "retrieved_semantic_name": "semantic_name" in channels,
            "retrieved_semantic_address": "semantic_address" in channels,
            "retrieved_semantic_combined": "semantic_combined" in channels,
            "retrieved_reciprocal": bool(recip.get("is_mutual_topk")),
            "semantic_name_score": sem.get("semantic_name", 0.0),
            "semantic_address_score": sem.get("semantic_address", 0.0),
            "semantic_combined_score": sem.get("semantic_combined", 0.0),
            "reciprocal_rank": recip.get("reciprocal_rank"),
            "cheap_rank_score": rank_score_map.get(rid, 0.0),
            "provenance_count": len([ch for ch in channels if ch != "geo"]) or (1 if "geo" in channels else 0),
        })

    return pd.DataFrame(rows)


def _empty_candidates() -> pd.DataFrame:
    from src.data.schema import CANDIDATE_PAIR_COLUMNS
    return pd.DataFrame(columns=CANDIDATE_PAIR_COLUMNS)
