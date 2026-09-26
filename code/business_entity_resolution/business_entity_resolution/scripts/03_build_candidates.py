"""Stage A: geographic blocking -> lexical channels -> cheap ranker ->
semantic retrieval -> pruning -> candidate_pairs.tsv.

Two passes, because the cheap ranker needs labeled examples to fit on:
  Pass 1: for entities that HAVE labels, build a geo+lexical-only pool
          (no ranker yet) and fit the logistic-regression cheap ranker on
          the resulting labeled pairs.
  Pass 2: run the full funnel (now using the fitted ranker) for every S1
          entity, train and test alike, to produce the final
          candidate_pairs.tsv — the file that is itself part of the scored
          submission.

Run: python scripts/03_build_candidates.py
"""
import os
import sys
import joblib
import pandas as pd

sys.path.insert(0, os.getcwd())
from src.utils.seed import set_seed
from src.utils.io import load_config, save_table
from src.utils.logging_utils import get_logger
from src.data.loader import load_source1, load_source23, load_labels
from src.normalization.build import normalize_records
from src.blocking.geographic import geo_pool
from src.blocking.lexical import token_candidates, char_ngram_candidates, concatenated_name_address_candidates
from src.blocking.phonetic import phonetic_candidates
from src.blocking.rare_token import build_token_idf, rare_token_candidates
from src.blocking.cheap_ranker import CheapRanker
from src.blocking.cheap_features import compute_cheap_feature_row
from src.blocking.semantic import SemanticRetriever
from src.pipeline.inference import build_candidate_pairs

logger = get_logger(__name__)


def _lexical_union_pool(s1_rec, s23_norm_df, idf, cfg):
    geo_cfg = cfg["geographic_blocking"]
    lex_cfg = cfg["lexical_blocking"]

    pool_df = geo_pool(pd.Series(s1_rec), s23_norm_df, keys=geo_cfg["keys"],
                        fallback_key=geo_cfg["fallback_key"],
                        max_pool_per_entity=geo_cfg["max_pool_per_entity"])
    pool_norm = pool_df.to_dict("records")
    if not pool_norm:
        return []

    ids = set()
    ids.update(token_candidates(s1_rec, pool_norm, lex_cfg["token"]["top_k"]).keys())
    ids.update(char_ngram_candidates(s1_rec, pool_norm, lex_cfg["char_ngram"]["top_k"]).keys())
    ids.update(phonetic_candidates(s1_rec, pool_norm, lex_cfg["phonetic"]["top_k"]).keys())
    ids.update(rare_token_candidates(s1_rec, pool_norm, idf, lex_cfg["rare_token"]["top_k"]).keys())
    ids.update(concatenated_name_address_candidates(s1_rec, pool_norm, lex_cfg["concatenated_name_address"]["top_k"]).keys())
    ids.update(c["record_id"] for c in pool_norm)  # geo channel itself

    return [c for c in pool_norm if c["record_id"] in ids]


def fit_cheap_ranker(s1_norm_df, s23_norm_df, labels_df, idf, cfg) -> CheapRanker:
    s1_lookup = s1_norm_df.set_index("record_id").to_dict("index")
    labeled_entities = labels_df["source1_entity_id"].unique()

    feature_rows, label_values = [], []
    for entity_id in labeled_entities:
        s1_rec = {**s1_lookup[entity_id], "record_id": entity_id}
        pool = _lexical_union_pool(s1_rec, s23_norm_df, idf, cfg)
        pool_ids = {c["record_id"] for c in pool}

        entity_labels = labels_df[labels_df["source1_entity_id"] == entity_id]
        label_map = dict(zip(entity_labels["candidate_record_id"], entity_labels["label"]))

        for c in pool:
            feature_rows.append(compute_cheap_feature_row(s1_rec, c))
            label_values.append(label_map.get(c["record_id"], 0))

    feature_df = pd.DataFrame(feature_rows)
    ranker = CheapRanker(min_positive_examples=cfg["cheap_ranker"]["min_positive_examples"])
    if not feature_df.empty:
        ranker.fit(feature_df, pd.Series(label_values).values)
    else:
        logger.warning("No labeled pool built at all — cheap ranker will use the fallback heuristic.")
    return ranker


def main():
    set_seed(42)
    cfg = load_config("config.yaml")
    raw_dir = cfg["paths"]["raw_dir"]

    s1_df = load_source1(raw_dir)
    s23_df = load_source23(raw_dir, cfg.get("student_data", {}).get("shared_candidates_dir"))
    labels_df = load_labels(raw_dir)

    s1_norm = normalize_records(s1_df, id_col="source1_entity_id").rename(columns={"source1_entity_id": "record_id"})
    s23_norm = normalize_records(s23_df, id_col="record_id")
    idf = build_token_idf(s23_norm.to_dict("records"))

    logger.info("Fitting cheap ranker on labeled entities...")
    ranker = fit_cheap_ranker(s1_norm, s23_norm, labels_df, idf, cfg)
    logger.info("Cheap ranker fitted (fallback=%s)", ranker.is_fallback)

    semantic_retriever = SemanticRetriever(backend_name=cfg["semantic_retrieval"]["embedding_backend"])

    logger.info("Building final candidate_pairs.tsv via full funnel for all %d S1 entities...", len(s1_df))
    candidates_df, s1_norm_full, s23_norm_full = build_candidate_pairs(s1_df, s23_df, cfg, ranker, semantic_retriever)

    save_table(candidates_df, os.path.join(cfg["paths"]["candidates_dir"], "candidate_pairs.tsv"))
    joblib.dump(ranker, os.path.join(cfg["paths"]["models_dir"], "cheap_ranker.joblib"))

    logger.info("Wrote %d candidate rows for %d S1 entities -> %s",
                len(candidates_df), candidates_df["source1_entity_id"].nunique() if not candidates_df.empty else 0,
                os.path.join(cfg["paths"]["candidates_dir"], "candidate_pairs.tsv"))


if __name__ == "__main__":
    main()
