"""Reciprocal retrieval: does the candidate ALSO rank S1 highly among its
own top-K S1 neighbors? Mutual nearest-neighbor status is strong evidence.
This produces feature flags — it never removes non-mutual candidates."""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def reciprocal_flags(s1_norm: dict, s1_pool: list, finalists_norm: list, top_k: int,
                      backend) -> dict:
    """s1_pool: a sample of OTHER S1 entities' normalized names, used so we
    can ask "if we queried FROM this candidate, would this S1 entity show
    up in its own top-K?" This is an approximation (querying against a
    sample of S1, not literally re-running full retrieval per candidate)
    kept deliberately cheap since it's a supplementary evidence feature,
    not a filter.
    """
    if not finalists_norm or not s1_pool:
        return {c["record_id"]: {"is_mutual_topk": False, "reciprocal_rank": None}
                for c in finalists_norm}

    s1_names = [r["normalized_name"] for r in s1_pool]
    backend.fit_corpus(s1_names)
    s1_vecs = backend.embed(s1_names)

    results = {}
    for c in finalists_norm:
        cand_vec = backend.embed([c["normalized_name"]])
        sims = cosine_similarity(cand_vec, s1_vecs).ravel()
        order = np.argsort(-sims)
        top_ids = [s1_pool[i]["record_id"] for i in order[:top_k]]
        is_mutual = s1_norm["record_id"] in top_ids
        rank = (top_ids.index(s1_norm["record_id"]) + 1) if is_mutual else None
        results[c["record_id"]] = {"is_mutual_topk": is_mutual, "reciprocal_rank": rank}
    return results
