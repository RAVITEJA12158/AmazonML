"""Lexical retrieval channels. These run against the (already geo-narrowed)
pool passed in — see src/blocking/union.py for the funnel order."""
from src.blocking.cheap_features import token_overlap, char_ngram_overlap


def token_candidates(s1_norm: dict, pool_norm: list, top_k: int) -> dict:
    scored = [(c["record_id"], token_overlap(s1_norm["name_tokens"], c["name_tokens"]))
              for c in pool_norm]
    scored.sort(key=lambda x: -x[1])
    return {rid: score for rid, score in scored[:top_k] if score > 0}


def char_ngram_candidates(s1_norm: dict, pool_norm: list, top_k: int, n: int = 4) -> dict:
    key = f"char_{n}grams"
    scored = [(c["record_id"], char_ngram_overlap(s1_norm[key], c[key])) for c in pool_norm]
    scored.sort(key=lambda x: -x[1])
    return {rid: score for rid, score in scored[:top_k] if score > 0}


def concatenated_name_address_candidates(s1_norm: dict, pool_norm: list, top_k: int) -> dict:
    """NEW channel: name and address concatenated into one string, blocked
    on token overlap over the joined string. Catches cases where neither
    field alone is distinctive but the combination is (a common name like
    "City Bakery" becomes distinctive once the street is included)."""
    s1_tokens = set(s1_norm["name_tokens"]) | set(s1_norm["address_tokens"])
    scored = []
    for c in pool_norm:
        c_tokens = set(c["name_tokens"]) | set(c["address_tokens"])
        if not s1_tokens or not c_tokens:
            scored.append((c["record_id"], 0.0))
            continue
        overlap = len(s1_tokens & c_tokens) / len(s1_tokens | c_tokens)
        scored.append((c["record_id"], overlap))
    scored.sort(key=lambda x: -x[1])
    return {rid: score for rid, score in scored[:top_k] if score > 0}
