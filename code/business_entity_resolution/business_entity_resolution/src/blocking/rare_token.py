"""Rare-token / IDF-weighted retrieval. A distinctive token ("Raviteja")
carries more retrieval weight than a common one ("Global", "Technologies").
This is a retrieval channel, not a final match score."""
import math
from collections import Counter


def build_token_idf(all_norm_records: list) -> dict:
    """all_norm_records: normalized records across the WHOLE corpus (used
    once, up front, to build global token document frequencies)."""
    df = Counter()
    n = len(all_norm_records)
    for r in all_norm_records:
        for t in set(r["name_tokens"]):
            df[t] += 1
    return {t: math.log((n + 1) / (d + 1)) + 1 for t, d in df.items()}


def rare_token_candidates(s1_norm: dict, pool_norm: list, idf: dict, top_k: int) -> dict:
    s1_tokens = set(s1_norm["name_tokens"])
    if not s1_tokens:
        return {}
    scored = []
    for c in pool_norm:
        c_tokens = set(c["name_tokens"])
        shared = s1_tokens & c_tokens
        if not shared:
            continue
        score = sum(idf.get(t, 1.0) for t in shared)
        scored.append((c["record_id"], score))
    scored.sort(key=lambda x: -x[1])
    return {rid: score for rid, score in scored[:top_k]}
