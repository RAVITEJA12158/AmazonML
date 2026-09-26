"""For an S1 entity with candidates in both S2 and S3, measure whether those
candidates agree with EACH OTHER — corroborating evidence neither pair alone
provides. Feature-only: graph connectivity never becomes an automatic match.
"""
from rapidfuzz.distance import Levenshtein


def cross_source_features(entity_candidates_norm: list) -> dict:
    """entity_candidates_norm: normalized records for ALL candidates of one
    S1 entity (both S2 and S3), each with a 'source' field. Returns a dict
    keyed by candidate_record_id -> {cross_source_support, neighbor_match_count}.
    """
    s2 = [c for c in entity_candidates_norm if c.get("source") == "S2"]
    s3 = [c for c in entity_candidates_norm if c.get("source") == "S3"]

    result = {rid["record_id"]: {"cross_source_support": 0.0, "neighbor_match_count": 0}
              for rid in entity_candidates_norm}

    for a in s2:
        best_sim, best_b = 0.0, None
        for b in s3:
            sim = Levenshtein.normalized_similarity(a["normalized_name"], b["normalized_name"])
            if sim > best_sim:
                best_sim, best_b = sim, b
        if best_b is not None:
            result[a["record_id"]]["cross_source_support"] = best_sim
            result[best_b["record_id"]]["cross_source_support"] = max(
                result[best_b["record_id"]]["cross_source_support"], best_sim)
            if best_sim > 0.85:
                result[a["record_id"]]["neighbor_match_count"] += 1
                result[best_b["record_id"]]["neighbor_match_count"] += 1

    return result


def triangular_consistency(name_sim_s1_a: float, name_sim_s1_b: float, cross_source_support: float) -> float:
    """High when all three legs of the S1-A-B triangle agree; low when any
    one leg disagrees with the other two."""
    values = [name_sim_s1_a, name_sim_s1_b, cross_source_support]
    return min(values) if all(v is not None for v in values) else 0.0
