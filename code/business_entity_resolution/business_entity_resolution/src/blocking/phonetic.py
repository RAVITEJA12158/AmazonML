"""Phonetic retrieval — supplementary channel only, never standalone, since
phonetic collisions alone would blow up the candidate set. Targets the
transliteration-variant noise pattern (e.g. "Sri Lakshmi" vs "Shree Laxmi")."""


def phonetic_candidates(s1_norm: dict, pool_norm: list, top_k: int) -> dict:
    s1_codes = s1_norm["phonetic_codes"]
    if not s1_codes:
        return {}
    scored = []
    for c in pool_norm:
        c_codes = c["phonetic_codes"]
        if not c_codes:
            continue
        overlap = len(s1_codes & c_codes)
        if overlap > 0:
            scored.append((c["record_id"], overlap / max(len(s1_codes), 1)))
    scored.sort(key=lambda x: -x[1])
    return {rid: score for rid, score in scored[:top_k]}
