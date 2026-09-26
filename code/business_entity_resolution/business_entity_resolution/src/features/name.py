from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz.fuzz import token_sort_ratio, token_set_ratio


def name_features(s1: dict, cand: dict) -> dict:
    a, b = s1["normalized_name"], cand["normalized_name"]
    a_ns, b_ns = s1["name_without_legal_suffix"], cand["name_without_legal_suffix"]
    a_compact, b_compact = s1["compact_name"], cand["compact_name"]

    return {
        "exact_name_match": float(a == b and a != ""),
        "no_suffix_exact": float(a_ns == b_ns and a_ns != ""),
        "levenshtein_ratio": Levenshtein.normalized_similarity(a, b) if a and b else 0.0,
        "jaro_winkler": JaroWinkler.similarity(a, b) if a and b else 0.0,
        "token_sort_ratio": token_sort_ratio(a, b) / 100.0 if a and b else 0.0,
        "token_set_ratio": token_set_ratio(a, b) / 100.0 if a and b else 0.0,
        "char_3gram": _jaccard(s1["char_3grams"], cand["char_3grams"]),
        "char_4gram": _jaccard(s1["char_4grams"], cand["char_4grams"]),
        "char_5gram": _jaccard(s1["char_5grams"], cand["char_5grams"]),
        "prefix_similarity": float(a_compact[:5] == b_compact[:5] and len(a_compact) >= 5 and len(b_compact) >= 5),
        "suffix_similarity": float(a_compact[-5:] == b_compact[-5:] and len(a_compact) >= 5 and len(b_compact) >= 5),
        "token_overlap": _jaccard(set(s1["name_tokens"]), set(cand["name_tokens"])),
        "phonetic_match": float(bool(s1["phonetic_codes"] & cand["phonetic_codes"])),
    }


def address_features(s1: dict, cand: dict) -> dict:
    a, b = s1["normalized_address"], cand["normalized_address"]
    return {
        "address_similarity": Levenshtein.normalized_similarity(a, b) if a and b else 0.0,
        "postal_exact": _exact(s1.get("postal_code"), cand.get("postal_code")),
        "postal_partial": _prefix_match(s1.get("postal_code"), cand.get("postal_code")),
        "city_exact": _exact(s1.get("city"), cand.get("city")),
        "state_exact": _exact(s1.get("state"), cand.get("state")),
        "street_number_exact": _exact(s1.get("street_number"), cand.get("street_number")),
        "street_name_similarity": (Levenshtein.normalized_similarity(s1.get("street_name") or "", cand.get("street_name") or "")
                                    if s1.get("street_name") and cand.get("street_name") else 0.0),
        "landmark_similarity": (Levenshtein.normalized_similarity(s1.get("landmark") or "", cand.get("landmark") or "")
                                 if s1.get("landmark") and cand.get("landmark") else 0.0),
        "address_token_overlap": _jaccard(set(s1["address_tokens"]), set(cand["address_tokens"])),
    }


def _jaccard(a, b) -> float:
    if not a or not b:
        return 0.0
    return len(set(a) & set(b)) / len(set(a) | set(b))


def _exact(a, b) -> float:
    if a is None or b is None:
        return 0.0
    return float(str(a).strip() == str(b).strip())


def _prefix_match(a, b, n: int = 3) -> float:
    if a is None or b is None:
        return 0.0
    return float(str(a).strip()[:n] == str(b).strip()[:n])
