"""Cheap, fast pairwise features — the ONLY features the pre-Qwen ranker is
allowed to use. Nothing here calls an embedding model; everything must be
cheap enough to run on thousands of candidates per S1 entity before the pool
is shrunk down to the size semantic retrieval will actually see.
"""
from rapidfuzz.distance import JaroWinkler
from src.normalization.phonetics import phonetic_codes_for_tokens


def token_overlap(tokens_a: list, tokens_b: list) -> float:
    a, b = set(tokens_a), set(tokens_b)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def char_ngram_overlap(ngrams_a: set, ngrams_b: set) -> float:
    if not ngrams_a or not ngrams_b:
        return 0.0
    return len(ngrams_a & ngrams_b) / len(ngrams_a | ngrams_b)


def name_jaro_winkler(name_a: str, name_b: str) -> float:
    if not name_a or not name_b:
        return 0.0
    return JaroWinkler.similarity(name_a, name_b)


def phonetic_match(tokens_a: list, tokens_b: list) -> float:
    codes_a = phonetic_codes_for_tokens(tokens_a)
    codes_b = phonetic_codes_for_tokens(tokens_b)
    if not codes_a or not codes_b:
        return 0.0
    return 1.0 if (codes_a & codes_b) else 0.0


def address_token_overlap(addr_tokens_a: list, addr_tokens_b: list) -> float:
    return token_overlap(addr_tokens_a, addr_tokens_b)


def postal_exact(postal_a, postal_b) -> float:
    if postal_a is None or postal_b is None:
        return 0.0
    return 1.0 if str(postal_a).strip() == str(postal_b).strip() else 0.0


def city_exact(city_a, city_b) -> float:
    if city_a is None or city_b is None:
        return 0.0
    return 1.0 if str(city_a).strip() == str(city_b).strip() else 0.0


def compute_cheap_feature_row(s1_norm: dict, cand_norm: dict) -> dict:
    """s1_norm / cand_norm are rows (as dicts) from the normalized tables —
    the output of src.normalization.build.normalize_records."""
    return {
        "token_overlap": token_overlap(s1_norm["name_tokens"], cand_norm["name_tokens"]),
        "char_ngram_overlap": char_ngram_overlap(s1_norm["char_4grams"], cand_norm["char_4grams"]),
        "name_jaro_winkler": name_jaro_winkler(s1_norm["normalized_name"], cand_norm["normalized_name"]),
        "address_token_overlap": address_token_overlap(s1_norm["address_tokens"], cand_norm["address_tokens"]),
        "postal_exact": postal_exact(s1_norm.get("postal_code"), cand_norm.get("postal_code")),
        "city_exact": city_exact(s1_norm.get("city"), cand_norm.get("city")),
        "phonetic_match": phonetic_match(s1_norm["name_tokens"], cand_norm["name_tokens"]),
    }


CHEAP_FEATURE_ORDER = [
    "token_overlap", "char_ngram_overlap", "name_jaro_winkler",
    "address_token_overlap", "postal_exact", "city_exact", "phonetic_match",
]
