from src.normalization.text import (
    basic_clean, expand_abbreviations, strip_punctuation, compact,
    tokenize, char_ngrams, LEGAL_SUFFIXES, is_missing,
)

STOPWORDS = {"of", "the", "a", "an"}


def remove_legal_suffix(normalized_name: str) -> str:
    s = normalized_name
    for suffix in sorted(LEGAL_SUFFIXES, key=len, reverse=True):
        if s.endswith(" " + suffix) or s == suffix:
            s = s[: len(s) - len(suffix)].strip()
            break
    return s.strip()


def remove_stopwords(s: str) -> str:
    return " ".join(t for t in s.split() if t not in STOPWORDS)


def build_name_views(raw_name: str) -> dict:
    """Multi-view normalization: every representation is retained, never
    collapsed into one canonical string, so downstream features can compare
    at whichever resolution is predictive."""
    # NaN (pandas' empty-cell representation, a float) is not caught by an
    # `is not None` check and previously stringified to the literal text
    # "nan" -- is_missing() catches None, NaN, and blank strings alike.
    original = "" if is_missing(raw_name) else str(raw_name)
    cleaned = basic_clean(original)
    cleaned = strip_punctuation(cleaned)
    cleaned = expand_abbreviations(cleaned)

    normalized_name = cleaned
    name_without_suffix = remove_legal_suffix(normalized_name)
    compact_name = compact(normalized_name)
    name_tokens = tokenize(normalized_name)
    stopword_removed = remove_stopwords(normalized_name)

    return {
        "original_name": original,
        "normalized_name": normalized_name,
        "name_without_legal_suffix": name_without_suffix,
        "compact_name": compact_name,
        "name_tokens": name_tokens,
        "stopword_removed_normalized": stopword_removed,
        "char_3grams": char_ngrams(normalized_name, 3),
        "char_4grams": char_ngrams(normalized_name, 4),
        "char_5grams": char_ngrams(normalized_name, 5),
        "name_missing": len(normalized_name.strip()) == 0,
    }
