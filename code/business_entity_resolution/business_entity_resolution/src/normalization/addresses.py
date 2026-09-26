import re
from src.normalization.text import basic_clean, expand_abbreviations, strip_punctuation, tokenize, is_missing

LANDMARK_PATTERN = re.compile(r"\b(near|opp\.?|opposite|behind|next to)\b.*", re.IGNORECASE)
STREET_NUMBER_PATTERN = re.compile(r"^\s*(\d+[a-zA-Z]?)\b")

_is_missing = is_missing  # kept as a local alias -- this file used to define its own copy


def _clean_code(v) -> str:
    """Postal codes etc. often come back as pandas floats (94105.0) after a
    numeric-looking column round-trips through CSV. Strip the trailing .0
    rather than let it silently break exact-match features."""
    s = str(v).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def extract_landmark(raw_address: str) -> str:
    if _is_missing(raw_address):
        return ""
    m = LANDMARK_PATTERN.search(str(raw_address))
    return m.group(0).strip() if m else ""


def extract_street_number(normalized_address: str) -> str:
    m = STREET_NUMBER_PATTERN.match(normalized_address)
    return m.group(1) if m else ""


def build_address_views(raw_address: str, postal_code=None, city=None, state=None,
                         street_number=None, street_name=None, country=None) -> dict:
    """Address fields are extracted independently where the source already
    provides them; only the free-text address string is parsed as a
    fallback. Missing is tracked explicitly (§ missingness features) and is
    never silently treated as similarity = 0 downstream."""
    # raw_address can be None (real Python null), NaN (pandas' representation
    # of an empty CSV cell -- a float, not None, which is what actually broke
    # this before), or a genuinely empty string. _is_missing() catches all three.
    original = "" if _is_missing(raw_address) else str(raw_address)
    landmark = extract_landmark(original)
    without_landmark = LANDMARK_PATTERN.sub("", original).strip()

    cleaned = basic_clean(without_landmark)
    cleaned = strip_punctuation(cleaned)
    normalized_address = expand_abbreviations(cleaned)
    address_tokens = tokenize(normalized_address)

    parsed_street_number = extract_street_number(normalized_address)

    return {
        "original_address": original,
        "normalized_address": normalized_address,
        "address_tokens": address_tokens,
        "landmark": landmark,
        "postal_code": None if _is_missing(postal_code) else _clean_code(postal_code),
        "city": None if _is_missing(city) else basic_clean(str(city)),
        "state": None if _is_missing(state) else basic_clean(str(state)),
        "street_number": (None if _is_missing(street_number)
                           else _clean_code(street_number)) or (parsed_street_number or None),
        "street_name": None if _is_missing(street_name) else basic_clean(str(street_name)),
        "country": None if _is_missing(country) else basic_clean(str(country)),
        "address_missing": len(normalized_address.strip()) == 0,
        "postal_missing": _is_missing(postal_code),
        "city_missing": _is_missing(city),
        "state_missing": _is_missing(state),
        "street_number_missing": _is_missing(street_number) and not parsed_street_number,
    }
