import re
import unicodedata
import pandas as pd


def is_missing(v) -> bool:
    """True for None, pandas/numpy NaN, or an all-whitespace string. Centralized
    here so every normalization function treats "missing" identically --
    NaN in particular is a float, not None, and silently stringifies to the
    literal text "nan" if you only check `is None` (a real bug this fixes)."""
    if v is None:
        return True
    if isinstance(v, float) and pd.isna(v):
        return True
    return str(v).strip() == ""


ABBREVIATIONS = {
    "rd": "road", "rd.": "road",
    "st": "street", "st.": "street",
    "ave": "avenue", "ave.": "avenue",
    "blvd": "boulevard", "blvd.": "boulevard",
    "ln": "lane", "ln.": "lane",
    "dr": "drive", "dr.": "drive",
    "apt": "apartment", "apt.": "apartment",
    "&": "and",
}

LEGAL_SUFFIXES = [
    "pvt ltd", "private limited", "pvt. ltd.", "pvt.ltd.",
    "ltd", "limited", "corp", "corporation", "inc", "incorporated",
    "co", "company", "llc", "llp", "gmbh", "sarl", "sas", "sa",
]


def unicode_normalize(s: str) -> str:
    if s is None:
        return ""
    # NFKD then strip combining marks handles most transliteration/accent
    # noise (é -> e) while staying language-agnostic, per the no-hardcoded
    # -vocabulary constraint on country/language handling.
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s


def basic_clean(s: str) -> str:
    if is_missing(s):
        return ""
    s = unicode_normalize(s)
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def expand_abbreviations(s: str) -> str:
    tokens = s.split()
    return " ".join(ABBREVIATIONS.get(t, t) for t in tokens)


def strip_punctuation(s: str, keep_spaces: bool = True) -> str:
    if keep_spaces:
        return re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"[^\w]", "", s)


def compact(s: str) -> str:
    """No spaces, no punctuation — for exact/prefix comparisons robust to
    spacing noise."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def tokenize(s: str) -> list:
    return [t for t in s.split() if t]


def char_ngrams(s: str, n: int) -> set:
    s = s.replace(" ", "")
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i + n] for i in range(len(s) - n + 1)}
