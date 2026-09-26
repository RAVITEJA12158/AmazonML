try:
    import jellyfish
    _HAS_JELLYFISH = True
except ImportError:
    _HAS_JELLYFISH = False


def _fallback_soundex(s: str) -> str:
    """Minimal pure-python soundex, used only if jellyfish isn't installed."""
    s = "".join(c for c in s.upper() if c.isalpha())
    if not s:
        return ""
    codes = {**{c: "1" for c in "BFPV"}, **{c: "2" for c in "CGJKQSXZ"},
             **{c: "3" for c in "DT"}, "L": "4", **{c: "5" for c in "MN"}, "R": "6"}
    first = s[0]
    tail = []
    prev = codes.get(first, "")
    for c in s[1:]:
        code = codes.get(c, "")
        if code and code != prev:
            tail.append(code)
        prev = code
    return (first + "".join(tail) + "000")[:4]


def soundex(s: str) -> str:
    if not s:
        return ""
    if _HAS_JELLYFISH:
        return jellyfish.soundex(s)
    return _fallback_soundex(s)


def double_metaphone(s: str):
    """Returns (primary, secondary) codes. Falls back to soundex-only if
    jellyfish's metaphone isn't available — degraded but non-fatal, since
    phonetic blocking is a supplementary channel, never a required one."""
    if not s:
        return "", ""
    if _HAS_JELLYFISH:
        try:
            return jellyfish.metaphone(s), ""
        except Exception:
            pass
    return soundex(s), ""


def phonetic_codes_for_tokens(tokens: list) -> set:
    """Phonetic codes computed per-token, not on the whole compacted name —
    this keeps transliteration matching from being wrecked by word order or
    a single missing token."""
    codes = set()
    for t in tokens:
        sdx = soundex(t)
        if sdx:
            codes.add(("soundex", sdx))
        primary, _ = double_metaphone(t)
        if primary:
            codes.add(("metaphone", primary))
    return codes
