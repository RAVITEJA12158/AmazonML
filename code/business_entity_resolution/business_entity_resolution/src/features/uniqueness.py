from collections import Counter


def build_name_frequency_table(norm_records: list) -> Counter:
    return Counter(r["normalized_name"] for r in norm_records if r["normalized_name"])


def build_token_frequency_table(norm_records: list) -> Counter:
    c = Counter()
    for r in norm_records:
        c.update(set(r["name_tokens"]))
    return c


def uniqueness_features(s1: dict, cand: dict, name_freq: Counter, token_freq: Counter) -> dict:
    name_frequency = name_freq.get(cand["normalized_name"], 0)
    tok_freqs = [token_freq.get(t, 0) for t in cand["name_tokens"]] or [0]
    avg_token_freq = sum(tok_freqs) / len(tok_freqs)
    return {
        "name_frequency": float(name_frequency),
        "token_frequency": float(avg_token_freq),
        "inverse_name_frequency": 1.0 / (1.0 + name_frequency),
    }
