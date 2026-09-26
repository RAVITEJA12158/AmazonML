import pandas as pd


def categorize_errors(scored_df: pd.DataFrame, labels_df: pd.DataFrame, accepted_df: pd.DataFrame) -> pd.DataFrame:
    """Buckets false positives and false negatives into the categories
    tracked in the methodology docs, using the same cheap similarity
    signals already computed in the feature table."""
    labels_map = labels_df.set_index(["source1_entity_id", "candidate_record_id"])["label"].to_dict()
    accepted_keys = set(zip(accepted_df["source1_entity_id"], accepted_df["candidate_record_id"]))

    rows = []
    for _, r in scored_df.iterrows():
        key = (r["source1_entity_id"], r["candidate_record_id"])
        true_label = labels_map.get(key, 0)
        predicted = key in accepted_keys

        if predicted and not true_label:
            category = _fp_category(r)
            rows.append({**key_dict(key), "error_type": "false_positive", "category": category})
        elif true_label and not predicted:
            category = _fn_category(r)
            rows.append({**key_dict(key), "error_type": "false_negative", "category": category})

    return pd.DataFrame(rows)


def key_dict(key):
    return {"source1_entity_id": key[0], "candidate_record_id": key[1]}


def _fp_category(r) -> str:
    name_sim = r.get("jaro_winkler", 0)
    addr_sim = r.get("address_similarity", 0)
    if name_sim > 0.9 and addr_sim < 0.4:
        return "similar_but_different_business"
    if r.get("postal_conflict") == 1 or r.get("city_conflict") == 1:
        return "singleton_false_positive_or_wrong_location"
    return "other_false_positive"


def _fn_category(r) -> str:
    name_sim = r.get("jaro_winkler", 0)
    addr_sim = r.get("address_similarity", 0)
    if name_sim < 0.5 and addr_sim < 0.5:
        return "likely_blocking_miss"
    if r.get("address_missing_s1") or r.get("address_missing_candidate"):
        return "missing_address_component"
    return "other_false_negative"
