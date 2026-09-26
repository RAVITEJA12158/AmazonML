"""Canonical column contracts. Every stage downstream trusts these names —
if your raw files use different column names, remap them in loader.py, not
scattered throughout the pipeline.
"""

SOURCE1_COLUMNS = [
    "source1_entity_id",
    "name",
    "address",
    "country",
    "state",
    "city",
    "postal_code",
    "street_number",
    "street_name",
    "phone",
    "email",
]

SOURCE23_COLUMNS = [
    "record_id",
    "source",          # "S2" | "S3"
    "name",
    "address",
    "country",
    "state",
    "city",
    "postal_code",
    "street_number",
    "street_name",
    "phone",
    "email",
]

LABELS_COLUMNS = [
    "source1_entity_id",
    "candidate_source",   # "S2" | "S3"
    "candidate_record_id",
    "label",              # 1 = true match, 0 = confirmed non-match (optional)
]

CANDIDATE_PAIR_COLUMNS = [
    "source1_entity_id",
    "candidate_source",
    "candidate_record_id",
    "retrieved_geo",
    "retrieved_exact",
    "retrieved_token",
    "retrieved_char_ngram",
    "retrieved_phonetic",
    "retrieved_rare_token",
    "retrieved_concat_name_address",
    "retrieved_semantic_name",
    "retrieved_semantic_address",
    "retrieved_semantic_combined",
    "retrieved_reciprocal",
    "cheap_rank_score",
    "provenance_count",
]
