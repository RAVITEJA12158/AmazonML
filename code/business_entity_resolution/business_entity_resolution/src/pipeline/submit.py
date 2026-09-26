import pandas as pd


def validate_submission(matching_results_path: str, candidate_pairs_path: str, s1_ids: set) -> list:
    errors = []
    results = pd.read_csv(matching_results_path, sep="\t")
    candidates = pd.read_csv(candidate_pairs_path, sep="\t")

    required_cols = {"source1_entity_id", "candidate_source", "candidate_record_id"}
    missing_cols = required_cols - set(results.columns)
    if missing_cols:
        errors.append(f"matching_results.tsv missing columns: {missing_cols}")

    result_entities = set(results["source1_entity_id"].unique())
    missing_entities = s1_ids - result_entities
    if missing_entities:
        errors.append(f"{len(missing_entities)} Source 1 entities have no representation "
                       f"in matching_results.tsv (need at least a singleton row/marker).")

    dupes = results.duplicated(subset=["source1_entity_id", "candidate_source", "candidate_record_id"]).sum()
    if dupes:
        errors.append(f"{dupes} duplicate rows in matching_results.tsv")

    # Singleton entities are represented by a NONE marker row (see
    # src/pipeline/inference.py write_matching_results) rather than being
    # absent from the file entirely — absence is ambiguous (did the pipeline
    # forget this entity, or correctly find no match?), a marker row isn't.
    cand_keys = set(zip(candidates["source1_entity_id"], candidates["candidate_record_id"]))
    real_matches = results[results["candidate_source"] != "NONE"]
    result_keys = set(zip(real_matches["source1_entity_id"], real_matches["candidate_record_id"]))
    invalid = result_keys - cand_keys
    if invalid:
        errors.append(f"{len(invalid)} matches in matching_results.tsv are not present in "
                       f"candidate_pairs.tsv — every accepted match must trace back to a "
                       f"candidate blocking actually generated.")

    valid_sources = {"S2", "S3", "NONE"}
    bad_sources = set(results["candidate_source"].unique()) - valid_sources
    if bad_sources:
        errors.append(f"Invalid candidate_source values: {bad_sources}")

    return errors
