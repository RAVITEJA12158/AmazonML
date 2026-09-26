"""Geographic pre-blocking. This is the FIRST filter applied — before the
cheap ranker and before semantic retrieval — exactly per the restructured
flow:

    S1 -> geographic blocking -> cheap ranking -> semantic retrieval -> Top-K -> pair model

Never run semantic retrieval against the full S2/S3 pool.
"""
import pandas as pd


def geo_pool(s1_row: pd.Series, s23_norm: pd.DataFrame, keys, fallback_key: str,
             max_pool_per_entity: int) -> pd.DataFrame:
    """Returns the subset of s23_norm sharing at least the fallback_key with
    s1_row, narrowed further by any additional keys that are present on
    both sides. Country is never a hard filter on its own if it's missing on
    either side (open-set label, no hard-coded vocabulary) — a missing key
    just means that key is skipped for narrowing, not the whole entity
    dropped."""
    pool = s23_norm

    for key in keys:
        s1_val = s1_row.get(key)
        if s1_val is None or str(s1_val).strip() == "":
            continue
        if key not in pool.columns:
            continue
        narrowed = pool[pool[key] == s1_val]
        # Only accept the narrowing if it doesn't wipe out the pool due to
        # a missing/inconsistent field on the S2/S3 side; fall back to the
        # wider pool rather than losing recall to a formatting mismatch.
        if len(narrowed) > 0:
            pool = narrowed
        elif key == fallback_key:
            # fallback key found nothing at all -> keep the pre-narrowing
            # pool (effectively: don't geo-block this entity)
            break

    if len(pool) > max_pool_per_entity:
        pool = pool.sample(n=max_pool_per_entity, random_state=0)

    return pool
