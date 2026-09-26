import pandas as pd
from src.normalization.names import build_name_views
from src.normalization.addresses import build_address_views
from src.normalization.phonetics import phonetic_codes_for_tokens


def normalize_records(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """Applies multi-view normalization to every record. Raw fields are
    always retained alongside derived ones (never overwritten), per the
    "raw -> normalized -> derived" rule — this is what makes error analysis
    possible later."""
    rows = []
    for _, r in df.iterrows():
        name_views = build_name_views(r.get("name"))
        addr_views = build_address_views(
            r.get("address"), r.get("postal_code"), r.get("city"),
            r.get("state"), r.get("street_number"), r.get("street_name"),
            r.get("country"),
        )
        phon = phonetic_codes_for_tokens(name_views["name_tokens"])

        row = {id_col: r[id_col]}
        if "source" in df.columns:
            row["source"] = r.get("source")
        row.update(name_views)
        row.update(addr_views)
        row["phonetic_codes"] = phon
        rows.append(row)
    return pd.DataFrame(rows)
