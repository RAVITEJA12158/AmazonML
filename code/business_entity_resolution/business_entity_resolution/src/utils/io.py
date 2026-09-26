import os
import yaml
import pandas as pd


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_table(df: pd.DataFrame, path: str) -> None:
    # explicit encoding="utf-8" everywhere -- pandas otherwise falls back to
    # the OS default codepage (cp1252 on Windows), which breaks on any
    # non-ASCII character in a business name or address.
    ensure_dir(os.path.dirname(path))
    if path.endswith(".tsv"):
        df.to_csv(path, sep="\t", index=False, encoding="utf-8")
    elif path.endswith(".csv"):
        df.to_csv(path, index=False, encoding="utf-8")
    elif path.endswith(".parquet"):
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported output format for {path}")


def load_table(path: str) -> pd.DataFrame:
    if path.endswith(".tsv"):
        return pd.read_csv(path, sep="\t", encoding="utf-8")
    elif path.endswith(".csv"):
        return pd.read_csv(path, encoding="utf-8")
    elif path.endswith(".parquet"):
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported input format for {path}")
