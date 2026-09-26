import numpy as np
import pandas as pd
from src.evaluation.metrics import macro_f05


def sweep_threshold(oof_scores_df: pd.DataFrame, labels_df: pd.DataFrame,
                     sweep_min: float, sweep_max: float, sweep_step: float) -> pd.DataFrame:
    """oof_scores_df needs: source1_entity_id, candidate_record_id, oof_score.
    Sweeps on OOF predictions ONLY — never on the final holdout, per the
    leakage checklist."""
    results = []
    for t in np.arange(sweep_min, sweep_max + 1e-9, sweep_step):
        accepted = oof_scores_df[oof_scores_df["oof_score"] >= t][["source1_entity_id", "candidate_record_id"]]
        m = macro_f05(accepted, labels_df)
        m["threshold"] = round(float(t), 4)
        results.append(m)
    return pd.DataFrame(results).sort_values("macro_f05", ascending=False)


def best_threshold(oof_scores_df: pd.DataFrame, labels_df: pd.DataFrame, cfg: dict) -> float:
    sweep = sweep_threshold(oof_scores_df, labels_df, cfg["sweep_min"], cfg["sweep_max"], cfg["sweep_step"])
    return float(sweep.iloc[0]["threshold"])
