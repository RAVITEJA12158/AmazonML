import numpy as np
import pandas as pd
from src.models.pair_model import get_pair_model, one_hot_source_pair
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def run_oof(feature_df: pd.DataFrame, folds_df: pd.DataFrame, model_name: str = "lightgbm",
            sample_weight_col: str = None) -> pd.DataFrame:
    """feature_df must include 'label' and 'source1_entity_id'. folds_df:
    source1_entity_id -> fold. Returns feature_df with an added 'oof_score'
    and 'fold' column — every row scored ONLY by a model that never saw it.
    """
    df = feature_df.merge(folds_df, on="source1_entity_id", how="left")
    df = one_hot_source_pair(df)
    df["oof_score"] = np.nan

    for fold in sorted(df["fold"].dropna().unique()):
        train_mask = df["fold"] != fold
        valid_mask = df["fold"] == fold

        model = get_pair_model(model_name)
        sw = df.loc[train_mask, sample_weight_col].values if sample_weight_col and sample_weight_col in df.columns else None
        model.fit(df.loc[train_mask], df.loc[train_mask, "label"].values, sample_weight=sw)

        df.loc[valid_mask, "oof_score"] = model.predict_proba(df.loc[valid_mask].copy())
        logger.info("Fold %s: trained on %d rows, scored %d OOF rows", fold, train_mask.sum(), valid_mask.sum())

    return df
