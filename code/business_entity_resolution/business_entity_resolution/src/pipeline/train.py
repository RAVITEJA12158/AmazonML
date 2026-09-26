import pandas as pd
from src.models.pair_model import get_pair_model, one_hot_source_pair
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def train_final_pair_model(feature_df: pd.DataFrame, model_name: str = "lightgbm",
                            sample_weight_col: str = None):
    df = one_hot_source_pair(feature_df.copy())
    model = get_pair_model(model_name)
    sw = df[sample_weight_col].values if sample_weight_col and sample_weight_col in df.columns else None
    model.fit(df, df["label"].values, sample_weight=sw)
    logger.info("Final pair model trained on %d rows (%d positive).", len(df), int(df["label"].sum()))
    return model
