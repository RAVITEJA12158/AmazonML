"""Pair model — answers "how likely is this S1<->candidate pair to be the
same business?" LightGBM is the default/primary; CatBoost/XGBoost plug into
the same interface if installed (see config.yaml `pair_model.candidates`),
benchmarked on identical features and identical GroupKFold splits.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb

NON_FEATURE_COLUMNS = {
    "source1_entity_id", "candidate_source", "candidate_record_id", "label",
    "fold", "hard_negative_type", "sample_weight", "reciprocal_rank",
}


def get_feature_columns(df: pd.DataFrame) -> list:
    cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    # keep only numeric columns — categorical-as-string (source_pair) is
    # one-hot encoded upstream for LightGBM; CatBoost could take it raw.
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    return numeric


def one_hot_source_pair(df: pd.DataFrame) -> pd.DataFrame:
    if "source_pair" in df.columns:
        dummies = pd.get_dummies(df["source_pair"], prefix="source_pair")
        return pd.concat([df, dummies], axis=1)
    return df


class LightGBMPairModel:
    def __init__(self, params: dict = None):
        self.params = params or {
            "objective": "binary",
            "metric": "binary_logloss",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "n_estimators": 200,
            "min_child_samples": 5,
            "verbosity": -1,
        }
        self.model = None
        self.feature_columns = None

    def fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray = None):
        self.feature_columns = get_feature_columns(X)
        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(X[self.feature_columns].fillna(0), y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        cols = self.feature_columns or get_feature_columns(X)
        for c in cols:
            if c not in X.columns:
                X[c] = 0.0
        return self.model.predict_proba(X[cols].fillna(0))[:, 1]

    def feature_importance(self) -> pd.DataFrame:
        if self.model is None:
            return pd.DataFrame()
        return pd.DataFrame({
            "feature": self.feature_columns,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False)


def get_pair_model(name: str = "lightgbm"):
    if name == "lightgbm":
        return LightGBMPairModel()
    raise ValueError(
        f"Pair model '{name}' not available in this environment. "
        f"CatBoost/XGBoost wrappers follow the same interface as "
        f"LightGBMPairModel (fit/predict_proba/feature_importance) — add "
        f"one here once the package is installed."
    )
