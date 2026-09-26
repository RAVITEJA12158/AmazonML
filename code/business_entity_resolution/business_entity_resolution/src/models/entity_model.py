"""Entity-level decision layer. Operates on ALL of a Source 1 entity's
candidates jointly — never `argmax(score)`, since the task allows zero,
one, or many true matches per entity.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb

ENTITY_FEATURE_COLUMNS = [
    "pair_score", "rank", "score_gap", "best_score", "second_best_score",
    "candidate_count", "is_source2", "is_source3",
    "city_conflict", "state_conflict", "postal_conflict", "street_number_conflict",
    "cross_source_support", "neighbor_match_count", "is_mutual_topk",
    "inverse_name_frequency", "jaro_winkler", "address_similarity",
]


def hand_rule_decision(row: pd.Series, threshold: float, singleton_score_gap_low: float,
                        singleton_best_score_low: float) -> int:
    """Baseline A/B combined: global threshold + a margin-aware singleton
    lean. Implemented first, validated first, per the blueprint's own
    ordering — promote to the learned model only once this is measured."""
    if row["pair_score"] < threshold:
        return 0
    # Weak-margin high scorers get slightly more scrutiny: require a bit
    # more separation from the runner-up when this candidate itself isn't
    # a clear standout, since ambiguous clusters are exactly where false
    # merges happen.
    if row["score_gap"] < singleton_score_gap_low and row["best_score"] < 1.0 and row["rank"] > 1:
        return 0
    return 1


def apply_hand_rule(entity_df: pd.DataFrame, threshold: float, cfg: dict) -> pd.DataFrame:
    df = entity_df.copy()
    df["accepted"] = df.apply(
        hand_rule_decision, axis=1, threshold=threshold,
        singleton_score_gap_low=cfg["singleton_score_gap_low"],
        singleton_best_score_low=cfg["singleton_best_score_low"],
    )
    return df


class EntityLightGBMModel:
    """Learned entity-level decision model — a small classifier over
    entity-context features. Kept only if it beats the hand-rule baseline
    on held-out macro F0.5 (see evaluation/metrics.py)."""

    def __init__(self):
        self.model = None
        self.feature_columns = ENTITY_FEATURE_COLUMNS

    def fit(self, entity_feature_df: pd.DataFrame, labels: np.ndarray):
        cols = [c for c in self.feature_columns if c in entity_feature_df.columns]
        self.feature_columns = cols
        self.model = lgb.LGBMClassifier(
            n_estimators=150, num_leaves=15, learning_rate=0.05,
            min_child_samples=5, verbosity=-1,
        )
        self.model.fit(entity_feature_df[cols].fillna(0), labels)
        return self

    def predict_proba(self, entity_feature_df: pd.DataFrame) -> np.ndarray:
        for c in self.feature_columns:
            if c not in entity_feature_df.columns:
                entity_feature_df[c] = 0.0
        return self.model.predict_proba(entity_feature_df[self.feature_columns].fillna(0))[:, 1]
