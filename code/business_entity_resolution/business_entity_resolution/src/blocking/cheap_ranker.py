"""Replaces hand-written pruning weights (0.25 * token_overlap + 0.15 *
phonetic_match + ...) with a tiny logistic regression fit on the same cheap
features. Its ONLY job is: which candidates survive long enough to reach
semantic retrieval / the pair model? It is deliberately not the final
classifier — that's src/models/pair_model.py, trained on the richer feature
set downstream.

    Candidate Pair
         |
         +-- token overlap
         +-- name jaro-winkler
         +-- address token overlap
         +-- postal / city exact
         +-- phonetic match
                |
                v
         Logistic Regression
                |
                v
          cheap_rank_score
                |
                v
              Top-K

If there are too few labeled positive examples to fit a stable model
(`min_positive_examples` in config.yaml), this falls back to a fixed-weight
heuristic — but that fallback is explicit and logged, never silent.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.blocking.cheap_features import compute_cheap_feature_row, CHEAP_FEATURE_ORDER
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Only used if there isn't enough labeled data to fit a real model. This is
# a fallback, not the primary mechanism — see module docstring.
_FALLBACK_WEIGHTS = {
    "token_overlap": 0.25, "char_ngram_overlap": 0.15, "name_jaro_winkler": 0.30,
    "address_token_overlap": 0.15, "postal_exact": 0.05, "city_exact": 0.05,
    "phonetic_match": 0.05,
}


class CheapRanker:
    def __init__(self, min_positive_examples: int = 20):
        self.min_positive_examples = min_positive_examples
        self.model = None
        self.is_fallback = True

    def fit(self, feature_df: pd.DataFrame, labels: np.ndarray) -> "CheapRanker":
        n_pos = int(labels.sum())
        if n_pos < self.min_positive_examples:
            logger.warning(
                "Only %d positive examples (< min_positive_examples=%d); "
                "falling back to fixed-weight heuristic for the cheap ranker. "
                "This should be revisited once more labeled data is available.",
                n_pos, self.min_positive_examples,
            )
            self.is_fallback = True
            return self

        X = feature_df[CHEAP_FEATURE_ORDER].values
        self.model = LogisticRegression(max_iter=1000, class_weight="balanced")
        self.model.fit(X, labels)
        self.is_fallback = False
        return self

    def score(self, feature_df: pd.DataFrame) -> np.ndarray:
        if self.is_fallback or self.model is None:
            weights = np.array([_FALLBACK_WEIGHTS[c] for c in CHEAP_FEATURE_ORDER])
            X = feature_df[CHEAP_FEATURE_ORDER].values
            return X @ weights
        X = feature_df[CHEAP_FEATURE_ORDER].values
        return self.model.predict_proba(X)[:, 1]

    def score_pairs(self, s1_norm: dict, candidates_norm: list) -> pd.DataFrame:
        rows = [compute_cheap_feature_row(s1_norm, c) for c in candidates_norm]
        feat_df = pd.DataFrame(rows)
        feat_df["cheap_rank_score"] = self.score(feat_df)
        return feat_df
