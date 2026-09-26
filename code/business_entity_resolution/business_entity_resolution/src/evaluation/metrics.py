"""The actual competition metric: macro-averaged F0.5 per Source 1 entity.
Singleton entities (no true matches) score 1.0 for a correctly-empty
prediction and 0.0 for any false positive match."""
import numpy as np
import pandas as pd


def _f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    if precision == 0 and recall == 0:
        return 0.0
    b2 = beta ** 2
    denom = (b2 * precision) + recall
    if denom == 0:
        return 0.0
    return (1 + b2) * precision * recall / denom


def entity_f05(predicted_ids: set, true_ids: set) -> float:
    """Per the problem statement: singleton (true_ids empty) scores 1.0 for
    an empty prediction, 0.0 for any false positive."""
    if not true_ids:
        return 1.0 if not predicted_ids else 0.0

    if not predicted_ids:
        return 0.0  # zero recall, zero precision-defined -> F-beta = 0

    tp = len(predicted_ids & true_ids)
    precision = tp / len(predicted_ids)
    recall = tp / len(true_ids)
    return _f_beta(precision, recall)


def macro_f05(predictions_df: pd.DataFrame, labels_df: pd.DataFrame,
               entity_col: str = "source1_entity_id") -> dict:
    """predictions_df: source1_entity_id, candidate_record_id (accepted matches only)
    labels_df: source1_entity_id, candidate_record_id, label (label==1 rows are true matches)
    Every S1 entity present in labels_df must be scored, including those
    with zero predictions and zero true matches (singletons)."""
    true_by_entity = (labels_df[labels_df["label"] == 1]
                       .groupby(entity_col)["candidate_record_id"].apply(set).to_dict())
    pred_by_entity = predictions_df.groupby(entity_col)["candidate_record_id"].apply(set).to_dict()

    all_entities = set(labels_df[entity_col].unique()) | set(predictions_df[entity_col].unique())

    scores = []
    for eid in all_entities:
        pred = pred_by_entity.get(eid, set())
        true = true_by_entity.get(eid, set())
        scores.append(entity_f05(pred, true))

    scores = np.array(scores)
    tp = fp = fn = 0
    for eid in all_entities:
        pred = pred_by_entity.get(eid, set())
        true = true_by_entity.get(eid, set())
        tp += len(pred & true)
        fp += len(pred - true)
        fn += len(true - pred)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    singleton_entities = [eid for eid in all_entities if not true_by_entity.get(eid)]
    singleton_correct = sum(1 for eid in singleton_entities if not pred_by_entity.get(eid))
    singleton_accuracy = singleton_correct / len(singleton_entities) if singleton_entities else None

    return {
        "macro_f05": float(scores.mean()) if len(scores) else 0.0,
        "precision": precision,
        "recall": recall,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "singleton_accuracy": singleton_accuracy,
        "n_entities": len(all_entities),
    }
