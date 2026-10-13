"""Classification metrics, subgroup summaries, and latency measurement."""

from __future__ import annotations

from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    target,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    """Calculate threshold-dependent and ranking metrics."""

    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(target, predictions, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(target, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(target, predictions)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
        "specificity": float(specificity),
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "pr_auc": float(average_precision_score(target, probabilities)),
        "brier_score": float(brier_score_loss(target, probabilities)),
        "log_loss": float(log_loss(target, probabilities, labels=[0, 1])),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def subgroup_metrics(
    target,
    probabilities: np.ndarray,
    groups: pd.Series,
    threshold: float,
) -> pd.DataFrame:
    """Calculate aggregate performance for each subgroup with both classes."""

    rows = []
    for group_name, indices in groups.groupby(groups).groups.items():
        group_target = target.iloc[list(indices)]
        if group_target.nunique() < 2:
            continue
        metrics = classification_metrics(
            group_target,
            probabilities[list(indices)],
            threshold,
        )
        rows.append({"group": group_name, "rows": len(indices), **metrics})
    return pd.DataFrame(rows)


def measure_batch_latency(model, features, repeats: int = 10) -> dict[str, float]:
    """Measure median and upper-tail batch prediction latency in milliseconds."""

    durations = []
    for _ in range(repeats):
        start = perf_counter()
        model.predict_proba(features)
        durations.append((perf_counter() - start) * 1_000)
    return {
        "median_ms": float(np.median(durations)),
        "p95_ms": float(np.percentile(durations, 95)),
    }
