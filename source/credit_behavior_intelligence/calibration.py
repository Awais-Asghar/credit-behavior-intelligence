"""Probability calibration and validation-only threshold selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, precision_recall_curve


@dataclass(frozen=True)
class ThresholdSelection:
    """Selected operating threshold and its validation F1 score."""

    threshold: float
    f1: float
    precision: float
    recall: float


def calibrate_classifier(
    estimator: BaseEstimator,
    features,
    target,
    *,
    folds: int = 3,
) -> CalibratedClassifierCV:
    """Fit sigmoid calibration across stratified folds."""

    calibrated = CalibratedClassifierCV(estimator=estimator, method="sigmoid", cv=folds)
    return calibrated.fit(features, target)


def select_f1_threshold(target, probabilities: np.ndarray) -> ThresholdSelection:
    """Select the probability threshold that maximizes validation F1."""

    precision, recall, thresholds = precision_recall_curve(target, probabilities)
    if not len(thresholds):
        raise ValueError("Threshold selection requires both target classes.")
    denominator = precision[:-1] + recall[:-1]
    f1_values = np.divide(
        2.0 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    best_index = int(np.argmax(f1_values))
    selected = float(thresholds[best_index])
    predictions = (probabilities >= selected).astype(int)
    return ThresholdSelection(
        threshold=selected,
        f1=float(f1_score(target, predictions, zero_division=0)),
        precision=float(precision[best_index]),
        recall=float(recall[best_index]),
    )
