"""Reusable plotting functions corresponding to the notebook figures."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve


def plot_explained_variance(model):
    """Plot per-component and cumulative PCA explained variance."""

    components = np.arange(1, model.n_components_ + 1)
    cumulative = np.cumsum(model.explained_variance_ratio_)
    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.bar(components, model.explained_variance_ratio_, alpha=0.55, label="Individual")
    axis.plot(components, cumulative, marker="o", label="Cumulative")
    axis.set(xlabel="Principal component", ylabel="Explained variance")
    axis.legend()
    figure.tight_layout()
    return figure


def plot_loss_curves(estimators):
    """Plot training and internal-validation log loss for fitted estimators."""

    figure, axes = plt.subplots(1, len(estimators), figsize=(5 * len(estimators), 4))
    axes = np.atleast_1d(axes)
    for index, (axis, estimator) in enumerate(zip(axes, estimators, strict=True), start=1):
        axis.plot(estimator.train_score_, label="Training loss")
        axis.plot(estimator.validation_score_, label="Validation loss")
        axis.set(title=f"Fold {index}", xlabel="Boosting iteration", ylabel="Score")
        axis.legend()
    figure.tight_layout()
    return figure


def plot_calibration(target, probabilities, bins: int = 10):
    """Plot observed positive frequency against predicted probability."""

    observed, predicted = calibration_curve(target, probabilities, n_bins=bins)
    figure, axis = plt.subplots(figsize=(5, 5))
    axis.plot([0, 1], [0, 1], linestyle="--", color="black", label="Ideal")
    axis.plot(predicted, observed, marker="o", label="Model")
    axis.set(xlabel="Mean predicted probability", ylabel="Observed positive rate")
    axis.legend()
    figure.tight_layout()
    return figure


def plot_confusion_and_roc(target, probabilities, threshold: float):
    """Plot the confusion matrix and ROC curve for a fixed test evaluation."""

    predictions = (probabilities >= threshold).astype(int)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ConfusionMatrixDisplay(
        confusion_matrix(target, predictions, labels=[0, 1]),
        display_labels=["No default", "Default"],
    ).plot(ax=axes[0], colorbar=False)
    false_positive_rate, true_positive_rate, _ = roc_curve(target, probabilities)
    axes[1].plot(false_positive_rate, true_positive_rate)
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="black")
    axes[1].set(xlabel="False positive rate", ylabel="True positive rate")
    figure.tight_layout()
    return figure
