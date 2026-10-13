"""Histogram-gradient-boosting construction and candidate comparison."""

from __future__ import annotations

from dataclasses import asdict
from typing import Iterable

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

from .config import BoostingConfig, DEFAULT_BOOSTING_CONFIG, RANDOM_STATE


DEFAULT_CANDIDATES = (
    BoostingConfig(learning_rate=0.01, max_leaf_nodes=15),
    BoostingConfig(learning_rate=0.03, max_leaf_nodes=15),
    BoostingConfig(learning_rate=0.05, max_leaf_nodes=15),
    BoostingConfig(learning_rate=0.08, max_leaf_nodes=15),
    BoostingConfig(learning_rate=0.03, max_leaf_nodes=31),
    BoostingConfig(learning_rate=0.05, max_leaf_nodes=31),
    BoostingConfig(learning_rate=0.08, max_leaf_nodes=31),
    BoostingConfig(learning_rate=0.03, max_leaf_nodes=63),
    BoostingConfig(learning_rate=0.05, max_leaf_nodes=63),
    BoostingConfig(learning_rate=0.08, max_leaf_nodes=63),
)


def build_classifier(
    config: BoostingConfig = DEFAULT_BOOSTING_CONFIG,
) -> HistGradientBoostingClassifier:
    """Build a classifier with explicit internal early stopping."""

    return HistGradientBoostingClassifier(
        learning_rate=config.learning_rate,
        max_iter=config.max_iter,
        max_leaf_nodes=config.max_leaf_nodes,
        min_samples_leaf=config.min_samples_leaf,
        l2_regularization=config.l2_regularization,
        early_stopping=True,
        validation_fraction=config.validation_fraction,
        n_iter_no_change=config.n_iter_no_change,
        tol=config.tol,
        scoring="loss",
        random_state=config.random_state,
    )


def compare_candidates(
    features: pd.DataFrame,
    target: pd.Series,
    candidates: Iterable[BoostingConfig] = DEFAULT_CANDIDATES,
    *,
    folds: int = 3,
) -> pd.DataFrame:
    """Rank candidate configurations using training-only cross-validation."""

    cross_validation = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    rows = []
    for index, config in enumerate(candidates, start=1):
        scores = cross_validate(
            build_classifier(config),
            features,
            target,
            cv=cross_validation,
            scoring={
                "pr_auc": "average_precision",
                "roc_auc": "roc_auc",
                "f1": "f1",
                "balanced_accuracy": "balanced_accuracy",
            },
            n_jobs=-1,
            return_train_score=False,
        )
        row = {"candidate": index, **asdict(config)}
        for metric in ("pr_auc", "roc_auc", "f1", "balanced_accuracy"):
            row[f"mean_{metric}"] = float(scores[f"test_{metric}"].mean())
            row[f"std_{metric}"] = float(scores[f"test_{metric}"].std())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mean_pr_auc", ascending=False)
