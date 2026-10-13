"""Stratified splitting and leakage-audit utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DEFAULT_SPLIT_CONFIG, SplitConfig


@dataclass(frozen=True)
class SupervisedSplit:
    """Training, validation, and test partitions with optional identifiers."""

    x_train: pd.DataFrame
    x_validation: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series
    ids_train: pd.DataFrame | None = None
    ids_validation: pd.DataFrame | None = None
    ids_test: pd.DataFrame | None = None


def make_supervised_split(
    features: pd.DataFrame,
    target: pd.Series,
    identifiers: pd.DataFrame | None = None,
    config: SplitConfig = DEFAULT_SPLIT_CONFIG,
) -> SupervisedSplit:
    """Create the fixed 80/10/10 shuffled and target-stratified split."""

    indices = np.arange(len(features))
    train_indices, holdout_indices = train_test_split(
        indices,
        train_size=config.train_size,
        stratify=target,
        random_state=config.random_state,
        shuffle=True,
    )
    validation_fraction = config.validation_size / (
        config.validation_size + config.test_size
    )
    validation_indices, test_indices = train_test_split(
        holdout_indices,
        train_size=validation_fraction,
        stratify=target.iloc[holdout_indices],
        random_state=config.random_state,
        shuffle=True,
    )

    def frame_rows(frame: pd.DataFrame, selected: np.ndarray) -> pd.DataFrame:
        return frame.iloc[selected].reset_index(drop=True)

    def target_rows(selected: np.ndarray) -> pd.Series:
        return target.iloc[selected].reset_index(drop=True)

    return SupervisedSplit(
        x_train=frame_rows(features, train_indices),
        x_validation=frame_rows(features, validation_indices),
        x_test=frame_rows(features, test_indices),
        y_train=target_rows(train_indices),
        y_validation=target_rows(validation_indices),
        y_test=target_rows(test_indices),
        ids_train=frame_rows(identifiers, train_indices) if identifiers is not None else None,
        ids_validation=(
            frame_rows(identifiers, validation_indices)
            if identifiers is not None
            else None
        ),
        ids_test=frame_rows(identifiers, test_indices) if identifiers is not None else None,
    )


def identifier_overlap(left: pd.DataFrame, right: pd.DataFrame) -> int:
    """Count identifier rows shared by two partitions."""

    left_keys = set(map(tuple, left.to_numpy()))
    right_keys = set(map(tuple, right.to_numpy()))
    return len(left_keys & right_keys)


def standardized_mean_difference(
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> pd.Series:
    """Measure numeric distribution differences between two partitions."""

    left_numeric = left.select_dtypes(include="number")
    right_numeric = right.loc[:, left_numeric.columns]
    pooled_variance = (left_numeric.var(ddof=1) + right_numeric.var(ddof=1)) / 2.0
    denominator = np.sqrt(pooled_variance).replace(0.0, np.nan)
    difference = (left_numeric.mean() - right_numeric.mean()).abs() / denominator
    return difference.fillna(0.0).sort_values(ascending=False)


def split_summary(split: SupervisedSplit) -> pd.DataFrame:
    """Return row counts and target prevalence for each partition."""

    rows = []
    for name, features, target in (
        ("train", split.x_train, split.y_train),
        ("validation", split.x_validation, split.y_validation),
        ("test", split.x_test, split.y_test),
    ):
        rows.append(
            {
                "partition": name,
                "rows": len(features),
                "positive_rate": float(target.mean()),
            }
        )
    return pd.DataFrame(rows).set_index("partition")
