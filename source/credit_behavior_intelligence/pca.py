"""Principal component analysis helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA

from .config import DEFAULT_PCA_CONFIG, PCAConfig
from .preprocessing import make_behavioral_preprocessor


@dataclass(frozen=True)
class PCAResult:
    """Fitted preprocessing, PCA model, and transformed training scores."""

    preprocessor: ColumnTransformer
    model: PCA
    scores: np.ndarray

    @property
    def cumulative_variance(self) -> np.ndarray:
        return np.cumsum(self.model.explained_variance_ratio_)


def fit_pca(
    training_features: pd.DataFrame,
    strategy: str = "mixed",
    config: PCAConfig = DEFAULT_PCA_CONFIG,
    *,
    whiten: bool = False,
) -> PCAResult:
    """Fit preprocessing and PCA using training data only."""

    preprocessor = make_behavioral_preprocessor(strategy)
    transformed = preprocessor.fit_transform(training_features)
    model = PCA(
        n_components=config.variance_threshold,
        whiten=whiten,
        svd_solver="full",
        random_state=config.random_state,
    )
    scores = model.fit_transform(transformed)
    return PCAResult(preprocessor=preprocessor, model=model, scores=scores)


def transform_pca(result: PCAResult, features: pd.DataFrame) -> np.ndarray:
    """Transform new records with training-fitted preprocessing and PCA."""

    return result.model.transform(result.preprocessor.transform(features))


def component_summary(result: PCAResult) -> pd.DataFrame:
    """Return explained and cumulative variance for every retained component."""

    return pd.DataFrame(
        {
            "component": np.arange(1, result.model.n_components_ + 1),
            "explained_variance": result.model.explained_variance_ratio_,
            "cumulative_variance": result.cumulative_variance,
        }
    )
