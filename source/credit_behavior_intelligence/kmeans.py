"""K-Means evaluation, selection, and stability analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from .config import DEFAULT_KMEANS_CONFIG, KMeansConfig


@dataclass(frozen=True)
class KMeansResult:
    """Fitted K-Means model and descriptive quality measurements."""

    model: KMeans
    labels: np.ndarray
    silhouette: float
    calinski_harabasz: float
    davies_bouldin: float
    minimum_cluster_share: float


def fit_kmeans(
    representation: np.ndarray,
    config: KMeansConfig = DEFAULT_KMEANS_CONFIG,
    *,
    silhouette_sample_size: int = 8_000,
) -> KMeansResult:
    """Fit the selected behavioral segmentation configuration."""

    values = np.asarray(representation)[:, : config.n_components]
    model = KMeans(
        n_clusters=config.n_clusters,
        n_init=config.n_init,
        random_state=config.random_state,
    )
    labels = model.fit_predict(values)
    counts = np.bincount(labels, minlength=config.n_clusters)
    return KMeansResult(
        model=model,
        labels=labels,
        silhouette=float(
            silhouette_score(
                values,
                labels,
                sample_size=min(silhouette_sample_size, len(values)),
                random_state=config.random_state,
            )
        ),
        calinski_harabasz=float(calinski_harabasz_score(values, labels)),
        davies_bouldin=float(davies_bouldin_score(values, labels)),
        minimum_cluster_share=float(counts.min() / counts.sum()),
    )


def initialization_stability(
    representation: np.ndarray,
    config: KMeansConfig = DEFAULT_KMEANS_CONFIG,
    seeds: tuple[int, ...] = (11, 23, 42, 71, 101),
) -> float:
    """Measure agreement among repeated initializations with adjusted Rand index."""

    values = np.asarray(representation)[:, : config.n_components]
    assignments = [
        KMeans(
            n_clusters=config.n_clusters,
            n_init=config.n_init,
            random_state=seed,
        ).fit_predict(values)
        for seed in seeds
    ]
    scores = [
        adjusted_rand_score(assignments[left], assignments[right])
        for left in range(len(assignments))
        for right in range(left + 1, len(assignments))
    ]
    return float(np.mean(scores))


def evaluate_cluster_counts(
    representation: np.ndarray,
    cluster_counts: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8),
    random_state: int = 42,
) -> pd.DataFrame:
    """Compare common internal clustering metrics across cluster counts."""

    rows = []
    for cluster_count in cluster_counts:
        config = KMeansConfig(
            n_components=np.asarray(representation).shape[1],
            n_clusters=cluster_count,
            random_state=random_state,
        )
        result = fit_kmeans(representation, config)
        rows.append(
            {
                "clusters": cluster_count,
                "inertia": float(result.model.inertia_),
                "silhouette": result.silhouette,
                "calinski_harabasz": result.calinski_harabasz,
                "davies_bouldin": result.davies_bouldin,
                "minimum_cluster_share": result.minimum_cluster_share,
            }
        )
    return pd.DataFrame(rows)
