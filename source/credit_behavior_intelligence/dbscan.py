"""DBSCAN density search and anomaly-exploration helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

from .config import DBSCANConfig, DEFAULT_DBSCAN_CONFIG


@dataclass(frozen=True)
class DBSCANResult:
    """DBSCAN assignments and density-structure summary."""

    model: DBSCAN
    labels: np.ndarray
    cluster_count: int
    noise_share: float
    minimum_cluster_share: float | None
    silhouette: float | None


def nearest_neighbor_distances(
    representation: np.ndarray,
    min_samples: int,
) -> np.ndarray:
    """Return sorted distances to the selected nearest neighbor."""

    neighbors = NearestNeighbors(n_neighbors=min_samples)
    distances, _ = neighbors.fit(representation).kneighbors(representation)
    return np.sort(distances[:, -1])


def fit_dbscan(
    representation: np.ndarray,
    config: DBSCANConfig = DEFAULT_DBSCAN_CONFIG,
) -> DBSCANResult:
    """Fit DBSCAN and summarize clusters and density-noise records."""

    values = np.asarray(representation)[:, : config.n_components]
    model = DBSCAN(eps=config.eps, min_samples=config.min_samples)
    labels = model.fit_predict(values)
    non_noise = labels != -1
    clusters = sorted(set(labels[non_noise]))
    cluster_count = len(clusters)
    noise_share = float((labels == -1).mean())

    minimum_cluster_share: float | None = None
    if cluster_count:
        counts = pd.Series(labels[non_noise]).value_counts()
        minimum_cluster_share = float(counts.min() / len(labels))

    silhouette: float | None = None
    if cluster_count >= 2 and non_noise.sum() > cluster_count:
        silhouette = float(silhouette_score(values[non_noise], labels[non_noise]))

    return DBSCANResult(
        model=model,
        labels=labels,
        cluster_count=cluster_count,
        noise_share=noise_share,
        minimum_cluster_share=minimum_cluster_share,
        silhouette=silhouette,
    )


def evaluate_dbscan_grid(
    representation: np.ndarray,
    eps_values: tuple[float, ...],
    min_samples_values: tuple[int, ...] = (10, 20, 40),
) -> pd.DataFrame:
    """Evaluate density configurations without using the supervised target."""

    rows = []
    for min_samples in min_samples_values:
        for eps in eps_values:
            result = fit_dbscan(
                representation,
                DBSCANConfig(
                    n_components=np.asarray(representation).shape[1],
                    eps=eps,
                    min_samples=min_samples,
                ),
            )
            rows.append(
                {
                    "eps": eps,
                    "min_samples": min_samples,
                    "clusters": result.cluster_count,
                    "noise_share": result.noise_share,
                    "minimum_cluster_share": result.minimum_cluster_share,
                    "silhouette": result.silhouette,
                }
            )
    return pd.DataFrame(rows)
