"""Central configuration for the Credit Behavior Intelligence workflow."""

from __future__ import annotations

from dataclasses import dataclass


RANDOM_STATE = 42


@dataclass(frozen=True)
class SplitConfig:
    """Sizes and seed for the stratified supervised split."""

    train_size: float = 0.80
    validation_size: float = 0.10
    test_size: float = 0.10
    random_state: int = RANDOM_STATE

    def __post_init__(self) -> None:
        total = self.train_size + self.validation_size + self.test_size
        if abs(total - 1.0) > 1e-9:
            raise ValueError("Train, validation, and test sizes must sum to 1.0.")
        if min(self.train_size, self.validation_size, self.test_size) <= 0:
            raise ValueError("Every split size must be positive.")


@dataclass(frozen=True)
class PCAConfig:
    """Configuration for leakage-safe principal component analysis."""

    variance_threshold: float = 0.95
    random_state: int = RANDOM_STATE


@dataclass(frozen=True)
class KMeansConfig:
    """Selected K-Means configuration from the notebook analysis."""

    n_components: int = 10
    n_clusters: int = 2
    n_init: int = 20
    random_state: int = RANDOM_STATE


@dataclass(frozen=True)
class DBSCANConfig:
    """Selected DBSCAN anomaly-exploration configuration."""

    n_components: int = 5
    eps: float = 1.1526
    min_samples: int = 10
    whiten: bool = True


@dataclass(frozen=True)
class BoostingConfig:
    """Selected histogram-gradient-boosting configuration."""

    learning_rate: float = 0.05
    max_iter: int = 600
    max_leaf_nodes: int = 15
    min_samples_leaf: int = 20
    l2_regularization: float = 1.0
    validation_fraction: float = 0.10
    n_iter_no_change: int = 30
    tol: float = 1e-7
    random_state: int = RANDOM_STATE


DEFAULT_SPLIT_CONFIG = SplitConfig()
DEFAULT_PCA_CONFIG = PCAConfig()
DEFAULT_KMEANS_CONFIG = KMeansConfig()
DEFAULT_DBSCAN_CONFIG = DBSCANConfig()
DEFAULT_BOOSTING_CONFIG = BoostingConfig()
