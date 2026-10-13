"""Reusable code for the Credit Behavior Intelligence project."""

from .artifacts import ARTIFACT_VERSION, ModelArtifact, load_artifact, save_artifact
from .calibration import ThresholdSelection, calibrate_classifier, select_f1_threshold
from .config import (
    DEFAULT_BOOSTING_CONFIG,
    DEFAULT_DBSCAN_CONFIG,
    DEFAULT_KMEANS_CONFIG,
    DEFAULT_PCA_CONFIG,
    DEFAULT_SPLIT_CONFIG,
    BoostingConfig,
    DBSCANConfig,
    KMeansConfig,
    PCAConfig,
    SplitConfig,
)

from .data import (
    BEHAVIORAL_FEATURES,
    DEMOGRAPHIC_FEATURES,
    DatasetBundle,
    DatasetExpectations,
    DatasetValidationError,
    fetch_credit_default_dataset,
    prepare_dataset,
    validate_dataset,
)
from .features import FEATURE_ENGINEERING_VERSION, engineer_behavioral_features
from .evaluation import classification_metrics, measure_batch_latency, subgroup_metrics
from .splitting import SupervisedSplit, make_supervised_split

__all__ = [
    "BEHAVIORAL_FEATURES",
    "DEMOGRAPHIC_FEATURES",
    "DatasetBundle",
    "DatasetExpectations",
    "DatasetValidationError",
    "fetch_credit_default_dataset",
    "prepare_dataset",
    "validate_dataset",
    "FEATURE_ENGINEERING_VERSION",
    "engineer_behavioral_features",
    "ARTIFACT_VERSION",
    "ModelArtifact",
    "load_artifact",
    "save_artifact",
    "ThresholdSelection",
    "calibrate_classifier",
    "select_f1_threshold",
    "BoostingConfig",
    "DBSCANConfig",
    "KMeansConfig",
    "PCAConfig",
    "SplitConfig",
    "DEFAULT_BOOSTING_CONFIG",
    "DEFAULT_DBSCAN_CONFIG",
    "DEFAULT_KMEANS_CONFIG",
    "DEFAULT_PCA_CONFIG",
    "DEFAULT_SPLIT_CONFIG",
    "classification_metrics",
    "measure_batch_latency",
    "subgroup_metrics",
    "SupervisedSplit",
    "make_supervised_split",
]
