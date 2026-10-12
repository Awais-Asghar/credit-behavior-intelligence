"""Reusable machine learning code for the Customer Intelligence project."""

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
]
