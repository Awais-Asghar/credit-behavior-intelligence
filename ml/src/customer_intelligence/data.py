"""Dataset loading, normalization, and validation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype

UCI_DATASET_ID = 350
TARGET_NAME = "default_next_month"

RAW_FEATURE_COLUMNS = tuple(f"X{index}" for index in range(1, 24))

FEATURE_NAME_MAP = {
    "X1": "credit_limit",
    "X2": "sex",
    "X3": "education",
    "X4": "marriage",
    "X5": "age",
    "X6": "repayment_status_sep",
    "X7": "repayment_status_aug",
    "X8": "repayment_status_jul",
    "X9": "repayment_status_jun",
    "X10": "repayment_status_may",
    "X11": "repayment_status_apr",
    "X12": "bill_statement_sep",
    "X13": "bill_statement_aug",
    "X14": "bill_statement_jul",
    "X15": "bill_statement_jun",
    "X16": "bill_statement_may",
    "X17": "bill_statement_apr",
    "X18": "previous_payment_sep",
    "X19": "previous_payment_aug",
    "X20": "previous_payment_jul",
    "X21": "previous_payment_jun",
    "X22": "previous_payment_may",
    "X23": "previous_payment_apr",
}

DEMOGRAPHIC_FEATURES = ("sex", "education", "marriage", "age")
BEHAVIORAL_FEATURES = (
    "credit_limit",
    "bill_statement_sep",
    "bill_statement_aug",
    "bill_statement_jul",
    "bill_statement_jun",
    "bill_statement_may",
    "bill_statement_apr",
    "previous_payment_sep",
    "previous_payment_aug",
    "previous_payment_jul",
    "previous_payment_jun",
    "previous_payment_may",
    "previous_payment_apr",
    "repayment_status_sep",
    "repayment_status_aug",
    "repayment_status_jul",
    "repayment_status_jun",
    "repayment_status_may",
    "repayment_status_apr",
)


class DatasetValidationError(ValueError):
    """Raised when downloaded data does not match the expected contract."""


@dataclass(frozen=True)
class DatasetExpectations:
    """Expected schema properties for a dataset validation run."""

    row_count: int = 30_000
    feature_count: int = 23
    target_values: frozenset[int] = frozenset({0, 1})
    require_no_missing_values: bool = True


@dataclass(frozen=True)
class DatasetBundle:
    """Normalized dataset and source metadata."""

    features: pd.DataFrame
    target: pd.Series
    ids: pd.DataFrame | None
    metadata: Any
    variables: pd.DataFrame | None
    validation: dict[str, object]


def _normalize_features(raw_features: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(raw_features, pd.DataFrame):
        raise DatasetValidationError("Features must be provided as a pandas DataFrame.")

    features = raw_features.copy()
    features.columns = [str(column) for column in features.columns]

    raw_columns = set(RAW_FEATURE_COLUMNS)
    semantic_columns = set(FEATURE_NAME_MAP.values())
    received_columns = set(features.columns)

    if received_columns == raw_columns:
        features = features.loc[:, RAW_FEATURE_COLUMNS].rename(columns=FEATURE_NAME_MAP)
    elif received_columns == semantic_columns:
        features = features.loc[:, tuple(FEATURE_NAME_MAP.values())]
    else:
        missing_raw = sorted(raw_columns - received_columns)
        unexpected = sorted(received_columns - raw_columns - semantic_columns)
        raise DatasetValidationError(
            "Feature columns do not match the UCI schema. "
            f"Missing raw columns: {missing_raw}. Unexpected columns: {unexpected}."
        )

    return features.reset_index(drop=True)


def _normalize_target(raw_target: pd.DataFrame | pd.Series) -> pd.Series:
    if isinstance(raw_target, pd.DataFrame):
        if raw_target.shape[1] != 1:
            raise DatasetValidationError("The target must contain exactly one column.")
        target = raw_target.iloc[:, 0]
    elif isinstance(raw_target, pd.Series):
        target = raw_target
    else:
        raise DatasetValidationError(
            "The target must be provided as a pandas DataFrame or Series."
        )

    try:
        numeric_target = pd.to_numeric(target, errors="raise")
    except (TypeError, ValueError) as error:
        raise DatasetValidationError("The target must contain numeric values.") from error

    non_missing_target = numeric_target.dropna()
    if not (non_missing_target % 1 == 0).all():
        raise DatasetValidationError(
            "The target must contain whole-number class labels."
        )

    normalized = numeric_target.astype("Int64")

    return normalized.rename(TARGET_NAME).reset_index(drop=True)


def validate_dataset(
    features: pd.DataFrame,
    target: pd.Series,
    ids: pd.DataFrame | None = None,
    expectations: DatasetExpectations | None = None,
) -> dict[str, object]:
    """Validate data and return an aggregate validation summary."""

    expected = expectations or DatasetExpectations()
    errors: list[str] = []

    if len(features) != expected.row_count:
        errors.append(
            f"Expected {expected.row_count} rows, received {len(features)} rows."
        )
    if features.shape[1] != expected.feature_count:
        errors.append(
            f"Expected {expected.feature_count} features, received {features.shape[1]}."
        )
    if len(target) != len(features):
        errors.append("Feature and target row counts do not match.")
    if features.columns.duplicated().any():
        errors.append("Feature names contain duplicates.")

    missing_feature_values = int(features.isna().sum().sum())
    missing_target_values = int(target.isna().sum())
    if expected.require_no_missing_values:
        if missing_feature_values:
            errors.append(f"Features contain {missing_feature_values} missing values.")
        if missing_target_values:
            errors.append(f"Target contains {missing_target_values} missing values.")

    observed_target_values = frozenset(int(value) for value in target.dropna().unique())
    if observed_target_values != expected.target_values:
        errors.append(
            "Target values do not match the expected binary classes. "
            f"Expected {sorted(expected.target_values)}, "
            f"received {sorted(observed_target_values)}."
        )

    non_numeric_features = [
        column for column in features.columns if not is_numeric_dtype(features[column])
    ]
    if non_numeric_features:
        errors.append(f"Non-numeric features found: {non_numeric_features}.")

    duplicate_id_count = 0
    if ids is not None:
        if len(ids) != len(features):
            errors.append("Identifier and feature row counts do not match.")
        duplicate_id_count = int(ids.duplicated().sum())
        if duplicate_id_count:
            errors.append(f"Identifiers contain {duplicate_id_count} duplicate rows.")

    if errors:
        raise DatasetValidationError(" ".join(errors))

    class_counts = {
        int(label): int(count)
        for label, count in target.value_counts().sort_index().items()
    }
    return {
        "row_count": int(len(features)),
        "feature_count": int(features.shape[1]),
        "missing_feature_values": missing_feature_values,
        "missing_target_values": missing_target_values,
        "duplicate_id_count": duplicate_id_count,
        "target_distribution": class_counts,
    }


def prepare_dataset(
    raw_features: pd.DataFrame,
    raw_target: pd.DataFrame | pd.Series,
    raw_ids: pd.DataFrame | None = None,
    metadata: Any = None,
    variables: pd.DataFrame | None = None,
    expectations: DatasetExpectations | None = None,
) -> DatasetBundle:
    """Normalize UCI data and enforce the project data contract."""

    features = _normalize_features(raw_features)
    target = _normalize_target(raw_target)
    ids = raw_ids.copy().reset_index(drop=True) if raw_ids is not None else None
    validation = validate_dataset(features, target, ids, expectations)

    return DatasetBundle(
        features=features,
        target=target,
        ids=ids,
        metadata=metadata,
        variables=variables.copy() if variables is not None else None,
        validation=validation,
    )


def fetch_credit_default_dataset() -> DatasetBundle:
    """Download UCI dataset 350 and return a validated, normalized bundle."""

    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as error:
        raise RuntimeError(
            "ucimlrepo is required. Install dependencies from ml/requirements.txt."
        ) from error

    dataset = fetch_ucirepo(id=UCI_DATASET_ID)
    return prepare_dataset(
        raw_features=dataset.data.features,
        raw_target=dataset.data.targets,
        raw_ids=getattr(dataset.data, "ids", None),
        metadata=dataset.metadata,
        variables=dataset.variables,
    )
