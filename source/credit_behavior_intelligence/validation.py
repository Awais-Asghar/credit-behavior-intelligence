"""Data-quality and category-audit helpers."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


DOCUMENTED_CODES: dict[str, set[int]] = {
    "sex": {1, 2},
    "education": {1, 2, 3, 4},
    "marriage": {1, 2, 3},
}


def data_quality_summary(features: pd.DataFrame, target: pd.Series) -> pd.Series:
    """Return aggregate quality checks without exposing individual records."""

    combined = pd.concat([features.reset_index(drop=True), target.reset_index(drop=True)], axis=1)
    bill_columns = features.filter(like="bill_statement_")
    payment_columns = features.filter(like="previous_payment_")
    return pd.Series(
        {
            "rows": len(features),
            "features": features.shape[1],
            "missing_feature_values": int(features.isna().sum().sum()),
            "missing_target_values": int(target.isna().sum()),
            "duplicate_feature_profiles": int(features.duplicated().sum()),
            "duplicate_feature_and_target_rows": int(combined.duplicated().sum()),
            "minimum_bill_balance": float(bill_columns.min().min()),
            "maximum_bill_balance": float(bill_columns.max().max()),
            "maximum_previous_payment": float(payment_columns.max().max()),
        },
        name="value",
    )


def audit_category_codes(
    features: pd.DataFrame,
    documented_codes: Mapping[str, set[int]] | None = None,
) -> pd.DataFrame:
    """Compare observed categorical codes with the published documentation."""

    expected = dict(documented_codes or DOCUMENTED_CODES)
    repayment_columns = [
        column for column in features if column.startswith("repayment_status_")
    ]
    for column in repayment_columns:
        expected.setdefault(column, {-1, 1, 2, 3, 4, 5, 6, 7, 8, 9})

    rows: list[dict[str, object]] = []
    for column, documented in expected.items():
        if column not in features:
            raise ValueError(f"Missing category-audit feature: {column}.")
        observed = {int(value) for value in features[column].dropna().unique()}
        rows.append(
            {
                "feature": column,
                "observed_codes": sorted(observed),
                "observed_but_undocumented": sorted(observed - documented),
                "documented_but_not_observed": sorted(documented - observed),
            }
        )
    return pd.DataFrame(rows).set_index("feature")


def conflicting_profile_count(features: pd.DataFrame, target: pd.Series) -> int:
    """Count exact feature profiles that appear with both target labels."""

    combined = features.copy()
    combined["__target__"] = target.to_numpy()
    unique_targets = combined.groupby(list(features.columns), dropna=False)[
        "__target__"
    ].nunique()
    return int((unique_targets > 1).sum())
