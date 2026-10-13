"""Behavioral feature engineering shared by training and inference."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import BEHAVIORAL_FEATURES

FEATURE_ENGINEERING_VERSION = "behavioral-v2"
MONTHS = ("sep", "aug", "jul", "jun", "may", "apr")
REPAYMENT_FEATURES = tuple(f"repayment_status_{month}" for month in MONTHS)
BILL_FEATURES = tuple(f"bill_statement_{month}" for month in MONTHS)
PAYMENT_FEATURES = tuple(f"previous_payment_{month}" for month in MONTHS)


def engineer_behavioral_features(features: pd.DataFrame) -> pd.DataFrame:
    """Create deterministic repayment, utilization, bill, and payment features."""

    missing = sorted(set(BEHAVIORAL_FEATURES) - set(features.columns))
    if missing:
        raise ValueError(f"Missing behavioral features: {missing}.")

    engineered = features.loc[:, BEHAVIORAL_FEATURES].astype(float).copy()
    credit_limit = engineered["credit_limit"].clip(lower=1.0)
    bill_values = engineered.loc[:, BILL_FEATURES]
    payment_values = engineered.loc[:, PAYMENT_FEATURES]
    repayment_values = engineered.loc[:, REPAYMENT_FEATURES]

    for month, bill_feature, payment_feature in zip(
        MONTHS,
        BILL_FEATURES,
        PAYMENT_FEATURES,
        strict=True,
    ):
        engineered[f"bill_to_limit_{month}"] = (
            engineered[bill_feature] / credit_limit
        ).clip(-5.0, 5.0)
        engineered[f"payment_to_limit_{month}"] = (
            engineered[payment_feature] / credit_limit
        ).clip(0.0, 5.0)
        bill_denominator = engineered[bill_feature].abs().clip(lower=100.0)
        engineered[f"payment_to_bill_{month}"] = (
            engineered[payment_feature] / bill_denominator
        ).clip(0.0, 10.0)

    engineered["bill_mean"] = bill_values.mean(axis=1)
    engineered["bill_std"] = bill_values.std(axis=1)
    engineered["bill_range"] = bill_values.max(axis=1) - bill_values.min(axis=1)
    engineered["bill_recent_change"] = (
        engineered["bill_statement_sep"] - engineered["bill_statement_apr"]
    )
    engineered["mean_bill_to_limit"] = (
        bill_values.mean(axis=1) / credit_limit
    ).clip(-5.0, 5.0)
    engineered["max_bill_to_limit"] = (
        bill_values.max(axis=1) / credit_limit
    ).clip(-5.0, 5.0)

    engineered["payment_sum"] = payment_values.sum(axis=1)
    engineered["payment_mean"] = payment_values.mean(axis=1)
    engineered["payment_std"] = payment_values.std(axis=1)
    engineered["zero_payment_months"] = (payment_values <= 0).sum(axis=1)
    positive_bill_total = bill_values.clip(lower=0.0).sum(axis=1).clip(lower=100.0)
    engineered["total_payment_to_bill"] = (
        payment_values.sum(axis=1) / positive_bill_total
    ).clip(0.0, 10.0)

    engineered["repayment_mean"] = repayment_values.mean(axis=1)
    engineered["repayment_max"] = repayment_values.max(axis=1)
    engineered["repayment_recent_change"] = (
        engineered["repayment_status_sep"]
        - engineered["repayment_status_apr"]
    )
    engineered["delayed_months"] = (repayment_values > 0).sum(axis=1)
    engineered["serious_delay_months"] = (repayment_values >= 2).sum(axis=1)
    engineered["severe_delay_months"] = (repayment_values >= 3).sum(axis=1)

    if not np.isfinite(engineered.to_numpy()).all():
        raise ValueError("Engineered behavioral features contain non-finite values.")

    return engineered
