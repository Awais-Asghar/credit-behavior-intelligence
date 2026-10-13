"""Leakage-safe preprocessing for PCA and distance-based models."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, RobustScaler, StandardScaler

from .data import BEHAVIORAL_FEATURES
from .features import BILL_FEATURES, PAYMENT_FEATURES, REPAYMENT_FEATURES


MONETARY_FEATURES = ("credit_limit", *BILL_FEATURES, *PAYMENT_FEATURES)


def make_behavioral_preprocessor(strategy: str = "mixed") -> ColumnTransformer:
    """Create one of the preprocessing strategies compared in the notebook."""

    if strategy == "standard":
        return ColumnTransformer(
            [("all", StandardScaler(), list(BEHAVIORAL_FEATURES))],
            verbose_feature_names_out=False,
        )
    if strategy == "robust":
        return ColumnTransformer(
            [("all", RobustScaler(), list(BEHAVIORAL_FEATURES))],
            verbose_feature_names_out=False,
        )
    if strategy == "mixed":
        return ColumnTransformer(
            [
                (
                    "money",
                    Pipeline(
                        [
                            (
                                "power",
                                PowerTransformer(
                                    method="yeo-johnson",
                                    standardize=True,
                                ),
                            )
                        ]
                    ),
                    list(MONETARY_FEATURES),
                ),
                ("repayment", StandardScaler(), list(REPAYMENT_FEATURES)),
            ],
            verbose_feature_names_out=False,
        )
    raise ValueError("Strategy must be 'standard', 'robust', or 'mixed'.")
