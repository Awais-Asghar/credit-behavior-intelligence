"""Tests for the behavioral feature-engineering contract."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRECTORY = PROJECT_ROOT / "src"
if str(SOURCE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIRECTORY))

from customer_intelligence import (  # noqa: E402
    BEHAVIORAL_FEATURES,
    engineer_behavioral_features,
)


class BehavioralFeatureEngineeringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.features = pd.DataFrame(
            {
                "credit_limit": [100_000, 200_000],
                **{
                    f"bill_statement_{month}": [10_000 + index, -500 + index]
                    for index, month in enumerate(
                        ("sep", "aug", "jul", "jun", "may", "apr")
                    )
                },
                **{
                    f"previous_payment_{month}": [2_000 + index, 0]
                    for index, month in enumerate(
                        ("sep", "aug", "jul", "jun", "may", "apr")
                    )
                },
                **{
                    f"repayment_status_{month}": [index % 3, -1]
                    for index, month in enumerate(
                        ("sep", "aug", "jul", "jun", "may", "apr")
                    )
                },
            }
        )

    def test_engineering_is_finite_and_preserves_rows(self) -> None:
        engineered = engineer_behavioral_features(self.features)

        self.assertEqual(len(engineered), len(self.features))
        self.assertEqual(engineered.shape[1], 54)
        self.assertTrue(np.isfinite(engineered.to_numpy()).all())

    def test_engineering_keeps_raw_behavioral_features(self) -> None:
        engineered = engineer_behavioral_features(self.features)

        self.assertEqual(
            list(engineered.columns[: len(BEHAVIORAL_FEATURES)]),
            list(BEHAVIORAL_FEATURES),
        )
        self.assertNotIn("sex", engineered.columns)
        self.assertNotIn("age", engineered.columns)

    def test_missing_behavioral_feature_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "Missing behavioral features"):
            engineer_behavioral_features(self.features.drop(columns="credit_limit"))


if __name__ == "__main__":
    unittest.main()
