"""Unit tests for dataset normalization and validation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT))

from credit_behavior_intelligence.data import (
    DatasetExpectations,
    DatasetValidationError,
    FEATURE_NAME_MAP,
    prepare_dataset,
)


class DataValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.features = pd.DataFrame(
            {
                raw_name: [index, index + 1, index + 2, index + 3]
                for index, raw_name in enumerate(FEATURE_NAME_MAP, start=1)
            }
        )
        self.target = pd.DataFrame({"Y": [0, 1, 0, 1]})
        self.ids = pd.DataFrame({"ID": [101, 102, 103, 104]})
        self.expectations = DatasetExpectations(row_count=4)

    def test_prepare_dataset_normalizes_and_validates_schema(self) -> None:
        dataset = prepare_dataset(
            self.features,
            self.target,
            self.ids,
            expectations=self.expectations,
        )

        self.assertEqual(dataset.features.shape, (4, 23))
        self.assertEqual(dataset.target.name, "default_next_month")
        self.assertEqual(dataset.validation["duplicate_id_count"], 0)
        self.assertEqual(dataset.validation["target_distribution"], {0: 2, 1: 2})
        self.assertIn("credit_limit", dataset.features.columns)
        self.assertNotIn("X1", dataset.features.columns)

    def test_missing_values_fail_validation(self) -> None:
        self.features.loc[0, "X1"] = None

        with self.assertRaisesRegex(DatasetValidationError, "missing values"):
            prepare_dataset(
                self.features,
                self.target,
                self.ids,
                expectations=self.expectations,
            )

    def test_duplicate_identifiers_fail_validation(self) -> None:
        self.ids.loc[3, "ID"] = 103

        with self.assertRaisesRegex(DatasetValidationError, "duplicate"):
            prepare_dataset(
                self.features,
                self.target,
                self.ids,
                expectations=self.expectations,
            )

    def test_unexpected_target_values_fail_validation(self) -> None:
        invalid_target = pd.DataFrame({"Y": [0, 1, 2, 1]})

        with self.assertRaisesRegex(DatasetValidationError, "Target values"):
            prepare_dataset(
                self.features,
                invalid_target,
                self.ids,
                expectations=self.expectations,
            )

    # Value: protects=fractional class labels are rejected without truncation;
    # fails_when=target normalization converts decimals into integer classes;
    # why_new=existing target tests cover unexpected integers only; seam=none;
    # no_claim=upstream dataset accuracy.
    def test_fractional_target_is_not_silently_truncated(self) -> None:
        invalid_target = pd.DataFrame({"Y": [0, 1, 0.5, 1]})

        with self.assertRaisesRegex(DatasetValidationError, "whole-number"):
            prepare_dataset(
                self.features,
                invalid_target,
                self.ids,
                expectations=self.expectations,
            )

    # Value: protects=the exact 23-column UCI input contract;
    # fails_when=missing or unexpected columns pass normalization;
    # why_new=existing tests cover only the valid raw schema; seam=none;
    # no_claim=semantic correctness of UCI values.
    def test_unexpected_feature_schema_fails_validation(self) -> None:
        invalid_features = self.features.rename(columns={"X23": "unexpected"})

        with self.assertRaisesRegex(DatasetValidationError, "UCI schema"):
            prepare_dataset(
                invalid_features,
                self.target,
                self.ids,
                expectations=self.expectations,
            )

    # Value: protects=model inputs remain numeric after normalization;
    # fails_when=text values pass dataset validation; why_new=no numeric-type
    # error path was exercised; seam=none; no_claim=plausible numeric ranges.
    def test_non_numeric_feature_fails_validation(self) -> None:
        invalid_features = self.features.copy()
        invalid_features["X1"] = invalid_features["X1"].astype(object)
        invalid_features.loc[0, "X1"] = "not-numeric"

        with self.assertRaisesRegex(DatasetValidationError, "Non-numeric"):
            prepare_dataset(
                invalid_features,
                self.target,
                self.ids,
                expectations=self.expectations,
            )

    # Value: protects=identifier rows remain aligned with feature rows;
    # fails_when=misaligned identifiers pass validation; why_new=duplicate-ID
    # coverage does not check row alignment; seam=none; no_claim=ID provenance.
    def test_identifier_row_mismatch_fails_validation(self) -> None:
        with self.assertRaisesRegex(DatasetValidationError, "Identifier and feature"):
            prepare_dataset(
                self.features,
                self.target,
                self.ids.iloc[:-1],
                expectations=self.expectations,
            )


if __name__ == "__main__":
    unittest.main()
