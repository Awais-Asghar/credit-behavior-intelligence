"""Tests for the modularized notebook workflow."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from credit_behavior_intelligence.calibration import select_f1_threshold
from credit_behavior_intelligence.config import (
    BoostingConfig,
    DBSCANConfig,
    KMeansConfig,
    PCAConfig,
    SplitConfig,
)
from credit_behavior_intelligence.data import BEHAVIORAL_FEATURES
from credit_behavior_intelligence.dbscan import fit_dbscan
from credit_behavior_intelligence.evaluation import classification_metrics
from credit_behavior_intelligence.kmeans import fit_kmeans, initialization_stability
from credit_behavior_intelligence.modeling import build_classifier
from credit_behavior_intelligence.pca import fit_pca, transform_pca
from credit_behavior_intelligence.splitting import (
    identifier_overlap,
    make_supervised_split,
    split_summary,
)


def behavioral_frame(rows: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    values: dict[str, np.ndarray] = {}
    for column in BEHAVIORAL_FEATURES:
        if column == "credit_limit":
            values[column] = rng.integers(20_000, 500_000, size=rows)
        elif column.startswith("repayment_status_"):
            values[column] = rng.integers(-2, 5, size=rows)
        elif column.startswith("bill_statement_"):
            values[column] = rng.normal(50_000, 25_000, size=rows)
        else:
            values[column] = rng.gamma(2.0, 4_000, size=rows)
    return pd.DataFrame(values)


class PipelineComponentTests(unittest.TestCase):
    def test_stratified_split_has_expected_sizes_and_no_id_overlap(self) -> None:
        features = behavioral_frame(200)
        target = pd.Series([0] * 150 + [1] * 50)
        identifiers = pd.DataFrame({"id": np.arange(200)})
        split = make_supervised_split(
            features,
            target,
            identifiers,
            SplitConfig(random_state=7),
        )

        summary = split_summary(split)
        self.assertEqual(summary["rows"].to_dict(), {"train": 160, "validation": 20, "test": 20})
        self.assertEqual(identifier_overlap(split.ids_train, split.ids_validation), 0)
        self.assertEqual(identifier_overlap(split.ids_train, split.ids_test), 0)
        self.assertTrue(np.allclose(summary["positive_rate"], 0.25))

    def test_pca_transform_uses_training_fitted_pipeline(self) -> None:
        features = behavioral_frame(140)
        result = fit_pca(features.iloc[:100], config=PCAConfig(variance_threshold=0.90))
        transformed = transform_pca(result, features.iloc[100:])

        self.assertEqual(transformed.shape[0], 40)
        self.assertEqual(transformed.shape[1], result.model.n_components_)
        self.assertGreaterEqual(result.cumulative_variance[-1], 0.90)

    def test_kmeans_and_dbscan_return_descriptive_summaries(self) -> None:
        rng = np.random.default_rng(42)
        first = rng.normal(-2.0, 0.25, size=(80, 3))
        second = rng.normal(2.0, 0.25, size=(80, 3))
        representation = np.vstack([first, second])

        config = KMeansConfig(n_components=3, n_clusters=2, n_init=10)
        result = fit_kmeans(representation, config, silhouette_sample_size=160)
        stability = initialization_stability(representation, config, seeds=(1, 2, 3))
        density = fit_dbscan(
            representation,
            DBSCANConfig(n_components=3, eps=0.8, min_samples=5),
        )

        self.assertEqual(len(result.labels), 160)
        self.assertGreater(result.silhouette, 0.8)
        self.assertGreater(stability, 0.95)
        self.assertGreaterEqual(density.cluster_count, 2)

    def test_threshold_selection_and_metrics_are_consistent(self) -> None:
        target = np.array([0, 0, 0, 1, 1, 1])
        probabilities = np.array([0.05, 0.20, 0.40, 0.35, 0.70, 0.90])
        selection = select_f1_threshold(target, probabilities)
        metrics = classification_metrics(target, probabilities, selection.threshold)

        self.assertGreaterEqual(selection.threshold, 0.0)
        self.assertLessEqual(selection.threshold, 1.0)
        self.assertAlmostEqual(selection.f1, metrics["f1"])
        self.assertEqual(
            metrics["true_negatives"]
            + metrics["false_positives"]
            + metrics["false_negatives"]
            + metrics["true_positives"],
            len(target),
        )

    def test_boosting_builder_exposes_early_stopping_configuration(self) -> None:
        config = BoostingConfig(learning_rate=0.03, n_iter_no_change=12)
        model = build_classifier(config)

        self.assertEqual(model.learning_rate, 0.03)
        self.assertTrue(model.early_stopping)
        self.assertEqual(model.n_iter_no_change, 12)
        self.assertEqual(model.scoring, "loss")


if __name__ == "__main__":
    unittest.main()
