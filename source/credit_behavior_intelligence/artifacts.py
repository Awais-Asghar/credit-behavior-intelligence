"""Versioned model artifact persistence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib


ARTIFACT_VERSION = "credit-behavior-intelligence-v1"


@dataclass(frozen=True)
class ModelArtifact:
    """Serializable model, feature contract, threshold, and metadata."""

    model: Any
    feature_names: tuple[str, ...]
    threshold: float
    metadata: dict[str, Any]
    version: str = ARTIFACT_VERSION


def save_artifact(artifact: ModelArtifact, path: str | Path) -> Path:
    """Persist a model artifact and create its parent directory if required."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(asdict(artifact), destination)
    return destination


def load_artifact(path: str | Path) -> ModelArtifact:
    """Load and validate a versioned model artifact."""

    payload = joblib.load(path)
    artifact = ModelArtifact(**payload)
    if artifact.version != ARTIFACT_VERSION:
        raise ValueError(
            f"Unsupported artifact version {artifact.version!r}; "
            f"expected {ARTIFACT_VERSION!r}."
        )
    return artifact
