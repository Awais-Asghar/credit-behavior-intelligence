"""Export selected PNG outputs from the executed portfolio notebook."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path


FIGURES = {
    (14, 0): "class-distribution.png",
    (18, 0): "behavioral-correlation.png",
    (22, 0): "pca-preprocessing-comparison.png",
    (25, 0): "pca-explained-variance.png",
    (26, 0): "pca-diagnostic-projection.png",
    (34, 0): "kmeans-model-selection.png",
    (41, 0): "kmeans-cluster-profiles.png",
    (42, 0): "kmeans-cluster-projection.png",
    (52, 0): "dbscan-distance-curve.png",
    (58, 0): "dbscan-anomaly-projection.png",
    (72, 0): "threshold-and-calibration.png",
    (72, 1): "training-validation-loss.png",
    (75, 0): "test-confusion-matrix-and-roc.png",
}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Export named figures from saved notebook outputs."
    )
    parser.add_argument(
        "--notebook",
        type=Path,
        default=project_root / "ml" / "notebooks" / "customer-intelligence.ipynb",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "reports" / "figures",
    )
    return parser.parse_args()


def image_outputs(cell: dict) -> list[bytes]:
    images = []
    for output in cell.get("outputs", []):
        payload = output.get("data", {}).get("image/png")
        if payload is None:
            continue
        encoded = "".join(payload) if isinstance(payload, list) else payload
        images.append(base64.b64decode(encoded))
    return images


def export_figures(notebook_path: Path, output_dir: Path) -> list[Path]:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    exported = []

    for (cell_index, image_index), filename in FIGURES.items():
        try:
            image = image_outputs(notebook["cells"][cell_index])[image_index]
        except (IndexError, KeyError) as error:
            raise RuntimeError(
                f"Missing expected image {image_index} in notebook cell {cell_index}."
            ) from error

        if not image.startswith(b"\x89PNG\r\n\x1a\n"):
            raise RuntimeError(f"Expected PNG data for {filename}.")

        destination = output_dir / filename
        destination.write_bytes(image)
        exported.append(destination)

    return exported


def main() -> None:
    args = parse_args()
    exported = export_figures(args.notebook.resolve(), args.output_dir.resolve())
    print(f"Exported {len(exported)} figures to {args.output_dir.resolve()}")
    for path in exported:
        print(f"- {path.name}")


if __name__ == "__main__":
    main()
