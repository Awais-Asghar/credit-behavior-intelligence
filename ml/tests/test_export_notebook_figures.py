"""Tests for exporting stable portfolio figures from notebook outputs."""

from __future__ import annotations

import base64
import json
import sys
import tempfile
import unittest
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIRECTORY = ML_ROOT / "scripts"
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from export_notebook_figures import FIGURES, export_figures  # noqa: E402


PNG_BYTES = b"\x89PNG\r\n\x1a\nportfolio-test-fixture"


# Value: protects=named figures are exported from the expected saved notebook outputs;
# fails_when=cell mapping, output selection, decoding, or PNG validation regresses;
# why_new=no existing test exercises the portfolio figure export boundary;
# seam=none; no_claim=visual correctness or model-result correctness.
class NotebookFigureExportTests(unittest.TestCase):
    def build_notebook(self, image_bytes: bytes = PNG_BYTES) -> dict:
        highest_cell = max(cell_index for cell_index, _ in FIGURES)
        cells = [
            {"cell_type": "code", "source": [], "outputs": []}
            for _ in range(highest_cell + 1)
        ]

        for cell_index, image_index in FIGURES:
            outputs = cells[cell_index]["outputs"]
            while len(outputs) <= image_index:
                outputs.append(
                    {
                        "output_type": "display_data",
                        "data": {
                            "image/png": base64.b64encode(image_bytes).decode("ascii")
                        },
                        "metadata": {},
                    }
                )

        return {
            "cells": cells,
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }

    def write_notebook(self, directory: Path, notebook: dict) -> Path:
        notebook_path = directory / "executed.ipynb"
        notebook_path.write_text(json.dumps(notebook), encoding="utf-8")
        return notebook_path

    def test_exports_every_named_figure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            notebook_path = self.write_notebook(root, self.build_notebook())
            output_directory = root / "figures"

            exported = export_figures(notebook_path, output_directory)

            self.assertEqual(len(exported), len(FIGURES))
            self.assertEqual(
                {path.name for path in exported},
                set(FIGURES.values()),
            )
            self.assertTrue(all(path.read_bytes() == PNG_BYTES for path in exported))

    def test_missing_expected_output_fails_with_cell_context(self) -> None:
        notebook = self.build_notebook()
        missing_cell, _ = next(iter(FIGURES))
        notebook["cells"][missing_cell]["outputs"] = []

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            notebook_path = self.write_notebook(root, notebook)

            with self.assertRaisesRegex(
                RuntimeError,
                rf"Missing expected image 0 in notebook cell {missing_cell}",
            ):
                export_figures(notebook_path, root / "figures")

    def test_non_png_output_fails_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            notebook_path = self.write_notebook(
                root,
                self.build_notebook(image_bytes=b"not-a-png"),
            )

            with self.assertRaisesRegex(RuntimeError, "Expected PNG data"):
                export_figures(notebook_path, root / "figures")


if __name__ == "__main__":
    unittest.main()
