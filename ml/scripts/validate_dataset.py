"""Fetch UCI dataset 350 and print its aggregate validation report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_ROOT / "src"))

from customer_intelligence.data import fetch_credit_default_dataset


def main() -> None:
    dataset = fetch_credit_default_dataset()
    print(json.dumps(dataset.validation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
