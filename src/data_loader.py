"""
Data loading utilities for the Teleconnect churn dataset.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "teleconnect.csv"


class DataLoadError(Exception):
    """Raised when the Teleconnect dataset cannot be loaded."""


def load_teleconnect_data(
    file_path: str | Path = DATA_PATH,
) -> pd.DataFrame:
    """
    Load the supplied Teleconnect CSV file.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise DataLoadError(
            f"Dataset not found at: {file_path}"
        )

    try:
        df = pd.read_csv(file_path)
    except Exception as exc:
        raise DataLoadError(
            f"Could not read dataset: {file_path}"
        ) from exc

    if df.empty:
        raise DataLoadError(
            "The loaded dataset is empty."
        )

    required = {
        "customerID",
        "TotalCharges",
        "Churn",
    }

    missing = required.difference(
        df.columns
    )

    if missing:
        raise DataLoadError(
            f"Missing required columns: "
            f"{sorted(missing)}"
        )

    return df