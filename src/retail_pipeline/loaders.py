
from pathlib import Path

import pandas as pd


class RetailDataLoader:
    """Load retail transaction data from supported files."""

    def __init__(self, file_path: str):
        self._file_path = Path(file_path)

    @property
    def file_path(self) -> Path:
        """Return the configured dataset path."""
        return self._file_path

    def load(self) -> pd.DataFrame:
        """Load the dataset and return a DataFrame."""
        raise NotImplementedError(
            "Implement dataset loading and validation"
        )

    def load_excel(self, sheet_names: tuple[str, ...]) -> pd.DataFrame:
        """Load specified Excel worksheets into one DataFrame."""
        frames = []

        for sheet_name in sheet_names:
            frame = pd.read_excel(
                self._file_path,
                sheet_name=sheet_name
            )
            frames.append(frame)

        return pd.concat(frames, ignore_index=True)
