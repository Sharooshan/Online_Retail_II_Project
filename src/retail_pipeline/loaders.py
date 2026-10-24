
from pathlib import Path

import pandas as pd

from .exceptions import RetailDataLoadError


class RetailDataLoader:
    """Load retail transaction data from supported files."""

    def __init__(self, file_path: str):
        self._file_path = Path(file_path)

    @property
    def file_path(self) -> Path:
        """Return the configured dataset path."""
        return self._file_path

    def load(self) -> pd.DataFrame:
        """Load both Online Retail II worksheets."""
        sheet_names = ("Year 2009-2010", "Year 2010-2011")
        return self.load_excel(sheet_names)

    def load_excel(self, sheet_names: tuple[str, ...]) -> pd.DataFrame:
        """Load specified worksheets and close the workbook afterward."""
        if not sheet_names:
            raise ValueError("At least one worksheet must be specified")

        frames = []
        workbook = None

        try:
            workbook = pd.ExcelFile(self._file_path)

            for sheet_name in sheet_names:
                frame = pd.read_excel(
                    workbook,
                    sheet_name=sheet_name
                )
                frames.append(frame)

        except (FileNotFoundError, ValueError, OSError) as exc:
            raise RetailDataLoadError(
                f"Unable to load retail data from '{self._file_path}'"
            ) from exc

        else:
            return pd.concat(frames, ignore_index=True)

        finally:
            if workbook is not None:
                workbook.close()
