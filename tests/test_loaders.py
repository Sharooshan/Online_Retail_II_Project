
import pandas as pd
import pytest

from src.retail_pipeline.loaders import RetailDataLoader
from src.retail_pipeline.exceptions import RetailDataLoadError


def test_missing_excel_file():
    """A missing Excel workbook should raise the custom exception."""

    loader = RetailDataLoader("data/nonexistent_file.xlsx")

    with pytest.raises(RetailDataLoadError):
        loader.load_excel(("Year 2009-2010",))


def test_empty_sheet_names():
    """Reject an empty worksheet selection."""

    loader = RetailDataLoader("data/example.xlsx")

    with pytest.raises(ValueError):
        loader.load_excel(())


def test_missing_worksheet(tmp_path):
    """Raise a custom exception for a missing worksheet."""

    file_path = tmp_path / "sample.xlsx"

    sample = pd.DataFrame({
        "Invoice": ["10001"],
        "Quantity": [2],
        "Price": [5.0],
    })

    sample.to_excel(
        file_path,
        sheet_name="ExistingSheet",
        index=False
    )

    loader = RetailDataLoader(str(file_path))

    with pytest.raises(RetailDataLoadError):
        loader.load_excel(("MissingSheet",))
