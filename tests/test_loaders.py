
import pytest

from src.retail_pipeline.loaders import RetailDataLoader
from src.retail_pipeline.exceptions import RetailDataLoadError


def test_missing_excel_file():
    """A missing Excel workbook should raise the custom exception."""

    loader = RetailDataLoader("data/nonexistent_file.xlsx")

    with pytest.raises(RetailDataLoadError):
        loader.load_excel(("Year 2009-2010",))
