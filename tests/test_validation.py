
import pandas as pd
import pytest

from src.retail_pipeline.validation import (
    RetailDataValidator,
    REQUIRED_COLUMNS,
)


def test_valid_schema():
    """Check that a valid dataset passes validation."""
    data = pd.DataFrame(columns=list(REQUIRED_COLUMNS))
    validator = RetailDataValidator(REQUIRED_COLUMNS)

    validator.validate_schema(data)


def test_missing_column():
    """Check that a missing required column raises ValueError."""
    columns = REQUIRED_COLUMNS - {"InvoiceDate"}
    data = pd.DataFrame(columns=list(columns))

    validator = RetailDataValidator(REQUIRED_COLUMNS)

    with pytest.raises(ValueError, match="InvoiceDate"):
        validator.validate_schema(data)


def test_invalid_input_type():
    """Check that a non-DataFrame input raises TypeError."""
    validator = RetailDataValidator(REQUIRED_COLUMNS)

    with pytest.raises(TypeError):
        validator.validate_schema(["not", "a", "dataframe"])


def test_assess_quality_missing_column():
    """Quality assessment must reject an incomplete dataset."""

    # Create a dataset without the InvoiceDate column.
    columns = REQUIRED_COLUMNS - {"InvoiceDate"}
    data = pd.DataFrame(columns=list(columns))

    # Create the validator.
    validator = RetailDataValidator(REQUIRED_COLUMNS)

    # Verify that the expected exception is raised.
    with pytest.raises(ValueError, match="InvoiceDate"):
        validator.assess_quality(data)



def test_assess_quality_counts():
    """Check data-quality counts using known sample records."""

    sample = pd.DataFrame([
        {
            "Invoice": "10001",
            "StockCode": "A01",
            "Description": "Notebook",
            "Quantity": 2,
            "InvoiceDate": "2010-01-01",
            "Price": 5.0,
            "Customer ID": None,
            "Country": "United Kingdom",
        },
        {
            "Invoice": "10001",
            "StockCode": "A01",
            "Description": "Notebook",
            "Quantity": 2,
            "InvoiceDate": "2010-01-01",
            "Price": 5.0,
            "Customer ID": None,
            "Country": "United Kingdom",
        },
        {
            "Invoice": "C10002",
            "StockCode": "B02",
            "Description": "Returned item",
            "Quantity": -1,
            "InvoiceDate": "2010-01-02",
            "Price": 0.0,
            "Customer ID": 12345,
            "Country": "United Kingdom",
        },
    ])

    validator = RetailDataValidator(REQUIRED_COLUMNS)
    report = validator.assess_quality(sample)

    assert report["total_rows"] == 3
    assert report["total_columns"] == 8
    assert report["duplicate_rows"] == 1
    assert report["cancelled_rows"] == 1
    assert report["negative_quantities"] == 1
    assert report["zero_prices"] == 1
    assert report["missing_values"]["Customer ID"] == 2
