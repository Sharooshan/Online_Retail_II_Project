
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
