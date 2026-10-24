
import pytest

from src.retail_pipeline.exceptions import (
    RetailPipelineError,
    RetailDataLoadError,
)


def test_custom_exception_inheritance():
    """Check the custom exception inheritance."""
    assert issubclass(RetailDataLoadError, RetailPipelineError)


def test_custom_exception_can_be_raised():
    """Check that the custom exception can be raised."""
    with pytest.raises(RetailDataLoadError):
        raise RetailDataLoadError("Sample loading failure")
