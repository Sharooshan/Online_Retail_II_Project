
import pandas as pd


REQUIRED_COLUMNS = {
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country",
}


class RetailDataValidator:
    """Validate the structure and quality of retail data."""

    def __init__(self, required_columns: set[str]):
        self._required_columns = frozenset(required_columns)

    @property
    def required_columns(self) -> frozenset[str]:
        """Return the required dataset columns."""
        return self._required_columns

    def validate_schema(self, data: pd.DataFrame) -> None:
        """Check whether the dataset contains all required columns."""

        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")

        missing_columns = self._required_columns - set(data.columns)

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {sorted(missing_columns)}"
            )

    def assess_quality(self, data: pd.DataFrame) -> dict:
        """Summarize important data-quality indicators."""
        raise NotImplementedError(
            "Implement data-quality assessment"
        )
