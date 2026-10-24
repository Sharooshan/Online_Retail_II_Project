
import pandas as pd
from .base import RetailPipelineStage


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


class RetailDataValidator(RetailPipelineStage):
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

        # Validate the structure before calculating quality metrics.
        self.validate_schema(data)

        # Count records and columns.
        total_rows = len(data)
        total_columns = len(data.columns)

        # Count missing values for each column.
        missing_values = {
            column: int(count)
            for column, count in data.isna().sum().items()
        }

        # Count exact duplicate records.
        duplicate_rows = int(data.duplicated().sum())

        # Identify cancellation invoices.
        cancelled_rows = int(
            data["Invoice"]
            .astype("string")
            .str.startswith("C", na=False)
            .sum()
        )

        # Count negative quantities and zero prices.
        negative_quantities = int(
            (data["Quantity"] < 0).sum()
        )

        zero_prices = int(
            (data["Price"] == 0).sum()
        )

        # Return the findings as a structured dictionary.
        return {
            "total_rows": total_rows,
            "total_columns": total_columns,
            "missing_values": missing_values,
            "duplicate_rows": duplicate_rows,
            "cancelled_rows": cancelled_rows,
            "negative_quantities": negative_quantities,
            "zero_prices": zero_prices,
        }

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate the schema and return the original DataFrame."""
        self.validate_schema(data)
        return data
    