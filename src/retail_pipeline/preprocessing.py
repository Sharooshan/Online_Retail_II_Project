
import pandas as pd

from .validation import RetailDataValidator, REQUIRED_COLUMNS


class RetailDataPreprocessor:
    """Prepare retail transactions for daily revenue analysis."""

    def __init__(self, remove_duplicates: bool = True):
        self._remove_duplicates = remove_duplicates
        self._cleaning_statistics = {}

    @property
    def remove_duplicates(self) -> bool:
        """Return whether duplicate removal is enabled."""
        return self._remove_duplicates

    @property
    def cleaning_statistics(self) -> dict:
        """Return a copy of the latest cleaning statistics."""
        return self._cleaning_statistics.copy()

    def remove_exact_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of the data without exact duplicate rows."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")

        if not self._remove_duplicates:
            return data.copy()

        return data.drop_duplicates().copy()

    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare positive-sales transactions and record cleaning statistics."""

        validator = RetailDataValidator(REQUIRED_COLUMNS)
        validator.validate_schema(data)

        cleaned = data.copy()
        initial_rows = len(cleaned)

        # Stage 1: Remove exact duplicate records.
        before = len(cleaned)
        cleaned = self.remove_exact_duplicates(cleaned)
        duplicates_removed = before - len(cleaned)

        # Stage 2: Exclude cancelled invoices.
        before = len(cleaned)
        cancelled = (
            cleaned["Invoice"]
            .astype("string")
            .str.startswith("C", na=False)
        )
        cleaned = cleaned.loc[~cancelled].copy()
        cancellations_removed = before - len(cleaned)

        # Stage 3: Convert quantities and prices to numeric values.
        cleaned["Quantity"] = pd.to_numeric(
            cleaned["Quantity"], errors="coerce"
        )
        cleaned["Price"] = pd.to_numeric(
            cleaned["Price"], errors="coerce"
        )

        # Stage 4: Remove invalid quantities.
        before = len(cleaned)
        cleaned = cleaned.loc[cleaned["Quantity"] > 0].copy()
        invalid_quantity_removed = before - len(cleaned)

        # Stage 5: Remove invalid prices.
        before = len(cleaned)
        cleaned = cleaned.loc[cleaned["Price"] > 0].copy()
        invalid_price_removed = before - len(cleaned)

        # Stage 6: Remove missing or invalid transaction dates.
        before = len(cleaned)
        cleaned["InvoiceDate"] = pd.to_datetime(
            cleaned["InvoiceDate"], errors="coerce"
        )
        cleaned = cleaned.dropna(subset=["InvoiceDate"]).copy()
        invalid_date_removed = before - len(cleaned)

        # Stage 7: Calculate transaction revenue.
        cleaned["Revenue"] = (
            cleaned["Quantity"] * cleaned["Price"]
        )

        # Record the cleaning audit.
        self._cleaning_statistics = {
            "initial_rows": initial_rows,
            "duplicates_removed": duplicates_removed,
            "cancellations_removed": cancellations_removed,
            "invalid_quantity_removed": invalid_quantity_removed,
            "invalid_price_removed": invalid_price_removed,
            "invalid_date_removed": invalid_date_removed,
            "final_rows": len(cleaned),
        }

        return cleaned
