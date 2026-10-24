
import pandas as pd

from src.retail_pipeline.preprocessing import RetailDataPreprocessor


def test_remove_exact_duplicates():
    """Verify that exact duplicate rows are removed."""

    sample = pd.DataFrame({
        "Invoice": ["10001", "10001", "10002"],
        "Quantity": [2, 2, 3],
        "Price": [5.0, 5.0, 10.0],
    })

    preprocessor = RetailDataPreprocessor(
        remove_duplicates=True
    )

    cleaned = preprocessor.remove_exact_duplicates(sample)

    assert len(cleaned) == 2
    assert len(sample) == 3
    assert cleaned["Invoice"].tolist() == ["10001", "10002"]


def test_duplicate_removal_disabled():
    """Duplicates should remain when removal is disabled."""
    sample = pd.DataFrame({
        "Invoice": ["10001", "10001", "10002"],
        "Quantity": [2, 2, 3],
        "Price": [5.0, 5.0, 10.0],
    })

    preprocessor = RetailDataPreprocessor(remove_duplicates=False)
    result = preprocessor.remove_exact_duplicates(sample)

    assert len(result) == 3
    assert result is not sample



def test_clean_transactions():
    """Check that only valid positive-sales transactions remain."""

    sample = pd.DataFrame({
        "Invoice": ["10001", "C10002", "10003", "10004", "10005"],
        "StockCode": ["A", "B", "C", "D", "E"],
        "Description": ["Item A", "Item B", "Item C", "Item D", "Item E"],
        "Quantity": [2, -1, 0, 3, 2],
        "InvoiceDate": ["2010-01-01"] * 5,
        "Price": [5.0, 5.0, 8.0, 0.0, 7.5],
        "Customer ID": [1, 2, 3, 4, None],
        "Country": ["United Kingdom"] * 5,
    })

    preprocessor = RetailDataPreprocessor()
    result = preprocessor.clean(sample)

    assert len(result) == 2
    assert result["Invoice"].tolist() == ["10001", "10005"]
    assert result["Revenue"].tolist() == [10.0, 15.0]
    assert len(sample) == 5
    assert "Revenue" not in sample.columns


    
def test_cleaning_statistics():
    """Verify that cleaning counts match the removed records."""

    sample = pd.DataFrame({
        "Invoice": ["10001", "10001", "C10002", "10003", "10004", "10005"],
        "StockCode": ["A", "A", "B", "C", "D", "E"],
        "Description": ["Item A", "Item A", "Item B", "Item C", "Item D", "Item E"],
        "Quantity": [2, 2, -1, 0, 3, 2],
        "InvoiceDate": ["2010-01-01"] * 6,
        "Price": [5.0, 5.0, 5.0, 8.0, 0.0, 7.5],
        "Customer ID": [1, 1, 2, 3, 4, None],
        "Country": ["United Kingdom"] * 6,
    })

    preprocessor = RetailDataPreprocessor()
    cleaned = preprocessor.clean(sample)
    stats = preprocessor.cleaning_statistics

    assert stats["initial_rows"] == 6
    assert stats["duplicates_removed"] == 1
    assert stats["cancellations_removed"] == 1
    assert stats["invalid_quantity_removed"] == 1
    assert stats["invalid_price_removed"] == 1
    assert stats["invalid_date_removed"] == 0
    assert stats["final_rows"] == 2

    total_removed = (
        stats["duplicates_removed"]
        + stats["cancellations_removed"]
        + stats["invalid_quantity_removed"]
        + stats["invalid_price_removed"]
        + stats["invalid_date_removed"]
    )

    assert stats["initial_rows"] - total_removed == stats["final_rows"]
    assert cleaned["Revenue"].sum() == 25.0


def test_invalid_dates_and_numeric_values():
    """Verify that invalid dates, quantities, and prices are excluded."""

    sample = pd.DataFrame({
        "Invoice": ["10001", "10002", "10003", "10004"],
        "StockCode": ["A", "B", "C", "D"],
        "Description": ["Item A", "Item B", "Item C", "Item D"],
        "Quantity": [2, "invalid", 3, 1],
        "InvoiceDate": [
            "2010-01-01",
            "2010-01-02",
            "not-a-date",
            "2010-01-04",
        ],
        "Price": [5.0, 10.0, 7.0, "invalid"],
        "Customer ID": [1, 2, None, 4],
        "Country": ["United Kingdom"] * 4,
    })

    preprocessor = RetailDataPreprocessor()
    result = preprocessor.clean(sample)
    stats = preprocessor.cleaning_statistics

    assert len(result) == 1
    assert result.iloc[0]["Invoice"] == "10001"
    assert result.iloc[0]["Revenue"] == 10.0

    assert stats["invalid_quantity_removed"] == 1
    assert stats["invalid_price_removed"] == 1
    assert stats["invalid_date_removed"] == 1
    assert stats["final_rows"] == 1
