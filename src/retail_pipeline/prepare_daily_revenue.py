
from pathlib import Path
import sys
import inspect

import pandas as pd


# ============================================================
# TASK 3 - DAILY REVENUE PREPARATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from retail_pipeline.loaders import RetailDataLoader
from retail_pipeline.preprocessing import RetailDataPreprocessor


# ============================================================
# FILE PATHS
# ============================================================

RAW_FILE = PROJECT_ROOT / "data" / "online_retail_II.xlsx"
CLEANED_FILE = PROJECT_ROOT / "outputs" / "cleaned_retail.parquet"

OUTPUT_DIR = PROJECT_ROOT / "results" / "task3"
DAILY_CSV = OUTPUT_DIR / "daily_revenue.csv"
DAILY_PARQUET = OUTPUT_DIR / "daily_revenue.parquet"


# ============================================================
# LOAD AND CLEAN DATA
# ============================================================

def load_cleaned_data():

    if CLEANED_FILE.exists():
        print("Loading existing cleaned Parquet dataset...")
        cleaned_df = pd.read_parquet(CLEANED_FILE)

    else:
        print("Cleaned Parquet file not found.")
        print("Loading original Online Retail II workbook...")

        if not RAW_FILE.exists():
            raise FileNotFoundError(
                f"Original Excel workbook not found:\n{RAW_FILE}"
            )

        loader = RetailDataLoader(RAW_FILE)

        # The existing loader's load() method accepts
        # no positional worksheet argument.
        raw_df = loader.load()

        print(f"Original records loaded: {len(raw_df):,}")

        preprocessor = RetailDataPreprocessor()
        cleaned_df = preprocessor.clean(raw_df)

        # Standardise mixed-type identifiers for Parquet.
        for column in ["Invoice", "StockCode"]:
            if column in cleaned_df.columns:
                cleaned_df[column] = (
                    cleaned_df[column].astype("string")
                )

        CLEANED_FILE.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        cleaned_df.to_parquet(
            CLEANED_FILE,
            index=False,
            engine="pyarrow",
            compression="snappy"
        )

        print(f"Cleaned Parquet saved:\n{CLEANED_FILE}")

    return cleaned_df


# ============================================================
# PREPARE DAILY REVENUE
# ============================================================

def prepare_daily_revenue(df):

    required = {"InvoiceDate", "Revenue"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Required columns missing: {sorted(missing)}"
        )

    data = df[["InvoiceDate", "Revenue"]].copy()

    data["InvoiceDate"] = pd.to_datetime(
        data["InvoiceDate"],
        errors="coerce"
    )

    data["Revenue"] = pd.to_numeric(
        data["Revenue"],
        errors="coerce"
    )

    if data.isna().any().any():
        raise ValueError(
            "Missing or invalid InvoiceDate/Revenue values."
        )

    if (data["Revenue"] <= 0).any():
        raise ValueError(
            "Expected positive-sales transactions only."
        )

    # Extract calendar date from transaction timestamp.
    data["Date"] = data["InvoiceDate"].dt.normalize()

    # Aggregate transaction revenue by calendar date.
    daily = (
        data.groupby("Date")["Revenue"]
        .sum()
        .sort_index()
        .to_frame()
    )

    # Create continuous calendar.
    calendar = pd.date_range(
        start=daily.index.min(),
        end=daily.index.max(),
        freq="D",
        name="Date"
    )

    daily = daily.reindex(calendar)

    # Identify dates containing qualifying transactions.
    daily["HasTransactions"] = (
        daily["Revenue"].notna().astype(int)
    )

    # Zero indicates no qualifying transactions recorded.
    # It does not prove that the retailer had zero actual sales.
    daily["Revenue"] = daily["Revenue"].fillna(0.0)

    # Forecast target: next calendar day's recorded revenue.
    daily["NextDayRevenue"] = (
        daily["Revenue"].shift(-1)
    )

    return daily.reset_index()


# ============================================================
# VALIDATE DAILY DATASET
# ============================================================

def validate_daily_data(daily):

    print("\nVALIDATION RESULTS")
    print("-" * 55)

    calendar_days = len(daily)
    transaction_days = int(
        daily["HasTransactions"].sum()
    )
    missing_days = int(
        (daily["HasTransactions"] == 0).sum()
    )
    missing_targets = int(
        daily["NextDayRevenue"].isna().sum()
    )

    print(f"Calendar days: {calendar_days:,}")
    print(f"Days with transactions: {transaction_days:,}")
    print(f"Days without transactions: {missing_days:,}")
    print(f"Missing next-day targets: {missing_targets:,}")

    assert calendar_days == transaction_days + missing_days
    assert missing_targets == 1
    assert daily["Date"].is_unique
    assert daily["Date"].is_monotonic_increasing

    print("Daily revenue validation: PASSED")

    # These are expected from the earlier calendar audit.
    expected = {
        "calendar_days": 739,
        "transaction_days": 604,
        "missing_days": 135
    }

    observed = {
        "calendar_days": calendar_days,
        "transaction_days": transaction_days,
        "missing_days": missing_days
    }

    if observed == expected:
        print("Earlier calendar audit: MATCHED")
    else:
        print("Earlier calendar audit: DIFFERENCE DETECTED")
        print("Expected:", expected)
        print("Observed:", observed)
        print("Investigate differences before modelling.")


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("=" * 55)
    print("TASK 3 - DAILY REVENUE DATA PREPARATION")
    print("=" * 55)

    print(f"\nProject root:\n{PROJECT_ROOT}")

    cleaned_df = load_cleaned_data()

    print(
        f"\nCleaned transactions: {len(cleaned_df):,}"
    )

    print(
        f"Cleaned attributes: {len(cleaned_df.columns)}"
    )

    daily_df = prepare_daily_revenue(cleaned_df)

    validate_daily_data(daily_df)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    daily_df.to_csv(
        DAILY_CSV,
        index=False
    )

    daily_df.to_parquet(
        DAILY_PARQUET,
        index=False,
        engine="pyarrow",
        compression="snappy"
    )

    print("\nDAILY REVENUE SUMMARY")
    print("-" * 55)

    print(
        "Date range:",
        daily_df["Date"].min().date(),
        "to",
        daily_df["Date"].max().date()
    )

    print("\nFirst five records:")
    print(
        daily_df.head().to_string(index=False)
    )

    print("\nLast five records:")
    print(
        daily_df.tail().to_string(index=False)
    )

    print("\nSAVED OUTPUT FILES")
    print("-" * 55)

    print(f"Cleaned dataset: {CLEANED_FILE}")
    print(f"Daily CSV: {DAILY_CSV}")
    print(f"Daily Parquet: {DAILY_PARQUET}")

    print("\nTASK 3 STEP 1 COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
