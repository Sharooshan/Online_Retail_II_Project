
from pathlib import Path
import platform
import sys
import time
import statistics
import shutil

import pandas as pd
import pyarrow

from retail_pipeline.loaders import RetailDataLoader
from retail_pipeline.preprocessing import RetailDataPreprocessor


# ==================================================
# 1. PROJECT SETTINGS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "online_retail_II.xlsx"

OUTPUT_DIR = PROJECT_ROOT / "results" / "storage"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WARMUP_RUNS = 1
MEASURED_RUNS = 5
TOTAL_RUNS = WARMUP_RUNS + MEASURED_RUNS

TARGET_COLUMN = "Revenue"

SUMMARY_PATH = OUTPUT_DIR / "benchmark_summary.csv"
TIMINGS_PATH = OUTPUT_DIR / "benchmark_timings.csv"

# True = run only Parquet and retain previous CSV/Pickle results
# False = rerun all six configurations on standardized data
PARQUET_ONLY = False


# ==================================================
# 2. STORAGE CONFIGURATIONS
# ==================================================

STORAGE_CONFIGURATIONS = [
    {
        "name": "csv_plain",
        "format": "csv",
        "compression": None,
        "filename": "retail_plain.csv",
    },
    {
        "name": "csv_gzip",
        "format": "csv",
        "compression": "gzip",
        "filename": "retail_compressed.csv.gz",
    },
    {
        "name": "pickle_plain",
        "format": "pickle",
        "compression": None,
        "filename": "retail_plain.pkl",
    },
    {
        "name": "pickle_gzip",
        "format": "pickle",
        "compression": "gzip",
        "filename": "retail_compressed.pkl.gz",
    },
    {
        "name": "parquet_plain",
        "format": "parquet",
        "compression": None,
        "filename": "retail_plain.parquet",
    },
    {
        "name": "parquet_snappy",
        "format": "parquet",
        "compression": "snappy",
        "filename": "retail_snappy.parquet",
    },
]


# ==================================================
# 3. PREPARE DATASET
# ==================================================

def prepare_dataset() -> pd.DataFrame:

    loader = RetailDataLoader(DATA_PATH)
    raw_data = loader.load()

    preprocessor = RetailDataPreprocessor()
    cleaned_data = preprocessor.process(raw_data)

    # Identifiers can contain both numbers and letters.
    # Standardize their representation for Parquet.
    for column in ["Invoice", "StockCode"]:
        cleaned_data[column] = cleaned_data[column].astype("string")

    return cleaned_data


# ==================================================
# 4. WRITE DATASET
# ==================================================

def write_dataset(
    data: pd.DataFrame,
    file_path: Path,
    file_format: str,
    compression=None,
):

    if file_format == "csv":

        data.to_csv(
            file_path,
            index=False,
            compression=compression,
        )

    elif file_format == "pickle":

        data.to_pickle(
            file_path,
            compression=compression,
        )

    elif file_format == "parquet":

        data.to_parquet(
            file_path,
            engine="pyarrow",
            index=False,
            compression=compression,
        )

    else:
        raise ValueError(
            f"Unsupported format: {file_format}"
        )


# ==================================================
# 5. READ FULL DATASET
# ==================================================

def read_dataset(
    file_path: Path,
    file_format: str,
    compression=None,
) -> pd.DataFrame:

    if file_format == "csv":

        return pd.read_csv(
            file_path,
            compression=compression,
            parse_dates=["InvoiceDate"],
            dtype={
                "Invoice": "str",
                "StockCode": "str",
            },
            low_memory=False,
        )

    elif file_format == "pickle":

        return pd.read_pickle(
            file_path,
            compression=compression,
        )

    elif file_format == "parquet":

        return pd.read_parquet(
            file_path,
            engine="pyarrow",
        )

    else:
        raise ValueError(
            f"Unsupported format: {file_format}"
        )


# ==================================================
# 6. READ SINGLE COLUMN
# ==================================================

def read_single_column(
    file_path: Path,
    file_format: str,
    compression=None,
) -> pd.Series:

    if file_format == "csv":

        data = pd.read_csv(
            file_path,
            usecols=[TARGET_COLUMN],
            compression=compression,
        )

    elif file_format == "pickle":

        data = pd.read_pickle(
            file_path,
            compression=compression,
        )[[TARGET_COLUMN]]

    elif file_format == "parquet":

        data = pd.read_parquet(
            file_path,
            engine="pyarrow",
            columns=[TARGET_COLUMN],
        )

    else:
        raise ValueError(
            f"Unsupported format: {file_format}"
        )

    return data[TARGET_COLUMN]


# ==================================================
# 7. MEASURE EXECUTION TIME
# ==================================================

def measure_operation(
    operation,
    repetitions=TOTAL_RUNS,
):

    timings = []

    for run_number in range(repetitions):

        start_time = time.perf_counter()

        result = operation()

        elapsed_time = (
            time.perf_counter() - start_time
        )

        timings.append(elapsed_time)

        print(
            f"Run {run_number + 1}: "
            f"{elapsed_time:.4f} seconds"
        )

        del result

    # Discard the first warm-up run
    measured_timings = timings[WARMUP_RUNS:]

    return {
        "median_seconds": statistics.median(
            measured_timings
        ),
        "minimum_seconds": min(measured_timings),
        "maximum_seconds": max(measured_timings),
        "range_seconds": (
            max(measured_timings) - min(measured_timings)
        ),
        "individual_timings": measured_timings,
    }


# ==================================================
# 8. VERIFY DATA INTEGRITY
# ==================================================

def verify_stored_data(
    original: pd.DataFrame,
    restored: pd.DataFrame,
):

    original = original.reset_index(drop=True).copy()
    restored = restored.reset_index(drop=True).copy()

    print("Original shape:", original.shape)
    print("Restored shape:", restored.shape)

    # Compare identifiers using consistent text types
    for column in ["Invoice", "StockCode"]:
        original[column] = original[column].astype("string")
        restored[column] = restored[column].astype("string")

    pd.testing.assert_frame_equal(
        original,
        restored,
        check_dtype=False,
        check_exact=False,
        rtol=1e-10,
        atol=1e-8,
    )

    print(
        "Data integrity verification passed "
        "for all nine columns."
    )


# ==================================================
# 9. BENCHMARK ONE CONFIGURATION
# ==================================================

def benchmark_configuration(
    data: pd.DataFrame,
    config: dict,
):

    file_path = OUTPUT_DIR / config["filename"]

    file_format = config["format"]
    compression = config["compression"]

    print(f"\nBenchmarking: {config['name']}")

    print("\nWriting dataset...")

    write_results = measure_operation(
        lambda: write_dataset(
            data,
            file_path,
            file_format,
            compression,
        )
    )

    file_size_mb = (
        file_path.stat().st_size / 1_000_000
    )

    print("\nVerifying data integrity...")

    restored_data = read_dataset(
        file_path,
        file_format,
        compression,
    )

    verify_stored_data(data, restored_data)

    del restored_data

    print("\nReading full dataset...")

    read_results = measure_operation(
        lambda: read_dataset(
            file_path,
            file_format,
            compression,
        )
    )

    print("\nReading Revenue column...")

    column_results = measure_operation(
        lambda: read_single_column(
            file_path,
            file_format,
            compression,
        )
    )

    return {
        "configuration": config["name"],
        "file_size_mb": file_size_mb,
        "write": write_results,
        "full_read": read_results,
        "column_read": column_results,
    }


# ==================================================
# 10. CONVERT RESULTS TO DATAFRAMES
# ==================================================

def results_to_dataframes(results):

    summary_rows = []
    timing_rows = []

    for result in results:

        for operation in [
            "write",
            "full_read",
            "column_read",
        ]:

            metrics = result[operation]

            summary_rows.append({
                "configuration": result["configuration"],
                "operation": operation,
                "file_size_mb": result["file_size_mb"],
                "median_seconds": metrics["median_seconds"],
                "minimum_seconds": metrics["minimum_seconds"],
                "maximum_seconds": metrics["maximum_seconds"],
                "range_seconds": metrics["range_seconds"],
            })

            for run_number, elapsed in enumerate(
                metrics["individual_timings"],
                start=1,
            ):

                timing_rows.append({
                    "configuration": result["configuration"],
                    "operation": operation,
                    "measured_run": run_number,
                    "elapsed_seconds": elapsed,
                })

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(timing_rows),
    )


# ==================================================
# 11. SAVE RESULTS
# ==================================================

def save_results(
    summary_df: pd.DataFrame,
    timings_df: pd.DataFrame,
):

    summary_df.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    timings_df.to_csv(
        TIMINGS_PATH,
        index=False,
    )

    print("\nBenchmark results saved successfully.")


# ==================================================
# 12. MAIN PROGRAM
# ==================================================

if __name__ == "__main__":

    print("Python version:", sys.version.split()[0])
    print("Platform:", platform.platform())
    print("Pandas version:", pd.__version__)
    print("PyArrow version:", pyarrow.__version__)

    print("\nPreparing cleaned dataset...")

    dataset = prepare_dataset()

    print("\nDataset prepared successfully!")
    print("Rows:", len(dataset))
    print("Columns:", len(dataset.columns))

    if PARQUET_ONLY:

        # Make sure previous results exist
        if not SUMMARY_PATH.exists() or not TIMINGS_PATH.exists():
            raise FileNotFoundError(
                "Previous benchmark results were not found. "
                "Set PARQUET_ONLY = False to run all formats."
            )

        # Load previous results
        previous_summary = pd.read_csv(SUMMARY_PATH)
        previous_timings = pd.read_csv(TIMINGS_PATH)

        completed = {
            "csv_plain",
            "csv_gzip",
            "pickle_plain",
            "pickle_gzip",
        }

        found = set(previous_summary["configuration"])

        if not completed.issubset(found):
            raise ValueError(
                "The previous results do not contain "
                "all four completed CSV/Pickle configurations."
            )

        # Keep only the four completed configurations
        previous_summary = previous_summary[
            previous_summary["configuration"].isin(completed)
        ].copy()

        previous_timings = previous_timings[
            previous_timings["configuration"].isin(completed)
        ].copy()

        # Validate previous result counts
        if len(previous_summary) != 12:
            raise ValueError(
                "Expected 12 CSV/Pickle summary rows."
            )

        if len(previous_timings) != 60:
            raise ValueError(
                "Expected 60 CSV/Pickle timing rows."
            )

        # Back up original result files
        shutil.copy2(
            SUMMARY_PATH,
            OUTPUT_DIR / "benchmark_summary_backup.csv",
        )

        shutil.copy2(
            TIMINGS_PATH,
            OUTPUT_DIR / "benchmark_timings_backup.csv",
        )

        print("\nPrevious CSV/Pickle results backed up.")

        configurations_to_run = [
            config
            for config in STORAGE_CONFIGURATIONS
            if config["format"] == "parquet"
        ]

    else:

        previous_summary = pd.DataFrame()
        previous_timings = pd.DataFrame()

        configurations_to_run = STORAGE_CONFIGURATIONS

    # Run remaining configurations
    new_results = []

    for config in configurations_to_run:

        result = benchmark_configuration(
            dataset,
            config,
        )

        new_results.append(result)

        new_summary, new_timings = results_to_dataframes(
            new_results
        )

        # Combine old and new results
        combined_summary = pd.concat(
            [previous_summary, new_summary],
            ignore_index=True,
        )

        combined_timings = pd.concat(
            [previous_timings, new_timings],
            ignore_index=True,
        )

        # Save progressively
        save_results(
            combined_summary,
            combined_timings,
        )

    print("\nAll requested benchmarks completed!")

    final_summary = pd.read_csv(SUMMARY_PATH)
    final_timings = pd.read_csv(TIMINGS_PATH)

    print(
        "Configurations:",
        final_summary["configuration"].nunique(),
    )

    print("Summary rows:", len(final_summary))
    print("Timing rows:", len(final_timings))
