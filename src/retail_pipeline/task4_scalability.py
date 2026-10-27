
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# ============================================================
# TASK 4 - PANDAS VS PYSPARK DAILY REVENUE AGGREGATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "cleaned_retail.parquet"
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "task4"

# Spark SQL operations are used without Python UDFs.
# Avoid Windows Python worker paths containing spaces.
os.environ.pop("PYSPARK_PYTHON", None)
os.environ.pop("PYSPARK_DRIVER_PYTHON", None)


# ============================================================
# 1. PANDAS AGGREGATION
# ============================================================

def pandas_aggregation(df):
    """Calculate total positive-sales revenue for each date."""

    data = df[["InvoiceDate", "Revenue"]].copy()

    data["Date"] = pd.to_datetime(
        data["InvoiceDate"]
    ).dt.date

    result = (
        data.groupby("Date", as_index=False)["Revenue"]
        .sum()
        .sort_values("Date")
        .reset_index(drop=True)
    )

    return result


# ============================================================
# 2. PYSPARK AGGREGATION
# ============================================================

def spark_aggregation(df):
    """Calculate equivalent daily revenue using Spark SQL."""

    return (
        df.withColumn(
            "Date",
            F.to_date(F.col("InvoiceDate"))
        )
        .groupBy("Date")
        .agg(
            F.sum(F.col("Revenue")).alias("Revenue")
        )
        .orderBy("Date")
    )


# ============================================================
# 3. RESULT VALIDATION
# ============================================================

def validate_results(pandas_result, spark_result):
    """Confirm pandas and PySpark produce matching daily totals."""

    pandas_result = pandas_result.copy()
    spark_result = spark_result.copy()

    pandas_result["Date"] = pd.to_datetime(
        pandas_result["Date"]
    )

    spark_result["Date"] = pd.to_datetime(
        spark_result["Date"]
    )

    pandas_result = (
        pandas_result
        .sort_values("Date")
        .reset_index(drop=True)
    )

    spark_result = (
        spark_result
        .sort_values("Date")
        .reset_index(drop=True)
    )

    if len(pandas_result) != len(spark_result):
        raise AssertionError(
            "Different numbers of daily records"
        )

    if not pandas_result["Date"].equals(
        spark_result["Date"]
    ):
        raise AssertionError(
            "Dates do not match"
        )

    np.testing.assert_allclose(
        pandas_result["Revenue"].to_numpy(),
        spark_result["Revenue"].to_numpy(),
        rtol=1e-8,
        atol=1e-5
    )

    return True


# ============================================================
# 4. MAIN EXPERIMENT
# ============================================================

def main():

    print("=" * 60)
    print("TASK 4 - PANDAS VS PYSPARK DAILY REVENUE")
    print("=" * 60)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # LOAD PANDAS DATA
    # --------------------------------------------------------

    print("\nLoading cleaned retail dataset...")

    pandas_df = pd.read_parquet(
        DATA_PATH
    )

    print(f"Input rows: {len(pandas_df):,}")
    print(f"Input columns: {len(pandas_df.columns)}")

    # --------------------------------------------------------
    # START SPARK
    # --------------------------------------------------------

    print("\nStarting local Spark session...")

    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("RetailTask4Scalability")
        .config(
            "spark.sql.shuffle.partitions",
            "4"
        )
        .config(
            "spark.sql.session.timeZone",
            "UTC"
        )
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    try:

        # Use a normal Windows path, not as_uri().
        spark_df = spark.read.parquet(
            str(DATA_PATH)
        )

        print(f"Spark version: {spark.version}")

        print(
            "Spark input partitions:",
            spark_df.rdd.getNumPartitions()
        )

        # ----------------------------------------------------
        # PANDAS EXECUTION
        # ----------------------------------------------------

        print("\nRunning pandas aggregation...")

        start = time.perf_counter()

        pandas_result = pandas_aggregation(
            pandas_df
        )

        pandas_seconds = (
            time.perf_counter() - start
        )

        # ----------------------------------------------------
        # PYSPARK EXECUTION
        # ----------------------------------------------------

        print("Running PySpark aggregation...")

        start = time.perf_counter()

        spark_query = spark_aggregation(
            spark_df
        )

        # Spark is lazy; collect() triggers computation.
        spark_rows = spark_query.collect()

        spark_seconds = (
            time.perf_counter() - start
        )

        # Avoid PySpark 3.5 toPandas() compatibility issues.
        spark_result = pd.DataFrame(
            [
                (row["Date"], row["Revenue"])
                for row in spark_rows
            ],
            columns=["Date", "Revenue"]
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        print("\nValidating aggregation results...")

        validate_results(
            pandas_result,
            spark_result
        )

        # ----------------------------------------------------
        # RESULTS
        # ----------------------------------------------------

        print("\n" + "=" * 60)
        print("TASK 4 - INITIAL COMPARISON RESULTS")
        print("=" * 60)

        print(
            f"Input transactions: {len(pandas_df):,}"
        )

        print(
            f"Aggregated daily records: {len(pandas_result)}"
        )

        print(
            f"\npandas execution time: "
            f"{pandas_seconds:.4f} seconds"
        )

        print(
            f"PySpark execution time: "
            f"{spark_seconds:.4f} seconds"
        )

        if pandas_seconds > 0:

            ratio = (
                spark_seconds / pandas_seconds
            )

            print(
                f"Spark / pandas time ratio: {ratio:.2f}x"
            )

        print("\nRevenue consistency: PASSED")

        print("=" * 60)

        # ----------------------------------------------------
        # SPARK EXECUTION PLAN
        # ----------------------------------------------------

        print("\nSPARK PHYSICAL EXECUTION PLAN")
        print("-" * 60)

        spark_query.explain(
            mode="formatted"
        )

        # ----------------------------------------------------
        # SAVE RESULTS
        # ----------------------------------------------------

        comparison = pd.DataFrame([
            {
                "Framework": "pandas",
                "InputRows": len(pandas_df),
                "DailyRecords": len(pandas_result),
                "ExecutionSeconds": pandas_seconds
            },
            {
                "Framework": "PySpark",
                "InputRows": len(pandas_df),
                "DailyRecords": len(spark_result),
                "ExecutionSeconds": spark_seconds
            }
        ])

        output_file = (
            OUTPUT_DIR / "initial_comparison.csv"
        )

        comparison.to_csv(
            output_file,
            index=False
        )

        print("\nResults saved to:")
        print(output_file)

        print(
            "\nTASK 4 INITIAL IMPLEMENTATION COMPLETED"
        )

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
