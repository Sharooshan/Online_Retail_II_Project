
import os
import time
import statistics
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# ============================================================
# TASK 4 - REPEATED PANDAS VS PYSPARK BENCHMARK
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "outputs" / "cleaned_retail.parquet"
OUTPUT_DIR = ROOT / "results" / "task4"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Avoid Windows Python worker launcher paths containing spaces.
# This benchmark uses Spark SQL functions only.
os.environ.pop("PYSPARK_PYTHON", None)
os.environ.pop("PYSPARK_DRIVER_PYTHON", None)

FRACTIONS = [0.25, 0.50, 1.00]
REPEATS = 3


def pandas_aggregate(df):
    """Daily revenue aggregation with pandas."""
    return (
        df.assign(Date=df["InvoiceDate"].dt.date)
        .groupby("Date", as_index=False)["Revenue"]
        .sum()
        .sort_values("Date")
        .reset_index(drop=True)
    )


def spark_aggregate(df):
    """Equivalent daily revenue aggregation with PySpark."""
    return (
        df.withColumn("Date", F.to_date("InvoiceDate"))
        .groupBy("Date")
        .agg(F.sum("Revenue").alias("Revenue"))
        .orderBy("Date")
    )


def check_equal(pd_result, spark_rows):
    """Verify that daily revenue totals agree."""

    spark_result = pd.DataFrame(
        [(r["Date"], r["Revenue"]) for r in spark_rows],
        columns=["Date", "Revenue"]
    )

    left = pd_result.copy()
    right = spark_result.copy()

    left["Date"] = pd.to_datetime(left["Date"])
    right["Date"] = pd.to_datetime(right["Date"])

    left = left.sort_values("Date").reset_index(drop=True)
    right = right.sort_values("Date").reset_index(drop=True)

    assert left["Date"].equals(right["Date"])

    np.testing.assert_allclose(
        left["Revenue"].to_numpy(),
        right["Revenue"].to_numpy(),
        rtol=1e-8,
        atol=1e-5
    )


def main():

    print("=" * 65)
    print("TASK 4 - REPEATED SCALABILITY BENCHMARK")
    print("=" * 65)

    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    print("\nLoading original cleaned dataset...")

    source = pd.read_parquet(
        DATA_PATH,
        columns=["InvoiceDate", "Revenue"]
    )

    source["InvoiceDate"] = pd.to_datetime(
        source["InvoiceDate"]
    )

    print(f"Total available rows: {len(source):,}")

    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("Task4RepeatedBenchmark")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    records = []

    try:
        for fraction in FRACTIONS:

            n = round(len(source) * fraction)

            # Deterministic subset of original records.
            subset = source.iloc[:n].copy()

            # Write a temporary Parquet dataset so Spark
            # can read exactly the same subset.
            temp_file = OUTPUT_DIR / "benchmark_input.parquet"

            subset.to_parquet(
                temp_file,
                index=False
            )

            spark_df = spark.read.parquet(
                str(temp_file)
            )

            print("\n" + "-" * 65)
            print(f"WORKLOAD: {fraction:.0%} | {n:,} rows")
            print("-" * 65)

            # Warm up both implementations.
            pandas_aggregate(subset)
            spark_aggregate(spark_df).collect()

            pandas_times = []
            spark_times = []

            for run in range(1, REPEATS + 1):

                # pandas
                start = time.perf_counter()
                pd_result = pandas_aggregate(subset)
                pandas_time = time.perf_counter() - start

                # PySpark
                start = time.perf_counter()
                spark_rows = spark_aggregate(
                    spark_df
                ).collect()
                spark_time = time.perf_counter() - start

                check_equal(pd_result, spark_rows)

                pandas_times.append(pandas_time)
                spark_times.append(spark_time)

                print(
                    f"Run {run}: "
                    f"pandas={pandas_time:.4f}s | "
                    f"PySpark={spark_time:.4f}s | "
                    f"Validation=PASSED"
                )

            pd_median = statistics.median(pandas_times)
            spark_median = statistics.median(spark_times)

            records.append({
                "WorkloadPercent": int(fraction * 100),
                "InputRows": n,
                "PandasMedianSeconds": pd_median,
                "PySparkMedianSeconds": spark_median,
                "SparkToPandasRatio": (
                    spark_median / pd_median
                    if pd_median > 0 else float("nan")
                ),
                "Repeats": REPEATS
            })

            print(
                f"MEDIAN: pandas={pd_median:.4f}s | "
                f"PySpark={spark_median:.4f}s"
            )

        # Save summary.
        results = pd.DataFrame(records)

        csv_path = OUTPUT_DIR / "benchmark_results.csv"
        results.to_csv(csv_path, index=False)

        # Generate performance comparison chart.
        plt.figure(figsize=(9, 5))

        plt.plot(
            results["InputRows"],
            results["PandasMedianSeconds"],
            marker="o",
            label="pandas"
        )

        plt.plot(
            results["InputRows"],
            results["PySparkMedianSeconds"],
            marker="s",
            label="PySpark"
        )

        plt.xlabel("Number of input transactions")
        plt.ylabel("Median execution time (seconds)")
        plt.title("Task 4: pandas vs PySpark Daily Revenue Aggregation")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        chart_path = OUTPUT_DIR / "benchmark_chart.png"
        plt.savefig(chart_path, dpi=200)
        plt.close()

        print("\n" + "=" * 65)
        print("FINAL BENCHMARK SUMMARY")
        print("=" * 65)
        print(results.to_string(index=False))

        print("\nSaved files:")
        print(csv_path)
        print(chart_path)

        print("\nTASK 4 BENCHMARK COMPLETED")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
