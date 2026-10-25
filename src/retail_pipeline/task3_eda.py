
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ==================================================
# PROJECT PATHS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT / "results" / "task3" / "daily_revenue.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "results" / "task3" / "charts"
)

SUMMARY_FILE = (
    PROJECT_ROOT / "results" / "task3" / "eda_summary.csv"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# LOAD DATASET
# ==================================================

def load_data():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Daily revenue dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE, parse_dates=["Date"])

    required = {
        "Date", "Revenue",
        "HasTransactions", "NextDayRevenue"
    }

    if not required.issubset(df.columns):
        raise ValueError("Required columns are missing.")

    df = df.sort_values("Date").reset_index(drop=True)

    print("Daily revenue dataset loaded successfully.")
    print(f"Total calendar days: {len(df):,}")

    return df


# ==================================================
# EXPLORATORY DATA ANALYSIS
# ==================================================

def run_eda(df):

    print("\n" + "=" * 55)
    print("TASK 3 - EXPLORATORY DATA ANALYSIS")
    print("=" * 55)

    df["YearMonth"] = df["Date"].dt.to_period("M")
    df["Weekday"] = df["Date"].dt.day_name()

    weekday_order = [
        "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday"
    ]

    # ----------------------------------------------
    # FIGURE 3.1 - DAILY REVENUE TREND
    # ----------------------------------------------

    plt.figure(figsize=(13, 5))

    plt.plot(
        df["Date"],
        df["Revenue"],
        linewidth=0.8
    )

    plt.title("Daily Retail Revenue Over Time")
    plt.xlabel("Date")
    plt.ylabel("Daily Revenue")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "01_daily_revenue_trend.png",
        dpi=300
    )
    plt.close()

    print("Saved Figure 3.1 - Daily revenue trend")

    # ----------------------------------------------
    # FIGURE 3.2 - MONTHLY REVENUE TREND
    # ----------------------------------------------

    monthly = (
        df.groupby("YearMonth")["Revenue"]
        .sum()
    )

    plt.figure(figsize=(13, 5))

    plt.plot(
        monthly.index.astype(str),
        monthly.values,
        marker="o",
        markersize=3
    )

    plt.title("Monthly Retail Revenue")
    plt.xlabel("Month")
    plt.ylabel("Total Revenue")
    plt.xticks(rotation=65)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "02_monthly_revenue_trend.png",
        dpi=300
    )
    plt.close()

    print("Saved Figure 3.2 - Monthly revenue trend")

    # ----------------------------------------------
    # FIGURE 3.3 - WEEKDAY REVENUE
    # ----------------------------------------------

    weekday_revenue = (
        df.groupby("Weekday")["Revenue"]
        .mean()
        .reindex(weekday_order)
    )

    plt.figure(figsize=(10, 5))

    plt.bar(
        weekday_revenue.index,
        weekday_revenue.values
    )

    plt.title("Average Daily Revenue by Weekday")
    plt.xlabel("Weekday")
    plt.ylabel("Average Daily Revenue")
    plt.xticks(rotation=35)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "03_weekday_revenue.png",
        dpi=300
    )
    plt.close()

    print("Saved Figure 3.3 - Weekday revenue")

    # ----------------------------------------------
    # FIGURE 3.4 - REVENUE DISTRIBUTION
    # ----------------------------------------------

    plt.figure(figsize=(10, 5))

    plt.hist(
        df["Revenue"],
        bins=40,
        edgecolor="black",
        linewidth=0.4
    )

    plt.title("Distribution of Daily Revenue")
    plt.xlabel("Daily Revenue")
    plt.ylabel("Number of Calendar Days")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "04_revenue_distribution.png",
        dpi=300
    )
    plt.close()

    print("Saved Figure 3.4 - Revenue distribution")

    # ----------------------------------------------
    # FIGURE 3.5 - CALENDAR COMPLETENESS
    # ----------------------------------------------

    day_counts = df["HasTransactions"].value_counts()

    counts = [
        int(day_counts.get(1, 0)),
        int(day_counts.get(0, 0))
    ]

    plt.figure(figsize=(8, 5))

    bars = plt.bar(
        ["Recorded transactions", "No recorded transactions"],
        counts
    )

    for bar, count in zip(bars, counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            str(count),
            ha="center"
        )

    plt.title("Calendar-Day Transaction Coverage")
    plt.ylabel("Number of Days")
    plt.ylim(0, max(counts) * 1.15)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "05_calendar_completeness.png",
        dpi=300
    )
    plt.close()

    print("Saved Figure 3.5 - Calendar completeness")

    # ----------------------------------------------
    # SUMMARY STATISTICS
    # ----------------------------------------------

    summary = {
        "Calendar Days": len(df),
        "Days With Transactions": int(
            df["HasTransactions"].sum()
        ),
        "Days Without Transactions": int(
            (df["HasTransactions"] == 0).sum()
        ),
        "Mean Daily Revenue": df["Revenue"].mean(),
        "Median Daily Revenue": df["Revenue"].median(),
        "Minimum Daily Revenue": df["Revenue"].min(),
        "Maximum Daily Revenue": df["Revenue"].max(),
        "Standard Deviation": df["Revenue"].std(),
        "Total Recorded Revenue": df["Revenue"].sum(),
        "Zero Revenue Days": int(
            (df["Revenue"] == 0).sum()
        ),
        "Missing Target Values": int(
            df["NextDayRevenue"].isna().sum()
        )
    }

    summary_df = pd.DataFrame(
        list(summary.items()),
        columns=["Metric", "Value"]
    )

    summary_df.to_csv(SUMMARY_FILE, index=False)

    print("\nSUMMARY STATISTICS")
    print("-" * 55)

    for metric, value in summary.items():
        if isinstance(value, float):
            print(f"{metric}: {value:,.2f}")
        else:
            print(f"{metric}: {value:,}")

    print("\nWEEKDAY AVERAGE REVENUE")
    print("-" * 55)

    print(weekday_revenue.round(2).to_string())

    print("\nOUTPUT LOCATION")
    print(OUTPUT_DIR)

    print("\nTASK 3 STEP 2 COMPLETED SUCCESSFULLY")


# ==================================================
# MAIN
# ==================================================

def main():
    df = load_data()
    run_eda(df)


if __name__ == "__main__":
    main()
