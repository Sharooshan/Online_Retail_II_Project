
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# TASK 3 - FEATURE ENGINEERING
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT / "results" / "task3" / "daily_revenue.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "task3"

OUTPUT_FILE = (
    OUTPUT_DIR / "modelling_dataset.csv"
)

OUTPUT_PARQUET = (
    OUTPUT_DIR / "modelling_dataset.parquet"
)

FEATURE_SUMMARY = (
    OUTPUT_DIR / "feature_engineering_summary.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Daily revenue file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["Date"]
    )

    required = {
        "Date",
        "Revenue",
        "HasTransactions",
        "NextDayRevenue"
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df = df.sort_values("Date").reset_index(drop=True)

    if df["Date"].duplicated().any():
        raise ValueError("Duplicate calendar dates detected.")

    if not df["Date"].diff().dropna().eq(
        pd.Timedelta(days=1)
    ).all():
        raise ValueError(
            "The input calendar must contain consecutive dates."
        )

    if not np.allclose(
        df["NextDayRevenue"].iloc[:-1].to_numpy(),
        df["Revenue"].shift(-1).iloc[:-1].to_numpy()
    ):
        raise ValueError(
            "Next-day targets do not match the revenue series."
        )

    print("Daily revenue dataset loaded successfully.")
    print(f"Input calendar days: {len(df):,}")

    return df


# ============================================================
# CREATE PREDICTIVE FEATURES
# ============================================================

def create_features(df):

    data = df.copy()

    # Each row predicts revenue for the following day.
    data["TargetDate"] = (
        data["Date"] + pd.Timedelta(days=1)
    )

    # Calendar attributes refer to the day being predicted.
    # Updated for Pandas 3.x compatibility.
    data["DayOfWeek"] = (
        data["TargetDate"].dt.day_of_week
    )

    data["Month"] = data["TargetDate"].dt.month

    data["DayOfMonth"] = (
        data["TargetDate"].dt.day
    )

    data["IsWeekend"] = (
        data["DayOfWeek"].isin([5, 6]).astype(int)
    )

    # Revenue on the current date is known when forecasting
    # the next date.
    data["Revenue_Lag1"] = data["Revenue"]

    data["Revenue_Lag2"] = (
        data["Revenue"].shift(1)
    )

    data["Revenue_Lag7"] = (
        data["Revenue"].shift(6)
    )

    # Historical rolling windows end on the forecast origin
    # (the current date), never on the future target date.
    data["Revenue_Rolling7"] = (
        data["Revenue"]
        .rolling(window=7, min_periods=7)
        .mean()
    )

    data["Revenue_Rolling14"] = (
        data["Revenue"]
        .rolling(window=14, min_periods=14)
        .mean()
    )

    data["Revenue_Rolling30"] = (
        data["Revenue"]
        .rolling(window=30, min_periods=30)
        .mean()
    )

    # Historical transaction-presence indicators.
    data["HasTransactions_Lag1"] = (
        data["HasTransactions"]
    )

    data["HasTransactions_Lag7"] = (
        data["HasTransactions"].shift(6)
    )

    # Exclude the final date (no next-day target) and
    # initial dates without enough rolling history.
    data = data.dropna().reset_index(drop=True)

    return data


# ============================================================
# VALIDATE FEATURES
# ============================================================

def validate_features(data):

    print("\nFEATURE VALIDATION")
    print("-" * 55)

    feature_columns = [
        "DayOfWeek",
        "Month",
        "DayOfMonth",
        "IsWeekend",
        "Revenue_Lag1",
        "Revenue_Lag2",
        "Revenue_Lag7",
        "Revenue_Rolling7",
        "Revenue_Rolling14",
        "Revenue_Rolling30",
        "HasTransactions_Lag1",
        "HasTransactions_Lag7"
    ]

    assert not data[feature_columns].isna().any().any()

    assert not data["NextDayRevenue"].isna().any()

    assert (
        data["TargetDate"] > data["Date"]
    ).all()

    assert (
        data["TargetDate"] - data["Date"]
    ).eq(pd.Timedelta(days=1)).all()

    assert np.allclose(
        data["Revenue_Lag1"],
        data["Revenue"]
    )

    print(f"Feature count: {len(feature_columns)}")
    print(f"Modelling observations: {len(data):,}")
    print("Missing predictor values: 0")
    print("Missing target values: 0")
    print("Target-date alignment: PASSED")
    print("Feature validation: PASSED")

    return feature_columns


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 55)
    print("TASK 3 - FEATURE ENGINEERING")
    print("=" * 55)

    df = load_data()

    model_df = create_features(df)

    feature_columns = validate_features(model_df)

    # Retain identifiers and target, but exclude the
    # current-day Revenue and HasTransactions from the
    # final model feature list because lag equivalents
    # are already included.
    output_columns = (
        ["Date", "TargetDate"]
        + feature_columns
        + ["NextDayRevenue"]
    )

    model_df = model_df[output_columns]

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    model_df.to_parquet(
        OUTPUT_PARQUET,
        index=False,
        compression="snappy"
    )

    summary = pd.DataFrame({
        "Metric": [
            "Input Calendar Days",
            "Modelling Observations",
            "Feature Count",
            "First Prediction Date",
            "Last Prediction Date",
            "Missing Feature Values",
            "Missing Target Values"
        ],
        "Value": [
            len(df),
            len(model_df),
            len(feature_columns),
            str(model_df["TargetDate"].min().date()),
            str(model_df["TargetDate"].max().date()),
            int(model_df[feature_columns].isna().sum().sum()),
            int(model_df["NextDayRevenue"].isna().sum())
        ]
    })

    summary.to_csv(
        FEATURE_SUMMARY,
        index=False
    )

    print("\nFEATURE ENGINEERING SUMMARY")
    print("-" * 55)

    print(summary.to_string(index=False))

    print("\nFIRST FIVE MODELLING RECORDS")
    print("-" * 55)

    print(
        model_df[
            [
                "Date",
                "TargetDate",
                "Revenue_Lag1",
                "Revenue_Lag7",
                "Revenue_Rolling7",
                "NextDayRevenue"
            ]
        ].head().to_string(index=False)
    )

    print("\nOUTPUT FILES")
    print("-" * 55)

    print(OUTPUT_FILE)
    print(OUTPUT_PARQUET)
    print(FEATURE_SUMMARY)

    print("\nTASK 3 STEP 3 COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
