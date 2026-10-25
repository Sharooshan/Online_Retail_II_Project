
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# TASK 3 - ADDITIONAL MODEL EVALUATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TASK3_DIR = PROJECT_ROOT / "results" / "task3"
MODEL_DIR = TASK3_DIR / "models"
CHART_DIR = MODEL_DIR / "charts"

PREDICTIONS_FILE = MODEL_DIR / "test_predictions.csv"
DAILY_FILE = TASK3_DIR / "daily_revenue.csv"
IMPORTANCE_FILE = MODEL_DIR / "feature_importance.csv"

CHART_DIR.mkdir(parents=True, exist_ok=True)

SELECTED_MODEL = "Random Forest"
PREDICTION_COLUMN = "Random_Forest_Prediction"


# ============================================================
# LOAD AND VALIDATE DATA
# ============================================================

def load_data():

    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Test predictions not found:\n{PREDICTIONS_FILE}"
        )

    if not DAILY_FILE.exists():
        raise FileNotFoundError(
            f"Daily revenue data not found:\n{DAILY_FILE}"
        )

    predictions = pd.read_csv(
        PREDICTIONS_FILE,
        parse_dates=["TargetDate"]
    )

    daily = pd.read_csv(
        DAILY_FILE,
        parse_dates=["Date"]
    )

    required_predictions = {
        "TargetDate",
        "ActualRevenue",
        PREDICTION_COLUMN
    }

    missing = required_predictions - set(predictions.columns)

    if missing:
        raise ValueError(
            f"Missing prediction columns: {sorted(missing)}"
        )

    required_daily = {
        "Date",
        "Revenue",
        "HasTransactions"
    }

    if not required_daily.issubset(daily.columns):
        raise ValueError(
            "Required daily revenue columns are missing."
        )

    if predictions["TargetDate"].duplicated().any():
        raise ValueError(
            "Duplicate prediction dates detected."
        )

    if daily["Date"].duplicated().any():
        raise ValueError(
            "Duplicate daily dates detected."
        )

    # Match test predictions to actual calendar-day status.
    merged = predictions.merge(
        daily[["Date", "Revenue", "HasTransactions"]],
        left_on="TargetDate",
        right_on="Date",
        how="left",
        validate="one_to_one"
    )

    if merged["HasTransactions"].isna().any():
        raise ValueError(
            "Some test dates could not be matched."
        )

    if not np.allclose(
        merged["ActualRevenue"],
        merged["Revenue"],
        rtol=1e-7,
        atol=0.01
    ):
        raise ValueError(
            "Test actuals do not match daily revenue."
        )

    merged = merged.sort_values(
        "TargetDate"
    ).reset_index(drop=True)

    merged["Prediction"] = merged[PREDICTION_COLUMN]

    merged["Error"] = (
        merged["ActualRevenue"] - merged["Prediction"]
    )

    merged["AbsoluteError"] = (
        merged["Error"].abs()
    )

    print("Test predictions loaded successfully.")
    print(f"Selected model: {SELECTED_MODEL}")
    print(f"Test observations: {len(merged):,}")

    return merged


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(data):

    if data.empty:
        return {
            "Observations": 0,
            "MAE": np.nan,
            "RMSE": np.nan,
            "R2": np.nan,
            "MeanError": np.nan
        }

    actual = data["ActualRevenue"].to_numpy()
    predicted = data["Prediction"].to_numpy()

    return {
        "Observations": len(data),

        "MAE": mean_absolute_error(
            actual,
            predicted
        ),

        "RMSE": np.sqrt(
            mean_squared_error(
                actual,
                predicted
            )
        ),

        "R2": (
            r2_score(actual, predicted)
            if len(data) >= 2
            else np.nan
        ),

        "MeanError": data["Error"].mean()
    }


# ============================================================
# GROUP-BASED EVALUATION
# ============================================================

def evaluate_groups(df):

    groups = {
        "All Test Days": df,

        "Recorded Transaction Days": df[
            df["HasTransactions"] == 1
        ],

        "No Recorded Transaction Days": df[
            df["HasTransactions"] == 0
        ]
    }

    records = []

    for name, subset in groups.items():

        metrics = calculate_metrics(subset)

        records.append({
            "Group": name,
            **metrics
        })

    results = pd.DataFrame(records)

    results.to_csv(
        MODEL_DIR / "group_evaluation.csv",
        index=False
    )

    print("\nGROUP-BASED EVALUATION")
    print("-" * 70)

    print(
        results.round(3).to_string(index=False)
    )

    return results


# ============================================================
# IDENTIFY LARGEST PREDICTION ERRORS
# ============================================================

def identify_largest_errors(df):

    largest_errors = (
        df.sort_values(
            "AbsoluteError",
            ascending=False
        )
        .head(10)
        [
            [
                "TargetDate",
                "ActualRevenue",
                "Prediction",
                "Error",
                "AbsoluteError",
                "HasTransactions"
            ]
        ]
    )

    # Save original numeric precision to CSV.
    largest_errors.to_csv(
        MODEL_DIR / "largest_prediction_errors.csv",
        index=False
    )

    print("\nTOP 10 LARGEST PREDICTION ERRORS")
    print("-" * 70)

    # Format only numeric columns to avoid datetime warnings.
    formatted_errors = largest_errors.copy()

    numeric_columns = [
        "ActualRevenue",
        "Prediction",
        "Error",
        "AbsoluteError"
    ]

    formatted_errors[numeric_columns] = (
        formatted_errors[numeric_columns].round(2)
    )

    print(
        formatted_errors.to_string(index=False)
    )

    return largest_errors


# ============================================================
# VISUALISATIONS
# ============================================================

def create_charts(df, group_results):

    # Figure 1 - MAE by transaction-day group.
    group_data = group_results[
        group_results["Group"] != "All Test Days"
    ]

    plt.figure(figsize=(10, 5))

    plt.bar(
        group_data["Group"],
        group_data["MAE"]
    )

    plt.title(
        "Random Forest MAE by Transaction-Day Status"
    )

    plt.ylabel("Mean Absolute Error")
    plt.xticks(rotation=15, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        CHART_DIR / "06_group_mae_comparison.png",
        dpi=300
    )

    plt.close()

    # Figure 2 - Error distribution.
    plt.figure(figsize=(10, 5))

    plt.hist(
        df["Error"],
        bins=30,
        edgecolor="black",
        linewidth=0.5
    )

    plt.axvline(
        x=0,
        linestyle="--",
        linewidth=1.5
    )

    plt.title(
        "Random Forest Prediction Error Distribution"
    )

    plt.xlabel(
        "Actual Revenue - Predicted Revenue"
    )

    plt.ylabel("Number of Test Days")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        CHART_DIR / "07_error_distribution.png",
        dpi=300
    )

    plt.close()

    # Figure 3 - Absolute errors over time.
    plt.figure(figsize=(13, 5))

    plt.plot(
        df["TargetDate"],
        df["AbsoluteError"],
        linewidth=1.2
    )

    plt.title(
        "Random Forest Absolute Prediction Error Over Time"
    )

    plt.xlabel("Prediction Date")
    plt.ylabel("Absolute Error")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        CHART_DIR / "08_absolute_error_over_time.png",
        dpi=300
    )

    plt.close()

    print(
        "\nThree additional evaluation charts saved."
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def examine_feature_importance():

    if not IMPORTANCE_FILE.exists():

        print(
            "\nFeature importance file not found. "
            "Skipping feature-importance summary."
        )

        return

    importance = pd.read_csv(
        IMPORTANCE_FILE
    )

    print("\nFEATURE IMPORTANCE FILE")
    print("-" * 55)

    print(
        importance.head(12).to_string(index=False)
    )

    print(
        "\nFeature importance indicates the model's "
        "relative reliance on features, not causation."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("TASK 3 - ADDITIONAL MODEL EVALUATION")
    print("=" * 65)

    df = load_data()

    group_results = evaluate_groups(df)

    identify_largest_errors(df)

    create_charts(
        df,
        group_results
    )

    examine_feature_importance()

    print("\nOUTPUT FILES")
    print("-" * 65)

    print(
        MODEL_DIR / "group_evaluation.csv"
    )

    print(
        MODEL_DIR / "largest_prediction_errors.csv"
    )

    print(
        CHART_DIR / "06_group_mae_comparison.png"
    )

    print(
        CHART_DIR / "07_error_distribution.png"
    )

    print(
        CHART_DIR / "08_absolute_error_over_time.png"
    )

    print(
        "\nTASK 3 STEP 5 COMPLETED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()
