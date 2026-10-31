
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .task3_modelling import FEATURES


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DAILY_REVENUE_FILE = (
    PROJECT_ROOT / "results" / "task3" / "daily_revenue.csv"
)

MODEL_FILE = (
    PROJECT_ROOT / "results" / "task3"
    / "models" / "selected_model.joblib"
)


def load_daily_history():
    """Load and validate historical daily revenue."""

    if not DAILY_REVENUE_FILE.exists():
        raise FileNotFoundError(
            f"Daily revenue file not found: {DAILY_REVENUE_FILE}"
        )

    df = pd.read_csv(
        DAILY_REVENUE_FILE,
        parse_dates=["Date"]
    )

    required = {"Date", "Revenue", "HasTransactions"}

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing historical columns: {sorted(missing)}"
        )

    df = df.sort_values("Date").reset_index(drop=True)

    if df["Date"].duplicated().any():
        raise ValueError("Duplicate calendar dates detected.")

    if not df["Date"].diff().dropna().eq(
        pd.Timedelta(days=1)
    ).all():
        raise ValueError(
            "Daily revenue history must contain consecutive dates."
        )

    if len(df) < 30:
        raise ValueError(
            "At least 30 historical days are required."
        )

    for column in ["Revenue", "HasTransactions"]:
        df[column] = pd.to_numeric(
            df[column], errors="raise"
        )

    if df[["Revenue", "HasTransactions"]].isna().any().any():
        raise ValueError("Historical features contain missing values.")

    if (df["Revenue"] < 0).any():
        raise ValueError("Revenue must not be negative.")

    if not df["HasTransactions"].isin([0, 1]).all():
        raise ValueError(
            "HasTransactions must contain only 0 or 1."
        )

    return df


def create_next_day_features(history):
    """Build one forecast row using only known historical data."""

    if len(history) < 30:
        raise ValueError("At least 30 days of history are required.")

    latest_date = history["Date"].iloc[-1]
    target_date = latest_date + pd.Timedelta(days=1)

    revenue = history["Revenue"]
    transactions = history["HasTransactions"]

    features = {
        "DayOfWeek": target_date.day_of_week,
        "Month": target_date.month,
        "DayOfMonth": target_date.day,
        "IsWeekend": int(target_date.day_of_week in [5, 6]),
        "Revenue_Lag1": float(revenue.iloc[-1]),
        "Revenue_Lag2": float(revenue.iloc[-2]),
        "Revenue_Lag7": float(revenue.iloc[-7]),
        "Revenue_Rolling7": float(revenue.iloc[-7:].mean()),
        "Revenue_Rolling14": float(revenue.iloc[-14:].mean()),
        "Revenue_Rolling30": float(revenue.iloc[-30:].mean()),
        "HasTransactions_Lag1": int(transactions.iloc[-1]),
        "HasTransactions_Lag7": int(transactions.iloc[-7]),
    }

    feature_frame = pd.DataFrame(
        [features],
        columns=FEATURES
    )

    if feature_frame.isna().any().any():
        raise ValueError("Forecast features contain missing values.")

    return target_date, feature_frame


def forecast_next_day():
    """Load the saved model and generate a new one-day forecast."""

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Saved model not found: {MODEL_FILE}"
        )

    history = load_daily_history()

    target_date, features = create_next_day_features(
        history
    )

    # Load only a model artifact created by this project.
    artifact = joblib.load(MODEL_FILE)

    model = artifact["model"]
    model_features = artifact["features"]
    model_name = artifact["model_name"]

    if list(model_features) != list(FEATURES):
        raise ValueError(
            "Saved model features do not match forecasting features."
        )

    if model_name != "Random Forest":
        raise ValueError(
            f"Expected Random Forest, found {model_name}."
        )

    prediction = float(
        np.maximum(model.predict(features)[0], 0.0)
    )

    return {
        "forecast_date": target_date,
        "latest_history_date": history["Date"].iloc[-1],
        "predicted_revenue": prediction,
        "model_name": model_name,
        "features": features,
    }


if __name__ == "__main__":

    result = forecast_next_day()

    print("\nNEXT-DAY REVENUE FORECAST")
    print("-" * 45)

    print(
        "Historical data through:",
        result["latest_history_date"].date()
    )

    print(
        "Forecast date:",
        result["forecast_date"].date()
    )

    print(
        "Model:",
        result["model_name"]
    )

    print(
        "Predicted revenue: "
        f"£{result['predicted_revenue']:,.2f}"
    )
