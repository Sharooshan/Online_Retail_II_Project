
from pathlib import Path

import pandas as pd
import streamlit as st

from src.retail_pipeline.task3_forecast import forecast_next_day


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Retail Revenue Forecasting",
    page_icon="📊",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# 2. FIND RESULT FILES AUTOMATICALLY
# ============================================================

def find_result_file(filename):
    """
    Search the Task 3 results directory first.
    If not found, search the results folder recursively.
    """

    task3_dir = BASE_DIR / "results" / "task3"

    direct_path = task3_dir / filename

    if direct_path.is_file():
        return direct_path

    if task3_dir.exists():
        matches = sorted(task3_dir.rglob(filename))

        if matches:
            return matches[0]

    results_dir = BASE_DIR / "results"

    if results_dir.exists():
        matches = sorted(results_dir.rglob(filename))

        if matches:
            return matches[0]

    return None


PREDICTIONS_FILE = find_result_file("test_predictions.csv")
METRICS_FILE = find_result_file("test_results.csv")


# ============================================================
# 3. APPLICATION HEADER
# ============================================================

st.title("📊 Retail Revenue Forecasting Dashboard")

st.caption(
    "Online Retail II | MSc Data Science | "
    "Machine Learning-Based Next-Day Revenue Prediction"
)


# ============================================================
# 4. LOAD PREDICTION RESULTS
# ============================================================

@st.cache_data
def load_predictions(file_path):

    df = pd.read_csv(file_path)

    required_columns = {
        "TargetDate",
        "ActualRevenue",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    df["TargetDate"] = pd.to_datetime(
        df["TargetDate"],
        errors="raise",
    )

    if df["TargetDate"].duplicated().any():
        raise ValueError(
            "Duplicate prediction dates detected."
        )

    return (
        df.sort_values("TargetDate")
        .reset_index(drop=True)
    )


@st.cache_data
def load_metrics(file_path):

    if file_path is None:
        return pd.DataFrame()

    return pd.read_csv(file_path)


if PREDICTIONS_FILE is None:

    st.error(
        "The file test_predictions.csv could not be found "
        "inside the project's results folder."
    )

    st.stop()


try:

    predictions = load_predictions(
        str(PREDICTIONS_FILE)
    )

    metrics = load_metrics(
        str(METRICS_FILE) if METRICS_FILE else None
    )

except (
    ValueError,
    OSError,
    pd.errors.ParserError,
) as error:

    st.error(
        f"Unable to load prediction results: {error}"
    )

    st.stop()


# ============================================================
# 5. IDENTIFY RANDOM FOREST PREDICTION COLUMN
# ============================================================

rf_columns = [
    column
    for column in predictions.columns
    if "random" in column.lower()
    and "forest" in column.lower()
    and "prediction" in column.lower()
]

if not rf_columns:

    st.error(
        "Random Forest prediction column not found."
    )

    st.write(
        "Available columns:",
        predictions.columns.tolist(),
    )

    st.stop()


RF_COLUMN = rf_columns[0]


# ============================================================
# 6. VALIDATE REVENUE VALUES
# ============================================================

predictions["ActualRevenue"] = pd.to_numeric(
    predictions["ActualRevenue"],
    errors="coerce",
)

predictions[RF_COLUMN] = pd.to_numeric(
    predictions[RF_COLUMN],
    errors="coerce",
)

if predictions[
    ["ActualRevenue", RF_COLUMN]
].isna().any().any():

    st.error(
        "Prediction results contain missing "
        "or invalid revenue values."
    )

    st.stop()


# ============================================================
# 7. SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("📊 Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Project Overview",
        "Predict Revenue",
        "Forecast Next Day",
        "Model Performance",
        "Dataset Information",
    ],
)

st.sidebar.divider()

st.sidebar.info(
    "This dashboard supports historical test "
    "predictions and genuine one-day-ahead "
    "forecasting using a trained Random Forest model."
)


# ============================================================
# 8. PROJECT OVERVIEW PAGE
# ============================================================

if page == "Project Overview":

    st.header("Project Overview")

    st.write(
        "This project applies machine learning "
        "techniques to historical Online Retail II "
        "transaction data to predict next-day revenue."
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Original Transactions",
        "1,067,371",
    )

    col2.metric(
        "Cleaned Transactions",
        "1,007,913",
    )

    col3.metric(
        "Calendar Observations",
        "739",
    )

    col4.metric(
        "Test Observations",
        f"{len(predictions):,}",
    )

    st.divider()

    st.subheader("Revenue Prediction Overview")

    chart_df = predictions[
        [
            "TargetDate",
            "ActualRevenue",
            RF_COLUMN,
        ]
    ].copy()

    chart_df = chart_df.rename(
        columns={
            "ActualRevenue": "Actual Revenue",
            RF_COLUMN: "Predicted Revenue",
        }
    )

    st.line_chart(
        chart_df.set_index("TargetDate")
    )

    st.caption(
        "Comparison between actual and predicted "
        "revenue for historical test observations."
    )

    st.subheader("Project Workflow")

    st.markdown(
        """
        1. Load Online Retail II transaction data.
        2. Validate and clean transaction records.
        3. Aggregate revenue into daily observations.
        4. Generate historical and calendar features.
        5. Train and evaluate regression models.
        6. Select Random Forest for prediction.
        7. Visualise historical prediction results.
        8. Generate a new next-day revenue forecast.
        """
    )


# ============================================================
# 9. PREDICT REVENUE PAGE
# ============================================================

elif page == "Predict Revenue":

    st.header("🔮 Revenue Prediction")

    st.write(
        "Select a historical test date and click "
        "**Predict Revenue** to display the "
        "Random Forest model's saved prediction."
    )

    available_dates = (
        predictions["TargetDate"]
        .dt.date
        .tolist()
    )

    selected_date = st.selectbox(
        "Select Prediction Date",
        options=available_dates,
        index=len(available_dates) - 1,
        format_func=lambda value: value.strftime(
            "%d %B %Y"
        ),
    )

    st.write(
        "Selected Date:",
        selected_date.strftime("%d %B %Y"),
    )

    if st.button(
        "🔮 Predict Revenue",
        type="primary",
        width="stretch",
    ):

        selected_row = predictions.loc[
            predictions["TargetDate"].dt.date
            == selected_date
        ].iloc[0]

        predicted_revenue = float(
            selected_row[RF_COLUMN]
        )

        actual_revenue = float(
            selected_row["ActualRevenue"]
        )

        prediction_error = (
            actual_revenue - predicted_revenue
        )

        absolute_error = abs(prediction_error)

        st.success(
            "Historical prediction loaded successfully!"
        )

        st.subheader("Prediction Results")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Predicted Revenue",
            f"£{predicted_revenue:,.2f}",
        )

        col2.metric(
            "Actual Revenue",
            f"£{actual_revenue:,.2f}",
        )

        col3.metric(
            "Absolute Prediction Error",
            f"£{absolute_error:,.2f}",
        )

        st.divider()

        st.subheader("Actual vs Predicted Revenue")

        comparison_df = pd.DataFrame(
            {
                "Revenue (£)": [
                    predicted_revenue,
                    actual_revenue,
                ]
            },
            index=[
                "Predicted Revenue",
                "Actual Revenue",
            ],
        )

        st.bar_chart(comparison_df)

        st.subheader("Prediction Interpretation")

        if predicted_revenue > actual_revenue:

            st.warning(
                "The model overestimated revenue "
                "for this historical date."
            )

        elif predicted_revenue < actual_revenue:

            st.info(
                "The model underestimated revenue "
                "for this historical date."
            )

        else:

            st.success(
                "The predicted revenue matches "
                "the actual recorded revenue."
            )

        st.caption(
            "This result is retrieved from previously "
            "generated out-of-sample predictions. "
            "It is not a newly calculated live forecast."
        )


# ============================================================
# 10. GENUINE NEXT-DAY FORECAST PAGE
# ============================================================

elif page == "Forecast Next Day":

    st.header("🔮 Genuine Next-Day Revenue Forecast")

    st.write(
        "This feature runs the saved Random Forest "
        "model to calculate a new revenue forecast "
        "for the day following the latest available "
        "historical observation."
    )

    st.info(
        "The Online Retail II dataset ends in December 2011. "
        "This feature forecasts the next historical calendar "
        "day, not tomorrow's date in 2026."
    )

    st.subheader("Generate Forecast")

    if st.button(
        "🚀 Generate Next-Day Forecast",
        type="primary",
        width="stretch",
    ):

        try:

            result = forecast_next_day()

            st.session_state["latest_forecast"] = result

        except (
            FileNotFoundError,
            ValueError,
            KeyError,
            OSError,
        ) as error:

            st.error(
                f"Unable to generate forecast: {error}"
            )

    if "latest_forecast" in st.session_state:

        result = st.session_state["latest_forecast"]

        st.success(
            "New model forecast generated successfully!"
        )

        st.subheader("Forecast Results")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Forecast Date",
            result["forecast_date"].strftime(
                "%d %B %Y"
            ),
        )

        col2.metric(
            "Predicted Revenue",
            f"£{result['predicted_revenue']:,.2f}",
        )

        col3.metric(
            "Machine Learning Model",
            result["model_name"],
        )

        st.divider()

        st.subheader("Forecast Information")

        st.write(
            "**Latest Historical Date:**",
            result["latest_history_date"].strftime(
                "%d %B %Y"
            ),
        )

        st.write(
            "**Prediction Date:**",
            result["forecast_date"].strftime(
                "%d %B %Y"
            ),
        )

        st.write(
            "**Forecast Horizon:** One calendar day"
        )

        st.write(
            "**Forecast Method:** "
            "Trained Random Forest regression model"
        )

        st.subheader("Model Input Features")

        st.write(
            "The following 12 features were calculated "
            "using historical data available before "
            "the prediction date."
        )

        st.dataframe(
            result["features"],
            width="stretch",
            hide_index=True,
        )

        st.subheader("Prediction Interpretation")

        st.info(
            "This prediction was calculated by executing "
            "the trained machine learning model. "
            "It was not retrieved from the saved "
            "test_predictions.csv file."
        )

        st.warning(
            "The forecast is an estimate, not a "
            "guaranteed revenue amount. Its accuracy "
            "for this date cannot be confirmed without "
            "the actual next-day revenue."
        )


# ============================================================
# 11. MODEL PERFORMANCE PAGE
# ============================================================

elif page == "Model Performance":

    st.header("📈 Model Performance")

    actual = predictions["ActualRevenue"]

    predicted = predictions[RF_COLUMN]

    errors = actual - predicted

    mae = errors.abs().mean()

    rmse = (
        errors.pow(2).mean()
    ) ** 0.5

    denominator = (
        (actual - actual.mean()) ** 2
    ).sum()

    if denominator > 0:

        r2 = (
            1 - errors.pow(2).sum()
            / denominator
        )

    else:

        r2 = float("nan")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Mean Absolute Error",
        f"£{mae:,.2f}",
    )

    col2.metric(
        "Root Mean Squared Error",
        f"£{rmse:,.2f}",
    )

    col3.metric(
        "R² Score",
        f"{r2:.3f}",
    )

    st.divider()

    st.subheader("Actual vs Predicted Revenue")

    performance_df = predictions[
        [
            "TargetDate",
            "ActualRevenue",
            RF_COLUMN,
        ]
    ].copy()

    performance_df = performance_df.rename(
        columns={
            "ActualRevenue": "Actual Revenue",
            RF_COLUMN: "Predicted Revenue",
        }
    )

    st.line_chart(
        performance_df.set_index("TargetDate")
    )

    st.subheader("Prediction Error Analysis")

    error_df = pd.DataFrame(
        {
            "TargetDate": predictions["TargetDate"],
            "Prediction Error": errors,
        }
    )

    st.bar_chart(
        error_df.set_index("TargetDate")
    )

    if not metrics.empty:

        st.subheader("Model Comparison")

        st.dataframe(
            metrics,
            width="stretch",
            hide_index=True,
        )

    st.info(
        "Lower MAE and RMSE values indicate "
        "smaller prediction errors. "
        "R² measures how much variation in "
        "revenue the model explains on the test set."
    )


# ============================================================
# 12. DATASET INFORMATION PAGE
# ============================================================

elif page == "Dataset Information":

    st.header("📁 Dataset Information")

    st.subheader("Online Retail II Dataset")

    st.write(
        "The Online Retail II dataset contains "
        "transaction records from a UK-based "
        "online retail business."
    )

    dataset_info = pd.DataFrame(
        {
            "Property": [
                "Dataset",
                "Source",
                "Original Records",
                "Cleaned Records",
                "Original Columns",
                "Target Variable",
                "Prediction Horizon",
                "Selected Model",
            ],
            "Value": [
                "Online Retail II",
                "UCI Machine Learning Repository",
                "1,067,371",
                "1,007,913",
                "8",
                "Next-Day Revenue",
                "One Calendar Day",
                "Random Forest Regressor",
            ],
        }
    )

    st.dataframe(
        dataset_info,
        width="stretch",
        hide_index=True,
    )

    st.subheader("Prediction Data Preview")

    st.dataframe(
        predictions.head(20),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Available Historical Test Period")

    minimum_date = predictions["TargetDate"].min()

    maximum_date = predictions["TargetDate"].max()

    st.write(
        "From:",
        minimum_date.strftime("%d %B %Y"),
    )

    st.write(
        "To:",
        maximum_date.strftime("%d %B %Y"),
    )

    st.warning(
        "The model was developed using historical "
        "retail transactions from 2009–2011. "
        "Historical test-date predictions and "
        "one-day-ahead forecasting are supported. "
        "Current-calendar forecasts require "
        "more recent sales data."
    )


# ============================================================
# 13. FOOTER
# ============================================================

st.divider()

st.caption(
    "Retail Revenue Forecasting Dashboard | "
    "MSc Data Science | "
    "Developed using Python, Pandas, "
    "Scikit-learn and Streamlit"
)
