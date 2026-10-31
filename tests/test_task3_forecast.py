
import numpy as np
import pandas as pd

from src.retail_pipeline.task3_feature_engineering import create_features
from src.retail_pipeline.task3_forecast import create_next_day_features
from src.retail_pipeline.task3_modelling import FEATURES


def test_next_day_forecast_features_match_training_pipeline():

    # 45 consecutive days of synthetic historical data.
    dates = pd.date_range(
        start="2011-10-01",
        periods=45,
        freq="D"
    )

    revenue = np.array(
        [float(1000 + i * 100) for i in range(45)]
    )

    history = pd.DataFrame({
        "Date": dates,
        "Revenue": revenue,
        "HasTransactions": np.ones(45, dtype=int),
    })

    # Create known next-day targets for the historical
    # feature-engineering pipeline.
    history["NextDayRevenue"] = (
        history["Revenue"].shift(-1)
    )

    historical_features = create_features(history)

    # Forecast from the day before the final historical date.
    forecast_history = history.iloc[:-1].copy()

    forecast_date, forecast_features = (
        create_next_day_features(forecast_history)
    )

    expected_row = historical_features.loc[
        historical_features["TargetDate"] == forecast_date
    ]

    assert len(expected_row) == 1

    assert list(forecast_features.columns) == FEATURES

    np.testing.assert_allclose(
        forecast_features.iloc[0].to_numpy(dtype=float),
        expected_row[FEATURES].iloc[0].to_numpy(dtype=float),
    )
