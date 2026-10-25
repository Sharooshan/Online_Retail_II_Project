
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# TASK 3 - SUPERVISED MACHINE LEARNING MODELLING
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT / "results" / "task3"
    / "modelling_dataset.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "results" / "task3" / "models"
)

CHART_DIR = OUTPUT_DIR / "charts"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

FEATURES = [
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

TARGET = "NextDayRevenue"


# ============================================================
# LOAD AND VALIDATE DATA
# ============================================================

def load_data():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Modelling dataset not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["Date", "TargetDate"]
    )

    required = set(
        FEATURES + [TARGET, "Date", "TargetDate"]
    )

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    df = df.sort_values(
        "TargetDate"
    ).reset_index(drop=True)

    if df[FEATURES + [TARGET]].isna().any().any():
        raise ValueError(
            "Missing predictor or target values."
        )

    if df["TargetDate"].duplicated().any():
        raise ValueError(
            "Duplicate prediction dates detected."
        )

    if not (
        df["TargetDate"] - df["Date"]
    ).eq(pd.Timedelta(days=1)).all():
        raise ValueError(
            "Forecast horizon must be exactly one day."
        )

    if len(df) < 100:
        raise ValueError(
            "Insufficient observations for evaluation."
        )

    print(f"Modelling observations: {len(df):,}")
    print(f"Predictive features: {len(FEATURES)}")

    return df


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

def split_data(df):

    n = len(df)

    train_end = int(n * 0.70)
    validation_end = int(n * 0.85)

    train = df.iloc[:train_end].copy()
    validation = df.iloc[
        train_end:validation_end
    ].copy()
    test = df.iloc[validation_end:].copy()

    assert (
        len(train) + len(validation) + len(test)
    ) == n

    assert (
        train["TargetDate"].max()
        < validation["TargetDate"].min()
    )

    assert (
        validation["TargetDate"].max()
        < test["TargetDate"].min()
    )

    print("\nCHRONOLOGICAL DATA SPLIT")
    print("-" * 55)

    for name, subset in [
        ("Training", train),
        ("Validation", validation),
        ("Testing", test)
    ]:
        print(
            f"{name}: {len(subset):,} observations | "
            f"{subset['TargetDate'].min().date()} "
            f"to {subset['TargetDate'].max().date()}"
        )

    return train, validation, test


# ============================================================
# MODEL DEFINITIONS
# ============================================================

def create_models():

    return {
        "Linear Regression": LinearRegression(),

        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE
        )
    }


# ============================================================
# EVALUATION METRICS
# ============================================================

def evaluate(y_true, y_pred):

    y_true = np.asarray(y_true, dtype=float)

    # Revenue predictions are constrained to be nonnegative.
    y_pred = np.maximum(
        np.asarray(y_pred, dtype=float),
        0.0
    )

    return {
        "MAE": float(
            mean_absolute_error(y_true, y_pred)
        ),
        "RMSE": float(
            np.sqrt(mean_squared_error(y_true, y_pred))
        ),
        "R2": float(
            r2_score(y_true, y_pred)
        )
    }


# ============================================================
# VALIDATION COMPARISON
# ============================================================

def validate_models(train, validation):

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_val = validation[FEATURES]
    y_val = validation[TARGET]

    models = create_models()

    records = []

    # Persistence baseline: tomorrow = today.
    baseline_predictions = validation[
        "Revenue_Lag1"
    ].to_numpy()

    baseline_metrics = evaluate(
        y_val,
        baseline_predictions
    )

    records.append({
        "Model": "Persistence Baseline",
        **baseline_metrics
    })

    for name, model in models.items():

        model.fit(X_train, y_train)

        predictions = model.predict(X_val)

        metrics = evaluate(
            y_val,
            predictions
        )

        records.append({
            "Model": name,
            **metrics
        })

    results = (
        pd.DataFrame(records)
        .sort_values("MAE")
        .reset_index(drop=True)
    )

    results.to_csv(
        OUTPUT_DIR / "validation_results.csv",
        index=False
    )

    print("\nVALIDATION RESULTS")
    print("-" * 55)
    print(results.round(3).to_string(index=False))

    best_name = results.iloc[0]["Model"]

    print(f"\nBest validation model: {best_name}")

    return best_name, results


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

def test_models(train, validation, test, best_name):

    # Refit using all data available before the test period.
    development = pd.concat(
        [train, validation],
        ignore_index=True
    )

    X_dev = development[FEATURES]
    y_dev = development[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    records = []
    predictions_table = pd.DataFrame({
        "TargetDate": test["TargetDate"].values,
        "ActualRevenue": y_test.to_numpy()
    })

    all_predictions = {}

    baseline = test["Revenue_Lag1"].to_numpy()

    all_predictions["Persistence Baseline"] = baseline

    records.append({
        "Model": "Persistence Baseline",
        **evaluate(y_test, baseline)
    })

    trained_models = {}

    for name, model in create_models().items():

        model.fit(X_dev, y_dev)

        predictions = np.maximum(
            model.predict(X_test),
            0.0
        )

        all_predictions[name] = predictions
        trained_models[name] = model

        records.append({
            "Model": name,
            **evaluate(y_test, predictions)
        })

    results = (
        pd.DataFrame(records)
        .sort_values("MAE")
        .reset_index(drop=True)
    )

    for name, predictions in all_predictions.items():

        column_name = (
            name.replace(" ", "_") + "_Prediction"
        )

        predictions_table[column_name] = predictions

    results.to_csv(
        OUTPUT_DIR / "test_results.csv",
        index=False
    )

    predictions_table.to_csv(
        OUTPUT_DIR / "test_predictions.csv",
        index=False
    )

    print("\nFINAL TEST RESULTS")
    print("-" * 55)
    print(results.round(3).to_string(index=False))

    print(
        f"\nSelected model (validation): {best_name}"
    )

    selected_metrics = results[
        results["Model"] == best_name
    ].iloc[0]

    print(
        f"Selected model test MAE: "
        f"{selected_metrics['MAE']:,.2f}"
    )

    print(
        f"Selected model test RMSE: "
        f"{selected_metrics['RMSE']:,.2f}"
    )

    print(
        f"Selected model test R2: "
        f"{selected_metrics['R2']:.4f}"
    )

    return (
        results,
        predictions_table,
        all_predictions,
        trained_models
    )


# ============================================================
# GENERATE EVALUATION CHARTS
# ============================================================

def create_charts(
    validation_results,
    test_results,
    predictions_table,
    all_predictions,
    best_name,
    trained_models
):

    # Chart 1 - Validation MAE comparison
    plt.figure(figsize=(10, 5))

    plt.bar(
        validation_results["Model"],
        validation_results["MAE"]
    )

    plt.title("Validation MAE by Model")
    plt.ylabel("Mean Absolute Error")
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        CHART_DIR / "01_validation_mae.png",
        dpi=300
    )
    plt.close()

    # Chart 2 - Test MAE comparison
    plt.figure(figsize=(10, 5))

    plt.bar(
        test_results["Model"],
        test_results["MAE"]
    )

    plt.title("Test MAE by Model")
    plt.ylabel("Mean Absolute Error")
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        CHART_DIR / "02_test_mae.png",
        dpi=300
    )
    plt.close()

    # Chart 3 - Selected model: actual vs predicted
    plt.figure(figsize=(13, 5))

    plt.plot(
        predictions_table["TargetDate"],
        predictions_table["ActualRevenue"],
        label="Actual Revenue",
        linewidth=1.5
    )

    plt.plot(
        predictions_table["TargetDate"],
        all_predictions[best_name],
        label=f"Predicted - {best_name}",
        linewidth=1.5
    )

    plt.title(
        f"Actual vs Predicted Revenue - {best_name}"
    )
    plt.xlabel("Date")
    plt.ylabel("Daily Revenue")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        CHART_DIR / "03_actual_vs_predicted.png",
        dpi=300
    )
    plt.close()

    # Chart 4 - Prediction residuals
    residuals = (
        predictions_table["ActualRevenue"].to_numpy()
        - np.asarray(all_predictions[best_name])
    )

    plt.figure(figsize=(10, 5))

    plt.scatter(
        all_predictions[best_name],
        residuals,
        alpha=0.6
    )

    plt.axhline(
        y=0,
        linestyle="--",
        linewidth=1
    )

    plt.title("Prediction Residuals")
    plt.xlabel("Predicted Revenue")
    plt.ylabel("Actual - Predicted Revenue")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        CHART_DIR / "04_prediction_residuals.png",
        dpi=300
    )
    plt.close()

    # Chart 5 - Feature importance (tree models only)
    if best_name in trained_models:

        selected_model = trained_models[best_name]

        if hasattr(selected_model, "feature_importances_"):

            importance = pd.Series(
                selected_model.feature_importances_,
                index=FEATURES
            ).sort_values()

            importance.to_csv(
                OUTPUT_DIR / "feature_importance.csv",
                header=["Importance"]
            )

            plt.figure(figsize=(10, 6))

            plt.barh(
                importance.index,
                importance.values
            )

            plt.title(
                f"Feature Importance - {best_name}"
            )
            plt.xlabel("Feature Importance")
            plt.tight_layout()

            plt.savefig(
                CHART_DIR / "05_feature_importance.png",
                dpi=300
            )
            plt.close()

    print("\nEvaluation charts saved successfully.")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("TASK 3 - SUPERVISED MACHINE LEARNING")
    print("=" * 60)

    df = load_data()

    train, validation, test = split_data(df)

    best_name, validation_results = validate_models(
        train,
        validation
    )

    (
        test_results,
        predictions_table,
        all_predictions,
        trained_models
    ) = test_models(
        train,
        validation,
        test,
        best_name
    )

    create_charts(
        validation_results,
        test_results,
        predictions_table,
        all_predictions,
        best_name,
        trained_models
    )

    experiment_summary = {
        "total_observations": len(df),
        "feature_count": len(FEATURES),
        "training_observations": len(train),
        "validation_observations": len(validation),
        "test_observations": len(test),
        "selection_metric": "Validation MAE",
        "selected_model": best_name,
        "test_start": str(
            test["TargetDate"].min().date()
        ),
        "test_end": str(
            test["TargetDate"].max().date()
        )
    }

    with open(
        OUTPUT_DIR / "experiment_summary.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            experiment_summary,
            file,
            indent=4
        )

    print("\nOUTPUT FILES")
    print("-" * 60)

    print(OUTPUT_DIR / "validation_results.csv")
    print(OUTPUT_DIR / "test_results.csv")
    print(OUTPUT_DIR / "test_predictions.csv")
    print(OUTPUT_DIR / "experiment_summary.json")
    print(CHART_DIR)

    print("\nTASK 3 STEP 4 COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
