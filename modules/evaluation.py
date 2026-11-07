
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.model_selection import TimeSeriesSplit

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor
)

from xgboost import XGBRegressor


# ==========================================
# DEFAULT MODEL CONFIGURATION
# ==========================================

def get_model_definitions():

    return {
        "Linear Regression": LinearRegression(),

        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=3,
            max_features=0.8,
            random_state=42,
            n_jobs=-1
        ),

        "Extra Trees": ExtraTreesRegressor(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        ),

        "XGBoost": XGBRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            reg_lambda=2,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
    }


# ==========================================
# COMMON METRIC CALCULATOR
# ==========================================

def calculate_metrics(actual, predicted):

    actual = np.asarray(actual, dtype=float).ravel()
    predicted = np.asarray(predicted, dtype=float).ravel()

    if len(actual) != len(predicted):
        raise ValueError(
            "Actual and predicted values must have equal lengths."
        )

    if len(actual) == 0:
        raise ValueError("Cannot evaluate empty predictions.")

    if not np.isfinite(actual).all():
        raise ValueError("Actual values contain invalid numbers.")

    if not np.isfinite(predicted).all():
        raise ValueError("Predictions contain invalid numbers.")

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    r2 = (
        r2_score(actual, predicted)
        if len(actual) >= 2 else np.nan
    )

    nonzero = actual != 0

    mape = (
        np.mean(
            np.abs(
                (actual[nonzero] - predicted[nonzero])
                / actual[nonzero]
            )
        ) * 100
        if nonzero.any() else np.nan
    )

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2 Score": float(r2),
        "MAPE": float(mape)
    }


# ==========================================
# DIRECTIONAL ACCURACY
# ==========================================

def calculate_directional_accuracy(
    actual,
    predicted,
    previous_close
):

    actual = np.asarray(actual).ravel()
    predicted = np.asarray(predicted).ravel()
    previous_close = np.asarray(previous_close).ravel()

    if not (
        len(actual) == len(predicted) == len(previous_close)
    ):
        raise ValueError("Directional inputs must have equal lengths.")

    actual_direction = actual > previous_close

    predicted_direction = predicted > previous_close

    return float(
        np.mean(
            actual_direction == predicted_direction
        ) * 100
    )


# ==========================================
# MODEL EVALUATION
# ==========================================

def evaluate_models(results):

    evaluation = {}

    for name, result in results.items():

        actual = result["actual"]
        predicted = result["predictions"]

        metrics = calculate_metrics(
            actual,
            predicted
        )

        # Directional accuracy
        if "previous_close" in result:

            metrics["Directional Accuracy"] = (
                calculate_directional_accuracy(
                    actual,
                    predicted,
                    result["previous_close"]
                )
            )

        # Training versus testing performance
        if (
            "train_actual" in result
            and "train_predictions" in result
        ):

            train_metrics = calculate_metrics(
                result["train_actual"],
                result["train_predictions"]
            )

            metrics["Training MAE"] = train_metrics["MAE"]

            metrics["Testing MAE"] = metrics["MAE"]

            metrics["Overfitting Gap"] = (
                metrics["Testing MAE"]
                - metrics["Training MAE"]
            )

        evaluation[name] = metrics

    return evaluation


# ==========================================
# NAIVE PREVIOUS-CLOSE BASELINE
# ==========================================

def evaluate_naive_baseline(results):

    baseline_results = {}

    for name, result in results.items():

        if "previous_close" not in result:
            raise ValueError(
                f"{name} is missing previous_close. "
                "Update models.py to store this value."
            )

        actual = result["actual"]

        previous_close = result["previous_close"]

        baseline_results[name] = calculate_metrics(
            actual,
            previous_close
        )

        baseline_results[name]["Model"] = name

    return baseline_results


# ==========================================
# OVERFITTING ANALYSIS
# ==========================================

def analyze_overfitting(results):

    analysis = {}

    for name, result in results.items():

        if (
            "train_actual" not in result
            or "train_predictions" not in result
        ):
            analysis[name] = {
                "Status": "Training predictions unavailable",
                "Training MAE": None,
                "Testing MAE": None,
                "Overfitting Gap": None
            }

            continue

        train_metrics = calculate_metrics(
            result["train_actual"],
            result["train_predictions"]
        )

        test_metrics = calculate_metrics(
            result["actual"],
            result["predictions"]
        )

        gap = (
            test_metrics["MAE"]
            - train_metrics["MAE"]
        )

        if gap > max(train_metrics["MAE"] * 0.25, 0.01):
            status = "Potential Overfitting"

        elif gap < -max(train_metrics["MAE"] * 0.25, 0.01):
            status = "Unusual Generalization Gap"

        else:
            status = "No Large MAE Gap Detected"

        analysis[name] = {
            "Training MAE": train_metrics["MAE"],
            "Testing MAE": test_metrics["MAE"],
            "Overfitting Gap": float(gap),
            "Status": status
        }

    return analysis


# ==========================================
# WALK-FORWARD VALIDATION
# ==========================================

def walk_forward_validation(
    data,
    feature_columns,
    target_column="Target_Close",
    n_splits=5,
    test_size=None,
    model_definitions=None
):

    if data is None or data.empty:
        raise ValueError("Input data is empty.")

    if n_splits < 2:
        raise ValueError("At least two validation splits are required.")

    required = list(feature_columns) + [
        target_column,
        "Close"
    ]

    missing = [
        col for col in required
        if col not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df = data.copy().sort_index()

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Exclude rows without a known target.
    # The latest row cannot be evaluated yet.
    df = df.dropna(subset=required)

    if len(df) < n_splits + 2:
        raise ValueError(
            "Insufficient observations for walk-forward validation."
        )

    X = df[feature_columns]
    y = df[target_column]

    # Previous day's close is the naive prediction.
    previous_close = df["Close"]

    splitter = TimeSeriesSplit(
        n_splits=n_splits,
        test_size=test_size
    )

    if model_definitions is None:
        model_definitions = get_model_definitions()

    fold_results = {
        name: [] for name in model_definitions
    }

    fold_results["Naive Baseline"] = []

    prediction_results = {
        name: {
            "actual": [],
            "predictions": [],
            "previous_close": []
        }
        for name in model_definitions
    }

    prediction_results["Naive Baseline"] = {
        "actual": [],
        "predictions": [],
        "previous_close": []
    }

    # ======================================
    # CHRONOLOGICAL VALIDATION
    # ======================================

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(X),
        start=1
    ):

        print(f"\nWalk-forward fold {fold}/{n_splits}")

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        baseline = previous_close.iloc[test_idx]

        # Naive baseline
        baseline_metrics = calculate_metrics(
            y_test,
            baseline
        )

        baseline_metrics["Fold"] = fold

        fold_results["Naive Baseline"].append(
            baseline_metrics
        )

        prediction_results["Naive Baseline"][
            "actual"
        ].extend(y_test.tolist())

        prediction_results["Naive Baseline"][
            "predictions"
        ].extend(baseline.tolist())

        prediction_results["Naive Baseline"][
            "previous_close"
        ].extend(baseline.tolist())

        # Machine learning models
        for name, template in model_definitions.items():

            from sklearn.base import clone

            model = clone(template)

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            metrics = calculate_metrics(
                y_test,
                predictions
            )

            metrics["Fold"] = fold

            metrics["Directional Accuracy"] = (
                calculate_directional_accuracy(
                    y_test,
                    predictions,
                    baseline
                )
            )

            fold_results[name].append(metrics)

            prediction_results[name]["actual"].extend(
                y_test.tolist()
            )

            prediction_results[name]["predictions"].extend(
                predictions.tolist()
            )

            prediction_results[name]["previous_close"].extend(
                baseline.tolist()
            )

            print(
                f"{name}: MAE={metrics['MAE']:.3f} | "
                f"RMSE={metrics['RMSE']:.3f} | "
                f"R2={metrics['R2 Score']:.4f}"
            )

    # ======================================
    # AGGREGATE RESULTS
    # ======================================

    summary = {}

    for name, result in prediction_results.items():

        metrics = calculate_metrics(
            result["actual"],
            result["predictions"]
        )

        metrics["Directional Accuracy"] = (
            calculate_directional_accuracy(
                result["actual"],
                result["predictions"],
                result["previous_close"]
            )
        )

        metrics["Average Fold MAE"] = float(
            np.mean([
                fold["MAE"]
                for fold in fold_results[name]
            ])
        )

        metrics["Fold Results"] = fold_results[name]

        summary[name] = metrics

    print("\nWalk-forward validation completed.")

    return summary


# ==========================================
# PREDICTION ERROR ANALYSIS
# ==========================================

def prediction_error_analysis(results):

    error_results = {}

    for name, result in results.items():

        actual = np.asarray(
            result["actual"],
            dtype=float
        ).ravel()

        predicted = np.asarray(
            result["predictions"],
            dtype=float
        ).ravel()

        errors = actual - predicted

        error_results[name] = {
            "Mean Error": float(np.mean(errors)),
            "Median Error": float(np.median(errors)),
            "Error Std": float(np.std(errors)),
            "Maximum Error": float(np.max(np.abs(errors))),
            "Error Distribution": errors
        }

    return error_results
