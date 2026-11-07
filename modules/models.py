
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor


# ==========================================
# FEATURE CONFIGURATION
# Must match feature_engineering.py
# ==========================================

FEATURE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Volume",
    "SMA_7",
    "SMA_20",
    "SMA_50",
    "SMA_100",
    "SMA_200",
    "EMA_12",
    "EMA_26",
    "RSI",
    "MACD",
    "MACD_Signal",
    "MACD_Histogram",
    "BB_Upper",
    "BB_Lower",
    "Daily_Return",
    "Volatility",
    "Momentum_5",
    "Momentum_10",
    "Volume_SMA_20",
    "Volume_Change"
]


# ==========================================
# MODEL DEFINITIONS
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
# TRAIN MACHINE LEARNING MODELS
# ==========================================

def train_models(data):

    if data is None or data.empty:
        raise ValueError("No stock data available.")

    required_columns = FEATURE_COLUMNS + [
        "Target_Close",
        "Close"
    ]

    missing = [
        col for col in required_columns
        if col not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # --------------------------------------
    # CLEAN TRAINING DATA
    # --------------------------------------

    training_data = data.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna(
        subset=required_columns
    ).copy()

    if len(training_data) < 80:
        raise ValueError(
            f"Only {len(training_data)} usable records. "
            "At least 80 records are required."
        )

    X = training_data[FEATURE_COLUMNS]

    y = training_data["Target_Close"]

    # --------------------------------------
    # CHRONOLOGICAL TRAIN-TEST SPLIT
    # --------------------------------------

    split_index = int(len(training_data) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    if len(X_test) < 10:
        raise ValueError(
            "Insufficient testing records."
        )

    # --------------------------------------
    # PREVIOUS CLOSE BASELINE
    # --------------------------------------

    previous_close_train = (
        training_data["Close"].iloc[:split_index]
    )

    previous_close_test = (
        training_data["Close"].iloc[split_index:]
    )

    # --------------------------------------
    # MODEL DEFINITIONS
    # --------------------------------------

    model_definitions = get_model_definitions()

    results = {}

    # ======================================
    # TRAIN AND EVALUATE EACH MODEL
    # ======================================

    for name, model in model_definitions.items():

        print(f"\nTraining {name}...")

        # Train model
        model.fit(X_train, y_train)

        # Training predictions
        train_predictions = model.predict(X_train)

        # Testing predictions
        predictions = model.predict(X_test)

        # ----------------------------------
        # REGRESSION METRICS
        # ----------------------------------

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        )

        r2 = r2_score(
            y_test,
            predictions
        )

        # ----------------------------------
        # DIRECTIONAL ACCURACY
        # ----------------------------------

        actual_direction = (
            y_test.values > previous_close_test.values
        )

        predicted_direction = (
            predictions > previous_close_test.values
        )

        directional_accuracy = (
            np.mean(
                predicted_direction == actual_direction
            ) * 100
        )

        # ----------------------------------
        # TRAINING METRICS
        # ----------------------------------

        train_mae = mean_absolute_error(
            y_train,
            train_predictions
        )

        train_rmse = np.sqrt(
            mean_squared_error(
                y_train,
                train_predictions
            )
        )

        train_r2 = r2_score(
            y_train,
            train_predictions
        )

        # ----------------------------------
        # SAVE RESULTS
        # ----------------------------------

        results[name] = {

            # Trained model
            "model": model,

            # Testing predictions
            "predictions": predictions,

            # Actual testing values
            "actual": y_test,

            # Training predictions
            "train_predictions": train_predictions,

            # Actual training values
            "train_actual": y_train,

            # Previous close baseline
            "previous_close": previous_close_test,

            # Training and testing features
            "X_train": X_train,
            "X_test": X_test,

            # Regression metrics
            "MAE": float(mae),
            "RMSE": float(rmse),
            "R2": float(r2),

            # Training metrics
            "Training MAE": float(train_mae),
            "Training RMSE": float(train_rmse),
            "Training R2": float(train_r2),

            # Directional accuracy
            "Directional Accuracy": float(
                directional_accuracy
            )
        }

        # ----------------------------------
        # DISPLAY MODEL PERFORMANCE
        # ----------------------------------

        print(
            f"{name} completed\n"
            f"Testing MAE: {mae:.2f}\n"
            f"Testing RMSE: {rmse:.2f}\n"
            f"Testing R2: {r2:.4f}\n"
            f"Training MAE: {train_mae:.2f}\n"
            f"Training RMSE: {train_rmse:.2f}\n"
            f"Training R2: {train_r2:.4f}\n"
            f"Directional Accuracy: "
            f"{directional_accuracy:.2f}%"
        )

    print("\nAll machine learning models trained successfully!")

    return results


# ==========================================
# NEXT TRADING DAY PREDICTION
# ==========================================

def predict_next_day(data, model):

    if data is None or data.empty:
        raise ValueError(
            "No data available for prediction."
        )

    missing = [
        col for col in FEATURE_COLUMNS
        if col not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing prediction features: {missing}"
        )

    # Get latest available features
    latest_features = data[
        FEATURE_COLUMNS
    ].iloc[[-1]].copy()

    # Handle invalid values
    latest_features = latest_features.replace(
        [np.inf, -np.inf],
        np.nan
    )

    if latest_features.isna().any().any():
        raise ValueError(
            "Latest record contains missing features."
        )

    # Predict next trading day's closing price
    prediction = model.predict(
        latest_features
    )

    return float(prediction[0])
