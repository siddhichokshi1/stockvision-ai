
import pandas as pd
import numpy as np


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


def create_features(data):

    if data is None or data.empty:
        raise ValueError("Input data is empty.")

    required = [
        "Open", "High", "Low", "Close", "Volume"
    ]

    missing = [
        col for col in required
        if col not in data.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = data.copy()
    df.sort_index(inplace=True)

    # -----------------------------
    # MOVING AVERAGES
    # -----------------------------

    for window in [7, 20, 50, 100, 200]:

        df[f"SMA_{window}"] = (
            df["Close"]
            .rolling(window=window, min_periods=1)
            .mean()
        )

    # -----------------------------
    # EXPONENTIAL MOVING AVERAGES
    # -----------------------------

    df["EMA_12"] = df["Close"].ewm(
        span=12,
        adjust=False
    ).mean()

    df["EMA_26"] = df["Close"].ewm(
        span=26,
        adjust=False
    ).mean()

    # -----------------------------
    # DAILY RETURNS
    # -----------------------------

    df["Daily_Return"] = df["Close"].pct_change()

    # -----------------------------
    # VOLATILITY
    # -----------------------------

    df["Volatility"] = (
        df["Daily_Return"]
        .rolling(window=20, min_periods=5)
        .std()
    )

    # -----------------------------
    # RSI
    # -----------------------------

    delta = df["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=5
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=5
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["RSI"] = 100 - (100 / (1 + rs))

    df.loc[
        (avg_loss == 0) & (avg_gain > 0),
        "RSI"
    ] = 100

    df.loc[
        (avg_loss == 0) & (avg_gain == 0),
        "RSI"
    ] = 50

    # -----------------------------
    # MACD
    # -----------------------------

    df["MACD"] = df["EMA_12"] - df["EMA_26"]

    df["MACD_Signal"] = df["MACD"].ewm(
        span=9,
        adjust=False
    ).mean()

    df["MACD_Histogram"] = (
        df["MACD"] - df["MACD_Signal"]
    )

    # -----------------------------
    # BOLLINGER BANDS
    # -----------------------------

    rolling_mean = df["Close"].rolling(
        20, min_periods=5
    ).mean()

    rolling_std = df["Close"].rolling(
        20, min_periods=5
    ).std()

    df["BB_Upper"] = rolling_mean + (2 * rolling_std)
    df["BB_Lower"] = rolling_mean - (2 * rolling_std)

    # -----------------------------
    # MOMENTUM
    # -----------------------------

    df["Momentum_5"] = df["Close"].pct_change(5)
    df["Momentum_10"] = df["Close"].pct_change(10)

    # -----------------------------
    # VOLUME INDICATORS
    # -----------------------------

    df["Volume_SMA_20"] = (
        df["Volume"]
        .rolling(20, min_periods=5)
        .mean()
    )

    df["Volume_Change"] = df["Volume"].pct_change()

    # -----------------------------
    # NEXT-DAY TARGET
    # -----------------------------

    df["Target_Close"] = df["Close"].shift(-1)

    # -----------------------------
    # CLEAN DATA
    # -----------------------------

    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    # Fill initial indicator warm-up values
    # using only information available up to that date.

    df[FEATURE_COLUMNS] = (
        df[FEATURE_COLUMNS].ffill()
    )

    # Remove rows with remaining missing features.
    # Preserve final row for next-day prediction.

    df.dropna(
        subset=FEATURE_COLUMNS,
        inplace=True
    )

    if len(df) < 50:
        raise ValueError(
            f"Only {len(df)} usable records remain. "
            "More historical data is required."
        )

    print(
        f"Feature engineering completed: {len(df)} usable records."
    )

    return df
