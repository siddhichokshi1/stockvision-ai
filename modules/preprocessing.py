
import pandas as pd
import numpy as np


REQUIRED_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]


def preprocess_data(data):
    """
    Clean and validate historical stock market data.

    Parameters
    ----------
    data : pd.DataFrame
        Raw OHLCV stock data.

    Returns
    -------
    pd.DataFrame
        Cleaned historical stock data.
    """

    if data is None or data.empty:
        raise ValueError("Input stock data is empty.")

    df = data.copy()

    # Remove accidental MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Validate required columns
    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df = df[REQUIRED_COLUMNS].copy()

    # Ensure datetime index
    df.index = pd.to_datetime(df.index)

    # Remove timezone information
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Sort chronologically
    df.sort_index(inplace=True)

    # Remove duplicate dates
    df = df[~df.index.duplicated(keep="last")]

    # Convert values to numeric
    for column in REQUIRED_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Replace infinite values
    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    # Remove rows with missing OHLC values
    df.dropna(
        subset=["Open", "High", "Low", "Close"],
        inplace=True
    )

    # Missing volume is treated as unavailable
    # rather than silently assumed to be zero.
    df.dropna(subset=["Volume"], inplace=True)

    # Remove invalid prices
    price_columns = ["Open", "High", "Low", "Close"]

    df = df[
        (df[price_columns] > 0).all(axis=1)
    ]

    # Remove negative volume
    df = df[df["Volume"] >= 0]

    # Validate OHLC relationships
    valid_ohlc = (
        (df["High"] >= df["Low"]) &
        (df["High"] >= df["Open"]) &
        (df["High"] >= df["Close"]) &
        (df["Low"] <= df["Open"]) &
        (df["Low"] <= df["Close"])
    )

    df = df[valid_ohlc]

    if df.empty:
        raise ValueError(
            "No valid records remain after preprocessing."
        )

    # Ensure numerical data types
    df = df.astype(float)

    print("Preprocessing completed successfully!")
    print(f"Clean records: {len(df)}")

    return df
