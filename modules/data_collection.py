
import yfinance as yf
import pandas as pd


VALID_PERIODS = [
    "1mo", "3mo", "6mo",
    "1y", "2y", "5y", "10y", "max"
]

REQUIRED_COLUMNS = [
    "Open", "High", "Low", "Close", "Volume"
]


def download_stock_data(ticker, period="5y"):
    """
    Download historical stock market data from Yahoo Finance.

    Parameters
    ----------
    ticker : str
        Yahoo Finance stock ticker, e.g. HDFCBANK.NS

    period : str
        Historical data period: 1y, 2y, 5y, etc.

    Returns
    -------
    pd.DataFrame
        Historical OHLCV stock data.
    """

    ticker = ticker.strip().upper()

    if not ticker:
        raise ValueError("Please provide a valid stock ticker.")

    if period not in VALID_PERIODS:
        raise ValueError(
            f"Invalid period: {period}. "
            f"Choose from {VALID_PERIODS}"
        )

    try:
        print(f"Downloading data for {ticker}...")

        data = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False
        )

        if data is None or data.empty:
            raise ValueError(
                f"No historical data found for {ticker}. "
                "Check the ticker symbol."
            )

        # Handle Yahoo Finance MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):

            if ticker in data.columns.get_level_values(-1):
                data = data.xs(
                    ticker,
                    axis=1,
                    level=-1
                )

            elif ticker in data.columns.get_level_values(0):
                data = data.xs(
                    ticker,
                    axis=1,
                    level=0
                )

            else:
                data.columns = data.columns.get_level_values(0)

        # Keep only required columns
        missing = [
            col for col in REQUIRED_COLUMNS
            if col not in data.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required market columns: {missing}"
            )

        data = data[REQUIRED_COLUMNS].copy()

        # Ensure chronological order
        data.sort_index(inplace=True)

        # Remove timezone information if present
        if isinstance(data.index, pd.DatetimeIndex):
            if data.index.tz is not None:
                data.index = data.index.tz_localize(None)

        # Convert columns to numeric
        for column in REQUIRED_COLUMNS:
            data[column] = pd.to_numeric(
                data[column],
                errors="coerce"
            )

        # Remove invalid records
        data.replace(
            [float("inf"), float("-inf")],
            pd.NA,
            inplace=True
        )

        data.dropna(
            subset=REQUIRED_COLUMNS,
            inplace=True
        )

        if data.empty:
            raise ValueError(
                "No usable historical records were returned."
            )

        print(
            f"Successfully downloaded {len(data)} records "
            f"for {ticker}."
        )

        return data

    except Exception as error:

        raise RuntimeError(
            f"Failed to download data for {ticker}: {error}"
        ) from error
