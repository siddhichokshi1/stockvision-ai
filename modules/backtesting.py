
import pandas as pd
import numpy as np


def moving_average_backtest(data, short_window=20, long_window=50):

    prices = data["Close"].copy()

    short_ma = prices.rolling(short_window).mean()
    long_ma = prices.rolling(long_window).mean()

    # Long-only strategy
    position = (short_ma > long_ma).astype(float)

    # Trade using the previous day's signal
    strategy_returns = position.shift(1) * prices.pct_change()

    strategy_returns = strategy_returns.dropna()

    # Buy-and-hold comparison
    market_returns = prices.pct_change().reindex(
        strategy_returns.index
    ).fillna(0)

    strategy_equity = (1 + strategy_returns).cumprod()
    market_equity = (1 + market_returns).cumprod()

    total_return = (strategy_equity.iloc[-1] - 1) * 100
    market_return = (market_equity.iloc[-1] - 1) * 100

    volatility = strategy_returns.std() * np.sqrt(252)

    sharpe = (
        strategy_returns.mean() / strategy_returns.std()
        * np.sqrt(252)
        if strategy_returns.std() > 0 else 0
    )

    return {
        "strategy_equity": strategy_equity,
        "market_equity": market_equity,
        "strategy_return": total_return,
        "market_return": market_return,
        "sharpe": sharpe,
        "volatility": volatility * 100
    }
