
import pandas as pd
import numpy as np


def analyze_portfolio(stock_data, allocations):

    """
    stock_data: dictionary of ticker -> historical DataFrame
    allocations: dictionary of ticker -> portfolio weight
    """

    returns = {}

    for ticker, data in stock_data.items():

        returns[ticker] = data["Close"].pct_change()

    returns_df = pd.DataFrame(returns).dropna()

    weights = pd.Series(allocations)

    weights = weights.reindex(returns_df.columns)

    weights = weights / weights.sum()

    portfolio_returns = returns_df.dot(weights)

    equity_curve = (1 + portfolio_returns).cumprod()

    annual_return = (
        (1 + portfolio_returns).prod()
        ** (252 / len(portfolio_returns)) - 1
    )

    annual_volatility = (
        portfolio_returns.std() * np.sqrt(252)
    )

    sharpe = (
        annual_return / annual_volatility
        if annual_volatility > 0 else 0
    )

    return {
        "returns": portfolio_returns,
        "equity_curve": equity_curve,
        "annual_return": annual_return * 100,
        "annual_volatility": annual_volatility * 100,
        "sharpe": sharpe,
        "weights": weights
    }
