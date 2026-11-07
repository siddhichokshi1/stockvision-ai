
import numpy as np
import pandas as pd


def calculate_risk_metrics(data):

    returns = data["Close"].pct_change().dropna()

    if len(returns) < 20:
        raise ValueError("Not enough observations for risk analysis.")

    annual_volatility = returns.std() * np.sqrt(252)

    annual_return = (
        (1 + returns).prod() ** (252 / len(returns)) - 1
    )

    # Historical 95% Value at Risk
    var_95 = returns.quantile(0.05)

    # Historical 99% Value at Risk
    var_99 = returns.quantile(0.01)

    wealth = (1 + returns).cumprod()

    drawdown = wealth / wealth.cummax() - 1

    max_drawdown = drawdown.min()

    risk_free_rate = 0.06

    sharpe_ratio = (
        (annual_return - risk_free_rate) / annual_volatility
        if annual_volatility != 0 else 0
    )

    return {
        "Annual Return": annual_return * 100,
        "Annual Volatility": annual_volatility * 100,
        "VaR 95%": var_95 * 100,
        "VaR 99%": var_99 * 100,
        "Maximum Drawdown": max_drawdown * 100,
        "Sharpe Ratio": sharpe_ratio,
        "Drawdown Series": drawdown * 100
    }
