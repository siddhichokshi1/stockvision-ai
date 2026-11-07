
import plotly.graph_objects as go
from plotly.subplots import make_subplots


BACKGROUND = "#0B1220"
PANEL = "#111C2E"
BLUE = "#38BDF8"
PURPLE = "#A78BFA"
GREEN = "#34D399"


def apply_theme(fig, title):

    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor=BACKGROUND,
        plot_bgcolor=PANEL,
        font=dict(color="#E5E7EB"),
        margin=dict(l=20, r=20, t=65, b=20),
        legend=dict(orientation="h", y=1.02)
    )

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#263449")

    return fig


def price_chart(data):

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data["Open"],
        high=data["High"],
        low=data["Low"],
        close=data["Close"],
        name="Price"
    ))

    for column, color in [
        ("SMA_20", BLUE),
        ("SMA_50", PURPLE),
        ("EMA_12", GREEN)
    ]:

        fig.add_trace(go.Scatter(
            x=data.index,
            y=data[column],
            name=column,
            line=dict(color=color, width=1.5)
        ))

    return apply_theme(fig, "Price Action & Moving Averages")


def technical_chart(data):

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.55, 0.22, 0.23],
        subplot_titles=("Stock Price", "RSI", "MACD")
    )

    fig.add_trace(
        go.Scatter(
            x=data.index, y=data["Close"],
            name="Close", line=dict(color=BLUE)
        ), row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=data.index, y=data["BB_Upper"],
            name="Upper Band",
            line=dict(color=PURPLE, dash="dot")
        ), row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=data.index, y=data["BB_Lower"],
            name="Lower Band",
            line=dict(color=PURPLE, dash="dot")
        ), row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=data.index, y=data["RSI"],
            name="RSI", line=dict(color="#FBBF24")
        ), row=2, col=1
    )

    fig.add_hline(y=70, line_dash="dash", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", row=2, col=1)

    fig.add_trace(
        go.Scatter(
            x=data.index, y=data["MACD"],
            name="MACD", line=dict(color=BLUE)
        ), row=3, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=data.index, y=data["MACD_Signal"],
            name="Signal", line=dict(color=PURPLE)
        ), row=3, col=1
    )

    return apply_theme(fig, "Technical Indicator Analysis")


def prediction_chart(results):

    fig = go.Figure()

    for name, result in results.items():

        actual = result["actual"]

        fig.add_trace(go.Scatter(
            x=actual.index,
            y=actual.values,
            name=f"{name} Actual",
            line=dict(color="#94A3B8", width=1)
        ))

        fig.add_trace(go.Scatter(
            x=actual.index,
            y=result["predictions"],
            name=f"{name} Predicted",
            line=dict(dash="dash", width=2)
        ))

    return apply_theme(fig, "Actual vs Predicted Prices")


def comparison_chart(evaluation):

    fig = go.Figure()

    names = list(evaluation.keys())

    fig.add_trace(go.Bar(
        x=names,
        y=[evaluation[n]["MAE"] for n in names],
        name="MAE",
        marker_color=BLUE
    ))

    fig.add_trace(go.Bar(
        x=names,
        y=[evaluation[n]["RMSE"] for n in names],
        name="RMSE",
        marker_color=PURPLE
    ))

    fig.update_layout(barmode="group")

    return apply_theme(fig, "Machine Learning Model Comparison")
