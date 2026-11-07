import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from modules.data_collection import download_stock_data
from modules.preprocessing import preprocess_data
from modules.feature_engineering import create_features
from modules.models import train_models, predict_next_day, FEATURE_COLUMNS
from modules.evaluation import evaluate_models


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="StockVision AI",
    page_icon="📈",
    layout="wide"
)


# ==========================================
# CUSTOM DESIGN
# ==========================================

st.markdown("""
<style>
.stApp {
    background-color: #0b1120;
    color: #f1f5f9;
}

section[data-testid="stSidebar"] {
    background-color: #111827;
}

h1, h2, h3 {
    color: #60a5fa;
}

div[data-testid="stMetric"] {
    background-color: #172033;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #26354d;
}

.stButton button {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
    height: 45px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# STOCK DATABASE
# ==========================================

STOCKS = {
    "HDFC Bank": "HDFCBANK.NS",
    "Reliance Industries": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "SBI": "SBIN.NS",
    "ITC": "ITC.NS",
    "Larsen & Toubro": "LT.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "Axis Bank": "AXISBANK.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "Tata Motors": "TMPV.NS",
    "Mahindra & Mahindra": "M&M.NS",
    "Sun Pharma": "SUNPHARMA.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Titan": "TITAN.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "NTPC": "NTPC.NS",
    "Power Grid": "POWERGRID.NS",
    "Tata Steel": "TATASTEEL.NS",
    "UltraTech Cement": "ULTRACEMCO.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Bajaj Finserv": "BAJAJFINSV.NS",
    "Wipro": "WIPRO.NS",
    "HCL Technologies": "HCLTECH.NS",
    "Tech Mahindra": "TECHM.NS",
    "Adani Enterprises": "ADANIENT.NS",
    "Adani Ports": "ADANIPORTS.NS",
    "Coal India": "COALINDIA.NS",
    "ONGC": "ONGC.NS",
    "Indian Oil": "IOC.NS",
    "Hindalco": "HINDALCO.NS",
    "JSW Steel": "JSWSTEEL.NS",
    "Grasim": "GRASIM.NS",
    "Nestle India": "NESTLEIND.NS",
    "Dr Reddys": "DRREDDY.NS",
    "Cipla": "CIPLA.NS",
    "Apollo Hospitals": "APOLLOHOSP.NS",
    "Eicher Motors": "EICHERMOT.NS",
    "Hero MotoCorp": "HEROMOTOCO.NS",
    "Bharat Electronics": "BEL.NS",
    "Trent": "TRENT.NS",
    "Shriram Finance": "SHRIRAMFIN.NS",
    "Tata Consumer": "TATACONSUM.NS",
    "IndusInd Bank": "INDUSINDBK.NS",
    "Bajaj Auto": "BAJAJ-AUTO.NS",
    "TVS Motor": "TVSMOTOR.NS",
    "Jio Financial": "JIOFIN.NS",
    "Eternal": "ETERNAL.NS"
}


# ==========================================
# HEADER
# ==========================================

st.title("📈 STOCKVISION AI")

st.markdown(
    "### Intelligent Stock Market Analysis & Forecasting"
)

st.caption(
    "Machine Learning • Technical Indicators • Risk Analytics"
)

st.divider()


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("⚙️ Analysis Configuration")

stock_name = st.sidebar.selectbox(
    "Select Company",
    list(STOCKS.keys())
)

custom_ticker = st.sidebar.text_input(
    "Custom Yahoo Finance Ticker",
    placeholder="Example: INFY.NS"
)

ticker = (
    custom_ticker.strip().upper()
    if custom_ticker.strip()
    else STOCKS[stock_name]
)

period_option = st.sidebar.selectbox(
    "Historical Period",
    ["1 Year", "2 Years", "5 Years"]
)

periods = {
    "1 Year": "1y",
    "2 Years": "2y",
    "5 Years": "5y"
}

period = periods[period_option]

model_option = st.sidebar.selectbox(
    "Machine Learning Model",
    [
        "Compare All Models",
        "Linear Regression",
        "Random Forest",
        "Extra Trees",
        "XGBoost"
    ]
)

st.sidebar.caption("Data source: Yahoo Finance")

run_analysis = st.sidebar.button(
    "🚀 Run Analysis",
    use_container_width=True
)


# ==========================================
# CHART FUNCTIONS
# ==========================================

def create_price_chart(data, ticker):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["Close"],
            name="Closing Price",
            line=dict(color="#38bdf8", width=2)
        )
    )

    for column, color in [
        ("SMA_20", "#facc15"),
        ("SMA_50", "#fb923c")
    ]:

        if column in data.columns:

            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data[column],
                    name=column,
                    line=dict(color=color, width=1.5)
                )
            )

    fig.update_layout(
        title=f"{ticker} - Historical Price",
        template="plotly_dark",
        height=450,
        xaxis_title="Date",
        yaxis_title="Price (INR)",
        hovermode="x unified"
    )

    return fig


def create_prediction_chart(results):

    fig = go.Figure()

    for name, result in results.items():

        actual = result["actual"]

        fig.add_trace(
            go.Scatter(
                x=actual.index,
                y=actual,
                name=f"{name} Actual"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=actual.index,
                y=result["predictions"],
                name=f"{name} Predicted",
                line=dict(dash="dash")
            )
        )

    fig.update_layout(
        title="Actual vs Predicted Prices",
        template="plotly_dark",
        height=450,
        hovermode="x unified"
    )

    return fig


def create_indicator_chart(data):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["RSI"],
            name="RSI",
            line=dict(color="#a78bfa")
        )
    )

    fig.add_hline(
        y=70,
        line_dash="dash",
        line_color="red"
    )

    fig.add_hline(
        y=30,
        line_dash="dash",
        line_color="green"
    )

    fig.update_layout(
        title="Relative Strength Index",
        template="plotly_dark",
        height=350,
        yaxis_title="RSI"
    )

    return fig


def create_error_chart(results):

    fig = go.Figure()

    for name, result in results.items():

        actual = result["actual"]

        predicted = np.asarray(result["predictions"])

        errors = actual.values - predicted

        fig.add_trace(
            go.Scatter(
                x=actual.index,
                y=errors,
                name=name,
                mode="lines"
            )
        )

    fig.add_hline(y=0, line_dash="dash")

    fig.update_layout(
        title="Prediction Error Analysis",
        template="plotly_dark",
        height=400,
        yaxis_title="Actual - Predicted"
    )

    return fig


# ==========================================
# MAIN ANALYSIS
# ==========================================

if run_analysis:

    try:

        # DOWNLOAD DATA

        with st.spinner("Downloading stock market data..."):

            raw_data = download_stock_data(
                ticker,
                period=period
            )

        if raw_data is None or raw_data.empty:

            st.error("No stock data received.")

            st.stop()

        st.success(
            f"Downloaded {len(raw_data)} historical records."
        )

        # PREPROCESS DATA

        with st.spinner("Preprocessing market data..."):

            clean_data = preprocess_data(raw_data)

        # FEATURE ENGINEERING

        with st.spinner("Generating technical indicators..."):

            featured_data = create_features(clean_data)

        usable_records = len(featured_data)

        st.info(
            f"Usable records after feature engineering: {usable_records}"
        )

        # VALIDATE TRAINING DATA

        training_data = featured_data.dropna(
            subset=["Target_Close"]
        )

        if len(training_data) < 60:

            st.error(
                f"Only {len(training_data)} training records "
                "are available. At least 60 are required."
            )

            st.stop()

        missing_features = [
            col for col in FEATURE_COLUMNS
            if col not in featured_data.columns
        ]

        if missing_features:

            st.error(
                f"Missing model features: {missing_features}"
            )

            st.stop()

        # TRAIN MODELS

        with st.spinner("Training machine learning models..."):

            results = train_models(featured_data)

        # EVALUATION

        with st.spinner("Evaluating model performance..."):

            evaluation_results = evaluate_models(results)

        # SELECT MODELS

        if model_option == "Compare All Models":

            selected_models = list(results.keys())

        else:

            selected_models = [model_option]

        # STOCK OVERVIEW

        latest_price = float(
            featured_data["Close"].iloc[-1]
        )

        latest_date = featured_data.index[-1].date()

        st.divider()

        st.subheader(
            f"📊 {stock_name} ({ticker})"
        )

        st.caption(
            f"Latest available market data: {latest_date}"
        )

        st.subheader("Market Overview")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Latest Price",
            f"₹{latest_price:,.2f}"
        )

        c2.metric(
            "SMA 20",
            f"₹{featured_data['SMA_20'].iloc[-1]:,.2f}"
        )

        c3.metric(
            "SMA 50",
            f"₹{featured_data['SMA_50'].iloc[-1]:,.2f}"
        )

        c4.metric(
            "RSI",
            f"{featured_data['RSI'].iloc[-1]:.2f}"
        )

        # NEXT-DAY PREDICTIONS

        st.divider()

        st.subheader("🤖 Next Trading Day Price Estimates")

        prediction_columns = st.columns(
            len(selected_models)
        )

        for i, name in enumerate(selected_models):

            model = results[name]["model"]

            predicted_price = predict_next_day(
                featured_data,
                model
            )

            change = predicted_price - latest_price

            percent_change = (
                change / latest_price
            ) * 100

            with prediction_columns[i]:

                st.metric(
                    name,
                    f"₹{predicted_price:,.2f}",
                    f"{percent_change:+.2f}%"
                )

                if change > 0:

                    st.success("Model estimate: Upward")

                elif change < 0:

                    st.error("Model estimate: Downward")

                else:

                    st.info("Model estimate: Flat")

        # HISTORICAL PRICE

        st.divider()

        st.subheader("📈 Historical Stock Performance")

        st.plotly_chart(
            create_price_chart(featured_data, ticker),
            use_container_width=True
        )

        # TECHNICAL INDICATORS

        st.divider()

        st.subheader("Technical Indicator Analysis")

        st.plotly_chart(
            create_indicator_chart(featured_data),
            use_container_width=True
        )

        # MODEL EVALUATION

        st.divider()

        st.subheader("Model Performance Evaluation")

        metric_data = []

        for name in selected_models:

            metrics = evaluation_results[name]

            metric_data.append({
                "Model": name,
                "MAE": round(metrics["MAE"], 4),
                "RMSE": round(metrics["RMSE"], 4),
                "R2 Score": round(metrics["R2 Score"], 4),
                "MAPE": round(metrics["MAPE"], 4),
                "Directional Accuracy": round(
                    metrics.get("Directional Accuracy", 0), 2
                )
            })

        metrics_df = pd.DataFrame(metric_data)

        st.dataframe(
            metrics_df,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Lower MAE, RMSE and MAPE indicate smaller prediction "
            "errors. R² is a goodness-of-fit metric."
        )

        # ACTUAL VS PREDICTED

        st.divider()

        st.subheader("Actual vs Predicted Performance")

        selected_results = {
            name: results[name]
            for name in selected_models
        }

        st.plotly_chart(
            create_prediction_chart(selected_results),
            use_container_width=True
        )

        # PREDICTION ERRORS

        st.divider()

        st.subheader("Prediction Error Analysis")

        st.plotly_chart(
            create_error_chart(selected_results),
            use_container_width=True
        )

        # RISK ANALYSIS

        st.divider()

        st.subheader("📉 Risk Analytics")

        daily_returns = (
            featured_data["Close"]
            .pct_change()
            .dropna()
        )

        if len(daily_returns) > 1:

            annual_volatility = (
                daily_returns.std() * np.sqrt(252) * 100
            )

            annual_return = (
                (1 + daily_returns).prod()
                ** (252 / len(daily_returns)) - 1
            ) * 100

            cumulative = (
                (1 + daily_returns).cumprod()
            )

            running_max = cumulative.cummax()

            drawdown = (
                cumulative / running_max - 1
            ) * 100

            max_drawdown = drawdown.min()

            r1, r2, r3 = st.columns(3)

            r1.metric(
                "Annualised Volatility",
                f"{annual_volatility:.2f}%"
            )

            r2.metric(
                "Annualised Historical Return",
                f"{annual_return:.2f}%"
            )

            r3.metric(
                "Maximum Drawdown",
                f"{max_drawdown:.2f}%"
            )

        # DOWNLOAD RESULTS

        st.divider()

        st.subheader("📥 Export Analysis")

        csv_data = featured_data.to_csv().encode("utf-8")

        st.download_button(
            "Download Analysis Data",
            data=csv_data,
            file_name=f"{ticker.replace('.', '_')}_analysis.csv",
            mime="text/csv"
        )

        st.success(
            "Analysis completed successfully!"
        )

        st.warning(
            "Disclaimer: This application provides experimental "
            "machine learning estimates based on historical data. "
            "It does not guarantee future prices and is not "
            "financial advice."
        )

    except Exception as error:

        st.error(
            f"Something went wrong: {error}"
        )


# WELCOME SCREEN

else:

    st.markdown(
        "## Welcome to StockVision AI"
    )

    st.write(
        "Your intelligent stock market analysis platform."
    )

    st.info(
        "Choose a company, select a historical period "
        "and choose a machine learning model from the sidebar."
    )

    a, b, c = st.columns(3)

    a.info("📈 Historical Market Analysis")

    b.info("🤖 Machine Learning Forecasting")

    c.info("📉 Risk Analytics")