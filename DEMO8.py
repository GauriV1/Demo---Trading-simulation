# trading_simulation.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ta import trend, momentum, volatility, volume
from datetime import datetime, timedelta
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import warnings
import logging
import requests
from bs4 import BeautifulSoup

# -----------------------
# Setup and Configuration
# -----------------------

warnings.filterwarnings('ignore')

logging.basicConfig(
    filename='trade_simulation.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

st.set_page_config(page_title="AI-Powered Trading Simulation", layout="wide")
st.title("📈 AI-Powered Trading Simulation Platform")

# -----------------------
# Sidebar: Investor Preferences
# -----------------------

st.sidebar.header("Investor Preferences")

# Initial Investment (in dollars)
initial_investment = st.sidebar.number_input("Initial Investment ($)", min_value=1000, value=1000, step=100)

# Investment Horizon
investment_horizon = st.sidebar.selectbox("Investment Horizon", options=["Long-term", "Short-term"], index=0)

# Grouping Preferences (Sectors)
# We'll fetch sectors from the S&P500 table.
@st.cache_data
def get_sp500_df():
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        df = pd.read_html(url, header=0)[0]
        df["Symbol"] = df["Symbol"].str.replace(".", "-")
        # Use "GICS Sector" column for grouping
        return df[["Symbol", "GICS Sector"]]
    except Exception as e:
        st.error(f"Error fetching S&P 500 data: {e}")
        logging.error(f"Error fetching S&P 500 data: {e}")
        return pd.DataFrame(columns=["Symbol", "GICS Sector"])

sp500_df = get_sp500_df()
unique_sectors = sorted(sp500_df["GICS Sector"].unique().tolist())

# Add an extra option for "Climate Friendly" (hardcoded list)
sector_options = unique_sectors + ["Climate Friendly"]
selected_groups = st.sidebar.multiselect("Grouping Preferences (Sectors)", options=sector_options, default=unique_sectors)

# Trading Frequency
trading_frequency = st.sidebar.selectbox("Trading Frequency", options=["Daily", "Weekly", "Monthly"], index=0)

# Date Range Input based on Investment Horizon
today = datetime.today()
if investment_horizon == "Long-term":
    date_range = st.sidebar.date_input("Select Date Range (Max 5 Years)", 
                                       value=[today - timedelta(days=365*2), today],
                                       min_value=today - timedelta(days=365*5),
                                       max_value=today)
else:
    date_range = st.sidebar.date_input("Select Date Range (Max 6 Months)", 
                                       value=[today - timedelta(days=90), today],
                                       min_value=today - timedelta(days=180),
                                       max_value=today)
start_date, end_date = date_range[0], date_range[1]

run_simulation = st.sidebar.button("Run Simulation")

# -----------------------
# Helper Functions
# -----------------------

# Function to determine the number of tickers to invest in based on initial investment
def get_num_tickers(capital):
    if 1000 <= capital < 2000:
        return 3
    elif 2000 <= capital < 5000:
        return 4
    elif capital >= 5000:
        extra = (capital - 5000) // 1000
        return 4 + int(extra)
    else:
        return 1

# Function to get the list of S&P500 tickers and sectors from the dataframe
def get_sp500_tickers_and_sectors():
    df = get_sp500_df()
    return df

# Filter tickers by grouping preference
def filter_tickers_by_group(selected_groups):
    df = get_sp500_tickers_and_sectors()
    # If "Climate Friendly" is selected, add a hardcoded list.
    climate_friendly = ["NEE", "ENPH", "FSLR", "RUN", "SEDG", "CSIQ", "SPWR"]
    # Filter by sectors that are not "Climate Friendly"
    if "Climate Friendly" in selected_groups:
        # Include tickers from the selected sectors (excluding "Climate Friendly") plus the hardcoded list
        groups = [g for g in selected_groups if g != "Climate Friendly"]
        filtered_df = df[df["GICS Sector"].isin(groups)]
        tickers = filtered_df["Symbol"].tolist() + climate_friendly
    else:
        filtered_df = df[df["GICS Sector"].isin(selected_groups)]
        tickers = filtered_df["Symbol"].tolist()
    return list(set(tickers))  # unique tickers

# -----------------------
# Data Fetching
# -----------------------

@st.cache_data
def fetch_data(tickers, start, end, interval):
    data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
            if not df.empty:
                data[ticker] = df
            else:
                logging.warning(f"No data fetched for {ticker}.")
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {e}")
            logging.error(f"Error fetching data for {ticker}: {e}")
    return data

# -----------------------
# Technical Indicators and Signal Generation
# -----------------------

def add_technical_indicators(df):
    # Ensure "Close" is a 1D Series
    close_series = df["Close"].squeeze()
    df["EMA10"] = trend.ema_indicator(close_series, window=10)
    df["EMA30"] = trend.ema_indicator(close_series, window=30)
    # For long-term strategy, add EMA50 and EMA200
    df["EMA50"] = trend.ema_indicator(close_series, window=50)
    df["EMA200"] = trend.ema_indicator(close_series, window=200)
    df["RSI"] = momentum.rsi(close_series, window=14)
    df.dropna(inplace=True)
    return df

def generate_trade_signals(df, investment_horizon):
    df = df.copy()
    if investment_horizon == "Long-term":
        # Buy when EMA50 > EMA200, else sell.
        df["Signal"] = np.where(df["EMA50"] > df["EMA200"], 1, 0)
    else:
        # Short-term: Use RSI. Buy when RSI < 30, sell when RSI > 70.
        df["Signal"] = 0
        df["Signal"] = np.where(df["RSI"] < 30, 1, df["Signal"])
        df["Signal"] = np.where(df["RSI"] > 70, 0, df["Signal"])
    return df

# -----------------------
# Trade Simulation (Per Ticker)
# -----------------------

def simulate_trades_single(df, capital, investment_horizon):
    """
    Simulate trades on a single ticker's DataFrame using a rule-based strategy.
    Capital is the allocated capital for this ticker.
    """
    df = add_technical_indicators(df)
    df = generate_trade_signals(df, investment_horizon)
    
    position = None
    entry_price = None
    shares = 0
    portfolio = capital
    portfolio_values = []
    trades = []
    transaction_cost_pct = 0.1 / 100  # 0.1% transaction cost
    
    for i in range(1, len(df)):
        current_signal = df["Signal"].iloc[i]
        prev_signal = df["Signal"].iloc[i-1]
        price = df["Close"].iloc[i]
        
        # Buy condition: no position and signal flips to 1
        if position is None and prev_signal == 0 and current_signal == 1:
            shares = portfolio / price
            cost = portfolio * transaction_cost_pct
            portfolio -= (shares * price + cost)
            entry_price = price
            position = "long"
            trades.append({
                "Type": "Buy",
                "Price": price,
                "Shares": shares,
                "Cost": shares * price + cost,
                "Date": df.index[i]
            })
            logging.info(f"Bought at ${price:.2f} on {df.index[i]}")
        
        # Sell condition: if in a long position and signal flips to 0
        elif position == "long" and prev_signal == 1 and current_signal == 0:
            proceeds = shares * price
            cost = proceeds * transaction_cost_pct
            portfolio += (proceeds - cost)
            profit = (price - entry_price) * shares - cost
            trades.append({
                "Type": "Sell",
                "Price": price,
                "Shares": shares,
                "Proceeds": proceeds - cost,
                "Profit": profit,
                "Date": df.index[i]
            })
            logging.info(f"Sold at ${price:.2f} on {df.index[i]} for profit ${profit:.2f}")
            position = None
            entry_price = None
            shares = 0
        
        portfolio_values.append(portfolio)
    
    # Force exit if still holding a position at end of data
    if position == "long":
        price = df["Close"].iloc[-1]
        proceeds = shares * price
        cost = proceeds * transaction_cost_pct
        portfolio += (proceeds - cost)
        profit = (price - entry_price) * shares - cost
        trades.append({
            "Type": "Sell",
            "Price": price,
            "Shares": shares,
            "Proceeds": proceeds - cost,
            "Profit": profit,
            "Date": df.index[-1]
        })
        logging.info(f"Forced sell at ${price:.2f} on {df.index[-1]} for profit ${profit:.2f}")
        portfolio_values.append(portfolio)
    
    return portfolio, trades, portfolio_values

# -----------------------
# Aggregated Simulation for Basket of Tickers
# -----------------------

def simulate_basket_trades(data, selected_tickers, investment_horizon, initial_investment):
    """
    Given a dictionary of data and a list of selected tickers, split the initial investment equally,
    simulate trades for each ticker individually, and aggregate the results.
    Returns overall final portfolio value, combined trades, and a dictionary of final portfolio per ticker.
    """
    num_tickers = len(selected_tickers)
    if num_tickers == 0:
        return 0, [], {}
    allocated_capital = initial_investment / num_tickers
    overall_final = 0
    combined_trades = []
    final_values = {}
    
    for ticker in selected_tickers:
        if ticker not in data:
            continue
        df = data[ticker]
        final_portfolio, trades, _ = simulate_trades_single(df, allocated_capital, investment_horizon)
        overall_final += final_portfolio
        final_values[ticker] = final_portfolio
        # Add ticker info to each trade record
        for t in trades:
            t["Ticker"] = ticker
        combined_trades.extend(trades)
    
    return overall_final, combined_trades, final_values

# -----------------------
# Performance Metrics & Visualization
# -----------------------

def calculate_performance(initial, final, trades):
    total_return = ((final - initial) / initial) * 100
    num_trades = len(trades)
    wins = sum(1 for trade in trades if trade.get("Profit", 0) > 0)
    win_rate = (wins / num_trades) * 100 if num_trades > 0 else 0
    profit_factor = (sum(trade["Profit"] for trade in trades if trade.get("Profit", 0) > 0) /
                     abs(sum(trade["Profit"] for trade in trades if trade.get("Profit", 0) < 0))) if (num_trades - wins) > 0 else np.inf
    return {
        "Initial Capital": f"${initial:.2f}",
        "Final Portfolio Value": f"${final:.2f}",
        "Total Return": f"{total_return:.2f}%",
        "Number of Trades": num_trades,
        "Win Rate": f"{win_rate:.2f}%",
        "Profit Factor": f"{profit_factor:.2f}"
    }

def plot_portfolio(portfolio_values):
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(list(range(len(portfolio_values))), portfolio_values, label="Portfolio Value")
    ax.set_xlabel("Trade Number")
    ax.set_ylabel("Portfolio Value ($)")
    ax.set_title("Portfolio Value Over Time")
    ax.legend()
    st.pyplot(fig)

def plot_trade_history(trades_df):
    st.subheader("📝 Trade History")
    if trades_df.empty:
        st.write("No trades were executed.")
    else:
        st.write(trades_df)

def plot_performance_metrics(performance):
    st.subheader("📊 Performance Metrics")
    metrics_df = pd.DataFrame.from_dict(performance, orient="index", columns=["Value"])
    st.write(metrics_df)

def plot_candlestick_with_signals(data, trades_df):
    for ticker, df in data.items():
        df = add_technical_indicators(df)
        trades = trades_df[trades_df["Ticker"] == ticker]
        if df.empty or trades.empty:
            continue
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                            row_width=[0.2, 0.7], subplot_titles=(f"{ticker} Price", "RSI"))
        fig.add_trace(go.Candlestick(x=df.index,
                                     open=df["Open"],
                                     high=df["High"],
                                     low=df["Low"],
                                     close=df["Close"],
                                     name="Candlestick"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], line=dict(color="blue", width=1), name="EMA50"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA200"], line=dict(color="orange", width=1), name="EMA200"), row=1, col=1)
        buy_signals = trades[trades["Type"] == "Buy"]
        fig.add_trace(go.Scatter(x=buy_signals["Date"], y=buy_signals["Price"],
                                 mode="markers", marker_symbol="triangle-up",
                                 marker_color="green", marker_size=10, name="Buy"), row=1, col=1)
        sell_signals = trades[trades["Type"] == "Sell"]
        fig.add_trace(go.Scatter(x=sell_signals["Date"], y=sell_signals["Price"],
                                 mode="markers", marker_symbol="triangle-down",
                                 marker_color="red", marker_size=10, name="Sell"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], line=dict(color="purple", width=1), name="RSI"), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        fig.update_layout(title=f"{ticker} Price Chart with Buy/Sell Signals and RSI", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

def plot_interactive_charts(data, trades_df, portfolio_values):
    st.subheader("📉 Price Charts with Buy/Sell Signals")
    plot_candlestick_with_signals(data, trades_df)
    st.subheader("📈 Portfolio Performance Over Time")
    plot_portfolio(portfolio_values)

def download_trade_history(trades_df):
    if trades_df.empty:
        st.write("No trades to download.")
    else:
        csv = trades_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Trade History as CSV",
            data=csv,
            file_name="trade_history.csv",
            mime="text/csv",
        )

# -----------------------
# Main Simulation Execution
# -----------------------

if run_simulation:
    st.header("🔍 Simulation Results")
    
    # Determine trading interval based on frequency
    if trading_frequency == "Daily":
        interval = "1d"
    elif trading_frequency == "Weekly":
        interval = "1wk"
    else:
        interval = "1mo"
    
    # Get tickers based on grouping preferences from S&P 500
    sp500_df = get_sp500_df()
    if selected_groups:
        # Separate out "Climate Friendly" from the sectors
        climate_friendly = ["NEE", "ENPH", "FSLR", "RUN", "SEDG", "CSIQ", "SPWR"]
        groups = [g for g in selected_groups if g != "Climate Friendly"]
        if groups:
            filtered_df = sp500_df[sp500_df["GICS Sector"].isin(groups)]
        else:
            filtered_df = sp500_df
        # Combine filtered tickers with climate-friendly ones if selected
        tickers = filtered_df["Symbol"].tolist()
        if "Climate Friendly" in selected_groups:
            tickers += climate_friendly
    else:
        tickers = get_sp500_tickers()
    
    selected_tickers = list(set(tickers))
    
    # Determine number of tickers based on initial investment
    num_tickers = get_num_tickers(initial_investment)
    st.write(f"Based on an initial investment of ${initial_investment}, the strategy will invest in {num_tickers} tickers.")
    
    # Fetch historical data for all selected tickers
    st.subheader("Fetching Historical Data...")
    data = fetch_data(selected_tickers, start_date, end_date, interval)
    if not data:
        st.error("No data fetched. Please check your ticker selections and date range.")
    else:
        st.success("Data fetched successfully!")
        st.subheader("Selected Tickers and Data Availability")
        ticker_availability = {ticker: "Available" if ticker in data else "Not Available" for ticker in selected_tickers}
        st.write(ticker_availability)
        
        # Calculate cumulative returns for filtering best-performing tickers
        st.subheader("Selecting Best Performing Tickers")
        performance = {}
        for ticker, df in data.items():
            try:
                if "Close" not in df.columns or len(df["Close"]) < 2:
                    logging.warning(f"Insufficient 'Close' data for {ticker}. Skipping.")
                    performance[ticker] = -np.inf
                    continue
                initial_price = df["Close"].iloc[0]
                final_price = df["Close"].iloc[-1]
                cumulative_return = ((final_price - initial_price) / initial_price) * 100
                performance[ticker] = cumulative_return
            except Exception as e:
                logging.error(f"Error calculating performance for {ticker}: {e}")
                performance[ticker] = -np.inf
        performance_df = pd.DataFrame(list(performance.items()), columns=["Ticker", "Cumulative_Return"])
        performance_df["Cumulative_Return"] = pd.to_numeric(performance_df["Cumulative_Return"], errors="coerce")
        performance_df.dropna(subset=["Cumulative_Return"], inplace=True)
        performance_df.sort_values(by="Cumulative_Return", ascending=False, inplace=True)
        
        # Select top num_tickers tickers based on performance
        best_tickers = performance_df.head(num_tickers)["Ticker"].tolist()
        st.write(f"**Best {num_tickers} tickers selected:**")
        st.write(best_tickers)
        # Filter data to include only these tickers
        filtered_data = {ticker: data[ticker] for ticker in best_tickers if ticker in data}
        
        # Simulate trades for each ticker in the basket.
        # Allocate the initial investment equally.
        allocated_capital = initial_investment / num_tickers
        overall_final = 0
        combined_trades = []
        per_ticker_final = {}
        
        for ticker, df in filtered_data.items():
            final_val, trades, _ = simulate_trades_single(df, allocated_capital, investment_horizon)
            overall_final += final_val
            per_ticker_final[ticker] = final_val
            for trade in trades:
                trade["Ticker"] = ticker
            combined_trades.extend(trades)
        
        st.write("Final Portfolio by Ticker:", per_ticker_final)
        st.write(f"Overall Final Portfolio Value: ${overall_final:.2f}")
        
        # For portfolio performance over time, we cannot easily aggregate different time series,
        # so we display final results and trade history.
        trades_df = pd.DataFrame(combined_trades)
        st.subheader("📝 Trade History")
        st.write(trades_df)
        
        performance_metrics = calculate_performance(initial_investment, overall_final, combined_trades)
        st.subheader("📊 Performance Metrics")
        st.write(pd.DataFrame.from_dict(performance_metrics, orient="index", columns=["Value"]))
        
        # Plot a simple bar chart for final portfolio by ticker
        st.subheader("Final Portfolio Allocation")
        fig, ax = plt.subplots(figsize=(10,6))
        ax.bar(per_ticker_final.keys(), per_ticker_final.values())
        ax.set_xlabel("Ticker")
        ax.set_ylabel("Final Portfolio Value ($)")
        ax.set_title("Final Portfolio Value by Ticker")
        st.pyplot(fig)
        
        download_trade_history(trades_df)
