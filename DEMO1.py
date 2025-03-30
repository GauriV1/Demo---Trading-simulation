import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ta import trend, momentum, volatility
from datetime import datetime
import plotly.graph_objs as go

# Suppress SettingWithCopyWarning
pd.options.mode.chained_assignment = None  # default='warn'

# Function to fetch data with caching to improve performance
@st.cache_data(show_spinner=False)
def fetch_data(ticker, start_date, end_date):
    """
    Fetches historical stock data for the specified ticker and date range.
    Handles MultiIndex columns by flattening them to single-level.
    
    Parameters:
        ticker (str): Stock ticker symbol.
        start_date (datetime): Start date for fetching data.
        end_date (datetime): End date for fetching data.
    
    Returns:
        tuple: (data DataFrame, selected price column)
    """
    try:
        # Ensure ticker is a single string without commas or spaces
        if ',' in ticker or ' ' in ticker:
            st.error("Please enter only one ticker symbol (e.g., AAPL).")
            return None, None
        
        # Fetch data
        data = yf.download(ticker, start=start_date, end=end_date, interval='1d')
    except Exception as e:
        st.error(f"An error occurred while fetching data: {e}")
        return None, None

    if data.empty:
        st.error("No data fetched. Please check the ticker symbol and date range.")
        return None, None

    # Handle MultiIndex columns by flattening them
    if isinstance(data.columns, pd.MultiIndex):
        # Since only one ticker is fetched, select the first level (e.g., 'Close', 'Open')
        data.columns = data.columns.get_level_values(0)
        st.warning("Detected MultiIndex columns. Flattened to single-level columns.")
    
    # Define priority order for price columns
    price_columns = ['Close', 'Adj Close', 'Open', 'High', 'Low']
    selected_price_column = None
    for col in price_columns:
        if col in data.columns:
            selected_price_column = col
            if col != 'Close':
                st.warning(f"Using '{col}' as the price column instead of 'Close'.")
            break

    if not selected_price_column:
        st.error("None of the required price columns ('Close', 'Adj Close', 'Open', 'High', 'Low') are present.")
        st.write("Available columns:", data.columns.tolist())
        return None, None

    # Drop rows where the selected price column is NaN
    if selected_price_column not in data.columns:
        st.error(f"Price column '{selected_price_column}' is missing from the data.")
        return None, None

    data = data.dropna(subset=[selected_price_column])

    # Display available columns for debugging
    st.sidebar.subheader("Fetched Data Columns")
    st.sidebar.write(data.columns.tolist())

    return data, selected_price_column

# Function to calculate technical indicators
def calculate_indicators(data, price_column):
    """
    Calculates technical indicators based on the selected price column.

    Parameters:
        data (DataFrame): Stock data.
        price_column (str): The column to use for price data.

    Returns:
        DataFrame: Stock data with technical indicators.
    """
    # Simple Moving Averages
    data['SMA50'] = trend.sma_indicator(close=data[price_column], window=50)
    data['SMA200'] = trend.sma_indicator(close=data[price_column], window=200)

    # Exponential Moving Averages
    data['EMA50'] = trend.ema_indicator(close=data[price_column], window=50)
    data['EMA200'] = trend.ema_indicator(close=data[price_column], window=200)

    # Relative Strength Index
    data['RSI'] = momentum.rsi(close=data[price_column], window=14)

    # Moving Average Convergence Divergence
    macd = trend.MACD(close=data[price_column])
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()
    data['MACD_Diff'] = macd.macd_diff()

    # Bollinger Bands
    bollinger = volatility.BollingerBands(close=data[price_column], window=20, window_dev=2)
    data['Bollinger_High'] = bollinger.bollinger_hband()
    data['Bollinger_Low'] = bollinger.bollinger_lband()

    # Drop rows with any NaN in indicators
    data.dropna(inplace=True)
    return data

# Function to generate buy/sell signals
def generate_signals(data):
    """
    Generates buy and sell signals based on moving average crossovers, RSI, and MACD.

    Parameters:
        data (DataFrame): Stock data with technical indicators.

    Returns:
        DataFrame: Stock data with buy/sell signals.
    """
    data['Signal'] = 0  # 1 for Buy, -1 for Sell, 0 for Hold

    # Moving Average Crossover Strategy with RSI and MACD Confirmation
    for i in range(1, len(data)):
        # Previous and current SMA50 and SMA200
        prev_sma50 = data['SMA50'].iloc[i-1]
        prev_sma200 = data['SMA200'].iloc[i-1]
        curr_sma50 = data['SMA50'].iloc[i]
        curr_sma200 = data['SMA200'].iloc[i]

        # Previous and current MACD
        prev_macd = data['MACD'].iloc[i-1]
        prev_macd_signal = data['MACD_Signal'].iloc[i-1]
        curr_macd = data['MACD'].iloc[i]
        curr_macd_signal = data['MACD_Signal'].iloc[i]

        # Buy Signal
        if (prev_sma50 < prev_sma200) and (curr_sma50 > curr_sma200):
            if data['RSI'].iloc[i] < 30:  # RSI Confirmation
                if (prev_macd < prev_macd_signal) and (curr_macd > curr_macd_signal):  # MACD Confirmation
                    data['Signal'].iloc[i] = 1  # Buy

        # Sell Signal
        elif (prev_sma50 > prev_sma200) and (curr_sma50 < curr_sma200):
            if data['RSI'].iloc[i] > 70:  # RSI Confirmation
                if (prev_macd > prev_macd_signal) and (curr_macd < curr_macd_signal):  # MACD Confirmation
                    data['Signal'].iloc[i] = -1  # Sell

    return data

# Function to simulate trades
def simulate_trades(data, price_column, initial_capital=100000):
    """
    Simulates trades based on generated signals.

    Parameters:
        data (DataFrame): Stock data with buy/sell signals.
        price_column (str): The column to use for price data.
        initial_capital (float): Starting capital for simulation.

    Returns:
        tuple: (data DataFrame with portfolio value, buy_dates list, sell_dates list, initial_capital, portfolio_values list)
    """
    capital = initial_capital
    position = 0  # Number of shares held
    portfolio_values = []
    buy_dates = []
    sell_dates = []

    for i in range(len(data)):
        signal = data['Signal'].iloc[i]
        price = data[price_column].iloc[i]

        if signal == 1 and capital > 0:
            # Buy as many shares as possible
            shares_to_buy = capital // price
            if shares_to_buy > 0:
                capital -= shares_to_buy * price
                position += shares_to_buy
                buy_dates.append(data.index[i])
                st.write(f"**Buy:** {data.index[i].date()} | **Price:** ${price:.2f} | **Shares:** {int(shares_to_buy)}")

        elif signal == -1 and position > 0:
            # Sell all shares
            capital += position * price
            sell_dates.append(data.index[i])
            st.write(f"**Sell:** {data.index[i].date()} | **Price:** ${price:.2f}")
            position = 0

        # Calculate current portfolio value
        current_value = capital + position * price
        portfolio_values.append(current_value)

    data = data.copy()
    data['Portfolio_Value'] = portfolio_values
    return data, buy_dates, sell_dates, initial_capital, portfolio_values

# Function to plot results using Matplotlib
def plot_results(data, buy_dates, sell_dates, price_column):
    """
    Plots the stock price with buy/sell signals and the portfolio value over time using Matplotlib.

    Parameters:
        data (DataFrame): Stock data with buy/sell signals.
        buy_dates (list): List of buy signal dates.
        sell_dates (list): List of sell signal dates.
        price_column (str): The column used for price data.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Plot Price
    ax1.plot(data[price_column], label=f'{price_column} Price', alpha=0.5)
    ax1.scatter(buy_dates, data.loc[buy_dates][price_column], marker='^', color='green', label='Buy Signal', s=100)
    ax1.scatter(sell_dates, data.loc[sell_dates][price_column], marker='v', color='red', label='Sell Signal', s=100)
    ax1.set_title(f'{price_column} Price with Buy/Sell Signals')
    ax1.set_ylabel('Price ($)')
    ax1.legend()

    # Plot Portfolio Value
    ax2.plot(data['Portfolio_Value'], label='Portfolio Value', color='blue')
    ax2.set_title('Portfolio Value Over Time')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Portfolio Value ($)')
    ax2.legend()

    plt.tight_layout()
    st.pyplot(fig)

# Function to plot results using Plotly (optional)
def plot_results_interactive(data, buy_dates, sell_dates, price_column):
    """
    Plots the stock price with buy/sell signals and the portfolio value over time using Plotly.

    Parameters:
        data (DataFrame): Stock data with buy/sell signals.
        buy_dates (list): List of buy signal dates.
        sell_dates (list): List of sell signal dates.
        price_column (str): The column used for price data.
    """
    fig = go.Figure()

    # Price
    fig.add_trace(go.Scatter(x=data.index, y=data[price_column], mode='lines', name=f'{price_column} Price'))

    # Buy Signals
    fig.add_trace(go.Scatter(
        x=buy_dates,
        y=data.loc[buy_dates][price_column],
        mode='markers',
        marker=dict(symbol='triangle-up', color='green', size=12),
        name='Buy Signal'
    ))

    # Sell Signals
    fig.add_trace(go.Scatter(
        x=sell_dates,
        y=data.loc[sell_dates][price_column],
        mode='markers',
        marker=dict(symbol='triangle-down', color='red', size=12),
        name='Sell Signal'
    ))

    # Portfolio Value
    fig.add_trace(go.Scatter(x=data.index, y=data['Portfolio_Value'], mode='lines', name='Portfolio Value', yaxis='y2'))

    # Layout
    fig.update_layout(
        title=f'{price_column} Price with Buy/Sell Signals and Portfolio Value',
        xaxis=dict(title='Date'),
        yaxis=dict(title=f'{price_column} Price ($)'),
        yaxis2=dict(title='Portfolio Value ($)', overlaying='y', side='right'),
        legend=dict(x=0, y=1.1, orientation='h')
    )

    st.plotly_chart(fig, use_container_width=True)

# Function to calculate performance metrics
def calculate_performance(data, initial_capital):
    """
    Calculates and displays performance metrics of the trading simulation.

    Parameters:
        data (DataFrame): Stock data with portfolio value.
        initial_capital (float): Starting capital for simulation.
    """
    final_portfolio = data['Portfolio_Value'].iloc[-1]
    total_return = (final_portfolio - initial_capital) / initial_capital * 100

    # Calculate CAGR
    start_date = data.index[0]
    end_date = data.index[-1]
    days = (end_date - start_date).days
    years = days / 365.25
    if years > 0:
        cagr = ((final_portfolio / initial_capital) ** (1 / years) - 1) * 100
    else:
        cagr = 0.0

    # Calculate daily returns
    data['Daily_Return'] = data['Portfolio_Value'].pct_change()
    sharpe_ratio = (data['Daily_Return'].mean() / data['Daily_Return'].std()) * np.sqrt(252) if data['Daily_Return'].std() != 0 else 0.0

    # Calculate Max Drawdown
    cumulative_max = data['Portfolio_Value'].cummax()
    drawdown = (data['Portfolio_Value'] - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min() * 100

    # Calculate Win Rate
    trades = data['Signal'].diff()
    buy_trades = trades[trades == 2].index
    sell_trades = trades[trades == -2].index
    wins = 0
    total_trades = min(len(buy_trades), len(sell_trades))
    for buy, sell in zip(buy_trades, sell_trades):
        if data.loc[sell, 'Portfolio_Value'] > data.loc[buy, 'Portfolio_Value']:
            wins += 1
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0

    st.subheader("Performance Metrics")
    metrics = {
        "Initial Capital": f"${initial_capital:,.2f}",
        "Final Portfolio Value": f"${final_portfolio:,.2f}",
        "Total Return": f"{total_return:.2f}%",
        "CAGR": f"{cagr:.2f}%",
        "Sharpe Ratio": f"{sharpe_ratio:.2f}",
        "Max Drawdown": f"{max_drawdown:.2f}%",
        "Win Rate": f"{win_rate:.2f}%"
    }

    for key, value in metrics.items():
        st.write(f"**{key}:** {value}")

# Main Streamlit App
def main():
    st.set_page_config(page_title="📈 Trading Simulation with Algorithmic Strategy", layout="wide")
    st.title("📈 Trading Simulation with Algorithmic Strategy")
    st.markdown("""
    This application simulates trading strategies based on historical stock data. 
    Users can input parameters such as date range and initial capital to see how the strategy would have performed.
    """)

    # Sidebar for user inputs
    st.sidebar.header("User Inputs")
    ticker = st.sidebar.text_input("Ticker Symbol", value='AAPL').upper().strip()

    # Date inputs
    start_date = st.sidebar.date_input("Start Date", datetime(2015, 1, 1))
    end_date = st.sidebar.date_input("End Date", datetime.today())

    # Initial capital
    initial_capital = st.sidebar.number_input("Initial Capital ($)", min_value=1000, value=100000, step=1000)

    # Validate dates
    if start_date >= end_date:
        st.sidebar.error("Error: End Date must fall after Start Date.")
        st.stop()

    st.sidebar.markdown("---")
    st.sidebar.markdown("Developed by [Your Name](https://www.example.com)")

    # Fetch and process data
    with st.spinner('Fetching and processing data...'):
        data, price_column = fetch_data(ticker, start_date, end_date)
        if data is None or price_column is None:
            st.error("Failed to fetch data. Please check the ticker symbol and try again.")
            st.stop()

        data = calculate_indicators(data, price_column)
        data = generate_signals(data)
        data, buy_dates, sell_dates, initial_capital, portfolio_values = simulate_trades(data, price_column, initial_capital)

    # Display data overview
    st.subheader(f"📊 {ticker} Stock Data from {start_date} to {end_date}")
    st.write(data.tail())

    # Plot results
    st.subheader("📈 Trading Simulation Results")

    # Choose between Matplotlib and Plotly plots
    plot_choice = st.selectbox("Choose Plot Type", ["Matplotlib", "Plotly"])

    if plot_choice == "Matplotlib":
        plot_results(data, buy_dates, sell_dates, price_column)
    else:
        plot_results_interactive(data, buy_dates, sell_dates, price_column)

    # Display performance metrics
    calculate_performance(data, initial_capital)

    # Download Portfolio Data as CSV
    st.download_button(
        label="Download Portfolio Data as CSV",
        data=data.to_csv(),
        file_name='portfolio_data.csv',
        mime='text/csv',
    )

    # Additional: Display data with signals
    if st.checkbox("Show Data with Signals"):
        st.subheader("Data with Buy/Sell Signals")
        st.write(data[[price_column, 'SMA50', 'SMA200', 'RSI', 'MACD', 'MACD_Signal', 'Signal', 'Portfolio_Value']])

# Run the app
if __name__ == "__main__":
    main()
