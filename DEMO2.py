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
        
        # Fetch data with minute interval for HFT simulation
        # Note: yfinance provides up to 7 days of minute data
        data = yf.download(tickers=ticker, start=start_date, end=end_date, interval='1m')
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
    # Exponential Moving Averages (Shorter periods for HFT simulation)
    data['EMA10'] = trend.ema_indicator(close=data[price_column], window=10)
    data['EMA30'] = trend.ema_indicator(close=data[price_column], window=30)
    data['EMA50'] = trend.ema_indicator(close=data[price_column], window=50)

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

# Function to generate buy/sell signals using vectorized operations
def generate_signals(data):
    """
    Generates buy and sell signals based on EMA crossovers and RSI.

    Parameters:
        data (DataFrame): Stock data with technical indicators.

    Returns:
        DataFrame: Stock data with buy/sell signals.
    """
    data['Signal'] = 0  # 1 for Buy, -1 for Sell, 0 for Hold

    # EMA Crossover Conditions
    data['EMA_Crossover'] = np.where(data['EMA10'] > data['EMA30'], 1, -1)
    data['EMA_Crossover_Signal'] = data['EMA_Crossover'].diff()

    # MACD Crossover Signal
    data['MACD_Crossover_Signal'] = data['MACD'] - data['MACD_Signal']
    data['MACD_Crossover_Signal'] = data['MACD_Crossover_Signal'].diff()

    # Buy Conditions
    buy_condition = (
        (data['EMA_Crossover_Signal'] == 2) &  # EMA10 crossed above EMA30
        (data['RSI'] < 35) &  # Lower RSI threshold for more buys
        (data['MACD_Crossover_Signal'] > 0)  # MACD crossed above signal line
    )
    data.loc[buy_condition, 'Signal'] = 1

    # Sell Conditions
    sell_condition = (
        (data['EMA_Crossover_Signal'] == -2) &  # EMA10 crossed below EMA30
        (data['RSI'] > 65) &  # Higher RSI threshold for more sells
        (data['MACD_Crossover_Signal'] < 0)  # MACD crossed below signal line
    )
    data.loc[sell_condition, 'Signal'] = -1

    return data

# Function to simulate trades with risk management and transaction costs
def simulate_trades(data, price_column, initial_capital=100000, transaction_cost=10, stop_loss=0.95, take_profit=1.10):
    """
    Simulates trades based on generated signals with risk management.

    Parameters:
        data (DataFrame): Stock data with buy/sell signals.
        price_column (str): The column to use for price data.
        initial_capital (float): Starting capital for simulation.
        transaction_cost (float): Fixed cost per transaction.
        stop_loss (float): Stop-loss multiplier (e.g., 0.95 for 5% loss).
        take_profit (float): Take-profit multiplier (e.g., 1.10 for 10% gain.

    Returns:
        tuple: (data DataFrame with portfolio value, buy_dates list, sell_dates list, initial_capital, portfolio_values list)
    """
    capital = initial_capital
    position = 0  # Number of shares held
    portfolio_values = []
    buy_dates = []
    sell_dates = []
    entry_price = 0  # Price at which the position was entered

    for i in range(len(data)):
        signal = data['Signal'].iloc[i]
        price = data[price_column].iloc[i]
        date = data.index[i]

        if signal == 1 and capital > (price + transaction_cost):
            # Buy as many shares as possible
            shares_to_buy = (capital - transaction_cost) // price
            if shares_to_buy > 0:
                capital -= shares_to_buy * price + transaction_cost
                position += shares_to_buy
                entry_price = price
                buy_dates.append(date)
                st.write(f"**Buy:** {date} | **Price:** ${price:.2f} | **Shares:** {int(shares_to_buy)}")

        elif position > 0:
            # Check for stop-loss or take-profit
            if price <= entry_price * stop_loss:
                # Stop-loss triggered
                capital += position * price - transaction_cost
                sell_dates.append(date)
                st.write(f"**Stop-Loss Sell:** {date} | **Price:** ${price:.2f}")
                position = 0
            elif price >= entry_price * take_profit:
                # Take-profit triggered
                capital += position * price - transaction_cost
                sell_dates.append(date)
                st.write(f"**Take-Profit Sell:** {date} | **Price:** ${price:.2f}")
                position = 0
            elif signal == -1:
                # Sell signal
                capital += position * price - transaction_cost
                sell_dates.append(date)
                st.write(f"**Sell:** {date} | **Price:** ${price:.2f}")
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
def calculate_performance(data, buy_dates, sell_dates, initial_capital):
    """
    Calculates and displays performance metrics of the trading simulation.

    Parameters:
        data (DataFrame): Stock data with portfolio value.
        buy_dates (list): List of buy signal dates.
        sell_dates (list): List of sell signal dates.
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

    # Calculate Profit Factor and Expectancy
    profits = []
    losses = []
    for buy, sell in zip(buy_dates, sell_dates):
        buy_price = data.loc[buy, 'Portfolio_Value']
        sell_price = data.loc[sell, 'Portfolio_Value']
        profit = sell_price - buy_price
        if profit > 0:
            profits.append(profit)
        else:
            losses.append(abs(profit))
    profit_factor = sum(profits) / sum(losses) if sum(losses) > 0 else np.inf
    expectancy = (sum(profits) - sum(losses)) / len(profits + losses) if len(profits + losses) > 0 else 0.0

    # Calculate Win Rate
    wins = len(profits)
    total_trades = len(profits) + len(losses)
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0

    st.subheader("Performance Metrics")
    metrics = {
        "Initial Capital": f"${initial_capital:,.2f}",
        "Final Portfolio Value": f"${final_portfolio:,.2f}",
        "Total Return": f"{total_return:.2f}%",
        "CAGR": f"{cagr:.2f}%",
        "Sharpe Ratio": f"{sharpe_ratio:.2f}",
        "Max Drawdown": f"{max_drawdown:.2f}%",
        "Profit Factor": f"{profit_factor:.2f}",
        "Expectancy": f"{expectancy:.2f}",
        "Win Rate": f"{win_rate:.2f}%"
    }

    for key, value in metrics.items():
        st.write(f"**{key}:** {value}")

# Main Streamlit App
def main():
    st.set_page_config(page_title="📈 High-Frequency Trading Simulation", layout="wide")
    st.title("📈 High-Frequency Trading Simulation with Algorithmic Strategy")
    st.markdown("""
    This application simulates a high-frequency trading (HFT) strategy based on historical stock data. 
    Users can input parameters such as date range, initial capital, and risk management settings to observe how the strategy would have performed.
    **Note:** High-frequency trading typically requires minute-level or tick-level data. This simulation uses minute-level data available via Yahoo Finance for the past 7 days.
    """)

    # Sidebar for user inputs
    st.sidebar.header("User Inputs")
    ticker = st.sidebar.text_input("Ticker Symbol", value='AAPL').upper().strip()

    # Date inputs
    today = datetime.today()
    default_start = today - pd.Timedelta(days=7)
    start_date = st.sidebar.date_input("Start Date (Max 7 days for 1m data)", default_start)
    end_date = st.sidebar.date_input("End Date", today)

    # Initial capital
    initial_capital = st.sidebar.number_input("Initial Capital ($)", min_value=1000, value=100000, step=1000)

    # Transaction cost
    transaction_cost = st.sidebar.number_input("Transaction Cost ($)", min_value=0, value=10, step=1)

    # Stop-loss and Take-profit
    stop_loss = st.sidebar.slider("Stop-Loss (%)", min_value=1, max_value=50, value=5) / 100  # 5%
    take_profit = st.sidebar.slider("Take-Profit (%)", min_value=1, max_value=100, value=10) / 100  # 10%

    # Validate dates
    if start_date >= end_date:
        st.sidebar.error("Error: End Date must fall after Start Date.")
        st.stop()
    if (end_date - start_date).days > 7:
        st.sidebar.error("Error: For 1-minute interval data, the maximum date range is 7 days.")
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

        # Debugging: Display number of buy and sell signals
        num_buys = data['Signal'].value_counts().get(1, 0)
        num_sells = data['Signal'].value_counts().get(-1, 0)
        st.write(f"**Number of Buy Signals:** {num_buys}")
        st.write(f"**Number of Sell Signals:** {num_sells}")

        # Simulate trades
        data, buy_dates, sell_dates, initial_capital, portfolio_values = simulate_trades(
            data, 
            price_column, 
            initial_capital=initial_capital, 
            transaction_cost=transaction_cost, 
            stop_loss=stop_loss, 
            take_profit=take_profit
        )

    # Display data overview
    st.subheader(f"📊 {ticker} Stock Data from {start_date} to {end_date}")
    st.write(data.tail())

    # Plot signals overview
    st.subheader("Buy/Sell Signals Overview")
    st.line_chart(data['Signal'])

    # Plot results
    st.subheader("📈 Trading Simulation Results")

    # Choose between Matplotlib and Plotly plots
    plot_choice = st.selectbox("Choose Plot Type", ["Matplotlib", "Plotly"])

    if plot_choice == "Matplotlib":
        plot_results(data, buy_dates, sell_dates, price_column)
    else:
        plot_results_interactive(data, buy_dates, sell_dates, price_column)

    # Display performance metrics
    calculate_performance(data, buy_dates, sell_dates, initial_capital)

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
        st.write(data[[price_column, 'EMA10', 'EMA30', 'EMA50', 'RSI', 'MACD', 'MACD_Signal', 'Signal', 'Portfolio_Value']])

    # Show Trade Details
    if st.checkbox("Show Trade Details"):
        if len(buy_dates) > 0 and len(sell_dates) > 0:
            trade_details = pd.DataFrame({
                'Buy Date': buy_dates,
                'Sell Date': sell_dates
            })
            trade_details['Buy Price'] = trade_details['Buy Date'].apply(lambda x: data.loc[x, price_column])
            trade_details['Sell Price'] = trade_details['Sell Date'].apply(lambda x: data.loc[x, price_column])
            trade_details['Profit/Loss'] = trade_details['Sell Price'] - trade_details['Buy Price']
            st.write(trade_details)
        else:
            st.write("No trades to display.")

# Run the app
if __name__ == "__main__":
    main()
