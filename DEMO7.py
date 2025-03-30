import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ta import trend, momentum, volume, volatility
from datetime import datetime, timedelta
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Suppress SettingWithCopyWarning
pd.options.mode.chained_assignment = None  # default='warn'

# Function to fetch minute-level data with caching
@st.cache_data(show_spinner=False)
def fetch_data(ticker, start_date, end_date):
    """
    Fetches historical minute-level stock data for the specified ticker and date range.
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

        # Fetch minute-level data (5m interval for Algo Trading)
        data = yf.download(tickers=ticker, start=start_date, end=end_date, interval='5m', progress=False)
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
    # Exponential Moving Averages
    data['EMA10'] = trend.ema_indicator(close=data[price_column], window=10)
    data['EMA30'] = trend.ema_indicator(close=data[price_column], window=30)

    # Relative Strength Index
    data['RSI'] = momentum.rsi(close=data[price_column], window=14)

    # Moving Average Convergence Divergence
    macd = trend.MACD(close=data[price_column], window_slow=26, window_fast=12, window_sign=9)
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()
    data['MACD_Diff'] = macd.macd_diff()

    # Average True Range
    data['ATR14'] = volatility.average_true_range(high=data['High'], low=data['Low'], close=data[price_column], window=14)

    # On-Balance Volume
    data['OBV'] = volume.on_balance_volume(close=data[price_column], volume=data['Volume'])

    # Drop rows with any NaN in indicators
    data.dropna(inplace=True)
    return data

# Function to generate buy/sell signals based on EMA crossovers
def generate_signals(data):
    """
    Generates buy and sell signals based on EMA crossovers.

    Parameters:
        data (DataFrame): Stock data with technical indicators.

    Returns:
        DataFrame: Stock data with buy/sell signals.
    """
    data['Signal'] = 0  # 1 for Buy, -1 for Sell, 0 for Hold

    # EMA Crossover Conditions
    data['EMA_Crossover'] = np.where(data['EMA10'] > data['EMA30'], 1, -1)
    data['EMA_Crossover_Signal'] = data['EMA_Crossover'].diff()

    # Buy Conditions
    buy_condition = (data['EMA_Crossover_Signal'] == 2)  # EMA10 crossed above EMA30
    data.loc[buy_condition, 'Signal'] = 1

    # Sell Conditions
    sell_condition = (data['EMA_Crossover_Signal'] == -2)  # EMA10 crossed below EMA30
    data.loc[sell_condition, 'Signal'] = -1

    # Debugging: Show the last few signals
    st.write("### Recent Signals and Indicators")
    st.write(data[['EMA10', 'EMA30', 'Signal']].tail())

    return data

# Function to simulate trades with risk management and transaction costs
def simulate_trades(data, price_column, initial_capital=100000, transaction_cost=10,
                   stop_loss_multiplier_percentage=100, risk_to_reward_percentage=150, risk_percentage=0.5):
    """
    Simulates trades based on generated signals with risk management.

    Parameters:
        data (DataFrame): Stock data with buy/sell signals.
        price_column (str): The column to use for price data.
        initial_capital (float): Starting capital for simulation.
        transaction_cost (float): Fixed cost per transaction.
        stop_loss_multiplier_percentage (float): ATR multiplier for stop loss as a percentage (10-200%).
        risk_to_reward_percentage (float): Risk to reward ratio as a percentage (50-200%).
        risk_percentage (float): Risk as a percentage of capital per trade (0.1-5%).

    Returns:
        tuple: (data DataFrame with portfolio value, trades DataFrame, final capital, portfolio_values list)
    """
    capital = initial_capital
    position = None  # 'long' or 'short'
    position_size = 0
    entry_price = 0
    stop_loss = 0
    take_profit = 0
    portfolio_values = []
    trades = []  # List to store trade details

    # Define maximum position size
    MAX_POSITION_SIZE = 1000  # Adjust based on asset characteristics

    # Convert percentage inputs to multipliers
    stop_loss_multiplier = (stop_loss_multiplier_percentage / 100) * 2.0  # Allows up to 2x ATR
    risk_to_reward_ratio = (risk_to_reward_percentage / 100) * 2.0  # Allows up to 2:1 ratio

    st.write(f"**Configured Stop Loss Multiplier:** {stop_loss_multiplier:.2f} ATR")
    st.write(f"**Configured Risk to Reward Ratio:** {risk_to_reward_ratio:.2f}:1")

    for i in range(len(data)):
        signal = data['Signal'].iloc[i]
        price = data[price_column].iloc[i]
        atr = data['ATR14'].iloc[i]
        date = data.index[i]

        st.write(f"---\n**Date:** {date}\n**Price:** ${price:.2f}\n**Signal:** {signal}\n**ATR:** {atr:.2f}")

        # Check if in a long position
        if position == 'long':
            st.write(f"**Current Position:** Long at ${entry_price:.2f} with size {position_size:.2f}")
            # Check for stop loss or take profit
            if price <= stop_loss or price >= take_profit:
                # Calculate profit
                profit = (price - entry_price) * position_size
                capital += position_size * price - transaction_cost
                trades.append({
                    'Type': 'Long',
                    'Entry_Date': entry_date,
                    'Entry_Price': entry_price,
                    'Exit_Date': date,
                    'Exit_Price': price,
                    'Position_Size': position_size,
                    'Profit_Loss': profit
                })
                st.write(f"**Long Sell Executed:** Price=${price:.2f}, Profit=${profit:.2f}")
                # Reset position
                position = None
                position_size = 0
                entry_price = 0
                stop_loss = 0
                take_profit = 0

        # Check if in a short position
        elif position == 'short':
            st.write(f"**Current Position:** Short at ${entry_price:.2f} with size {position_size:.2f}")
            # Check for stop loss or take profit
            if price >= stop_loss or price <= take_profit:
                # Calculate profit
                profit = (entry_price - price) * position_size
                capital += position_size * entry_price - position_size * price - transaction_cost
                trades.append({
                    'Type': 'Short',
                    'Entry_Date': entry_date,
                    'Entry_Price': entry_price,
                    'Exit_Date': date,
                    'Exit_Price': price,
                    'Position_Size': position_size,
                    'Profit_Loss': profit
                })
                st.write(f"**Short Cover Executed:** Price=${price:.2f}, Profit=${profit:.2f}")
                # Reset position
                position = None
                position_size = 0
                entry_price = 0
                stop_loss = 0
                take_profit = 0

        # If not in any position, check for new signals
        if position is None:
            if signal == 1:
                st.write("**Buy Signal Detected**")
                # Calculate risk amount
                risk_amount = capital * (risk_percentage / 100)
                st.write(f"**Risk Amount:** ${risk_amount:.2f}")

                # Calculate stop loss and take profit
                stop_loss = price - (atr * stop_loss_multiplier)
                take_profit = price + ((price - stop_loss) * risk_to_reward_ratio)
                st.write(f"**Stop Loss:** ${stop_loss:.2f}, **Take Profit:** ${take_profit:.2f}")

                # Calculate position size
                price_diff = price - stop_loss
                if price_diff <= 0:
                    st.warning(f"Invalid price difference (${price_diff:.2f}). Skipping trade.")
                    current_portfolio_value = capital
                else:
                    # Validate ATR
                    MIN_ATR = 0.01  # Define a minimum ATR to prevent excessively large positions
                    if atr < MIN_ATR:
                        st.warning(f"ATR ({atr:.2f}) is below the minimum threshold (${MIN_ATR}). Skipping trade.")
                        current_portfolio_value = capital
                    else:
                        position_size = risk_amount / price_diff
                        st.write(f"**Calculated Position Size:** {position_size:.2f}")

                        # Check for minimum and maximum position size
                        MIN_POSITION_SIZE = 1  # Define as per asset characteristics
                        if position_size < MIN_POSITION_SIZE:
                            st.warning(f"Position size {position_size:.2f} is below the minimum threshold. Skipping trade.")
                            current_portfolio_value = capital
                        elif position_size > MAX_POSITION_SIZE:
                            st.warning(f"Position size {position_size:.2f} exceeds the maximum allowed ({MAX_POSITION_SIZE}). Skipping trade.")
                            current_portfolio_value = capital
                        else:
                            # Execute buy
                            position = 'long'
                            entry_date = date
                            entry_price = price
                            capital -= position_size * price + transaction_cost
                            st.write(f"**Buy Executed:** {position_size:.2f} shares at ${price:.2f} each. Cost: ${position_size * price + transaction_cost:.2f}")
                            current_portfolio_value = capital + (position_size * price)

                # Append to portfolio_values
                portfolio_values.append(current_portfolio_value)
                st.write(f"**Current Portfolio Value:** ${current_portfolio_value:.2f}")

            elif signal == -1:
                st.write("**Sell Signal Detected**")
                # Calculate risk amount
                risk_amount = capital * (risk_percentage / 100)
                st.write(f"**Risk Amount:** ${risk_amount:.2f}")

                # Calculate stop loss and take profit
                stop_loss = price + (atr * stop_loss_multiplier)
                take_profit = price - ((stop_loss - price) * risk_to_reward_ratio)
                st.write(f"**Stop Loss:** ${stop_loss:.2f}, **Take Profit:** ${take_profit:.2f}")

                # Calculate position size
                price_diff = stop_loss - price
                if price_diff <= 0:
                    st.warning(f"Invalid price difference (${price_diff:.2f}). Skipping trade.")
                    current_portfolio_value = capital
                else:
                    # Validate ATR
                    MIN_ATR = 0.01  # Define a minimum ATR to prevent excessively large positions
                    if atr < MIN_ATR:
                        st.warning(f"ATR ({atr:.2f}) is below the minimum threshold (${MIN_ATR}). Skipping trade.")
                        current_portfolio_value = capital
                    else:
                        position_size = risk_amount / price_diff
                        st.write(f"**Calculated Position Size:** {position_size:.2f}")

                        # Check for minimum and maximum position size
                        MIN_POSITION_SIZE = 1  # Define as per asset characteristics
                        if position_size < MIN_POSITION_SIZE:
                            st.warning(f"Position size {position_size:.2f} is below the minimum threshold. Skipping trade.")
                            current_portfolio_value = capital
                        elif position_size > MAX_POSITION_SIZE:
                            st.warning(f"Position size {position_size:.2f} exceeds the maximum allowed ({MAX_POSITION_SIZE}). Skipping trade.")
                            current_portfolio_value = capital
                        else:
                            # Execute short sell
                            position = 'short'
                            entry_date = date
                            entry_price = price
                            capital += position_size * price - transaction_cost
                            st.write(f"**Short Sell Executed:** {position_size:.2f} shares at ${price:.2f} each. Proceeds: ${position_size * price - transaction_cost:.2f}")
                            current_portfolio_value = capital - (position_size * price) + (position_size * entry_price)

                # Append to portfolio_values
                portfolio_values.append(current_portfolio_value)
                st.write(f"**Current Portfolio Value:** ${current_portfolio_value:.2f}")

        else:
            # If in a position, append current portfolio value
            if position == 'long':
                current_portfolio_value = capital + (position_size * price)
            elif position == 'short':
                current_portfolio_value = capital - (position_size * price) + (position_size * entry_price)
            else:
                current_portfolio_value = capital

            portfolio_values.append(current_portfolio_value)
            st.write(f"**Current Portfolio Value:** ${current_portfolio_value:.2f}")

    # Handle any open positions at the end of the data
    if position == 'long':
        st.write("**Closing Final Long Position**")
        price = data[price_column].iloc[-1]
        profit = (price - entry_price) * position_size
        capital += position_size * price - transaction_cost
        trades.append({
            'Type': 'Long',
            'Entry_Date': entry_date,
            'Entry_Price': entry_price,
            'Exit_Date': data.index[-1],
            'Exit_Price': price,
            'Position_Size': position_size,
            'Profit_Loss': profit
        })
        st.write(f"**Final Long Sell:** {data.index[-1]} | **Price:** ${price:.2f} | **Profit:** ${profit:.2f}")
        # Update portfolio value after closing
        portfolio_values.append(capital)

    elif position == 'short':
        st.write("**Closing Final Short Position**")
        price = data[price_column].iloc[-1]
        profit = (entry_price - price) * position_size
        capital += position_size * entry_price - position_size * price - transaction_cost
        trades.append({
            'Type': 'Short',
            'Entry_Date': entry_date,
            'Entry_Price': entry_price,
            'Exit_Date': data.index[-1],
            'Exit_Price': price,
            'Position_Size': position_size,
            'Profit_Loss': profit
        })
        st.write(f"**Final Short Cover:** {data.index[-1]} | **Price:** ${price:.2f} | **Profit:** ${profit:.2f}")
        # Update portfolio value after closing
        portfolio_values.append(capital)

    # Final portfolio value
    final_portfolio = capital

    data = data.copy()
    data['Portfolio_Value'] = portfolio_values

    # Create trades DataFrame
    trades_df = pd.DataFrame(trades)

    # Debugging: Display trades_df
    if not trades_df.empty:
        st.write("### **Executed Trades:**")
        st.write(trades_df)
    else:
        st.warning("No trades were executed.")

    return data, trades_df, final_portfolio, portfolio_values

# Function to plot technical indicators
def plot_indicators(data, price_column):
    """
    Plots technical indicators alongside the price for visual inspection.

    Parameters:
        data (DataFrame): Stock data with technical indicators.
        price_column (str): The column used for price data.
    """
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Plot Price
    ax1.plot(data.index, data[price_column], label=f'{price_column} Price', color='black')
    ax1.set_ylabel('Price ($)', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.legend(loc='upper left')

    # Plot EMA10 and EMA30
    ax1.plot(data.index, data['EMA10'], label='EMA10', color='blue', linestyle='--')
    ax1.plot(data.index, data['EMA30'], label='EMA30', color='orange', linestyle='--')
    ax1.legend(loc='upper left')

    # Create a twin Axes sharing the xaxis
    ax2 = ax1.twinx()

    # Plot RSI
    ax2.plot(data.index, data['RSI'], label='RSI', color='green', alpha=0.3)
    ax2.axhline(30, color='red', linestyle='--', linewidth=0.5)
    ax2.axhline(70, color='red', linestyle='--', linewidth=0.5)
    ax2.set_ylabel('RSI', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.legend(loc='upper right')

    plt.title('Price with EMA10, EMA30, and RSI')
    plt.tight_layout()
    st.pyplot(fig)

# Function to plot additional indicators using Plotly
def plot_additional_indicators_plotly(data):
    """
    Plots MACD and OBV indicators using Plotly for interactive analysis.

    Parameters:
        data (DataFrame): Stock data with technical indicators.
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("MACD Indicator", "On-Balance Volume (OBV)"),
                        vertical_spacing=0.15)

    # MACD and Signal Line
    fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], mode='lines', name='MACD', line=dict(color='purple')),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['MACD_Signal'], mode='lines', name='MACD Signal', line=dict(color='red', dash='dash')),
                  row=1, col=1)
    fig.add_trace(go.Bar(x=data.index, y=data['MACD_Diff'], name='MACD Histogram', marker_color='gray', opacity=0.3),
                  row=1, col=1)

    # OBV
    fig.add_trace(go.Scatter(x=data.index, y=data['OBV'], mode='lines', name='OBV', line=dict(color='brown')),
                  row=2, col=1)

    fig.update_layout(
        height=800,
        title_text="Technical Indicators",
        showlegend=False
    )

    fig.update_yaxes(title_text="MACD", row=1, col=1)
    fig.update_yaxes(title_text="OBV", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

# Function to plot buy/sell signals on price chart using Plotly
def plot_signals_plotly(data, trades_df, price_column):
    """
    Plots the stock price with buy/sell signals using Plotly.

    Parameters:
        data (DataFrame): Stock data with buy/sell signals.
        trades_df (DataFrame): DataFrame containing executed trades.
        price_column (str): The column used for price data.
    """
    fig = go.Figure()

    # Price
    fig.add_trace(go.Scatter(x=data.index, y=data[price_column], mode='lines', name=f'{price_column} Price'))

    if not trades_df.empty:
        # Buy Signals
        buy_trades = trades_df[trades_df['Type'] == 'Long']
        fig.add_trace(go.Scatter(
            x=buy_trades['Entry_Date'],
            y=buy_trades['Entry_Price'],
            mode='markers',
            marker=dict(symbol='triangle-up', color='green', size=12),
            name='Buy Signal'
        ))

        # Sell Signals
        sell_trades = trades_df[trades_df['Type'] == 'Short']
        fig.add_trace(go.Scatter(
            x=sell_trades['Exit_Date'],
            y=sell_trades['Exit_Price'],
            mode='markers',
            marker=dict(symbol='triangle-down', color='red', size=12),
            name='Sell Signal'
        ))
    else:
        fig.add_annotation(
            x=0.5, y=0.5,
            text="No trades were executed to plot buy/sell signals.",
            showarrow=False,
            xref="paper", yref="paper",
            font=dict(size=16),
            align='center'
        )

    # Layout
    fig.update_layout(
        title=f'{price_column} Price with Buy/Sell Signals',
        xaxis=dict(title='Date'),
        yaxis=dict(title=f'{price_column} Price ($)'),
        legend=dict(x=0, y=1.1, orientation='h'),
        autosize=True
    )

    st.plotly_chart(fig, use_container_width=True)

# Function to plot portfolio value using Plotly
def plot_portfolio_value_plotly(data):
    """
    Plots the portfolio value over time using Plotly.

    Parameters:
        data (DataFrame): Stock data with portfolio value.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=data.index, y=data['Portfolio_Value'], mode='lines', name='Portfolio Value', line=dict(color='blue')))

    fig.update_layout(
        title="Portfolio Value Over Time",
        xaxis=dict(title='Date'),
        yaxis=dict(title='Portfolio Value ($)'),
        legend=dict(x=0, y=1),
        autosize=True
    )

    st.plotly_chart(fig, use_container_width=True)

# Function to plot additional indicators using Matplotlib
def plot_additional_indicators_matplotlib(data):
    """
    Plots MACD and OBV indicators using Matplotlib for visual inspection.

    Parameters:
        data (DataFrame): Stock data with technical indicators.
    """
    fig, ax = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Plot MACD and Signal Line
    ax[0].plot(data.index, data['MACD'], label='MACD', color='purple')
    ax[0].plot(data.index, data['MACD_Signal'], label='MACD Signal', color='red', linestyle='--')
    ax[0].bar(data.index, data['MACD_Diff'], label='MACD Histogram', color='gray', alpha=0.3)
    ax[0].set_title('MACD Indicator')
    ax[0].set_ylabel('MACD')
    ax[0].legend()

    # Plot On-Balance Volume (OBV)
    ax[1].plot(data.index, data['OBV'], label='OBV', color='brown')
    ax[1].set_title('On-Balance Volume (OBV)')
    ax[1].set_ylabel('OBV')
    ax[1].legend()

    plt.tight_layout()
    st.pyplot(fig)

# Function to plot results using Matplotlib
def plot_results(data, trades_df, price_column):
    """
    Plots the stock price with buy/sell signals and the portfolio value over time using Matplotlib.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Plot Price
    ax1.plot(data[price_column], label=f'{price_column} Price', alpha=0.5)

    if not trades_df.empty:
        buy_trades = trades_df[trades_df['Type'] == 'Long']
        sell_trades = trades_df[trades_df['Type'] == 'Short']
        ax1.scatter(buy_trades['Entry_Date'], buy_trades['Entry_Price'], marker='^', color='green', label='Buy Signal', s=100)
        ax1.scatter(sell_trades['Exit_Date'], sell_trades['Exit_Price'], marker='v', color='red', label='Sell Signal', s=100)
    else:
        ax1.text(0.5, 0.5, 'No trades were executed to plot buy/sell signals.',
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax1.transAxes, fontsize=12, color='red')

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

# Function to calculate performance metrics
def calculate_performance(trades_df, initial_capital, final_capital):
    """
    Calculates and displays performance metrics of the trading simulation.

    Parameters:
        trades_df (DataFrame): DataFrame containing trade details.
        initial_capital (float): Starting capital for simulation.
        final_capital (float): Final portfolio value after simulation.
    """
    if trades_df.empty:
        st.subheader("📈 Performance Metrics")
        st.write("**No trades were executed, so performance metrics are not available.**")
        return

    total_return = (final_capital - initial_capital) / initial_capital * 100

    # Calculate CAGR
    start_date = trades_df['Entry_Date'].min()
    end_date = trades_df['Exit_Date'].max()
    days = (end_date - start_date).days
    years = days / 365.25
    if years > 0:
        cagr = ((final_capital / initial_capital) ** (1 / years) - 1) * 100
    else:
        cagr = 0.0

    # Calculate Sharpe Ratio
    # Using daily returns is not applicable for minute-level data without proper aggregation.
    # For simplicity, we will skip Sharpe Ratio or use total return as a proxy.
    sharpe_ratio = 0.0
    if not trades_df.empty:
        returns = trades_df['Profit_Loss'] / initial_capital
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0.0

    # Calculate Max Drawdown
    portfolio_series = trades_df['Profit_Loss'].cumsum() + initial_capital
    cumulative_max = portfolio_series.cummax()
    drawdown = (portfolio_series - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min() * 100

    # Calculate Profit Factor and Expectancy
    profits = trades_df[trades_df['Profit_Loss'] > 0]['Profit_Loss']
    losses = trades_df[trades_df['Profit_Loss'] < 0]['Profit_Loss'].abs()
    profit_factor = profits.sum() / losses.sum() if losses.sum() > 0 else np.inf
    expectancy = (profits.sum() - losses.sum()) / len(trades_df) if len(trades_df) > 0 else 0.0

    # Calculate Win Rate
    wins = len(profits)
    total_trades = len(trades_df)
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0

    st.subheader("📈 Performance Metrics")
    metrics = {
        "Initial Capital": f"${initial_capital:,.2f}",
        "Final Portfolio Value": f"${final_capital:,.2f}",
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
    st.set_page_config(page_title="📈 Algo Trading Simulation", layout="wide")
    st.title("📈 Algorithmic Trading Simulation with EMA Crossovers")
    st.markdown("""
    This application simulates an algorithmic trading (Algo Trading) strategy based on historical 5-minute interval stock data. 
    Users can input parameters such as date range, initial capital, and risk management settings to observe how the strategy would have performed.
    **Note:** Due to data limitations, minute-level data is available only for the past 7 days.
    """)

    # Sidebar for user inputs
    st.sidebar.header("User Inputs")
    ticker = st.sidebar.text_input("Ticker Symbol", value='BTC-USD').upper().strip()

    # Date inputs
    today = datetime.today()
    default_start = today - timedelta(days=7)
    start_date = st.sidebar.date_input("Start Date (Max 7 days for 5m data)", default_start)
    end_date = st.sidebar.date_input("End Date", today)

    # Initial capital
    initial_capital = st.sidebar.number_input("Initial Capital ($)", min_value=1000, value=100000, step=1000)

    # Transaction cost
    transaction_cost = st.sidebar.number_input("Transaction Cost ($)", min_value=0, value=10, step=1)

    # Stop-loss and Take-profit multipliers (Percentage Inputs)
    stop_loss_multiplier_percentage = st.sidebar.slider(
        "Stop-Loss ATR Multiplier (%)",
        min_value=10.0,
        max_value=200.0,
        value=100.0,
        step=10.0,
        help="Set the stop-loss multiplier as a percentage of ATR (e.g., 100% = 1x ATR)."
    )

    risk_to_reward_percentage = st.sidebar.slider(
        "Risk to Reward Ratio (%)",
        min_value=50.0,
        max_value=200.0,
        value=150.0,
        step=10.0,
        help="Set the risk to reward ratio as a percentage (e.g., 150% = 1.5:1)."
    )

    # Risk percentage per trade
    risk_percentage = st.sidebar.slider(
        "Risk Percentage per Trade (%)",
        min_value=0.1,
        max_value=5.0,
        value=0.5,
        step=0.1,
        help="Set the percentage of capital to risk per trade."
    )

    # Validate dates
    if start_date >= end_date:
        st.sidebar.error("Error: End Date must fall after Start Date.")
        st.stop()
    if (end_date - start_date).days > 7:
        st.sidebar.error("Error: For 5-minute interval data, the maximum date range is 7 days.")
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
        num_buys = (data['Signal'] == 1).sum()
        num_sells = (data['Signal'] == -1).sum()
        st.write(f"**Number of Buy Signals:** {num_buys}")
        st.write(f"**Number of Sell Signals:** {num_sells}")

        # Simulate trades
        data, trades_df, final_capital, portfolio_values = simulate_trades(
            data, 
            price_column, 
            initial_capital=initial_capital, 
            transaction_cost=transaction_cost, 
            stop_loss_multiplier_percentage=stop_loss_multiplier_percentage, 
            risk_to_reward_percentage=risk_to_reward_percentage, 
            risk_percentage=risk_percentage
        )

    # Display data overview
    st.subheader(f"📊 {ticker} Stock Data from {start_date} to {end_date}")
    st.write(data.tail())

    # Tabbed Layout for Organized Visualization
    tab1, tab2, tab3, tab4 = st.tabs(["Technical Indicators", "Additional Indicators", "Buy/Sell Signals", "Portfolio Value"])

    with tab1:
        st.markdown("### Technical Indicators Overview")
        plot_indicators(data, price_column)

    with tab2:
        st.markdown("### Additional Indicators")
        # Choose plotting method
        plot_method = st.radio("Choose Plotting Method for Additional Indicators", ["Matplotlib", "Plotly"], key="plot_method")
        if plot_method == "Matplotlib":
            plot_additional_indicators_matplotlib(data)
        else:
            plot_additional_indicators_plotly(data)

    with tab3:
        st.markdown("### Buy/Sell Signals on Price Chart")
        # Choose plotting method
        plot_method_signals = st.radio("Choose Plotting Method for Buy/Sell Signals", ["Matplotlib", "Plotly"], key="plot_method_signals")
        if plot_method_signals == "Matplotlib":
            plot_results(data, trades_df, price_column)
        else:
            plot_signals_plotly(data, trades_df, price_column)

    with tab4:
        st.markdown("### Portfolio Value Over Time")
        # Choose plotting method
        plot_method_portfolio = st.radio("Choose Plotting Method for Portfolio Value", ["Matplotlib", "Plotly"], key="plot_method_portfolio")
        if plot_method_portfolio == "Matplotlib":
            plot_results(data, trades_df, price_column)  # Already includes portfolio plot
        else:
            plot_portfolio_value_plotly(data)

    # Display performance metrics
    calculate_performance(trades_df, initial_capital, final_capital)

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
        st.write(data[[price_column, 'EMA10', 'EMA30', 'RSI', 'Signal', 'Portfolio_Value']])

    # Show Trade Details
    if st.checkbox("Show Trade Details"):
        if not trades_df.empty:
            st.subheader("Trade Details")
            st.write(trades_df)
        else:
            st.write("No trades to display.")

# Run the app
if __name__ == "__main__":
    main()
