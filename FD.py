import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def fetch_stock_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches historical data for the specified symbol and date range using yfinance.
    Returns a DataFrame with columns: [Open, High, Low, Close, Adj Close, Volume].
    """
    df = yf.download(symbol, start=start_date, end=end_date)
    return df

def moving_average_crossover_strategy(df: pd.DataFrame, short_window: int = 20, long_window: int = 50) -> pd.DataFrame:
    """
    Implements a simple moving average crossover strategy.
    short_window: period for the short moving average (MA_Short)
    long_window: period for the long moving average (MA_Long)

    Steps:
    1. Calculate MA_Short and MA_Long on the 'Close' price.
    2. 'Signal' = 1 if MA_Short > MA_Long, else 0 (vectorized).
    3. 'Position' = difference in 'Signal' to determine buy/sell events.
    """
    # Calculate the short and long moving averages
    df['MA_Short'] = df['Close'].rolling(window=short_window).mean()
    df['MA_Long']  = df['Close'].rolling(window=long_window).mean()

    # Generate signals: 1 (buy/hold) when MA_Short > MA_Long, else 0
    df['Signal'] = (df['MA_Short'] > df['MA_Long']).astype(int)

    # Position changes: 1 = Buy, -1 = Sell, NaN or 0 = No change
    df['Position'] = df['Signal'].diff()

    return df

def backtest(df: pd.DataFrame, initial_capital: float = 10000.0) -> pd.DataFrame:
    """
    Backtests the crossover strategy by simulating trades.
    - Buys as many shares as possible on a 'Buy' signal (Position = 1).
    - Sells all shares on a 'Sell' signal (Position = -1).
    - Tracks daily portfolio value in 'Total'.
    - Assumes no transaction fees or slippage.

    Returns the DataFrame with updated columns:
    [Holdings, Cash, Total, Daily_Return].
    """
    # Daily return of the stock (for reference, not used directly in this example)
    df['Daily_Return'] = df['Close'].pct_change()

    # Initialize portfolio columns
    df['Holdings'] = 0.0
    df['Cash'] = initial_capital
    df['Total'] = initial_capital

    in_position = False

    for i in range(1, len(df)):
        if df['Position'].iloc[i] == 1:  # BUY signal
            if not in_position:
                # Buy as many shares as possible
                shares_to_buy = df['Cash'].iloc[i-1] // df['Close'].iloc[i]
                df.loc[df.index[i], 'Holdings'] = shares_to_buy
                df.loc[df.index[i], 'Cash'] = df['Cash'].iloc[i-1] - (shares_to_buy * df['Close'].iloc[i])
                in_position = True
            else:
                # Already in position, carry over previous values
                df.loc[df.index[i], 'Holdings'] = df['Holdings'].iloc[i-1]
                df.loc[df.index[i], 'Cash'] = df['Cash'].iloc[i-1]

        elif df['Position'].iloc[i] == -1:  # SELL signal
            if in_position:
                # Sell all holdings
                df.loc[df.index[i], 'Cash'] = df['Holdings'].iloc[i-1] * df['Close'].iloc[i] + df['Cash'].iloc[i-1]
                df.loc[df.index[i], 'Holdings'] = 0
                in_position = False
            else:
                # Not in position, just carry over
                df.loc[df.index[i], 'Holdings'] = 0
                df.loc[df.index[i], 'Cash'] = df['Cash'].iloc[i-1]

        else:
            # No change in signal; carry over
            df.loc[df.index[i], 'Holdings'] = df['Holdings'].iloc[i-1]
            df.loc[df.index[i], 'Cash'] = df['Cash'].iloc[i-1]

        # Compute total portfolio value
        df.loc[df.index[i], 'Total'] = (
            df['Holdings'].iloc[i] * df['Close'].iloc[i] +
            df['Cash'].iloc[i]
        )

    return df

def plot_results(df: pd.DataFrame, symbol: str):
    """
    Plots the stock price (with buy/sell signals) and portfolio value over time.
    """
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 10), sharex=True)

    # --- Price & Moving Averages ---
    ax1.plot(df.index, df['Close'], label='Close Price', alpha=0.5)
    ax1.plot(df.index, df['MA_Short'], label='MA Short', alpha=0.7)
    ax1.plot(df.index, df['MA_Long'], label='MA Long', alpha=0.7)

    # Buy signals (Position = 1) and Sell signals (Position = -1)
    buy_signals = df[df['Position'] == 1]
    sell_signals = df[df['Position'] == -1]

    ax1.scatter(buy_signals.index,
                df.loc[buy_signals.index, 'Close'],
                label='Buy Signal',
                marker='^', color='green', s=100)

    ax1.scatter(sell_signals.index,
                df.loc[sell_signals.index, 'Close'],
                label='Sell Signal',
                marker='v', color='red', s=100)

    ax1.set_title(f"{symbol} - Moving Average Crossover Strategy")
    ax1.set_ylabel("Price (USD)")
    ax1.legend(loc='best')

    # --- Portfolio Value ---
    ax2.plot(df.index, df['Total'], label='Portfolio Value', color='blue')
    ax2.set_title("Portfolio Value Over Time")
    ax2.set_ylabel("Value (USD)")
    ax2.legend(loc='best')

    plt.xlabel("Date")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    SYMBOL = "AAPL"              # Example stock symbol
    START_DATE = "2020-01-01"    # Start date for historical data
    END_DATE = "2021-01-01"      # End date for historical data
    INITIAL_CAPITAL = 10000.0    # Starting capital for backtest

    # 1. Fetch Data
    data = fetch_stock_data(SYMBOL, START_DATE, END_DATE)

    # 2. Moving Average Crossover Strategy
    data = moving_average_crossover_strategy(data, short_window=20, long_window=50)

    # 3. Backtest the Strategy
    data = backtest(data, initial_capital=INITIAL_CAPITAL)

    # 4. Plot the Results
    plot_results(data, SYMBOL)

    # 5. Print Final Statistics
    final_value = data['Total'].iloc[-1]
    profit = final_value - INITIAL_CAPITAL
    print(f"Final Portfolio Value: ${final_value:,.2f}")
    print(f"Net Profit: ${profit:,.2f} ({profit / INITIAL_CAPITAL * 100:.2f}%)")
