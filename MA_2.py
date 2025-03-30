import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import datetime

def fetch_stock_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches historical data from yfinance for a given symbol and date range.
    """
    return yf.download(symbol, start=start_date, end=end_date)

def moving_average_crossover_strategy(df: pd.DataFrame, short_window: int, long_window: int) -> pd.DataFrame:
    """
    Adds columns for short and long moving averages, along with buy/sell signals.
    """
    # Calculate rolling averages
    df['MA_Short'] = df['Close'].rolling(window=short_window).mean()
    df['MA_Long']  = df['Close'].rolling(window=long_window).mean()

    # 1 if short MA is above long MA, else 0
    df['Signal'] = (df['MA_Short'] > df['MA_Long']).astype(int)
    # Position changes: +1 = buy, -1 = sell
    df['Position'] = df['Signal'].diff()

    return df

def backtest(df: pd.DataFrame, initial_capital: float = 10000.0) -> pd.DataFrame:
    """
    Simulates trading for the crossover strategy:
    - Buys on Position=1, sells on Position=-1.
    - Tracks Holdings, Cash, and Total portfolio value.
    """
    df['Daily_Return'] = df['Close'].pct_change()
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
                # Already in position, carry forward
                df.loc[df.index[i], 'Holdings'] = df['Holdings'].iloc[i-1]
                df.loc[df.index[i], 'Cash'] = df['Cash'].iloc[i-1]

        elif df['Position'].iloc[i] == -1:  # SELL signal
            if in_position:
                # Sell all holdings
                df.loc[df.index[i], 'Cash'] = df['Holdings'].iloc[i-1] * df['Close'].iloc[i] + df['Cash'].iloc[i-1]
                df.loc[df.index[i], 'Holdings'] = 0
                in_position = False
            else:
                # Not in position, carry forward
                df.loc[df.index[i], 'Holdings'] = 0
                df.loc[df.index[i], 'Cash'] = df['Cash'].iloc[i-1]
        else:
            # No change => carry previous day
            df.loc[df.index[i], 'Holdings'] = df['Holdings'].iloc[i-1]
            df.loc[df.index[i], 'Cash'] = df['Cash'].iloc[i-1]

        # Recalculate total portfolio value
        df.loc[df.index[i], 'Total'] = (
            df['Holdings'].iloc[i] * df['Close'].iloc[i] +
            df['Cash'].iloc[i]
        )

    return df

def plot_results(df: pd.DataFrame, symbol: str):
    """
    Plots the stock price with buy/sell signals AND the portfolio value over time.
    """
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)

    # --- PRICE & MOVING AVERAGES ---
    ax1.plot(df.index, df['Close'], label='Close Price', alpha=0.5)
    ax1.plot(df.index, df['MA_Short'], label='MA Short', alpha=0.7)
    ax1.plot(df.index, df['MA_Long'], label='MA Long', alpha=0.7)

    # Mark buy/sell signals
    buy_signals = df[df['Position'] == 1]
    sell_signals = df[df['Position'] == -1]

    ax1.scatter(buy_signals.index, df.loc[buy_signals.index, 'Close'],
                label='Buy Signal', marker='^', color='green', s=100)
    ax1.scatter(sell_signals.index, df.loc[sell_signals.index, 'Close'],
                label='Sell Signal', marker='v', color='red', s=100)

    ax1.set_title(f"{symbol} - Moving Average Crossover Strategy")
    ax1.set_ylabel("Price (USD)")
    ax1.legend(loc='best')

    # --- PORTFOLIO VALUE ---
    ax2.plot(df.index, df['Total'], label='Portfolio Value', color='blue')
    ax2.set_title("Portfolio Value Over Time")
    ax2.set_ylabel("Value (USD)")
    ax2.legend(loc='best')

    plt.xlabel("Date")
    plt.tight_layout()
    st.pyplot(fig)

def main():
    st.title("Moving Average Crossover Simulation")

    # --- SIDEBAR INPUTS ---
    symbol = st.sidebar.text_input("Stock Symbol", "AAPL")
    start_date = st.sidebar.date_input("Start Date", datetime.date(2020, 1, 1))
    end_date = st.sidebar.date_input("End Date", datetime.date.today())
    short_window = st.sidebar.number_input("Short Window", min_value=5, max_value=100, value=20)
    long_window = st.sidebar.number_input("Long Window", min_value=10, max_value=200, value=50)
    initial_capital = st.sidebar.number_input("Initial Capital", min_value=1000, value=10000)

    if st.sidebar.button("Run Simulation"):
        st.write(f"## Simulation Results for {symbol}")
        df = fetch_stock_data(symbol, str(start_date), str(end_date))

        if df.empty:
            st.warning("No data found for this date range. Check the symbol and dates.")
            return

        # 1) Compute moving averages and signals
        df = moving_average_crossover_strategy(df, short_window, long_window)

        # 2) Drop rows where MA_Short or MA_Long is NaN
        df.dropna(subset=['MA_Short', 'MA_Long'], inplace=True)

        # 3) Run backtest
        df = backtest(df, initial_capital)

        # 4) Plot
        plot_results(df, symbol)

        # 5) Show final stats
        if not df.empty:
            final_value = df['Total'].iloc[-1]
            profit = final_value - initial_capital
            roi = (profit / initial_capital) * 100 if initial_capital != 0 else 0

            st.write(f"**Final Portfolio Value:** ${final_value:,.2f}")
            st.write(f"**Total Profit:** ${profit:,.2f}")
            st.write(f"**Return on Investment:** {roi:.2f}%")
        else:
            st.warning("No valid rows remain after dropping NaNs. Check your moving average windows and date range.")

if __name__ == "__main__":
    main()
