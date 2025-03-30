import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

########################################
# 1. HELPERS
########################################

def flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    If yfinance returns multi-level columns (e.g., ('Close','AAPL')),
    flatten them into single-level column names like 'Close_AAPL'.
    Also, if 'Close' becomes multiple columns, pick the first one by default.
    """
    if isinstance(df.columns, pd.MultiIndex):
        # Flatten multi-level
        df.columns = ['_'.join(col).rstrip('_') for col in df.columns.values]
    return df

def fix_close_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure 'Close' is a single Series, not multiple columns.
    If multiple 'Close' columns exist, pick the first one.
    """
    # e.g. you might have 'Close_AAPL' or 'Close' or 'Close_etc'
    # Try to find a column that starts with 'Close'
    close_cols = [c for c in df.columns if c.startswith('Close')]
    if len(close_cols) == 0:
        # No close column at all => can't do anything
        return df
    # If there's more than one (e.g., 'Close_MSFT' and 'Close_AAPL'), pick the first
    main_close = close_cols[0]
    # We'll rename it to 'Close' for consistency
    df['Close'] = df[main_close]
    return df


########################################
# 2. FETCH DATA
########################################

def fetch_data(ticker, start_date, end_date, use_hft=False):
    """
    Fetches data for the chosen ticker from yfinance.
    If use_hft=True, we use 15-minute intervals to simulate 'high-frequency' trading.
    Otherwise, 1-day intervals.
    """
    interval = "1d"
    if use_hft:
        interval = "15m"  # Simplified demonstration of HFT

    df = yf.download(ticker, start=start_date, end=end_date, interval=interval)

    # Flatten multi-index columns if any
    df = flatten_yfinance_columns(df)
    # Ensure we have a single 'Close' column
    df = fix_close_column(df)

    return df


########################################
# 3. STRATEGIES
########################################

def strategy_price_level(df):
    """
    Price Level Movements:
      - Calculate a single reference level (mean Close).
      - Buy if Close > this level, sell if Close < this level.
    """
    if 'Close' not in df.columns:
        return df  # Can't run

    # Single float: the mean close
    ref_level = df['Close'].mean(skipna=True)
    df['RefLevel'] = ref_level
    # Compare each row's Close to RefLevel => 1 or 0
    df['Signal'] = (df['Close'] > ref_level).astype(int)
    df['Position'] = df['Signal'].diff()

    return df

def strategy_channel_breakout(df, window=20):
    """
    Channel Breakouts:
      - 20-day rolling high/low channel.
      - Buy if price breaks above the channel high, sell if breaks below channel low.
    """
    if 'High' not in df.columns or 'Low' not in df.columns:
        return df
    df['ChannelHigh'] = df['High'].rolling(window).max()
    df['ChannelLow']  = df['Low'].rolling(window).min()
    df['Signal'] = 0
    df.loc[df['Close'] > df['ChannelHigh'], 'Signal'] = 1
    df.loc[df['Close'] < df['ChannelLow'], 'Signal'] = 0
    df['Position'] = df['Signal'].diff()
    return df

def strategy_arbitrage(df):
    """
    Arbitrage (Dummy):
      - Random signals with small probability of 'buy' each day, otherwise 'sell'.
    """
    if len(df) == 0:
        return df
    np.random.seed(42)
    df['Signal'] = np.random.choice([0,1], p=[0.995, 0.005], size=len(df))
    df['Position'] = df['Signal'].diff()
    return df

def strategy_index_rebalance(df):
    """
    Index Fund Rebalancing (Simplified):
      - Buy at the end of each quarter, sell otherwise.
    """
    df['Signal'] = 0
    if len(df) > 0:
        for idx in df.index:
            # If it's the last trading day of Q1, Q2, Q3, Q4 => 'Signal'=1 => re-enter
            if (idx.month in [3,6,9,12]):
                month_data = df.loc[str(idx.year)+'-'+str(idx.month)]
                if not month_data.empty and idx == month_data.index[-1]:
                    df.loc[idx, 'Signal'] = 1
    df['Position'] = df['Signal'].diff()
    return df

def strategy_mean_reversion(df, lookback=5):
    """
    Mean Reversion:
      - Buys if price is < 98% of a short SMA, sells if price is > 102% of that SMA
    """
    if 'Close' not in df.columns:
        return df
    df['ShortSMA'] = df['Close'].rolling(lookback).mean()
    buy_condition  = df['Close'] < df['ShortSMA']*0.98
    sell_condition = df['Close'] > df['ShortSMA']*1.02
    df['Signal'] = 0
    df.loc[buy_condition, 'Signal'] = 1
    df.loc[sell_condition, 'Signal'] = 0
    df['Position'] = df['Signal'].diff()
    return df

def strategy_moving_average_crossover(df, short_window=20, long_window=50):
    """
    Moving Average Crossover (Market Timing):
      - If short MA crosses above long MA => buy
      - If short MA crosses below => sell
    """
    if 'Close' not in df.columns:
        return df
    df['MA_Short'] = df['Close'].rolling(short_window).mean()
    df['MA_Long']  = df['Close'].rolling(long_window).mean()
    df['Signal'] = (df['MA_Short'] > df['MA_Long']).astype(int)
    df['Position'] = df['Signal'].diff()
    return df


########################################
# 4. BACKTEST
########################################

def backtest(df, initial_capital=10000):
    """
    Simple buy/sell backtest:
      - Buy on Position= +1, Sell on Position= -1
    """
    if 'Close' not in df.columns or 'Position' not in df.columns:
        return df

    df['Holdings'] = 0.0
    df['Cash'] = initial_capital
    df['Total'] = initial_capital
    in_position = False

    for i in range(1, len(df)):
        if df['Position'].iloc[i] == 1:  # BUY
            if not in_position:
                shares_to_buy = df['Cash'].iloc[i-1] // df['Close'].iloc[i]
                df.loc[df.index[i], 'Holdings'] = shares_to_buy
                df.loc[df.index[i], 'Cash'] = df['Cash'].iloc[i-1] - (shares_to_buy * df['Close'].iloc[i])
                in_position = True
            else:
                df.loc[df.index[i], 'Holdings'] = df['Holdings'].iloc[i-1]
                df.loc[df.index[i], 'Cash'] = df['Cash'].iloc[i-1]
        elif df['Position'].iloc[i] == -1:  # SELL
            if in_position:
                df.loc[df.index[i], 'Cash'] = df['Holdings'].iloc[i-1] * df['Close'].iloc[i] + df['Cash'].iloc[i-1]
                df.loc[df.index[i], 'Holdings'] = 0
                in_position = False
            else:
                df.loc[df.index[i], 'Holdings'] = 0
                df.loc[df.index[i], 'Cash'] = df['Cash'].iloc[i-1]
        else:
            df.loc[df.index[i], 'Holdings'] = df['Holdings'].iloc[i-1]
            df.loc[df.index[i], 'Cash'] = df['Cash'].iloc[i-1]
        
        df.loc[df.index[i], 'Total'] = (
            df['Holdings'].iloc[i] * df['Close'].iloc[i] +
            df['Cash'].iloc[i]
        )
    return df


########################################
# 5. STREAMLIT APP
########################################

def main():
    st.title("Multi-Strategy AI Trading Model (S&P 500, 2024)")

    # -- USER INPUTS --
    st.sidebar.header("User Settings")
    ticker = st.sidebar.text_input("Stock Ticker (S&P 500 member e.g. AAPL, MSFT, etc.)", value="AAPL")
    strategy = st.sidebar.selectbox("Choose Strategy:", [
        "Price Level Movements",
        "Channel Breakouts",
        "Arbitrage",
        "Index Fund Rebalancing",
        "Mean Reversion",
        "Moving Average Crossover"
    ])
    short_window = st.sidebar.number_input("Short Window (MA Crossover)", min_value=5, max_value=100, value=20)
    long_window  = st.sidebar.number_input("Long  Window (MA Crossover)", min_value=10, max_value=200, value=50)
    max_loss_tolerance = st.sidebar.number_input("Max Annual Loss Tolerance (%)", min_value=0.0, max_value=100.0, value=10.0)
    use_hft = (max_loss_tolerance > 15.0)
    initial_capital = st.sidebar.number_input("Initial Capital", min_value=1000, value=100000)

    if st.sidebar.button("Run Simulation"):
        st.write(f"**Strategy**: {strategy}")
        st.write(f"**Ticker**: {ticker}")
        st.write(f"**Max Loss Tolerance**: {max_loss_tolerance}% => HFT: {use_hft}")
        st.write(f"**Short/Long Windows**: {short_window}/{long_window} (MA Crossover)")

        # Data range
        start_date = "2023-12-01"
        end_date   = "2025-01-15"

        # 1) FETCH
        df = fetch_data(ticker, start_date, end_date, use_hft=use_hft)
        if df.empty:
            st.error("No data returned. Check the ticker or date range.")
            return

        # Restrict to 2024
        df = df.loc["2024-01-01":"2024-12-31"].copy()
        if df.empty:
            st.error("No trading data in 2024 for this ticker.")
            return

        # 2) STRATEGY
        if strategy == "Price Level Movements":
            df = strategy_price_level(df)
        elif strategy == "Channel Breakouts":
            df = strategy_channel_breakout(df, window=20)
        elif strategy == "Arbitrage":
            df = strategy_arbitrage(df)
        elif strategy == "Index Fund Rebalancing":
            df = strategy_index_rebalance(df)
        elif strategy == "Mean Reversion":
            df = strategy_mean_reversion(df, lookback=5)
        elif strategy == "Moving Average Crossover":
            df = strategy_moving_average_crossover(df, short_window, long_window)

        if 'Position' not in df.columns:
            st.error("Strategy failed to produce a 'Position' column. Aborting.")
            return

        # 3) BACKTEST
        df = backtest(df, initial_capital=initial_capital)

        # 4) RESULTS
        final_val = df['Total'].iloc[-1]
        profit = final_val - initial_capital
        st.write(f"**Initial Capital**: ${initial_capital:,.2f}")
        st.write(f"**Final Portfolio Value**: ${final_val:,.2f}")
        st.write(f"**Total Profit (2024)**: ${profit:,.2f}")

        if profit >= 0:
            st.success(f"The AI model made a profit of ${profit:,.2f} in 2024!")
        else:
            st.warning(f"The AI model lost ${-profit:,.2f} in 2024.")

        # 5) PLOTS
        if 'Close' in df.columns:
            fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12,8), sharex=True)

            # Price
            ax1.plot(df.index, df['Close'], label='Close Price', color='blue', alpha=0.6)
            buy_signals  = df[df['Position'] == 1]
            sell_signals = df[df['Position'] == -1]
            ax1.scatter(buy_signals.index, df.loc[buy_signals.index, 'Close'],
                        marker='^', color='green', label='Buy', s=100)
            ax1.scatter(sell_signals.index, df.loc[sell_signals.index, 'Close'],
                        marker='v', color='red', label='Sell', s=100)
            ax1.set_ylabel("Price (USD)")
            ax1.set_title(f"{ticker} - Price & Trades in 2024")
            ax1.legend(loc='best')

            # Volume (if exists)
            if 'Volume' in df.columns:
                ax2.bar(df.index, df['Volume'], color='orange', alpha=0.5)
                ax2.set_ylabel("Volume")
                ax2.set_title("Volume Traded")

            plt.tight_layout()
            st.pyplot(fig)

        # Portfolio value chart
        if 'Total' in df.columns:
            fig2, axp = plt.subplots(figsize=(12,5))
            axp.plot(df.index, df['Total'], label='Portfolio Value', color='purple')
            axp.set_ylabel("Portfolio Value (USD)")
            axp.set_title("AI Trades - Portfolio Value in 2024")
            axp.legend(loc='best')
            st.pyplot(fig2)

        # Show last 10 trades
        st.subheader("Sample of Trades (Last 10 rows)")
        cols_to_show = ['Close','Position','Holdings','Cash','Total']
        existing_cols = [c for c in cols_to_show if c in df.columns]
        st.dataframe(df[existing_cols].tail(10))

if __name__ == "__main__":
    main()
