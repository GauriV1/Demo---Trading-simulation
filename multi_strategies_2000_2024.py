import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

############################
# 1) Helper Functions
############################

def flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).rstrip('_') for col in df.columns.values]
    return df

def fix_close_column(df: pd.DataFrame) -> pd.DataFrame:
    close_cols = [c for c in df.columns if c.startswith('Close')]
    if len(close_cols) == 0:
        return df
    main_close = close_cols[0]
    df['Close'] = df[main_close]
    return df

def valid_intraday_range(start_dt, end_dt):
    """
    Yahoo Finance only provides about 60 days of intraday data (15m, 30m, etc.).
    Return True if (end_dt - start_dt) <= 60 days, otherwise False.
    """
    delta = end_dt - start_dt
    return delta.days <= 60

def fetch_data(ticker, start_date, end_date, use_hft=False):
    """
    Fetch data from yfinance for the chosen ticker.
    If use_hft=True, we try 15m intervals, but if the date range is > 60 days, we revert to daily.
    """
    # Decide on interval
    if use_hft:
        # Check if intraday range is allowed
        start_dt = pd.to_datetime(start_date)
        end_dt   = pd.to_datetime(end_date)
        if valid_intraday_range(start_dt, end_dt):
            interval = "15m"
        else:
            # Fallback to 1d if range is too large
            interval = "1d"
            st.warning("Requested intraday data beyond 60 days. Falling back to daily interval.")
    else:
        interval = "1d"

    try:
        df = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

    df = flatten_yfinance_columns(df)
    df = fix_close_column(df)

    return df

############################
# 2) Standard Strategies
############################

def strategy_price_level(df: pd.DataFrame):
    if 'Close' not in df.columns:
        return df
    ref_level = df['Close'].mean(skipna=True)
    df['RefLevel'] = ref_level
    df['Signal'] = (df['Close'] > ref_level).astype(int)
    df['Position'] = df['Signal'].diff()
    return df

def strategy_channel_breakout(df: pd.DataFrame, window=20):
    if 'High' not in df.columns or 'Low' not in df.columns:
        return df
    df['ChannelHigh'] = df['High'].rolling(window).max()
    df['ChannelLow']  = df['Low'].rolling(window).min()
    df['Signal'] = 0
    df.loc[df['Close'] > df['ChannelHigh'], 'Signal'] = 1
    df.loc[df['Close'] < df['ChannelLow'], 'Signal'] = 0
    df['Position'] = df['Signal'].diff()
    return df

def strategy_arbitrage(df: pd.DataFrame):
    if len(df) == 0:
        return df
    np.random.seed(42)
    df['Signal'] = np.random.choice([0,1], p=[0.995,0.005], size=len(df))
    df['Position'] = df['Signal'].diff()
    return df

def strategy_index_rebalance(df: pd.DataFrame):
    df['Signal'] = 0
    for idx in df.index:
        if idx.month in [3,6,9,12]:
            month_data = df.loc[str(idx.year)+'-'+str(idx.month)]
            if not month_data.empty and idx == month_data.index[-1]:
                df.loc[idx, 'Signal'] = 1
    df['Position'] = df['Signal'].diff()
    return df

def strategy_mean_reversion(df: pd.DataFrame, lookback=5):
    if 'Close' not in df.columns:
        return df
    df['ShortSMA'] = df['Close'].rolling(lookback).mean()
    buy_cond  = df['Close'] < df['ShortSMA']*0.98
    sell_cond = df['Close'] > df['ShortSMA']*1.02
    df['Signal'] = 0
    df.loc[buy_cond, 'Signal'] = 1
    df.loc[sell_cond, 'Signal'] = 0
    df['Position'] = df['Signal'].diff()
    return df

def strategy_moving_average_crossover(df: pd.DataFrame, short_window=20, long_window=50):
    if 'Close' not in df.columns:
        return df
    df['MA_Short'] = df['Close'].rolling(short_window).mean()
    df['MA_Long']  = df['Close'].rolling(long_window).mean()
    df['Signal'] = (df['MA_Short'] > df['MA_Long']).astype(int)
    df['Position'] = df['Signal'].diff()
    return df

def strategy_market_timing(df: pd.DataFrame, ma_period=200):
    if 'Close' not in df.columns:
        return df
    df['MarketMA'] = df['Close'].rolling(ma_period).mean()
    df['Signal'] = (df['Close'] > df['MarketMA']).astype(int)
    df['Position'] = df['Signal'].diff()
    return df

############################
# 3) HFT 'All Success' Example
############################

def strategy_hft_all_success(df: pd.DataFrame):
    """
    Alternate buy/sell each row, ensuring frequent trades, artificially forced to be profitable.
    """
    if 'Close' not in df.columns:
        return df
    df['Signal'] = 0
    signals = np.tile([1,0], int(len(df)/2)+1)[:len(df)]
    df['Signal'] = signals
    df['Position'] = df['Signal'].diff()
    df['strategy_hft_all_success'] = True
    return df

############################
# 4) Backtest with dtype fix
############################

def backtest(df, initial_capital=10000):
    if 'Close' not in df.columns or 'Position' not in df.columns:
        return df, pd.DataFrame()

    # Convert or ensure columns can hold floats
    df['Holdings'] = df.get('Holdings', pd.Series(np.zeros(len(df)), index=df.index, dtype='float64'))
    df['Cash'] = df.get('Cash', pd.Series(np.full(len(df), initial_capital), index=df.index, dtype='float64'))
    df['Total'] = df.get('Total', pd.Series(np.full(len(df), initial_capital), index=df.index, dtype='float64'))

    in_position = False
    trade_log = []

    for i in range(1, len(df)):
        date_i = df.index[i]
        close_price = float(df['Close'].iloc[i])  # ensure float
        prev_cash = float(df['Cash'].iloc[i-1])
        prev_holdings = float(df['Holdings'].iloc[i-1])

        pos_value_before = prev_holdings * close_price + prev_cash

        if df['Position'].iloc[i] == 1:  # BUY
            if not in_position:
                shares_to_buy = int(prev_cash // close_price)  # integer shares
                new_cash = float(prev_cash - (shares_to_buy * close_price))
                new_holdings = float(shares_to_buy)
                trade_log.append({
                    'Date': date_i,
                    'Action': 'BUY',
                    'Price': close_price,
                    'Shares': shares_to_buy,
                    'Cash_After': new_cash
                })
                in_position = True
            else:
                new_cash = prev_cash
                new_holdings = prev_holdings

        elif df['Position'].iloc[i] == -1:  # SELL
            if in_position:
                new_cash = float(prev_cash + (prev_holdings * close_price))
                trade_log.append({
                    'Date': date_i,
                    'Action': 'SELL',
                    'Price': close_price,
                    'Shares': prev_holdings,
                    'Cash_After': new_cash
                })
                new_holdings = 0.0
                in_position = False
            else:
                new_cash = prev_cash
                new_holdings = 0.0
        else:
            new_cash = prev_cash
            new_holdings = prev_holdings

        df.loc[date_i, 'Holdings'] = float(new_holdings)
        df.loc[date_i, 'Cash'] = float(new_cash)
        df.loc[date_i, 'Total'] = float(new_holdings * close_price + new_cash)

        # If forced HFT success
        if 'strategy_hft_all_success' in df.columns:
            pos_value_after = float(df.loc[date_i, 'Total'])
            if pos_value_after < pos_value_before:
                # Force a small profit
                forced_win = pos_value_before + 1.0
                df.loc[date_i, 'Total'] = forced_win
                df.loc[date_i, 'Cash'] = forced_win  # holdings = 0
                df.loc[date_i, 'Holdings'] = 0.0

    trades_df = pd.DataFrame(trade_log)
    if not trades_df.empty:
        trades_df.set_index('Date', inplace=True)
    return df, trades_df

############################
# 5) Streamlit App
############################

def main():
    st.title("Multi-Strategy AI Trading Model (2000 - 2024) - HFT & Dtype Fix")

    # USER INPUTS
    st.sidebar.header("User Settings")
    ticker = st.sidebar.text_input("Stock Ticker", value="AAPL")

    strategy_choice = st.sidebar.selectbox(
        "Choose Strategy:",
        [
            "Price Level Movements",
            "Channel Breakouts",
            "Arbitrage",
            "Index Fund Rebalancing",
            "Mean Reversion",
            "Moving Average Crossover",
            "Market Timing"
        ]
    )

    short_window = st.sidebar.number_input("Short Window (MA Crossover)", min_value=5, max_value=100, value=20)
    long_window  = st.sidebar.number_input("Long Window (MA Crossover)",  min_value=10, max_value=400, value=50)
    market_timing_ma = st.sidebar.number_input("Market Timing MA Period", min_value=50, max_value=400, value=200)

    # Lower threshold from 15 to 5
    max_loss_tolerance = st.sidebar.number_input("Max Annual Loss Tolerance (%)", min_value=0.0, max_value=100.0, value=3.0)
    use_hft = (max_loss_tolerance > 5.0)

    initial_capital = st.sidebar.number_input("Initial Capital", min_value=1000, value=100000)

    start_date = st.sidebar.date_input("Start Date", datetime.date(2000,1,1))
    end_date   = st.sidebar.date_input("End Date", datetime.date(2024,12,31))

    if st.sidebar.button("Run Simulation"):
        st.subheader(f"Strategy: {strategy_choice}")
        st.write(f"**Ticker**: {ticker}")
        st.write(f"**Loss Tolerance**: {max_loss_tolerance}% => High-Freq Mode: {use_hft}")
        st.write(f"**Short/Long Windows**: {short_window}/{long_window}")
        st.write(f"**Market Timing MA**: {market_timing_ma}")
        st.write(f"**Date Range**: {start_date} to {end_date}")

        # 1) Fetch Data
        df = fetch_data(ticker, start_date, end_date, use_hft=use_hft)
        if df.empty:
            st.error("No data returned. Check the ticker or date range.")
            return

        # 2) Apply Strategy
        if use_hft:
            st.info("High-Frequency Trading All-Success Demo ON!")
            df = strategy_hft_all_success(df)
        else:
            if strategy_choice == "Price Level Movements":
                df = strategy_price_level(df)
            elif strategy_choice == "Channel Breakouts":
                df = strategy_channel_breakout(df, window=20)
            elif strategy_choice == "Arbitrage":
                df = strategy_arbitrage(df)
            elif strategy_choice == "Index Fund Rebalancing":
                df = strategy_index_rebalance(df)
            elif strategy_choice == "Mean Reversion":
                df = strategy_mean_reversion(df, lookback=5)
            elif strategy_choice == "Moving Average Crossover":
                df = strategy_moving_average_crossover(df, short_window, long_window)
            elif strategy_choice == "Market Timing":
                df = strategy_market_timing(df, ma_period=market_timing_ma)

        if 'Position' not in df.columns:
            st.error("Strategy did not produce a Position column. Unable to backtest.")
            return

        # 3) Backtest
        df, trades_df = backtest(df, initial_capital=initial_capital)
        if len(df) == 0:
            st.error("After backtest, DataFrame is empty. No results.")
            return

        # 4) Final Stats
        final_val = float(df['Total'].iloc[-1])
        profit = final_val - initial_capital

        st.write(f"**Initial Capital**: ${initial_capital:,.2f}")
        st.write(f"**Final Portfolio Value**: ${final_val:,.2f}")
        st.write(f"**Total Profit**: ${profit:,.2f}")
        if profit >= 0:
            st.success(f"Profit of ${profit:,.2f}")
        else:
            st.warning(f"Loss of ${-profit:,.2f}")

        # 5) PLOTS
        if 'Close' in df.columns:
            fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12,8), sharex=True)

            ax1.plot(df.index, df['Close'], label='Close', color='blue', alpha=0.6)
            buy_signals = df[df['Position'] == 1]
            sell_signals = df[df['Position'] == -1]
            ax1.scatter(buy_signals.index, buy_signals['Close'],
                        marker='^', color='green', s=100, label='BUY')
            ax1.scatter(sell_signals.index, sell_signals['Close'],
                        marker='v', color='red', s=100, label='SELL')
            ax1.set_ylabel("Price (USD)")
            ax1.set_title(f"{ticker} Price & Trades ({start_date} to {end_date})")
            ax1.legend(loc='best')

            if 'Volume' in df.columns:
                ax2.bar(df.index, df['Volume'], color='orange', alpha=0.5)
                ax2.set_ylabel("Volume")
                ax2.set_title("Volume Traded")

            plt.tight_layout()
            st.pyplot(fig)

        if 'Total' in df.columns:
            fig2, axp = plt.subplots(figsize=(12,5))
            axp.plot(df.index, df['Total'], label='Portfolio Value', color='purple')
            axp.set_ylabel("Value (USD)")
            axp.set_title("Portfolio Value Over Time")
            axp.legend(loc='best')
            st.pyplot(fig2)

        # 6) Trade Log
        st.subheader("Trade Log")
        if not trades_df.empty:
            st.dataframe(trades_df)
        else:
            st.write("No trades were made.")

        # Show tail of final DataFrame
        st.subheader("Final Data (Last 10 rows)")
        show_cols = ['Close','Position','Holdings','Cash','Total']
        existing = [c for c in df.columns if c in show_cols]
        st.dataframe(df[existing].tail(10))


if __name__ == "__main__":
    main()
