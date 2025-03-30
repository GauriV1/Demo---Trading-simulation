import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import datetime

from alpha_vantage.timeseries import TimeSeries

# -------------------------------------------------------------------
# 1. Fetch Intraday Data from Alpha Vantage
# -------------------------------------------------------------------
def fetch_alpha_vantage_intraday(ticker, interval, outputsize, api_key):
    """
    Fetch intraday data for the given 'ticker' from Alpha Vantage,
    with 'interval' in ['1min','5min','15min','30min','60min'],
    and outputsize in ['compact','full'].

    Returns a DataFrame with columns:
      [Open, High, Low, Close, Volume, Price, Date]
    Where 'Price' = 'Close', 'Date' is a datetime index converted to a column.
    """
    try:
        ts = TimeSeries(key=api_key, output_format='pandas')
        data, meta_data = ts.get_intraday(
            symbol=ticker, 
            interval=interval, 
            outputsize=outputsize
        )
    except Exception as e:
        st.error(f"Error fetching data from Alpha Vantage: {e}")
        return pd.DataFrame()
    
    if data.empty:
        st.warning("No intraday data returned. Possibly invalid ticker or plan limit.")
        return pd.DataFrame()

    data.rename(columns={
        '1. open': 'Open',
        '2. high': 'High',
        '3. low':  'Low',
        '4. close': 'Close',
        '5. volume': 'Volume'
    }, inplace=True)

    data.sort_index(inplace=True)  # ascending by datetime
    data['Date'] = data.index
    data.reset_index(drop=True, inplace=True)

    data['Price'] = data['Close']
    return data

# -------------------------------------------------------------------
# 2. Indicator Calculations
# -------------------------------------------------------------------
def compute_rsi(series, period=14):
    delta = series.diff().dropna()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period).mean()
    rs = gain / loss
    return 100 - 100/(1+rs)

def add_indicators(df, short_window=20, long_window=50):
    """
    Computes rolling SMAs using smaller windows (20, 50) to reduce risk of 
    dropping all rows if data is limited. Also calculates RSI & Volatility.
    
    Drops NaNs afterward. If that yields an empty DF, we handle it later.
    """
    # Only proceed if there's enough data
    if len(df) < long_window:
        # Not enough bars to compute the longest SMA
        st.warning(f"Not enough data to compute a {long_window}-bar SMA. Found only {len(df)} bars.")
        return

    df['SMA_short'] = df['Price'].rolling(window=short_window).mean()
    df['SMA_long']  = df['Price'].rolling(window=long_window).mean()
    df['RSI'] = compute_rsi(df['Price'])
    df['Volatility'] = df['Price'].rolling(window=20).std()
    
    df.dropna(inplace=True)

# -------------------------------------------------------------------
# 3. Strategy Logic
# -------------------------------------------------------------------
def strategy_sma_crossover(row, position, risk_params):
    sma_s = float(row['SMA_short'])
    sma_l = float(row['SMA_long'])
    thr   = risk_params['sma_threshold']
    if sma_l == 0:
        return 0
    diff = (sma_s - sma_l) / abs(sma_l)
    if diff >= thr and position == 0:
        return 1
    elif diff <= -thr and position == 1:
        return -1
    return 0

def strategy_rsi(row, position, risk_params):
    rsi_val = float(row['RSI'])
    if np.isnan(rsi_val):
        return 0
    if rsi_val < risk_params['rsi_buy'] and position == 0:
        return 1
    elif rsi_val > risk_params['rsi_sell'] and position == 1:
        return -1
    return 0

def strategy_buy_and_hold(row, position, risk_params):
    if position == 0 and row.name == 0:
        return 1
    return 0

def select_strategy_auto(row, current_strategy, position, df, risk_params):
    # Before accessing row['SMA_short'], row['SMA_long'], ensure they're not missing
    sma_s = row.get('SMA_short', np.nan)
    sma_l = row.get('SMA_long', np.nan)
    vol   = row.get('Volatility', np.nan)
    
    if any(pd.isna([sma_s, sma_l, vol])):
        return current_strategy, "No change"

    vol_thresh = df['Volatility'].quantile(0.75)
    trend = 'bullish' if (sma_s > sma_l) else 'bearish'
    if vol >= vol_thresh:
        return strategy_rsi, "RSI"
    else:
        if trend == 'bullish':
            return strategy_sma_crossover, "SMA Crossover"
        else:
            return strategy_buy_and_hold, "Buy & Hold"

def run_algo_highfreq(df, initial_cap, risk_params):
    capital = initial_cap
    position = 0
    shares_held = 0
    trades_log = []
    portfolio_vals = []

    curr_strat = strategy_buy_and_hold
    curr_strat_name = "Buy & Hold"

    for i in range(len(df)):
        row = df.iloc[i]
        price = float(row['Price'])
        if pd.isna(price) or price <= 0:
            portfolio_vals.append(capital + shares_held * 0)
            continue

        # Possibly switch
        chosen_strat, chosen_name = select_strategy_auto(row, curr_strat, position, df, risk_params)
        if chosen_name != curr_strat_name:
            trades_log.append({
                'Date': row['Date'],
                'Event': 'Strategy Switch',
                'From': curr_strat_name,
                'To': chosen_name
            })
            curr_strat = chosen_strat
            curr_strat_name = chosen_name

        # Execute
        action = curr_strat(row, position, risk_params)
        if action == 1 and position == 0:
            shares_to_buy = int(capital // price)
            cost = shares_to_buy * price
            fee = cost * risk_params['cost_rate']
            capital -= (cost + fee)
            shares_held += shares_to_buy
            position = 1
            trades_log.append({
                'Date': row['Date'],
                'Event': 'BUY',
                'Price': price,
                'Shares': shares_to_buy,
                'Strategy': curr_strat_name,
                'Fee': fee
            })
        elif action == -1 and position == 1:
            proceeds = shares_held * price
            fee = proceeds * risk_params['cost_rate']
            capital += (proceeds - fee)
            trades_log.append({
                'Date': row['Date'],
                'Event': 'SELL',
                'Price': price,
                'Shares': shares_held,
                'Strategy': curr_strat_name,
                'Fee': fee
            })
            shares_held = 0
            position = 0

        portfolio_vals.append(capital + shares_held * price)

    # Final sell if still in position
    if position == 1 and shares_held > 0:
        last_price = float(df.iloc[-1]['Price'])
        proceeds = shares_held * last_price
        fee = proceeds * risk_params['cost_rate']
        capital += (proceeds - fee)
        trades_log.append({
            'Date': df.iloc[-1]['Date'],
            'Event': 'SELL (End)',
            'Price': last_price,
            'Shares': shares_held,
            'Strategy': curr_strat_name,
            'Fee': fee
        })
        shares_held = 0
        position = 0

    final_val = capital
    return final_val, portfolio_vals, trades_log

# -------------------------------------------------------------------
# Manual Buy-and-Hold
# -------------------------------------------------------------------
def run_manual_buy_hold(df, initial_cap):
    if df.empty:
        return initial_cap, [], []
    capital = initial_cap
    position = 0
    shares_held = 0
    trades_log = []
    portfolio_vals = []

    for i in range(len(df)):
        row = df.iloc[i]
        price = float(row['Price'])
        if pd.isna(price) or price <= 0:
            portfolio_vals.append(capital + shares_held * 0)
            continue

        # Buy on first bar
        if i == 0 and position == 0:
            qty = int(capital // price)
            cost = qty * price
            capital -= cost
            shares_held += qty
            position = 1
            trades_log.append({
                'Date': row['Date'],
                'Event': 'BUY (Start)',
                'Price': price,
                'Shares': qty,
                'Strategy': 'Manual-BuyHold',
                'Fee': 0.0
            })

        # Sell on last bar
        if i == len(df) - 1 and position == 1:
            proceeds = shares_held * price
            capital += proceeds
            trades_log.append({
                'Date': row['Date'],
                'Event': 'SELL (End)',
                'Price': price,
                'Shares': shares_held,
                'Strategy': 'Manual-BuyHold',
                'Fee': 0.0
            })
            shares_held = 0
            position = 0

        portfolio_vals.append(capital + shares_held * price)

    return capital, portfolio_vals, trades_log

# -------------------------------------------------------------------
# 5. Streamlit UI
# -------------------------------------------------------------------
def main():
    st.title("Intraday Strategy Comparison with Fixes to Avoid 'SMA_short' Error")

    # 1. Alpha Vantage inputs
    alpha_key = st.text_input("Alpha Vantage API Key", "YOUR_ALPHA_VANTAGE_API_KEY", type="password")
    ticker = st.text_input("Ticker (e.g., 'AAPL')", "AAPL")
    interval = st.selectbox("Intraday Interval", ['1min','5min','15min','30min','60min'], index=1)
    outputsize = st.selectbox("Output Size", ['compact','full'], index=0)

    # 2. Strategy / capital inputs
    initial_cap = st.number_input("Initial Capital", value=100000, step=1000)

    st.write("### Algorithmic Strategy Settings")
    risk_choice = st.selectbox("Risk Tolerance", ["low","medium","high"])
    if risk_choice == "low":
        sma_thr = 0.03
        rsi_buy, rsi_sell = 25, 75
        default_fee = 0.0005
    elif risk_choice == "medium":
        sma_thr = 0.07
        rsi_buy, rsi_sell = 30, 70
        default_fee = 0.001
    else:
        sma_thr = 0.10
        rsi_buy, rsi_sell = 35, 65
        default_fee = 0.002

    cost_rate = st.number_input("Transaction cost rate (0.001=0.1%)", 
                                value=default_fee, min_value=0.0, step=0.0001)

    # 3. Go button
    if st.button("Run Comparison"):
        if not alpha_key or alpha_key=="YOUR_ALPHA_VANTAGE_API_KEY":
            st.warning("Please provide a valid Alpha Vantage API key.")
            return

        st.write("**Fetching intraday data...**")
        df = fetch_alpha_vantage_intraday(ticker, interval, outputsize, alpha_key)
        if df.empty:
            st.warning("DataFrame is empty or invalid. Exiting.")
            return

        # 4. Add indicators with smaller rolling windows
        st.write("**Adding Indicators** (SMA windows = 20 & 50, RSI, Volatility)...")
        add_indicators(df, short_window=20, long_window=50)
        if df.empty or len(df)<10:
            st.warning("Not enough data after dropping NaNs. Possibly too few bars for rolling windows.")
            return

        # 5. Build strategy params
        risk_params = {
            'sma_threshold': sma_thr,
            'rsi_buy': rsi_buy,
            'rsi_sell': rsi_sell,
            'cost_rate': cost_rate
        }

        # 6. Run strategies
        st.write("**Running Algorithmic (HF) Strategy**")
        algo_final, algo_series, algo_log = run_algo_highfreq(df, initial_cap, risk_params)
        algo_pnl = algo_final - initial_cap

        st.write("**Running Manual (Buy&Hold) Strategy**")
        man_final, man_series, man_log = run_manual_buy_hold(df, initial_cap)
        man_pnl = man_final - initial_cap

        # 7. Plot
        st.write("## Portfolio Value Over Time")
        fig, ax = plt.subplots(figsize=(10,6))
        ax.plot(range(len(algo_series)), algo_series, label='Algorithmic (HF)', color='blue')
        ax.plot(range(len(man_series)), man_series, label='Manual (Buy&Hold)', color='orange')
        ax.set_title(f"{ticker} Intraday: HF vs Buy&Hold (Alpha Vantage) - SMA fix")
        ax.set_ylabel("Portfolio Value")
        ax.legend(loc='best')
        st.pyplot(fig)

        # 8. Output results
        st.write("### Final Results")
        st.write(f"**Algorithmic** final value: `${algo_final:,.2f}` | PnL: `${algo_pnl:,.2f}`")
        st.write(f"**Manual (Buy&Hold)** final value: `${man_final:,.2f}` | PnL: `${man_pnl:,.2f}`")

        st.write("### Trade Logs")
        st.write("**Algorithmic Trades**:")
        st.dataframe(pd.DataFrame(algo_log))
        st.write("**Manual Buy&Hold Trades**:")
        st.dataframe(pd.DataFrame(man_log))

# -------------------------------------------------------------------
if __name__ == "__main__":
    main()
