import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from alpha_vantage.timeseries import TimeSeries
import datetime


# ----------------------------------------------------------------------------
# 1. Data Retrieval
# ----------------------------------------------------------------------------
def fetch_alpha_vantage_data(ticker, start_date, end_date, api_key):
    """Returns DataFrame with [Open, High, Low, Close, Volume, Price, Date]."""
    try:
        ts = TimeSeries(key=api_key, output_format='pandas')
        df, meta_data = ts.get_daily(symbol=ticker, outputsize='full')
    except Exception as e:
        st.error(f"Error fetching data from Alpha Vantage: {e}")
        return pd.DataFrame()

    if df.empty:
        return df
    
    # Sort ascending by date
    df.sort_index(inplace=True)

    # Rename columns
    df.rename(columns={
        '1. open': 'Open',
        '2. high': 'High',
        '3. low': 'Low',
        '4. close': 'Close',
        '5. volume': 'Volume'
    }, inplace=True)

    # Filter date range
    df = df.loc[start_date:end_date].copy()
    if df.empty:
        return df

    df['Price'] = df['Close']
    df.dropna(subset=['Price'], inplace=True)
    if df.empty:
        return df

    # Put index (DateTimeIndex) into a column named 'Date'
    df['Date'] = df.index
    df.reset_index(drop=True, inplace=True)
    return df


# ----------------------------------------------------------------------------
# 2. Indicators
# ----------------------------------------------------------------------------
def compute_rsi(series, period=14):
    delta = series.diff().dropna()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period).mean()
    rs = gain / loss
    return 100 - 100/(1+rs)

def add_indicators(df):
    df['SMA_short'] = df['Price'].rolling(window=50).mean()
    df['SMA_long']  = df['Price'].rolling(window=200).mean()
    df['RSI'] = compute_rsi(df['Price'])
    df['Volatility'] = df['Price'].rolling(window=20).std()
    df.dropna(inplace=True)


# ----------------------------------------------------------------------------
# 3. Algorithmic Strategy: unlimited trades
# ----------------------------------------------------------------------------
def strategy_sma_crossover(row, position, risk_params):
    """Buy if (SMA_short - SMA_long)/SMA_long >= sma_threshold; Sell if <= -threshold."""
    sma_short_val = float(row['SMA_short'])
    sma_long_val  = float(row['SMA_long'])
    thr = risk_params['sma_threshold']
    
    if sma_long_val == 0:
        return 0
    
    diff_pct = (sma_short_val - sma_long_val) / abs(sma_long_val)
    if diff_pct >= thr and position == 0:
        return 1  # BUY
    elif diff_pct <= -thr and position == 1:
        return -1 # SELL
    return 0

def strategy_rsi(row, position, risk_params):
    """Buy if RSI < rsi_buy; Sell if RSI > rsi_sell."""
    rsi_val = float(row['RSI'])
    buy_thr = risk_params['rsi_buy']
    sell_thr= risk_params['rsi_sell']
    if np.isnan(rsi_val):
        return 0
    if rsi_val < buy_thr and position == 0:
        return 1
    elif rsi_val > sell_thr and position == 1:
        return -1
    return 0

def strategy_buy_and_hold(row, position, risk_params):
    """Buy on the first row, hold."""
    if position == 0 and row.name == 0:
        return 1
    return 0

def select_strategy_algo(row, current_strategy, position, df, risk_params):
    """
    For the algorithmic approach, auto-switch among RSI, SMA, B&H 
    depending on volatility & trend. 
    """
    sma_s = row['SMA_short']
    sma_l = row['SMA_long']
    vol   = row['Volatility']
    if any(pd.isna([sma_s, sma_l, vol])):
        return current_strategy, "No change"
    
    high_vol_thresh = df['Volatility'].quantile(0.75)
    trend = 'bullish' if (sma_s > sma_l) else 'bearish_or_sideways'
    
    if vol >= high_vol_thresh:
        return strategy_rsi, "RSI"
    else:
        if trend == 'bullish':
            return strategy_sma_crossover, "SMA Crossover"
        else:
            return strategy_buy_and_hold, "Buy & Hold"

def run_simulation_algo_unlimited(df, initial_capital, risk_params):
    """
    Algorithmic approach with unlimited trades. 
    """
    capital = initial_capital
    position = 0
    shares_held = 0
    trades_log = []
    portfolio_vals = []

    current_strategy = strategy_buy_and_hold
    current_strat_name = "Buy & Hold"

    for i in range(len(df)):
        row = df.iloc[i]
        price = float(row['Price'])
        if price <= 0 or pd.isna(price):
            portfolio_vals.append(capital + shares_held * 0)
            continue
        
        # Possibly switch strategy
        chosen_strat, chosen_name = select_strategy_algo(row, current_strategy, position, df, risk_params)
        if chosen_name != current_strat_name:
            trades_log.append({
                'Date': row['Date'],
                'Event': 'Strategy Switch',
                'From': current_strat_name,
                'To': chosen_name
            })
            current_strategy = chosen_strat
            current_strat_name = chosen_name

        # Execute strategy
        action = current_strategy(row, position, risk_params)
        if action == 1 and position == 0:
            # BUY
            shares_to_buy = int(capital // price)
            cost_of_trade = shares_to_buy * price
            fee = risk_params['cost_rate'] * cost_of_trade
            capital -= (cost_of_trade + fee)
            position = 1
            shares_held += shares_to_buy
            trades_log.append({
                'Date': row['Date'],
                'Event': 'BUY',
                'Price': price,
                'Shares': shares_to_buy,
                'Strategy': current_strat_name,
                'Fee': fee
            })
        elif action == -1 and position == 1:
            # SELL
            proceeds = shares_held * price
            fee = risk_params['cost_rate'] * proceeds
            capital += (proceeds - fee)
            trades_log.append({
                'Date': row['Date'],
                'Event': 'SELL',
                'Price': price,
                'Shares': shares_held,
                'Strategy': current_strat_name,
                'Fee': fee
            })
            shares_held = 0
            position = 0

        daily_val = capital + shares_held * price
        portfolio_vals.append(daily_val)

    # close open position
    if position == 1 and shares_held > 0:
        last_price = df.iloc[-1]['Price']
        proceeds = shares_held * last_price
        fee = risk_params['cost_rate'] * proceeds
        capital += (proceeds - fee)
        shares_held = 0
        position = 0

    final_val = capital
    return final_val, portfolio_vals, trades_log


# ----------------------------------------------------------------------------
# 4. Manual / Tentative Approach: max 3 trades/year
#    We'll do a simple RSI-based approach but limit to 3 trades/year total.
# ----------------------------------------------------------------------------
def run_simulation_manual_3(df, initial_capital, risk_params):
    """
    Manual approach with max 3 trades/year. 
    For demonstration, let's say:
     - We'll use a simplified RSI check:
       * Buy if RSI < rsi_buy
       * Sell if RSI > rsi_sell
     - But we only allow up to 3 trades each year 
       (Buy or Sell each count as a trade).
    """
    capital = initial_capital
    position = 0
    shares_held = 0
    trades_log = []
    portfolio_vals = []

    # track trades/year
    trades_this_year = {}

    for i in range(len(df)):
        row = df.iloc[i]
        price = float(row['Price'])
        if price <= 0 or pd.isna(price):
            portfolio_vals.append(capital + shares_held * 0)
            continue
        
        # figure out year
        row_year = pd.to_datetime(row['Date']).year
        if row_year not in trades_this_year:
            trades_this_year[row_year] = 0

        # If already 3 trades this year, do nothing
        if trades_this_year[row_year] >= 3:
            daily_val = capital + shares_held * price
            portfolio_vals.append(daily_val)
            continue

        # Simplistic "manual" rule: RSI approach
        rsi_val = row['RSI']
        buy_thr = risk_params['rsi_buy']
        sell_thr= risk_params['rsi_sell']

        action = 0
        if position == 0 and rsi_val < buy_thr:
            action = 1
        elif position == 1 and rsi_val > sell_thr:
            action = -1

        if action == 1 and position == 0:
            shares_to_buy = int(capital // price)
            cost_of_trade = shares_to_buy * price
            fee = risk_params['cost_rate'] * cost_of_trade
            capital -= (cost_of_trade + fee)
            position = 1
            shares_held += shares_to_buy
            trades_log.append({
                'Date': row['Date'],
                'Event': 'BUY',
                'Price': price,
                'Shares': shares_to_buy,
                'Strategy': 'Manual(3-yr)',
                'Fee': fee
            })
            trades_this_year[row_year] += 1

        elif action == -1 and position == 1:
            proceeds = shares_held * price
            fee = risk_params['cost_rate'] * proceeds
            capital += (proceeds - fee)
            trades_log.append({
                'Date': row['Date'],
                'Event': 'SELL',
                'Price': price,
                'Shares': shares_held,
                'Strategy': 'Manual(3-yr)',
                'Fee': fee
            })
            shares_held = 0
            position = 0
            trades_this_year[row_year] += 1

        daily_val = capital + shares_held * price
        portfolio_vals.append(daily_val)

    # close open position at end
    if position == 1 and shares_held > 0:
        last_price = df.iloc[-1]['Price']
        proceeds = shares_held * last_price
        fee = risk_params['cost_rate'] * proceeds
        capital += (proceeds - fee)
        shares_held = 0
        position = 0

    final_val = capital
    return final_val, portfolio_vals, trades_log


# ----------------------------------------------------------------------------
# 5. Overfitting Check: Train/Test
# ----------------------------------------------------------------------------
def run_train_test_split(df, initial_capital, risk_params):
    """
    Splits data 50/50 into train/test and runs:
      - 'Algorithmic' unlimited trades
      - 'Manual' max 3 trades/year
    on each portion (fresh capital).
    """
    df.sort_values(by='Date', inplace=True)
    mid_idx = len(df) // 2
    train_df = df.iloc[:mid_idx].copy()
    test_df  = df.iloc[mid_idx:].copy()

    # =============== TRAIN ===============
    # Algo
    train_algo_final, train_algo_vals, train_algo_log = run_simulation_algo_unlimited(
        train_df, initial_capital, risk_params
    )
    # Manual
    train_man_final, train_man_vals, train_man_log = run_simulation_manual_3(
        train_df, initial_capital, risk_params
    )

    # =============== TEST ===============
    test_algo_final, test_algo_vals, test_algo_log = run_simulation_algo_unlimited(
        test_df, initial_capital, risk_params
    )
    test_man_final, test_man_vals, test_man_log = run_simulation_manual_3(
        test_df, initial_capital, risk_params
    )

    return {
        'train_algo_val': train_algo_final,
        'train_algo_series': train_algo_vals,
        'train_algo_log': train_algo_log,
        'train_man_val': train_man_final,
        'train_man_series': train_man_vals,
        'train_man_log': train_man_log,

        'test_algo_val': test_algo_final,
        'test_algo_series': test_algo_vals,
        'test_algo_log': test_algo_log,
        'test_man_val': test_man_final,
        'test_man_series': test_man_vals,
        'test_man_log': test_man_log
    }


# ----------------------------------------------------------------------------
# 6. Streamlit UI
# ----------------------------------------------------------------------------
def main():
    st.title("Algorithmic vs. Manual(3 trades/year) — Overfitting & Transaction Test")
    st.write("""
    **Algorithmic (Unlimited Trades)**: Auto-switch among RSI, SMA crossover, 
    or Buy & Hold based on volatility & trend.\n
    **Manual (Max 3 Trades/Year)**: A simplified RSI approach with a 3/year limit.\n
    **Overfitting**: We split data 50/50 into Train & Test sets, each starting with fresh capital.\n
    **Transaction Costs**: A user-defined % of notional for each trade.\n
    """)

    ticker = st.text_input("Ticker (e.g. 'AAPL')", "AAPL")
    start_date = st.date_input("Start Date", datetime.date(2000,1,1))
    end_date   = st.date_input("End Date", datetime.date(2024,1,1))

    risk_choice = st.selectbox("Risk Tolerance", ["low","medium","high"])
    if risk_choice == "low":
        sma_thr = 0.03
        rsi_buy, rsi_sell = 25, 75
        default_cost = 0.0005
    elif risk_choice == "medium":
        sma_thr = 0.07
        rsi_buy, rsi_sell = 30, 70
        default_cost = 0.001
    else:
        sma_thr = 0.10
        rsi_buy, rsi_sell = 35, 65
        default_cost = 0.002

    cost_rate = st.number_input("Transaction cost rate (0.001 = 0.1%)", value=default_cost, min_value=0.0, step=0.0001)
    initial_cap = st.number_input("Initial Capital", value=100000, min_value=0, step=1000)
    api_key = st.text_input("Alpha Vantage API Key", "YOUR_ALPHA_VANTAGE_API_KEY", type="password")

    if st.button("Run Overfitting Test"):
        st.write("**Fetching Data...**")
        df = fetch_alpha_vantage_data(ticker, str(start_date), str(end_date), api_key)
        if df.empty:
            st.warning("No data found or all NaN. Exiting.")
            return

        st.write("**Adding Indicators...**")
        add_indicators(df)
        if len(df) < 10:
            st.warning("Not enough data after indicators. Exiting.")
            return

        # risk params
        risk_params = {
            'sma_threshold': sma_thr,
            'rsi_buy': rsi_buy,
            'rsi_sell': rsi_sell,
            'cost_rate': cost_rate
        }

        st.write("**Splitting into Train/Test & Running...**")
        results = run_train_test_split(df, initial_cap, risk_params)

        # TRAIN
        train_algo_val = results['train_algo_val']
        train_algo_pnl = train_algo_val - initial_cap
        train_man_val  = results['train_man_val']
        train_man_pnl  = train_man_val - initial_cap

        # TEST
        test_algo_val  = results['test_algo_val']
        test_algo_pnl  = test_algo_val - initial_cap
        test_man_val   = results['test_man_val']
        test_man_pnl   = test_man_val - initial_cap

        # Display
        st.subheader("TRAIN Results")
        st.write(f"**Algorithmic (Unlimited)** Final Value: ${train_algo_val:,.2f} | PnL: ${train_algo_pnl:,.2f}")
        st.write(f"**Manual (3/year)**        Final Value: ${train_man_val:,.2f} | PnL: ${train_man_pnl:,.2f}")

        st.subheader("TEST Results (Overfitting Check)")
        st.write(f"**Algorithmic (Unlimited)** Final Value: ${test_algo_val:,.2f} | PnL: ${test_algo_pnl:,.2f}")
        st.write(f"**Manual (3/year)**        Final Value: ${test_man_val:,.2f} | PnL: ${test_man_pnl:,.2f}")

        st.write("---")
        st.write("**If the test performance is much worse than train, it may indicate overfitting.**")

        # Combine for plotting: We'll create 2 separate lines for Algo + Manual
        # for train portion and test portion, but let's unify them into single arrays for each approach
        train_len = len(results['train_algo_series'])  # This is how many days in train

        # Merge ALGO
        merged_algo = results['train_algo_series'] + results['test_algo_series']
        # Merge MANUAL
        merged_man = results['train_man_series'] + results['test_man_series']
        idxs = range(len(merged_algo))

        fig, ax = plt.subplots(figsize=(10,6))
        ax.plot(idxs, merged_algo, label='Algo (Unlimited)', color='blue')
        ax.plot(idxs, merged_man,  label='Manual (3/year)', color='orange')
        ax.axvline(train_len, color='gray', linestyle='--', label='Train/Test Split')
        ax.set_title("Portfolio Value Over Train + Test")
        ax.set_ylabel("Portfolio Value")
        ax.legend(loc='upper left')
        st.pyplot(fig)

        # Logs
        st.write("**Train Logs (Algo)**")
        st.dataframe(pd.DataFrame(results['train_algo_log']))
        st.write("**Train Logs (Manual)**")
        st.dataframe(pd.DataFrame(results['train_man_log']))

        st.write("**Test Logs (Algo)**")
        st.dataframe(pd.DataFrame(results['test_algo_log']))
        st.write("**Test Logs (Manual)**")
        st.dataframe(pd.DataFrame(results['test_man_log']))


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
