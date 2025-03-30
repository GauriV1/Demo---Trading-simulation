import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from alpha_vantage.timeseries import TimeSeries
import datetime

# -----------------------------------------
# 1. Helper: We'll define a function that retrieves daily data from Alpha Vantage
# -----------------------------------------
def fetch_alpha_vantage_data(ticker, start_date, end_date, api_key):
    """Returns a DataFrame with columns: [Open, High, Low, Close, Volume, Price, Date]."""
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

    # Create 'Price' from 'Close'
    df['Price'] = df['Close']

    # Drop NaN Price rows
    df.dropna(subset=['Price'], inplace=True)
    if df.empty:
        return df

    # Put index into 'Date' col
    df['Date'] = df.index
    df.reset_index(drop=True, inplace=True)

    return df

# -----------------------------------------
# 2. Indicators & Strategies
# -----------------------------------------
def compute_rsi(series, period=14):
    delta = series.diff().dropna()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period).mean()
    rs = gain / loss
    return 100 - 100/(1+rs)

def add_indicators(df):
    """Adds SMA, RSI, Volatility, etc. to the DataFrame in-place."""
    df['SMA_short'] = df['Price'].rolling(window=50).mean()
    df['SMA_long']  = df['Price'].rolling(window=200).mean()
    df['RSI'] = compute_rsi(df['Price'])
    df['Volatility'] = df['Price'].rolling(window=20).std()
    df.dropna(inplace=True)

def strategy_sma_crossover(row, position, risk_params):
    """
    Buys if SMA_short > SMA_long + threshold,
    Sells if SMA_short < SMA_long - threshold.
    """
    sma_short_val = float(row['SMA_short'])
    sma_long_val  = float(row['SMA_long'])
    threshold     = risk_params['sma_threshold']  # e.g. 0.03 = 3%
    
    # If the difference is more than threshold % of the 'SMA_long_val'
    # or we can interpret threshold as absolute difference. 
    # Here, let's treat it as a fraction of the SMA_long:
    # i.e. buy if (SMA_short - SMA_long) / SMA_long >= threshold
    # That means an ~X% difference from the long SMA.
    if sma_long_val <= 0:
        return 0

    diff_percent = (sma_short_val - sma_long_val) / abs(sma_long_val)

    if diff_percent >= threshold and position == 0:
        return 1  # BUY
    elif diff_percent <= -threshold and position == 1:
        return -1  # SELL
    return 0

def strategy_rsi(row, position, risk_params):
    """
    Buys if RSI < buy_thresh,
    Sells if RSI > sell_thresh.
    """
    rsi_val = float(row['RSI'])
    buy_thresh  = risk_params['rsi_buy']
    sell_thresh = risk_params['rsi_sell']
    if np.isnan(rsi_val):
        return 0
    
    if rsi_val < buy_thresh and position == 0:
        return 1
    elif rsi_val > sell_thresh and position == 1:
        return -1
    return 0

def strategy_buy_and_hold(row, position, risk_params):
    """
    Buy on the first row, hold until the end. 
    """
    if position == 0 and row.name == 0:
        return 1
    return 0

def select_strategy(row, current_strategy, position, df, risk_params):
    """
    Automatic strategy switching. 
    For example:
     - If volatility > 75th percentile => use RSI
     - If short SMA > long => use SMA crossover
     - else => buy & hold
    """
    sma_short_val = row['SMA_short']
    sma_long_val  = row['SMA_long']
    vol_val       = row['Volatility']

    if any(pd.isna([sma_short_val, sma_long_val, vol_val])):
        # Keep the same strategy if indicators aren't ready
        return current_strategy, "No change"

    high_vol_thresh = df['Volatility'].quantile(0.75)
    trend = 'bullish' if (sma_short_val > sma_long_val) else 'bearish_or_sideways'

    if vol_val >= high_vol_thresh:
        return strategy_rsi, "RSI"
    else:
        if trend == 'bullish':
            return strategy_sma_crossover, "SMA Crossover"
        else:
            return strategy_buy_and_hold, "Buy & Hold"

# -----------------------------------------
# 3. Main simulation function 
# -----------------------------------------
def run_simulation(df, initial_capital, risk_params, enable_auto_switch=True):
    """
    Runs a single pass of the backtest on the given DataFrame `df` 
    with the specified risk_params and returns final metrics 
    plus daily portfolio values. 
    enable_auto_switch determines if we do strategy-switch or just pick 1 strategy.
    """

    # We start with buy-and-hold as the default strategy
    current_strategy = strategy_buy_and_hold
    current_strategy_name = 'Buy & Hold'

    capital = initial_capital
    position = 0
    shares_held = 0

    trades_log = []
    portfolio_vals = []
    
    # Naive "manual" approach for comparison
    manual_capital = initial_capital
    manual_position = 0
    manual_shares  = 0
    manual_vals    = []

    for i in range(len(df)):
        row = df.iloc[i]
        price = float(row['Price'])
        if price <= 0 or pd.isna(price):
            portfolio_vals.append(capital + shares_held * 0)
            manual_vals.append(manual_capital + manual_shares * 0)
            continue

        # Possibly switch strategy if auto-switch is enabled
        if enable_auto_switch:
            chosen_strat, strat_name = select_strategy(row, current_strategy, position, df, risk_params)
            if strat_name != current_strategy_name:
                # Log the strategy switch
                trades_log.append({
                    'Date': row['Date'],
                    'Event': 'Strategy Switch',
                    'From': current_strategy_name,
                    'To': strat_name
                })
                current_strategy = chosen_strat
                current_strategy_name = strat_name

        # Execute strategy
        action = current_strategy(row, position, risk_params)
        if action == 1 and position == 0:
            # BUY
            shares_to_buy = int(capital // price)
            cost_of_trade = shares_to_buy * price
            
            # Transaction cost 
            transaction_fee = risk_params['cost_rate'] * cost_of_trade
            
            capital -= (cost_of_trade + transaction_fee)
            position = 1
            shares_held += shares_to_buy
            trades_log.append({
                'Date': row['Date'],
                'Event': 'BUY',
                'Price': price,
                'Shares': shares_to_buy,
                'Strategy': current_strategy_name,
                'Fee': transaction_fee
            })

        elif action == -1 and position == 1:
            # SELL
            proceeds = shares_held * price
            transaction_fee = risk_params['cost_rate'] * proceeds
            
            capital += (proceeds - transaction_fee)
            trades_log.append({
                'Date': row['Date'],
                'Event': 'SELL',
                'Price': price,
                'Shares': shares_held,
                'Strategy': current_strategy_name,
                'Fee': transaction_fee
            })
            shares_held = 0
            position = 0

        # daily portfolio val
        daily_val = capital + shares_held * price
        portfolio_vals.append(daily_val)

        # manual approach
        if i == 0:
            # buy on first day
            m_shares_to_buy = int(manual_capital // price)
            manual_capital -= (m_shares_to_buy * price)
            manual_position = 1
            manual_shares += m_shares_to_buy
        elif i == len(df) - 1:
            # sell on last day
            if manual_position == 1:
                manual_capital += (manual_shares * price)
                manual_shares = 0
                manual_position = 0

        manual_vals.append(manual_capital + manual_shares * price)

    # If we ended with an open position
    if position == 1 and shares_held > 0:
        last_price = float(df.iloc[-1]['Price'])
        proceeds = shares_held * last_price
        transaction_fee = risk_params['cost_rate'] * proceeds
        capital += (proceeds - transaction_fee)
        shares_held = 0
        position = 0

    final_val = capital
    final_manual_val = manual_capital

    return final_val, final_manual_val, portfolio_vals, manual_vals, trades_log

# -----------------------------------------
# 4. Simple Overfitting Check
#    We'll split the data into train/test halves 
#    and run the same logic on both.
# -----------------------------------------
def run_train_test_split(df, initial_capital, risk_params):
    """
    Splits df into 2 halves: 
      - train part
      - test part
    and runs the same strategy logic on each portion.
    """

    # Sort by date just in case
    df.sort_values(by='Date', inplace=True)

    # We'll do a 50/50 split
    mid_idx = len(df) // 2
    train_df = df.iloc[:mid_idx].copy()
    test_df  = df.iloc[mid_idx:].copy()

    # Run the strategy on train
    final_train, final_train_man, train_vals, train_man_vals, train_log = run_simulation(train_df, initial_capital, risk_params, enable_auto_switch=True)

    # We'll keep the same risk_params 
    # (the idea is we "learned" or "chose" them on train),
    # then run on test with FRESH capital to see if it overfits
    # i.e. not carrying over the capital from train 
    # because we want to see out-of-sample performance from scratch
    final_test, final_test_man, test_vals, test_man_vals, test_log = run_simulation(test_df, initial_capital, risk_params, enable_auto_switch=True)

    return {
        'train_final': final_train,
        'train_manual': final_train_man,
        'test_final': final_test,
        'test_manual': final_test_man,
        'train_vals': train_vals,
        'test_vals': test_vals,
        'train_man_vals': train_man_vals,
        'test_man_vals': test_man_vals,
        'train_log': train_log,
        'test_log': test_log
    }

# -----------------------------------------
# 5. Streamlit UI
# -----------------------------------------
def main():
    st.title("Alpha Vantage: Multi-Strategy with Overfitting Check & Transaction Costs")

    # Ticker & Date Inputs
    ticker = st.text_input("Ticker (e.g. AAPL)", "AAPL")
    start_date = st.date_input("Start Date", datetime.date(2000,1,1))
    end_date   = st.date_input("End Date", datetime.date(2024,1,1))
    
    # Risk Tolerance Selection
    # We'll interpret 0-3% (low), 3-7% (medium), 7-10% (high)
    risk_choice = st.selectbox("Risk Tolerance", ["low","medium","high"])
    
    # Map these to strategy thresholds
    # e.g. for 'low' => sma_threshold=0.03, rsi_buy=25, rsi_sell=75, cost_rate=0.0005
    # for 'medium'=>0.07 => ...
    # for 'high'  =>0.10 => ...
    # also let user specify transaction cost. 
    # (You could do it automatically, but we'll let them override.)
    
    default_cost = 0.0005
    if risk_choice == 'low':
        sma_thr = 0.03  # 3% difference
        rsi_buy, rsi_sell = 25, 75
        default_cost = 0.0005
    elif risk_choice == 'medium':
        sma_thr = 0.07  # 7%
        rsi_buy, rsi_sell = 30, 70
        default_cost = 0.001
    else:
        sma_thr = 0.10  # 10%
        rsi_buy, rsi_sell = 35, 65
        default_cost = 0.002

    # Let user override transaction cost 
    cost_rate = st.number_input("Transaction cost rate (e.g. 0.001 = 0.1%)", value=default_cost, min_value=0.0, step=0.0001)

    # Starting capital
    capital_input = st.number_input("Initial Capital", value=100000, min_value=0, step=1000)

    # API Key
    alpha_key = st.text_input("Alpha Vantage API Key", "YOUR_ALPHA_VANTAGE_API_KEY", type="password")

    if st.button("Run Backtest & Overfitting Check"):
        # 1) fetch data
        df = fetch_alpha_vantage_data(ticker, str(start_date), str(end_date), alpha_key)
        if df.empty:
            st.warning("No data returned or all NaN. Exiting.")
            return

        # 2) add indicators
        add_indicators(df)
        if len(df) < 10:
            st.warning("Not enough data after indicators. Exiting.")
            return

        # 3) define the risk_params object
        risk_params = {
            'sma_threshold': sma_thr,
            'rsi_buy': rsi_buy,
            'rsi_sell': rsi_sell,
            'cost_rate': cost_rate
        }

        # 4) run train-test
        results = run_train_test_split(df, capital_input, risk_params)
        
        # 5) show results 
        # Train
        st.subheader("Training Period Results")
        train_strat_pnl = results['train_final'] - capital_input
        train_manual_pnl = results['train_manual'] - capital_input
        st.write(f"Final Strategy Value (Train): ${results['train_final']:,.2f} (PnL={train_strat_pnl:,.2f})")
        st.write(f"Final Manual Value (Train):   ${results['train_manual']:,.2f} (PnL={train_manual_pnl:,.2f})")

        # Test
        st.subheader("Testing Period Results (Out-of-Sample)")
        test_strat_pnl = results['test_final'] - capital_input
        test_manual_pnl = results['test_manual'] - capital_input
        st.write(f"Final Strategy Value (Test):  ${results['test_final']:,.2f} (PnL={test_strat_pnl:,.2f})")
        st.write(f"Final Manual Value (Test):    ${results['test_manual']:,.2f} (PnL={test_manual_pnl:,.2f})")

        # Overfitting sign = if train is super good but test is poor
        st.write("---")
        st.write("**Overfitting Check**: If the strategy performance is much lower in test vs train, it may be overfit.")
        st.write("---")

        # 6) Plot 
        # We'll plot train portion portfolio & test portion portfolio 
        # side by side or in one plot with a time offset
        train_len = len(results['train_vals'])
        # Make them consecutive 
        # We'll re-index the test portion to appear right after train
        # so we can see them in one continuous chart 
        # (Though in reality there's a break in time.)
        merged_vals = results['train_vals'] + results['test_vals']
        merged_man_vals = results['train_man_vals'] + results['test_man_vals']

        # We can also combine 'Date' from train & test, 
        # but for simplicity let's just do integer indexes
        idxs = range(len(merged_vals))
        
        fig, ax = plt.subplots(figsize=(10,6))
        ax.plot(idxs, merged_vals, label='Strategy', color='green')
        ax.plot(idxs, merged_man_vals, label='Manual', color='red', alpha=0.7)
        ax.axvline(train_len, color='gray', linestyle='--', label='Train/Test Split')
        ax.set_title("Train + Test Portfolio Value Over Time (Index-based)")
        ax.set_ylabel("Portfolio Value")
        ax.legend(loc='upper left')
        st.pyplot(fig)

        # 7) Show trade logs
        st.write("**Training Trade Log:**")
        st.dataframe(pd.DataFrame(results['train_log']))

        st.write("**Testing Trade Log:**")
        st.dataframe(pd.DataFrame(results['test_log']))

# -----------------------------------------
# Run
# -----------------------------------------
if __name__ == "__main__":
    main()
