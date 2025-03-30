import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from alpha_vantage.timeseries import TimeSeries
import datetime

# ---------------------------------------------------------------------------
# 1. Data Retrieval
# ---------------------------------------------------------------------------
def fetch_alpha_vantage_data(ticker, start_date, end_date, api_key):
    """
    Fetches daily data (not intraday) from Alpha Vantage for the given 'ticker'
    and date range. Returns a DataFrame with columns:
      [Open, High, Low, Close, Volume, Price, Date]
    'Price' = 'Close', 'Date' is a normal column.
    """
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
        '3. low':  'Low',
        '4. close': 'Close',
        '5. volume': 'Volume'
    }, inplace=True)

    # Filter date range
    df = df.loc[start_date:end_date].copy()
    if df.empty:
        return df

    # 'Price' = 'Close'
    df['Price'] = df['Close']
    
    df.dropna(subset=['Price'], inplace=True)
    if df.empty:
        return df

    df['Date'] = df.index
    df.reset_index(drop=True, inplace=True)
    return df

# ---------------------------------------------------------------------------
# 2. Indicators
# ---------------------------------------------------------------------------
def compute_rsi(series, period=14):
    delta = series.diff().dropna()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period).mean()
    rs = gain / loss
    return 100 - 100/(1+rs)

def add_indicators(df):
    """
    Adds:
      - 50-day SMA => 'SMA_short'
      - 200-day SMA => 'SMA_long'
      - 100-day SMA => 'SMA_100' (for Jim's manual approach)
      - RSI => 'RSI'
      - 20-day rolling std => 'Volatility'
    Then drops rows with NaNs.
    """
    df['SMA_short'] = df['Price'].rolling(window=50).mean()
    df['SMA_long']  = df['Price'].rolling(window=200).mean()
    df['SMA_100']   = df['Price'].rolling(window=100).mean()
    df['RSI'] = compute_rsi(df['Price'])
    df['Volatility'] = df['Price'].rolling(window=20).std()
    df.dropna(inplace=True)

# ---------------------------------------------------------------------------
# 3. Algorithmic Strategies
# ---------------------------------------------------------------------------
def strat_rsi(row, position, thresholds):
    """Simple RSI strategy: buy if RSI < thresholds['rsi_buy'], sell if RSI > thresholds['rsi_sell']."""
    rsi_val = row['RSI']
    if np.isnan(rsi_val):
        return 0
    if (rsi_val < thresholds['rsi_buy']) and (position == 0):
        return 1
    elif (rsi_val > thresholds['rsi_sell']) and (position == 1):
        return -1
    return 0

def strat_sma_cross(row, position, thresholds):
    """
    SMA crossover: buy if (SMA_short - SMA_long)/SMA_long >= sma_threshold,
    sell if <= -sma_threshold
    """
    sma_s = row['SMA_short']
    sma_l = row['SMA_long']
    thr   = thresholds['sma_threshold']
    if sma_l == 0:
        return 0
    diff_pct = (sma_s - sma_l)/abs(sma_l)
    if diff_pct >= thr and position==0:
        return 1
    elif diff_pct <= -thr and position==1:
        return -1
    return 0

def run_backtest(df, initial_cap, strategy_func, thresholds):
    """
    Runs a single-strategy backtest: strategy_func can be 'strat_rsi' or 'strat_sma_cross'.
    No transaction fee. Buys or sells entire capital/holding each time.
    """
    capital = initial_cap
    position = 0
    shares_held = 0
    portfolio_vals = []
    trades_log = []

    for i in range(len(df)):
        row = df.iloc[i]
        price = float(row['Price'])
        if price <= 0 or pd.isna(price):
            portfolio_vals.append(capital + shares_held*0)
            continue

        action = strategy_func(row, position, thresholds)
        if action==1 and position==0:
            # BUY
            shares_to_buy = int(capital // price)
            cost = shares_to_buy * price
            capital -= cost
            shares_held += shares_to_buy
            position=1
            trades_log.append({
                'Date': row['Date'],
                'Event': 'BUY',
                'Price': price,
                'Shares': shares_to_buy
            })
        elif action==-1 and position==1:
            # SELL
            proceeds = shares_held*price
            capital += proceeds
            trades_log.append({
                'Date': row['Date'],
                'Event': 'SELL',
                'Price': price,
                'Shares': shares_held
            })
            shares_held=0
            position=0

        portfolio_vals.append(capital + shares_held*price)

    # final close if still in position
    if position==1 and shares_held>0:
        last_price = float(df.iloc[-1]['Price'])
        proceeds = shares_held * last_price
        capital += proceeds
        trades_log.append({
            'Date': df.iloc[-1]['Date'],
            'Event': 'SELL (End)',
            'Price': last_price,
            'Shares': shares_held
        })
        position=0
        shares_held=0

    final_val = capital
    return final_val, portfolio_vals, trades_log

def run_algo_train(df, initial_cap, thresholds):
    """
    In the train portion, we test two strategies: RSI-based or SMA-based.
    We'll see which ends with a higher final portfolio. Then we pick that as 'best strategy'.
    Return best_strategy_name, best_strategy_func
    """
    # 1) RSI approach
    rsi_final, rsi_vals, _ = run_backtest(df, initial_cap, strat_rsi, thresholds)
    # 2) SMA approach
    sma_final, sma_vals, _ = run_backtest(df, initial_cap, strat_sma_cross, thresholds)

    if rsi_final >= sma_final:
        return 'RSI', strat_rsi
    else:
        return 'SMA', strat_sma_cross

def run_algo_test(df, initial_cap, best_strategy_func, thresholds):
    """Apply the best strategy from train to the test portion."""
    final_val, port_vals, trade_log = run_backtest(df, initial_cap, best_strategy_func, thresholds)
    return final_val, port_vals, trade_log

# ---------------------------------------------------------------------------
# 4. Manual Strategy (Jim's 100-day approach)
# ---------------------------------------------------------------------------
def run_manual_jim_100ma(df, initial_cap):
    """
    Jim's approach:
      - We need a 100-day SMA. The price must be in an uptrend => we check if the slope of SMA_100 is up.
      - Buy if the price crosses above the 100-day from below (and 100-day slope is up).
      - Sell if the price crosses back below the 100-day.
      - If it whipsaws or is sideways, we skip trades. (We'll interpret 'sideways' as the slope <= 0).
    """
    capital = initial_cap
    position=0
    shares_held=0
    trades_log=[]
    portfolio_vals=[]

    for i in range(len(df)):
        row = df.iloc[i]
        price = float(row['Price'])
        sma_100 = float(row['SMA_100'])
        # We'll check slope of 100-day by comparing today's SMA_100 vs e.g. 5 bars ago
        if i>=5:
            old_sma_100 = float(df.iloc[i-5]['SMA_100'])
        else:
            old_sma_100 = np.nan

        # skip if we can't evaluate
        if pd.isna(price) or pd.isna(sma_100) or pd.isna(old_sma_100):
            portfolio_vals.append(capital + shares_held*0)
            continue

        # Uptrend if slope > 0
        slope = (sma_100 - old_sma_100)
        uptrend = (slope > 0)

        # Are we crossing above from below?
        # We'll check if previous bar's price < sma_100, and now price > sma_100
        action=0
        if i>0:
            prev_price = float(df.iloc[i-1]['Price'])
            prev_sma_100= float(df.iloc[i-1]['SMA_100'])
            # CROSS above
            cross_above = (prev_price < prev_sma_100) and (price > sma_100)
            # CROSS below
            cross_below = (prev_price > prev_sma_100) and (price < sma_100)

            if position==0:
                if cross_above and uptrend:
                    action=1  # BUY
            elif position==1:
                if cross_below:
                    action=-1 # SELL

        if action==1 and position==0:
            shares_to_buy= int(capital//price)
            cost= shares_to_buy*price
            capital-= cost
            shares_held+= shares_to_buy
            position=1
            trades_log.append({
                'Date': row['Date'],
                'Event': 'BUY',
                'Price': price,
                'Shares': shares_to_buy
            })
        elif action==-1 and position==1:
            proceeds= shares_held*price
            capital+= proceeds
            trades_log.append({
                'Date': row['Date'],
                'Event': 'SELL',
                'Price': price,
                'Shares': shares_held
            })
            position=0
            shares_held=0

        portfolio_vals.append(capital + shares_held*price)

    # final close if still in position
    if position==1 and shares_held>0:
        last_price= float(df.iloc[-1]['Price'])
        proceeds= shares_held*last_price
        capital+= proceeds
        trades_log.append({
            'Date': df.iloc[-1]['Date'],
            'Event': 'SELL (End)',
            'Price': last_price,
            'Shares': shares_held
        })
        position=0
        shares_held=0

    return capital, portfolio_vals, trades_log

# ---------------------------------------------------------------------------
# 5. Train/Test with Manual + Algo
# ---------------------------------------------------------------------------
def run_train_test_split(df, initial_cap, thresholds):
    """
    Split data 50/50 for train & test.
    - ALGO: In train => pick best strategy (RSI vs SMA). In test => apply that strategy.
    - MANUAL: In train => run Jim's approach. In test => run Jim's approach.
    """
    df.sort_values(by='Date', inplace=True)
    mid_idx = len(df)//2
    train_df= df.iloc[:mid_idx].copy()
    test_df= df.iloc[mid_idx:].copy()

    # =============== ALGO ===============
    # 1) Train => pick best among RSI or SMA
    best_name, best_func= run_algo_train(train_df, initial_cap, thresholds)
    # run that best_func on train
    train_final_algo, train_vals_algo, train_log_algo= run_backtest(train_df, initial_cap, best_func, thresholds)
    # 2) Test => apply best_func
    test_final_algo, test_vals_algo, test_log_algo= run_backtest(test_df, initial_cap, best_func, thresholds)

    # =============== MANUAL ===============
    # We'll just run Jim's approach on train, then again on test
    train_final_man, train_vals_man, train_log_man= run_manual_jim_100ma(train_df, initial_cap)
    test_final_man, test_vals_man, test_log_man= run_manual_jim_100ma(test_df, initial_cap)

    return {
      'train_algo_val': train_final_algo,
      'train_algo_series': train_vals_algo,
      'train_algo_log': train_log_algo,
      'test_algo_val': test_final_algo,
      'test_algo_series': test_vals_algo,
      'test_algo_log': test_log_algo,

      'train_man_val': train_final_man,
      'train_man_series': train_vals_man,
      'train_man_log': train_log_man,
      'test_man_val': test_final_man,
      'test_man_series': test_vals_man,
      'test_man_log': test_log_man
    }

# ---------------------------------------------------------------------------
# 6. Streamlit UI
# ---------------------------------------------------------------------------
def main():
    st.title("Algo vs. Manual (Jim's 100-day) with Train/Test & No Fees")

    st.write("""
    **Algorithmic**: picks the best among RSI or SMA-crossover in the train set, then applies it in test.\n
    **Manual**: Jim's 100-day approach (buy on cross above 100-day in an uptrend, sell on cross below).
    """)

    # 1. Ticker + Dates + API Key
    ticker= st.text_input("Ticker", "AAPL")
    start_date= st.date_input("Start Date", datetime.date(2000,1,1))
    end_date= st.date_input("End Date", datetime.date(2024,1,1))
    alpha_key= st.text_input("Alpha Vantage API Key", "YOUR_ALPHA_VANTAGE_API_KEY", type="password")
    
    # 2. Capital
    initial_cap= st.number_input("Initial Capital", value=100000, step=1000)

    # 3. Risk Tolerance => thresholds
    risk_choice= st.selectbox("Risk Tolerance", ["low","medium","high"])
    if risk_choice=="low":
        sma_thr= 0.03
        rsi_buy, rsi_sell= 25, 75
    elif risk_choice=="medium":
        sma_thr= 0.07
        rsi_buy, rsi_sell= 30, 70
    else:
        sma_thr= 0.10
        rsi_buy, rsi_sell= 35, 65

    thresholds= {
      'sma_threshold': sma_thr,
      'rsi_buy': rsi_buy,
      'rsi_sell': rsi_sell
    }

    # 4. Go
    if st.button("Run Overfitting Test"):
        st.write("**Fetching daily data from Alpha Vantage**...")
        df= fetch_alpha_vantage_data(ticker, str(start_date), str(end_date), alpha_key)
        if df.empty:
            st.warning("No data or empty after fetch. Exiting.")
            return

        st.write("**Adding Indicators** (SMA_50, SMA_100, SMA_200, RSI, etc.)...")
        add_indicators(df)
        if len(df)<10:
            st.warning("Not enough data after indicators. Exiting.")
            return

        st.write("**Splitting 50/50 into Train & Test**")
        results= run_train_test_split(df, initial_cap, thresholds)

        # Extract values
        train_algo_val= results['train_algo_val']
        train_algo_series= results['train_algo_series']
        train_algo_log= results['train_algo_log']

        test_algo_val= results['test_algo_val']
        test_algo_series= results['test_algo_series']
        test_algo_log= results['test_algo_log']

        train_man_val= results['train_man_val']
        train_man_series= results['train_man_series']
        train_man_log= results['train_man_log']

        test_man_val= results['test_man_val']
        test_man_series= results['test_man_series']
        test_man_log= results['test_man_log']

        train_algo_pnl= train_algo_val - initial_cap
        test_algo_pnl= test_algo_val - initial_cap
        train_man_pnl= train_man_val - initial_cap
        test_man_pnl= test_man_val - initial_cap

        # ---- OUTPUT
        st.subheader("TRAIN Results")
        st.write(f"**Algorithmic** Final: ${train_algo_val:,.2f}  PnL= ${train_algo_pnl:,.2f}")
        st.write(f"**Manual (Jim)** Final: ${train_man_val:,.2f}  PnL= ${train_man_pnl:,.2f}")

        st.subheader("TEST Results")
        st.write(f"**Algorithmic** Final: ${test_algo_val:,.2f}  PnL= ${test_algo_pnl:,.2f}")
        st.write(f"**Manual (Jim)** Final: ${test_man_val:,.2f}  PnL= ${test_man_pnl:,.2f}")

        # Merge to single arrays for train + test
        train_len= len(train_algo_series)
        algo_merged= train_algo_series + test_algo_series
        man_merged= train_man_series + test_man_series
        idxs= range(len(algo_merged))

        # Price Growth Chart
        st.write("## Chart 1: Stock Price Growth Over Time")
        fig1, ax1= plt.subplots(figsize=(10,6))
        ax1.plot(df['Date'], df['Price'], label=f"{ticker} Price", color='blue')
        ax1.set_title("Stock Price Growth")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Price")
        ax1.legend(loc='best')
        ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        st.pyplot(fig1)

        # RSI Chart
        st.write("## Chart 2: RSI Growth Over Time")
        fig2, ax2= plt.subplots(figsize=(10,6))
        ax2.plot(df['Date'], df['RSI'], label="RSI", color='purple')
        ax2.axhline(rsi_buy, color='green', linestyle='--', label="RSI Buy Threshold")
        ax2.axhline(rsi_sell, color='red', linestyle='--', label="RSI Sell Threshold")
        ax2.set_title("RSI Over Time")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("RSI")
        ax2.legend(loc='best')
        ax2.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        st.pyplot(fig2)

        # Train/Test Portfolio Chart
        st.write("## Chart 3: Train + Test Portfolio Comparison")
        fig3, ax3= plt.subplots(figsize=(10,6))
        ax3.plot(idxs, algo_merged, label='Algorithmic', color='blue')
        ax3.plot(idxs, man_merged, label='Manual (Jim)', color='orange')
        ax3.axvline(train_len, color='gray', linestyle='--', label='Train/Test Split')
        ax3.set_title("Portfolio Value Over Train + Test")
        ax3.set_ylabel("Portfolio Value")
        ax3.legend(loc='best')
        st.pyplot(fig3)

        st.write("### Logs")
        st.write("**Train Logs (Algo)**")
        st.dataframe(pd.DataFrame(train_algo_log))
        st.write("**Train Logs (Manual)**")
        st.dataframe(pd.DataFrame(train_man_log))

        st.write("**Test Logs (Algo)**")
        st.dataframe(pd.DataFrame(test_algo_log))
        st.write("**Test Logs (Manual)**")
        st.dataframe(pd.DataFrame(test_man_log))


# ---------------------------------------------------------------------------
if __name__=="__main__":
    main()
