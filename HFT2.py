import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from alpha_vantage.timeseries import TimeSeries

def run_trading_simulation(
    ticker='AAPL',
    start_date='2000-01-01',
    end_date='2024-01-01',
    initial_capital=100000,
    risk_tolerance='medium',
    alpha_vantage_api_key='YOUR_ALPHA_VANTAGE_API_KEY'
):
    """
    Fetch daily data from Alpha Vantage and run a multi-strategy backtest,
    comparing the automated strategy profit vs. a naive manual approach.
    Displays charts and logs in a Streamlit UI.
    """

    # ------------------------------------------------------------
    # 1. Data Retrieval from Alpha Vantage
    # ------------------------------------------------------------
    st.write(f"**Fetching data for {ticker} from Alpha Vantage...**")
    try:
        ts = TimeSeries(key=alpha_vantage_api_key, output_format='pandas')
        data_av, meta_data = ts.get_daily(symbol=ticker, outputsize='full')
    except Exception as e:
        st.error(f"Error fetching data from Alpha Vantage: {e}")
        return

    # data_av has columns like '1. open', '2. high', '3. low', '4. close', '5. volume'
    if data_av.empty:
        st.warning("No data returned. Check ticker or Alpha Vantage usage limits.")
        return

    # Often returns newest date first, so sort ascending
    data_av.sort_index(inplace=True)

    # Rename columns to something standard
    data_av.rename(columns={
        '1. open': 'Open',
        '2. high': 'High',
        '3. low': 'Low',
        '4. close': 'Close',
        '5. volume': 'Volume'
    }, inplace=True)

    # Filter date range
    data_av = data_av.loc[start_date:end_date].copy()
    if data_av.empty:
        st.warning("No data after applying date range. Exiting.")
        return

    # Use 'Close' as our 'Price'
    data_av['Price'] = data_av['Close']

    # Drop rows with NaN Price
    data_av.dropna(subset=['Price'], inplace=True)
    if data_av.empty:
        st.warning("All Price data is NaN. Exiting.")
        return

    # Create a working DataFrame
    data = data_av

    # ------------------------------------------------------------
    # 2. Ensure a 'Date' column for plotting/logs
    # ------------------------------------------------------------
    data['Date'] = data.index
    data.reset_index(drop=True, inplace=True)

    # ------------------------------------------------------------
    # 3. Indicator Calculations
    # ------------------------------------------------------------
    data['SMA_short'] = data['Price'].rolling(window=50).mean()
    data['SMA_long'] = data['Price'].rolling(window=200).mean()

    def compute_rsi(series, period=14):
        delta = series.diff().dropna()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period).mean()
        rs = gain / loss
        return 100 - 100/(1+rs)

    data['RSI'] = compute_rsi(data['Price'])
    data['Volatility'] = data['Price'].rolling(window=20).std()

    # Drop rows with NaN in indicators
    data.dropna(inplace=True)
    if len(data) < 10:
        st.warning("Not enough data after dropping NaNs. Exiting.")
        return

    # ------------------------------------------------------------
    # 4. Strategy Definitions
    # ------------------------------------------------------------
    def strategy_sma_crossover(row, position, risk_tol):
        """
        SMA Short vs. SMA Long Crossover strategy.
        """
        if risk_tol == 'low':
            threshold = 0
        elif risk_tol == 'medium':
            threshold = 1
        else:
            threshold = 2

        sma_short_val = float(row['SMA_short'])
        sma_long_val  = float(row['SMA_long'])
        if np.isnan(sma_short_val) or np.isnan(sma_long_val):
            return 0

        if (sma_short_val > sma_long_val + threshold) and (position == 0):
            return 1
        elif (sma_short_val < sma_long_val - threshold) and (position == 1):
            return -1
        return 0

    def strategy_rsi(row, position, risk_tol):
        """
        Buys if RSI < ~30, sells if RSI > ~70, with risk-based thresholds.
        """
        if risk_tol == 'low':
            buy_thresh, sell_thresh = 25, 75
        elif risk_tol == 'medium':
            buy_thresh, sell_thresh = 30, 70
        else:
            buy_thresh, sell_thresh = 35, 65

        rsi_val = float(row['RSI'])
        if np.isnan(rsi_val):
            return 0

        if (rsi_val < buy_thresh) and (position == 0):
            return 1
        elif (rsi_val > sell_thresh) and (position == 1):
            return -1
        return 0

    def strategy_buy_and_hold(row, position, risk_tol):
        """
        Buy on the very first loop iteration, then hold.
        """
        if position == 0 and row.name == 0:
            return 1
        return 0

    # ------------------------------------------------------------
    # 5. Automatic Strategy Switching
    # ------------------------------------------------------------
    def select_strategy(row, current_strategy, current_position, df):
        """
        Decide which strategy to use each day, possibly switching 
        from the current strategy to another based on volatility or trend.
        """
        sma_short_val = float(row['SMA_short'])
        sma_long_val  = float(row['SMA_long'])
        current_vol   = float(row['Volatility'])

        if any(np.isnan([sma_short_val, sma_long_val, current_vol])):
            return current_strategy, "No change"

        high_vol_thresh = df['Volatility'].quantile(0.75)
        trend = 'bullish' if (sma_short_val > sma_long_val) else 'bearish_or_sideways'

        if current_vol >= high_vol_thresh:
            return strategy_rsi, "RSI"
        else:
            if trend == 'bullish':
                return strategy_sma_crossover, "SMA Crossover"
            else:
                return strategy_buy_and_hold, "Buy & Hold"

    # ------------------------------------------------------------
    # 6. Simulation Setup
    # ------------------------------------------------------------
    capital = initial_capital
    position = 0
    shares_held = 0

    current_strategy = strategy_buy_and_hold
    current_strategy_name = 'Buy & Hold'

    trades_log = []
    portfolio_values = []

    # Define a naive manual approach for comparison
    manual_shares_held = 0
    manual_position = 0
    manual_portfolio_values = []
    manual_capital = initial_capital

    # ------------------------------------------------------------
    # 7. Simulation Loop
    # ------------------------------------------------------------
    for i in range(len(data)):
        row = data.iloc[i]

        # Possibly switch strategy
        strategy_func, strategy_name = select_strategy(row, current_strategy, position, data)
        if strategy_name != current_strategy_name:
            trades_log.append({
                'Date': row['Date'],
                'Event': 'Strategy Switch',
                'From': current_strategy_name,
                'To': strategy_name,
                'Reason': 'Vol/Trend shift'
            })
            current_strategy = strategy_func
            current_strategy_name = strategy_name

        price = float(row['Price']) if not pd.isna(row['Price']) else 0
        if price <= 0:
            portfolio_values.append(capital + shares_held * 0)
            manual_portfolio_values.append(manual_capital + manual_shares_held * 0)
            continue

        # Execute the chosen strategy
        action = current_strategy(row, position, risk_tolerance)

        if action == 1 and position == 0:
            # BUY
            shares_to_buy = int(capital // price)
            capital -= (shares_to_buy * price)
            shares_held += shares_to_buy
            position = 1
            trades_log.append({
                'Date': row['Date'],
                'Event': 'BUY',
                'Price': price,
                'Shares': shares_to_buy,
                'Strategy': current_strategy_name
            })

        elif action == -1 and position == 1:
            # SELL
            proceeds = shares_held * price
            capital += proceeds
            trades_log.append({
                'Date': row['Date'],
                'Event': 'SELL',
                'Price': price,
                'Shares': shares_held,
                'Strategy': current_strategy_name
            })
            shares_held = 0
            position = 0

        # Track daily portfolio value
        daily_value = capital + shares_held * price
        portfolio_values.append(daily_value)

        # Manual baseline
        if i == 0:
            # Buy on first day
            manual_shares_held = int(manual_capital // price)
            manual_capital -= manual_shares_held * price
            manual_position = 1
        elif i == len(data) - 1:
            # Sell on last day
            if manual_position == 1:
                manual_capital += manual_shares_held * price
                manual_shares_held = 0
                manual_position = 0

        manual_portfolio_values.append(manual_capital + manual_shares_held * price)

    # Close any open position at the end
    if position == 1 and shares_held > 0:
        last_price = float(data.iloc[-1]['Price'])
        capital += shares_held * last_price
        shares_held = 0
        position = 0

    final_portfolio_value = capital
    final_manual_value = manual_capital

    # **Calculate Profit** for both strategies
    strategy_profit = final_portfolio_value - initial_capital
    manual_profit = final_manual_value - initial_capital

    trades_df = pd.DataFrame(trades_log)

    # ------------------------------------------------------------
    # 8. Visualization (Streamlit)
    # ------------------------------------------------------------
    data['Strategy_Portfolio'] = portfolio_values
    data['Manual_Portfolio'] = manual_portfolio_values

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # Plot 1: Stock Price
    ax1.plot(data['Date'], data['Price'], label=f"{ticker} Price", color='blue')
    ax1.set_title(f"{ticker} Price Over Time (Alpha Vantage Daily)")
    ax1.set_ylabel("Price")
    ax1.legend(loc='upper left')

    # Plot 2: Portfolio Value
    ax2.plot(data['Date'], data['Strategy_Portfolio'], label='Strategy', color='green')
    ax2.plot(data['Date'], data['Manual_Portfolio'], label='Manual', color='red', alpha=0.7)
    ax2.set_title("Portfolio Value Over Time")
    ax2.set_ylabel("Value")
    ax2.legend(loc='upper left')

    # Plot 3: RSI
    ax3.plot(data['Date'], data['RSI'], label='RSI', color='purple')
    ax3.axhline(30, color='gray', linestyle='--')
    ax3.axhline(70, color='gray', linestyle='--')
    ax3.set_title("RSI Over Time")
    ax3.set_ylabel("RSI")
    ax3.legend(loc='upper left')

    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))

    plt.tight_layout()
    st.pyplot(fig)  # Renders Matplotlib figure in Streamlit

    # ------------------------------------------------------------
    # 9. Results & Trade Log
    # ------------------------------------------------------------
    days_in_future = 365
    portfolio_returns = pd.Series(portfolio_values).pct_change().dropna()
    avg_daily_return = portfolio_returns.mean() if not portfolio_returns.empty else 0
    future_value = final_portfolio_value * ((1 + avg_daily_return) ** days_in_future)

    st.write("============================================================")
    st.write(f"**Initial Capital:** `${initial_capital:,.2f}`")
    st.write(f"**Final Strategy Portfolio Value:**  `${final_portfolio_value:,.2f}`")
    st.write(f"**Strategy Profit:**                `${strategy_profit:,.2f}`")
    st.write("")
    st.write(f"**Final Manual Portfolio Value:**   `${final_manual_value:,.2f}`")
    st.write(f"**Manual Profit:**                  `${manual_profit:,.2f}`")
    st.write("")
    st.write(f"**Naive 1-year Projection:**        `${future_value:,.2f}`")
    st.write("============================================================")

    st.write("**Trade Log:**")
    st.dataframe(trades_df)

# -----------------------------------------------------------------
# Streamlit Entry Point
# -----------------------------------------------------------------
if __name__ == "__main__":
    st.title("Alpha Vantage Trading Simulation")
    st.write("Fetch daily data from Alpha Vantage and simulate a multi-strategy backtest.")
    
    # Let the user specify ticker, date range, alpha vantage key, etc.
    ticker_input = st.text_input("Enter Ticker (e.g., 'AAPL'):", "AAPL")
    start_date_input = st.date_input("Start Date", value=pd.to_datetime("2000-01-01"))
    end_date_input = st.date_input("End Date", value=pd.to_datetime("2024-01-01"))
    
    alpha_key_input = st.text_input("Alpha Vantage API Key", "YOUR_ALPHA_VANTAGE_API_KEY", type="password")
    risk_tol_input = st.selectbox("Risk Tolerance", ["low", "medium", "high"])
    
    # **User input for initial capital**:
    capital_input = st.number_input("Initial Capital", value=100000, min_value=0, step=1000)
    
    if st.button("Run Simulation"):
        run_trading_simulation(
            ticker=ticker_input,
            start_date=str(start_date_input),
            end_date=str(end_date_input),
            initial_capital=capital_input,
            risk_tolerance=risk_tol_input,
            alpha_vantage_api_key=alpha_key_input
        )
