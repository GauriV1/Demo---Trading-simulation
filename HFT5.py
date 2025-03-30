import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ta import trend, momentum, volatility
from datetime import datetime

# Suppress SettingWithCopyWarning
pd.options.mode.chained_assignment = None  # default='warn'

# 1. Data Retrieval
def fetch_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date, interval='1d')
    data.dropna(inplace=True)
    return data

# 2. Data Preparation and Technical Indicators
def calculate_indicators(data):
    # Simple Moving Averages
    data['SMA50'] = trend.sma_indicator(data['Close'], window=50)
    data['SMA200'] = trend.sma_indicator(data['Close'], window=200)
    
    # Exponential Moving Averages
    data['EMA50'] = trend.ema_indicator(data['Close'], window=50)
    data['EMA200'] = trend.ema_indicator(data['Close'], window=200)
    
    # Relative Strength Index
    data['RSI'] = momentum.rsi(data['Close'], window=14)
    
    # Moving Average Convergence Divergence
    macd = trend.MACD(data['Close'])
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()
    data['MACD_Diff'] = macd.macd_diff()
    
    # Bollinger Bands
    bollinger = volatility.BollingerBands(data['Close'], window=20, window_dev=2)
    data['Bollinger_High'] = bollinger.bollinger_hband()
    data['Bollinger_Low'] = bollinger.bollinger_lband()
    
    data.dropna(inplace=True)
    return data

# 3. Signal Functions
def generate_signals(data):
    data['Signal'] = 0  # 1 for Buy, -1 for Sell, 0 for Hold
    
    # Moving Average Crossover Strategy with RSI and MACD Confirmation
    for i in range(1, len(data)):
        # Previous and current SMA50 and SMA200
        prev_sma50 = data['SMA50'].iloc[i-1]
        prev_sma200 = data['SMA200'].iloc[i-1]
        curr_sma50 = data['SMA50'].iloc[i]
        curr_sma200 = data['SMA200'].iloc[i]
        
        # Previous and current MACD
        prev_macd = data['MACD'].iloc[i-1]
        prev_macd_signal = data['MACD_Signal'].iloc[i-1]
        curr_macd = data['MACD'].iloc[i]
        curr_macd_signal = data['MACD_Signal'].iloc[i]
        
        # Buy Signal
        if (prev_sma50 < prev_sma200) and (curr_sma50 > curr_sma200):
            if data['RSI'].iloc[i] < 30:  # RSI Confirmation
                if (prev_macd < prev_macd_signal) and (curr_macd > curr_macd_signal):  # MACD Confirmation
                    data['Signal'].iloc[i] = 1  # Buy
        
        # Sell Signal
        elif (prev_sma50 > prev_sma200) and (curr_sma50 < curr_sma200):
            if data['RSI'].iloc[i] > 70:  # RSI Confirmation
                if (prev_macd > prev_macd_signal) and (curr_macd < curr_macd_signal):  # MACD Confirmation
                    data['Signal'].iloc[i] = -1  # Sell
                    
    return data

# 4. Simulation (Backtesting)
def simulate_trades(data, initial_capital=100000):
    capital = initial_capital
    position = 0  # Number of shares held
    portfolio = []
    buy_dates = []
    sell_dates = []
    portfolio_values = []
    
    for i in range(len(data)):
        signal = data['Signal'].iloc[i]
        price = data['Close'].iloc[i]
        
        if signal == 1 and capital > 0:
            # Buy as many shares as possible
            position = capital // price
            capital -= position * price
            buy_dates.append(data.index[i])
            print(f"Buy: {data.index[i].date()} | Price: {price:.2f} | Shares: {position}")
        
        elif signal == -1 and position > 0:
            # Sell all shares
            capital += position * price
            position = 0
            sell_dates.append(data.index[i])
            print(f"Sell: {data.index[i].date()} | Price: {price:.2f}")
        
        # Calculate current portfolio value
        current_value = capital + position * price
        portfolio_values.append(current_value)
        portfolio.append(current_value)
    
    data['Portfolio_Value'] = portfolio_values
    return data, buy_dates, sell_dates, initial_capital, portfolio_values

# 5. Visualization
def plot_results(data, buy_dates, sell_dates, initial_capital, portfolio_values):
    plt.figure(figsize=(14, 7))
    
    # Plot Close Price
    plt.subplot(2, 1, 1)
    plt.plot(data['Close'], label='AAPL Close Price', alpha=0.5)
    plt.scatter(data.loc[buy_dates].index, data.loc[buy_dates]['Close'], marker='^', color='green', label='Buy Signal', s=100)
    plt.scatter(data.loc[sell_dates].index, data.loc[sell_dates]['Close'], marker='v', color='red', label='Sell Signal', s=100)
    plt.title('AAPL Close Price with Buy/Sell Signals')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    
    # Plot Portfolio Value
    plt.subplot(2, 1, 2)
    plt.plot(data['Portfolio_Value'], label='Portfolio Value', color='blue')
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# 6. Performance Metrics
def calculate_performance(data, initial_capital):
    final_portfolio = data['Portfolio_Value'].iloc[-1]
    total_return = (final_portfolio - initial_capital) / initial_capital * 100
    
    # Calculate CAGR
    start_date = data.index[0]
    end_date = data.index[-1]
    days = (end_date - start_date).days
    years = days / 365.25
    cagr = ((final_portfolio / initial_capital) ** (1 / years) - 1) * 100
    
    print(f"Initial Capital: ${initial_capital:.2f}")
    print(f"Final Portfolio Value: ${final_portfolio:.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"CAGR: {cagr:.2f}%")

# Main Execution
def main():
    ticker = 'AAPL'
    start_date = '2000-01-01'
    end_date = '2024-12-31'
    initial_capital = 100000  # Starting with $100,000
    
    print("Fetching data...")
    data = fetch_data(ticker, start_date, end_date)
    
    print("Calculating technical indicators...")
    data = calculate_indicators(data)
    
    print("Generating signals...")
    data = generate_signals(data)
    
    print("Simulating trades...")
    data, buy_dates, sell_dates, initial_capital, portfolio_values = simulate_trades(data, initial_capital)
    
    print("Plotting results...")
    plot_results(data, buy_dates, sell_dates, initial_capital, portfolio_values)
    
    print("Calculating performance metrics...")
    calculate_performance(data, initial_capital)

if __name__ == "__main__":
    main()
