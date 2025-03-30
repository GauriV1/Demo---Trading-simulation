import streamlit as st
import pandas as pd
import datetime
from alpha_vantage.timeseries import TimeSeries
from prophet import Prophet
import matplotlib.pyplot as plt

def fetch_data_av(ticker, start_date, end_date):
    api_key = "YOUR_API_KEY"  # Replace with your actual Alpha Vantage API key
    ts = TimeSeries(key=api_key, output_format="pandas")
    data, meta_data = ts.get_daily(symbol=ticker, outputsize="full")

    # Clean and format
    data.rename(columns={
        "4. close": "Close"
    }, inplace=True)
    data.index = pd.to_datetime(data.index)
    data = data.sort_index()
    return data.loc[start_date:end_date][['Close']]

def train_forecast_model(df):
    df_prophet = df.reset_index().rename(columns={"index": "ds", "Close": "y"})
    model = Prophet(daily_seasonality=True)
    model.fit(df_prophet)
    return model

def make_forecast(model, periods=365):
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast

def plot_forecast(model, forecast):
    fig1 = model.plot(forecast)
    plt.title("Forecasted Stock Price")
    st.pyplot(fig1)

def main():
    st.title("Stock Price 12-Month Predictor")

    ticker = st.sidebar.text_input("Ticker", "AAPL")
    start_date = st.sidebar.date_input("Start Date", datetime.date(2010, 1, 1))
    end_date = st.sidebar.date_input("End Date", datetime.date.today())

    if st.sidebar.button("Run Forecast"):
        with st.spinner("Fetching data and training model..."):
            df = fetch_data_av(ticker, start_date, end_date)

            if df.empty:
                st.error("No data available for the selected range.")
                return

            st.write(f"Fetched {len(df)} data points.")
            st.line_chart(df)

            model = train_forecast_model(df)
            forecast = make_forecast(model, periods=365)

            st.subheader("12-Month Forecast")
            plot_forecast(model, forecast)

            # Show data for final day
            final_prediction = forecast[['ds', 'yhat']].tail(1)
            st.metric(label="Predicted Closing Price (1 year later)", value=f"${final_prediction['yhat'].values[0]:.2f}")

if __name__ == "__main__":
    main()
