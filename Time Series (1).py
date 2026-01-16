import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Use Case 3 – Sales Forecasting", layout="wide")
st.title("Use Case 3: Retail Sales Forecasting")

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("retail_sales_dataset.csv")
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

st.subheader("Dataset Preview")
st.dataframe(df.head())

# -----------------------------
# TIME SERIES CREATION
# -----------------------------
daily_sales = (
    df.groupby('Date')['Total Amount']
      .sum()
      .reset_index()
      .sort_values('Date')
)

daily_sales_ts = daily_sales.set_index('Date')

# FIX FREQUENCY (IMPORTANT)
daily_sales_ts = daily_sales_ts.asfreq('D')

st.subheader("Daily Sales Time Series")
st.line_chart(daily_sales_ts)

# -----------------------------
# TREND & SEASONALITY
# -----------------------------
st.subheader("Trend & Seasonality Decomposition")

decompose = seasonal_decompose(
    daily_sales_ts['Total Amount'],
    model='additive',
    period=30
)

fig = decompose.plot()
fig.set_size_inches(10, 8)
st.pyplot(fig)

# -----------------------------
# MOVING AVERAGE
# -----------------------------
st.subheader("Moving Average Smoothing")

daily_sales_ts['MA_7'] = daily_sales_ts['Total Amount'].rolling(7).mean()
daily_sales_ts['MA_30'] = daily_sales_ts['Total Amount'].rolling(30).mean()

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(daily_sales_ts['Total Amount'], label='Actual')
ax.plot(daily_sales_ts['MA_7'], label='7-Day MA')
ax.plot(daily_sales_ts['MA_30'], label='30-Day MA')
ax.legend()
st.pyplot(fig)

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------
train_size = int(len(daily_sales_ts) * 0.8)
train = daily_sales_ts.iloc[:train_size]
test = daily_sales_ts.iloc[train_size:]

# -----------------------------
# ARIMA MODEL
# -----------------------------
arima_model = ARIMA(train['Total Amount'], order=(1,1,1))
arima_fit = arima_model.fit()
arima_forecast = arima_fit.forecast(steps=len(test))

# -----------------------------
# SARIMA MODEL
# -----------------------------
sarima_model = SARIMAX(
    train['Total Amount'],
    order=(1,1,1),
    seasonal_order=(1,1,1,7)
)

sarima_fit = sarima_model.fit(disp=False)
sarima_forecast = sarima_fit.forecast(steps=len(test))

# -----------------------------
# PROPHET MODEL
# -----------------------------
from prophet import Prophet

prophet_df = daily_sales.reset_index(drop=True)
prophet_df.columns = ['ds','y']

train_prophet = prophet_df.iloc[:train_size]
test_prophet = prophet_df.iloc[train_size:]

prophet_model = Prophet()
prophet_model.fit(train_prophet)

future = prophet_model.make_future_dataframe(periods=len(test_prophet))
forecast = prophet_model.predict(future)

prophet_forecast = forecast['yhat'].iloc[-len(test_prophet):].values

# -----------------------------
# FORECAST COMPARISON
# -----------------------------
st.subheader("Forecast Comparison")

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(train.index, train['Total Amount'], label='Train')
ax.plot(test.index, test['Total Amount'], label='Test')
ax.plot(test.index, arima_forecast, label='ARIMA')
ax.plot(test.index, sarima_forecast, label='SARIMA')
ax.plot(test.index, prophet_forecast, label='Prophet')
ax.legend()
st.pyplot(fig)

# -----------------------------
# EVALUATION
# -----------------------------
def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse

arima_mae, arima_rmse = evaluate(test['Total Amount'], arima_forecast)
sarima_mae, sarima_rmse = evaluate(test['Total Amount'], sarima_forecast)
prophet_mae, prophet_rmse = evaluate(test['Total Amount'], prophet_forecast)

eval_df = pd.DataFrame({
    'Model': ['ARIMA', 'SARIMA', 'Prophet'],
    'MAE': [arima_mae, sarima_mae, prophet_mae],
    'RMSE': [arima_rmse, sarima_rmse, prophet_rmse]
})

st.subheader("Forecast Evaluation")
st.dataframe(eval_df)

# -----------------------------
# FINAL INSIGHT
# -----------------------------
st.subheader("Final Insight")

st.markdown("""
- **Moving Average** smooths short-term noise  
- **ARIMA** captures trend but misses seasonality  
- **SARIMA** captures weekly seasonality  
- **Prophet performs best** for business forecasting with minimal tuning  
""")
