import pandas as pd
# from fbprophet import Prophet
import yfinance as yf
import streamlit as st
import datetime
from datetime import date, timedelta
import plotly.graph_objects as go
from yahooquery import Screener


import pandas as pd
import matplotlib.pyplot as plt
import requests
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import warnings
warnings.filterwarnings("ignore")


st.set_page_config(layout="wide")
hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer{visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)
st.sidebar.header("FB PROPHET")

today = date.today()

d1 = today.strftime("%Y-%m-%d")
end_date = d1
d2 = date.today() - timedelta(days=730)
d2 = d2.strftime("%Y-%m-%d")
start_date = d2
# with open('style.css') as f:
#     st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


st.title("CRYPTOCURRENCY PREDICTION")

s = Screener()
data2 = s.get_screeners('all_cryptocurrencies_us', count=250)

# data is in the quotes key
dicts=data2['all_cryptocurrencies_us']['quotes']
symbols = [d['symbol'] for d in dicts]
# symbols
with st.form(key='my_form'):
  user_input=st.selectbox('Enter coin symbol: ', symbols, key=1)
  # user_input=st.text_input("Enter coin name: ", 'BTC-USD')
  coin_name=user_input.upper()
  # period=int(input('Enter number of days: '))
  period=st.slider(label='Enter number of days:', min_value=1, max_value=7, key=3)

  # period=st.number_input("Enter number of days", 10)
  submit_button = st.form_submit_button(label='Submit')
# if submit:
period=int(period)
data = yf.download(coin_name, 
                        start=start_date, 
                        end=end_date, 
                        progress=False)
data["Date"] = data.index
data = data[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
data.reset_index(drop=True, inplace=True)
  # print(data.head())
# df = data[["Date", "Close"]]
# df.columns = ["ds", "y"]
df=data
# Fetching data from the server
# url = "https://web-api.coinmarketcap.com/v1/cryptocurrency/ohlcv/historical"
# param = {"convert":"USD","slug":"bitcoin","time_end":"1601510400","time_start":"1367107200"}
# content = requests.get(url=url, params=param).json()
# df = pd.json_normalize(content['data']['quotes'])

# # Extracting and renaming the important variables
# df['Date']=pd.to_datetime(df['quote.USD.timestamp']).dt.tz_localize(None)
# df['Low'] = df['quote.USD.low']
# df['High'] = df['quote.USD.high']
# df['Open'] = df['quote.USD.open']
# df['Close'] = df['quote.USD.close']
# df['Volume'] = df['quote.USD.volume']

# # Drop original and redundant columns
# df=df.drop(columns=['time_open','time_close','time_high','time_low', 'quote.USD.low', 'quote.USD.high', 'quote.USD.open', 'quote.USD.close', 'quote.USD.volume', 'quote.USD.market_cap', 'quote.USD.timestamp'])

# Creating a new feature for better representing day-wise values
df['Mean'] = (df['Low'] + df['High'])/2

# Cleaning the data for any NaN or Null fields
df = df.dropna()



# Creating a copy for making small changes
dataset_for_prediction = df.copy()
dataset_for_prediction['Actual']=dataset_for_prediction['Mean'].shift()
dataset_for_prediction=dataset_for_prediction.dropna()

# date time typecast
dataset_for_prediction['Date'] =pd.to_datetime(dataset_for_prediction['Date'])
dataset_for_prediction.index= dataset_for_prediction['Date']



# normalizing the exogeneous variables
from sklearn.preprocessing import MinMaxScaler
sc_in = MinMaxScaler(feature_range=(0, 1))
scaled_input = sc_in.fit_transform(dataset_for_prediction[['Low', 'High', 'Open', 'Close', 'Volume', 'Mean']])
scaled_input = pd.DataFrame(scaled_input, index=dataset_for_prediction.index)
X=scaled_input
X.rename(columns={0:'Low', 1:'High', 2:'Open', 3:'Close', 4:'Volume', 5:'Mean'}, inplace=True)
print("Normalized X")
print(X.head())


# normalizing the time series
sc_out = MinMaxScaler(feature_range=(0, 1))
scaler_output = sc_out.fit_transform(dataset_for_prediction[['Actual']])
scaler_output =pd.DataFrame(scaler_output, index=dataset_for_prediction.index)
y=scaler_output
y.rename(columns={0:'BTC Price next day'}, inplace= True)
y.index=dataset_for_prediction.index
print("Normalized y")
print(y.head())


# train-test split (cannot shuffle in case of time series)
train_size=int(len(df) *0.9)
test_size = int(len(df)) - train_size
train_X, train_y = X[:train_size].dropna(), y[:train_size].dropna()
test_X, test_y = X[train_size:].dropna(), y[train_size:].dropna()


import statsmodels.api as sm
# Init the best SARIMAX model
# from statsmodels.tsa.arima_model import ARIMA
model= sm.tsa.arima.ARIMA(
    train_y,
    exog=train_X,
    order=(0,1,1)
)

# training the model
results = model.fit()

# get predictions
predictions = results.predict(start =train_size, end=train_size+test_size-2,exog=test_X)


# setting up for plots
act = pd.DataFrame(scaler_output.iloc[train_size:, 0])
predictions=pd.DataFrame(predictions)
predictions.reset_index(drop=True, inplace=True)
predictions.index=test_X.index
predictions['Actual'] = act['BTC Price next day']
predictions.rename(columns={0:'Pred', 'predicted_mean':'Pred'}, inplace=True)


# post-processing inverting normalization
testPredict = sc_out.inverse_transform(predictions[['Pred']])
testActual = sc_out.inverse_transform(predictions[['Actual']])

# prediction plots
figure = plt.Figure()
plt.figure(figsize=(20,10))
plt.plot(predictions.index, testActual, label='Pred', color='blue')
plt.plot(predictions.index, testPredict, label='Actual', color='red')
plt.legend()
plt.show()
st.write(figure, use_container_width=True, outliers=False)
# st.pyplot(figure, use_container_width=True)
# (figure, outliers=False)
# figure1 = go.Figure(data=[go.Candlestick(x=data["Date"],
#                                           open=data["Open"], 
#                                           high=data["High"],
#                                           low=data["Low"], 
#                                           close=data["Close"])])
# figure.update_layout(title = coin_name+" Price Analysis", 
#                       xaxis_rangeslider_visible=True)

# print RMSE
import numpy as np
from statsmodels.tools.eval_measures import rmse
print("RMSE:", np.long(rmse(testActual, testPredict)))
rmse= np.long(rmse(testActual, testPredict))
st.write("RMSE:",(rmse))