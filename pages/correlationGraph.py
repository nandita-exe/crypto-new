from datetime import datetime, timedelta

import pandas as pd
import pandas_datareader as pdr
import plotly.graph_objects as go

import pandas as pd
import yfinance as yf
import streamlit as st
import datetime
from datetime import date, timedelta
import plotly.graph_objects as go
from yahooquery import Screener

st.set_page_config(layout="wide")
hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer{visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)
st.title("CRYPTOCURRENCY CORRELATION GRAPH")

# df = pd.read_csv('BTC-USD.csv')
# df = pd.read_csv(data)
s = Screener()
data2 = s.get_screeners('all_cryptocurrencies_us', count=250)

# data is in the quotes key
dicts=data2['all_cryptocurrencies_us']['quotes']
symbols = [d['symbol'] for d in dicts]
with st.form(key='my_form'):
    col1_selection = st.selectbox('Coin 1', symbols[:30], list(symbols[:30]).index('BTC-USD'))
    col2_selection = st.selectbox('Coin 2', symbols[:30], list(symbols[:30]).index('ETH-USD'))
    col3_selection = st.selectbox('Coin 3', symbols[:30], list(symbols[:30]).index('DOGE-USD'))
    submit_button = st.form_submit_button(label='Submit')
# CRYPTOS = ['BTC', 'ETH', 'LTC', 'XRP']
CRYPTOS = [col1_selection, col2_selection, col3_selection]
CURRENCY = 'USD'

def getData(cryptocurrency):
    now = date.today()
    current_date = now.strftime("%Y-%m-%d")
    last_year_date = (now - timedelta(days=365)).strftime("%Y-%m-%d")

    start = pd.to_datetime(last_year_date)
    end = pd.to_datetime(current_date)

    data = pdr.get_data_yahoo(f'{cryptocurrency}', start, end)

    return data

crypto_data = {crypto:getData(crypto) for crypto in CRYPTOS}


fig = go.Figure()

    # Scatter
for idx, name in enumerate(crypto_data):
        fig = fig.add_trace(
            go.Scatter(
                x = crypto_data[name].index,
                y = crypto_data[name].Close,
                name = name,
            )
        )

fig.update_layout(
        title = 'The Correlation between Different Cryptocurrencies',
        xaxis_title = 'Date',
        yaxis_title = f'Closing price ({CURRENCY})',
        legend_title = 'Cryptocurrencies'
    )
fig.update_yaxes(type='log', tickprefix='£')

# fig.show()
st.plotly_chart(fig, use_container_width=True)