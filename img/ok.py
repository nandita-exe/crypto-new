import streamlit as st
st.markdown("# ok ❄️")
st.sidebar.markdown("#  ok ❄️")





from keras.models import Sequential
from keras.layers import Dense
import numpy as np

def gen_main(data, seed=123):
	#local variables
	NUM_EPOCHS = 1000
	BATCH_SIZE = 15

	#seed
	np.random.seed(seed)

	#format data
	x_data = range(len(data), 0, -1)
	train = (np.array(x_data[0:-1]), np.array(data[0:-1]))

	#creating the model
	model = Sequential() #initialize model
	model.add(Dense(5, input_dim=1, activation='relu')) #add hidden layer
	model.add(Dense(5, input_dim=1, activation='relu')) #add another hidden layer
	model.add(Dense(5, input_dim=1, activation='relu')) #add another hidden layer
	model.add(Dense(5, input_dim=1, activation='relu')) #add another hidden layer
	model.add(Dense(5, input_dim=1, activation='relu')) #add another hidden layer
	model.add(Dense(5, input_dim=1, activation='relu')) #add another hidden layer
	model.add(Dense(5, input_dim=1, activation='relu')) #add another hidden layer
	model.add(Dense(5, input_dim=1, activation='relu')) #add another hidden layer
	model.add(Dense(1, activation='sigmoid')) #add output layer
	model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]) #compile model

	model_fitted = model.fit(train[0], train[1], epochs=NUM_EPOCHS, verbose=0,
		batch_size=BATCH_SIZE, callbacks=[], initial_epoch=0) #train model

	predictions = model.predict(np.array(range(len(data), 0, -1))) #make predictions with model

	return predictions


#dependencies
import requests
import matplotlib.pyplot as plot
import json
import numpy as np
import sys
import os
import argparse

# from genData import gen_main

#crypto currency price prediction program
#made by: Lachi Balabanski
from yahooquery import Screener
import yfinance as yf
from datetime import date, timedelta
today = date.today()

d1 = today.strftime("%Y-%m-%d")
end_date = d1
d2 = date.today() - timedelta(days=730)
d2 = d2.strftime("%Y-%m-%d")
start_date = d2
#wrappers for main
def getUserCoin():
    s = Screener()
    data2 = s.get_screeners('all_cryptocurrencies_us', count=250)

    # data is in the quotes key
    dicts=data2['all_cryptocurrencies_us']['quotes']
    symbols = [d['symbol'] for d in dicts]
    # symbols
    with st.form(key='my_form'):
        coin=st.selectbox('Enter coin symbol: ', symbols, key=1)
        # user_input=st.text_input("Enter coin name: ", 'BTC-USD')
        coin_name=coin.upper()
        # period=int(input('Enter number of days: '))
        period=st.slider(label='Enter number of days:', min_value=0, max_value=365, key=3)

        # period=st.number_input("Enter number of days", 10)
        submit_button = st.form_submit_button(label='Submit')

    period=int(period)
    data = yf.download(coin_name, 
                        start=start_date, 
                        end=end_date, 
                        progress=False)
    # coin = input('Please enter cryptocurrency abbreviation (btc, eth, etc.): ')

    # while True:

    #     temp = requests.get('https://coinbin.org/' + str(coin))

    #     if temp.status_code == 200:
    #         return coin.lower()

    #     else:
    #         print('Invalid Coin')
    #         coin = input('Please enter cryptocurrency abbreviation (btc, eth, etc.): ')

def showGraph(data, coinName, title_mod='History'):
    xList = range(len(data), 0, -1)
    
    yList = data

    plot.plot(xList, yList)
    plot.title('Coin ' + title_mod + ' of ' + str(coinName).upper())
    plot.xlabel('Time')
    plot.ylabel('Price (USD)')
    plot.show()

def getData(coin, time):
    content = requests.get('https://coinbin.org/' + str(coin) + '/history').content
    data = json.loads(content)['history']
    data = data[:time]
    list_data = list(map(lambda x: x['value'], data))
    return list_data

def generate_data(dataArray, seed):
    return gen_main(dataArray, seed)

#main
def main(): 
    # This program has data points from October 22, 2017 and before.
    # All predictions are from data from that point and before.
    # The program cannot factor in data points from 10/22/17 to present day

    #parse arguments
    p = argparse.ArgumentParser()
    p.add_argument('-s', '--seed', help="seed to be used", type=int, default=123)
    args = p.parse_args()
    seed = args.seed

    
    coin = getUserCoin()

    try:
        time = int(input('From how long ago would you like to view data from?(days): '))

    except ValueError:
        print("Error while parsing previous input, will revert to default=31")
        time = 31
    
    print('\nLoading Data. . .', end='   ')
    data = getData(coin, time)
    print('[DONE]')

    print('\nGenerating Data. . .', end='   ')
    gen_data = generate_data(data, seed=seed)
    print('[DONE]')
    
    showGraph(data, coin)

    showGraph(gen_data, coin, title_mod='Prediction')


if __name__ == "__main__":
    main()