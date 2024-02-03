# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 14:31:36 2024

@author: Hemant
"""

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def calculate_moving_average(data, window_size):
    return data['Close'].rolling(window=window_size).mean()

def plot_stock_with_moving_average(symbol, start_date, end_date, window_size=44):
    # Download stock data from Yahoo Finance
    stock_data = yf.download(symbol, start=start_date, end=end_date)

    # Calculate the 44-day moving average
    stock_data['44_MA'] = calculate_moving_average(stock_data, window_size)

    # Plotting the stock price and the moving average
    plt.figure(figsize=(10, 6))
    plt.plot(stock_data.index, stock_data['Close'], label=f'{symbol} Close Price')
    plt.plot(stock_data.index, stock_data['44_MA'], label=f'{symbol} 44-Day Moving Average', color='orange')
    plt.title(f'{symbol} Stock Price with 44-Day Moving Average')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# Example usage
symbol = 'RVNL.NS'  # Replace with the desired stock symbol
start_date = '2023-01-01'
end_date = '2024-01-01'
plot_stock_with_moving_average(symbol, start_date, end_date)


# =============================================================================
# 
# =============================================================================
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def calculate_moving_average(data, window_size):
    return data['Close'].rolling(window=window_size).mean()

def plot_stock_with_moving_average(symbol, days_back, window_size=44):
    # Calculate the start and end dates based on the number of days back
    end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
    start_date = (pd.Timestamp.today() - pd.DateOffset(days=days_back)).strftime('%Y-%m-%d')

    # Download 5-minute interval stock data from Yahoo Finance
    stock_data = yf.download(symbol, start=start_date, end=end_date, interval='5m')

    # Calculate the 44-period moving average
    stock_data['44_MA'] = calculate_moving_average(stock_data, window_size)

    # Plotting the stock price and the moving average
    plt.figure(figsize=(10, 6))
    plt.plot(stock_data.index, stock_data['Close'], label=f'{symbol} Close Price')
    plt.plot(stock_data.index, stock_data['44_MA'], label=f'{symbol} 44-Period Moving Average', color='orange')
    plt.title(f'{symbol} 5-Minute Interval Stock Price with 44-Period Moving Average')
    plt.xlabel('Date and Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# Example usage
symbol = 'NAM-INDIA.NS'  # Replace with the desired stock symbol
days_back = 1
plot_stock_with_moving_average(symbol, days_back)
