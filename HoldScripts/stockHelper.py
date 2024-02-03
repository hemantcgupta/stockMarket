# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 21:53:37 2024

@author: Hemant
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def yfDownload(tickerName, period):
    df = yf.Ticker(tickerName).history(period=period).reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Day'] = df['Date'].dt.day_name()
    df['Date'] = df['Date'].dt.date
    df[['Open', 'High', 'Low', 'Close']] = df[['Open', 'High', 'Low', 'Close']].round(2)
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Day']]
    return df

def formulaPercentage(df):
    df['P/L'] = ((1-df['Open']/df['Close'])*100).round(2)    
    df['maxHigh'] = ((1-df['Open']/df['High'])*100).round(2)
    df['maxLow'] = ((1-df['Open']/df['Low'])*100).round(2)
    df['closeTolerance'] = df.apply(lambda row: row['P/L'] - row['maxHigh'] if row['P/L'] > 0 else row['P/L'] - row['maxLow'] if row['P/L'] < 0 else 0, axis=1)
    df['priceBand'] = (((df['High'] - df['Low'])/df['Open'])*100).round(2)
    return df

def plot_line_chart(df):
    plt.figure(figsize=(10, 6))
    plt.plot(df['Date'], df['maxHigh'], label='Max High', marker='o', color='green')
    plt.plot(df['Date'], df['maxLow'], label='Max Low', marker='o', color='red')
    for i, row in df.iterrows():
        plt.text(row['Date'], row['maxHigh'], f'{row["maxHigh"]:.2f}', ha='center', va='bottom', color='black')
        plt.text(row['Date'], row['maxLow'], f'{row["maxLow"]:.2f}', ha='center', va='top', color='black')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Sample Line Chart with Positive and Negative Y-Axes')
    plt.xticks(rotation=90)
    plt.legend()
    plt.show()
     
def buy_sell_probability_in_profit_and_loss(df):
    BuyInProfit = len(df[df['maxLow'] == 0])
    SellInLoss = len(df[df['maxHigh'] == 0])
    BuyInLoss = len(df[(df['maxLow'] != 0) & (df['maxHigh'] != 0) & (df['Open'] > df['Close'])])
    SellInProfit = len(df[(df['maxLow'] != 0) & (df['maxHigh'] != 0) & (df['Open'] < df['Close'])])
    Total = BuyInProfit+SellInLoss+BuyInLoss+SellInProfit
    ProbabilityOfCloseTolerance = df['closeTolerance'].astype(int).value_counts()
    ProbabilityOfCloseTolerance = round((ProbabilityOfCloseTolerance/ProbabilityOfCloseTolerance.sum())*100, 2).to_dict()
    ProbabilityOfProfitLoss = df['P/L'].astype(int).value_counts()
    ProbabilityOfProfitLoss = round((ProbabilityOfProfitLoss/ProbabilityOfProfitLoss.sum())*100, 2).to_dict()  
    ProbabilityOfProfitLossTomorrow = {'Profit': round(sum(value for key, value in ProbabilityOfProfitLoss.items() if key >= 0), 2), 'Loss': round(sum(value for key, value in ProbabilityOfProfitLoss.items() if key < 0), 2)}
    ProbabilityOfProfitMT2Percent= round(sum(value for key, value in ProbabilityOfProfitLoss.items() if key >= 2), 2)
    ProbabilityOfLoss1ratio3Percent= round(sum(value for key, value in ProbabilityOfProfitLoss.items() if key <= -1), 2)
    ProbabilityOfmaxHigh = df['maxHigh'].astype(int).value_counts()
    ProbabilityOfmaxHigh = round((ProbabilityOfmaxHigh/ProbabilityOfmaxHigh.sum())*100, 2).to_dict()  
    ProbabilityOfmaxLow = df['maxLow'].astype(int).value_counts()
    ProbabilityOfmaxLow = round((ProbabilityOfmaxLow/ProbabilityOfmaxLow.sum())*100, 2).to_dict()  
    ProbabilityOfpriceBand = df['priceBand'].astype(int).value_counts()
    ProbabilityOfpriceBand = round((ProbabilityOfpriceBand/ProbabilityOfpriceBand.sum())*100, 2).to_dict() 
    buysellProbability = {
        'BuyInProfit MP::HP::MP::HP': round((BuyInProfit/Total)*100, 2),
        'SellInLoss MP::MP::LP::LP': round((SellInLoss/Total)*100, 2),
        'BuyInLoss MP::HP::LP::HP': round((BuyInLoss/Total)*100, 2),
        'SellInProfit MP::HP::LP::LP': round((SellInProfit/Total)*100, 2),
        'ProbabilityOfCloseTolerance': ProbabilityOfCloseTolerance,
        'ProbabilityOfProfitLoss': ProbabilityOfProfitLoss,
        'ProbabilityOfProfitTomorrow': ProbabilityOfProfitLossTomorrow.get('Profit'),
        'ProbabilityOfLossTomorrow': ProbabilityOfProfitLossTomorrow.get('Loss'),
        'ProbabilityOfProfitMT2Percent': ProbabilityOfProfitMT2Percent,
        'ProbabilityOfLoss1ratio3Percent': ProbabilityOfLoss1ratio3Percent,
        'ProbabilityOfmaxHigh': ProbabilityOfmaxHigh,
        'ProbabilityOfmaxLow': ProbabilityOfmaxLow,
        'ProbabilityOfpriceBand': ProbabilityOfpriceBand
        }
    return buysellProbability