# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 19:48:18 2024

@author: Hemant
"""
import yfinance as yf
import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime

def yfDownload(tickerName, period, interval):
    df = yf.Ticker(tickerName).history(period=period, interval=interval).reset_index()
    df['Date'] = df['Datetime'].dt.date
    df['Time'] = df['Datetime'].dt.time
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df['Day'] = df['Datetime'].dt.day_name()
    df[['Open', 'High', 'Low', 'Close']] = df[['Open', 'High', 'Low', 'Close']].round(2)
    df = df[['Date', 'Time', 'Datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'Day']]
    return df


df = yfDownload('BLS.NS', '1mo', '5m')
df['timeSpendP&L'] = df.apply(lambda row: 5 if row['Open'] <= row['Close'] else -5, axis=1)
df.groupby('Date') ['timeSpendP&L'].value_counts().reset_index(name='count')


dfEtEx = df.groupby('Date')['High'].max().reset_index()
dfEtEx = pd.merge(dfEtEx, df[['Date', 'High', 'Datetime']], how='left', on=['Date', 'High']).drop_duplicates(subset=['Date', 'High'], keep='last').reset_index(drop=True).rename(columns={'High': 'Exit', 'Datetime': 'exitDatetime'})
dfEtEx[['Entry', 'entryDatetime']] = dfEtEx.apply(lambda row: df.loc[df[(df['Date'] == row['Date']) & (df['Datetime'] <= row['exitDatetime'])]['Low'].idxmin(), ['Low', 'Datetime']], axis=1)
dfEtEx = dfEtEx[['Date', 'entryDatetime', 'exitDatetime', 'Entry', 'Exit']]
dfEtEx['EtExProfit'] = (((dfEtEx['Exit'] - dfEtEx['Entry'])/dfEtEx['Exit'])*100).round(2)







# tt = dfEtEx['EtExProfit'].astype(int).value_counts() 
# tt = round((tt/tt.sum())*100, 2).to_dict()  




































# df[df['Time'] == datetime.time(9, 15, 0)][['Date', 'Open']].reset_index(drop=True)


