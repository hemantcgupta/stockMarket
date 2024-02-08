# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 14:30:00 2024

@author: Hemant
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def yfDownload(tickerName, period, interval):
    yfTicker = yf.Ticker(tickerName)
    df = yfTicker.history(period=period).dropna().reset_index()
    df = yfDownloadProcessing(df)
    dfInterval = yfTicker.history(period='1mo', interval=interval).dropna().reset_index()
    dfInterval = yfDownloadProcessingInterval(df, dfInterval)
    return df, dfInterval

def yfDownloadProcessing(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Day'] = df['Date'].dt.day_name()
    df['Date'] = df['Date'].dt.date
    df[['Open', 'High', 'Low', 'Close']] = df[['Open', 'High', 'Low', 'Close']].round(2)
    df['PvClose'] = df['Close'].shift(1)
    df = df.dropna().reset_index(drop=True)
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'PvClose', 'Volume', 'Day']]
    return df

def yfDownloadProcessingInterval(dfDay, df):
    df['Date'] = df['Datetime'].dt.date
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df[['Open', 'High', 'Low', 'Close']] = df[['Open', 'High', 'Low', 'Close']].round(2)
    df = pd.merge(df, dfDay[['Date', 'Open']].rename(columns={'Open': 'OpenDay'}), how='left', on='Date')
    df['CandleP/N_OpenDay'] = df.apply(lambda row: 1 if row['OpenDay'] <= row['Open'] else -1, axis=1)
    df['CandleP/N'] = df.apply(lambda row: 1 if row['Open'] <= row['Close'] else -1, axis=1)
    df = df[['Date', 'Datetime', 'Open', 'High', 'Low', 'Close', 'OpenDay', 'CandleP/N_OpenDay', 'CandleP/N']]
    return df

def formulaPercentage(df):
    df['P/L'] = ((1-df['Open']/df['Close'])*100).round(2)    
    df['maxHigh'] = ((df['High']/df['Open']-1)*100).round(2)
    df['maxLow'] = ((df['Low']/df['Open']-1)*100).round(2)
    df['Open-PvClose'] = (df['Open']-df['PvClose']).round(2)  
    df['closeTolerance'] = df.apply(lambda row: row['P/L'] - row['maxHigh'] if row['P/L'] > 0 else row['P/L'] - row['maxLow'] if row['P/L'] < 0 else 0, axis=1)
    df['priceBand'] = (((df['High'] - df['Low'])/df['Open'])*100).round(2)
    return df

def EntryExitMinToMax(dfInterval):
    dfEtEx = dfInterval.groupby('Date')['Low'].min().reset_index()
    dfEtEx = pd.merge(dfEtEx, dfInterval[['Date', 'Low', 'Datetime']], how='left', on=['Date', 'Low']).drop_duplicates(subset=['Date', 'Low'], keep='last').reset_index(drop=True).rename(columns={'Low': 'Entry', 'Datetime': 'entryDatetime'})
    dfEtEx[['Exit', 'exitDatetime']] = dfEtEx.apply(lambda row: dfInterval.loc[dfInterval[(dfInterval['Date'] == row['Date']) & (dfInterval['Datetime'] >= row['entryDatetime'])]['High'].idxmax(), ['High', 'Datetime']], axis=1)
    dfEtEx = pd.merge(dfEtEx, dfInterval[['Date', 'Datetime', 'OpenDay']].drop_duplicates(subset='Date', keep='first').reset_index(drop=True).rename(columns={'Datetime': 'OpenDayDatetime'}), how='left', on='Date')
    dfEtEx = dfEtEx[['Date', 'OpenDayDatetime', 'entryDatetime', 'exitDatetime', 'OpenDay', 'Entry','Exit']]
    dfEtEx['entrytimeDiff'] = ((dfEtEx['entryDatetime'] - dfEtEx['OpenDayDatetime']).dt.total_seconds()/60).astype(int)
    dfEtEx['exittimeDiff'] = ((dfEtEx['exitDatetime'] - dfEtEx['entryDatetime']).dt.total_seconds()/60).astype(int)
    dfEtEx['OpenToEntryLoss'] = ((1-dfEtEx['OpenDay']/dfEtEx['Entry'])*100).round(2)
    dfEtEx['OpenToExitProfit'] = ((1-dfEtEx['OpenDay']/dfEtEx['Exit'])*100).round(2)
    dfEtEx['EtExProfit'] = ((1-dfEtEx['Entry']/dfEtEx['Exit'])*100).round(2)
    dfEtEx = dfEtEx[['Date', 'Entry', 'Exit', 'entrytimeDiff', 'exittimeDiff', 'OpenToEntryLoss', 'OpenToExitProfit', 'EtExProfit']].rename(columns=lambda x: x + '1' if x != 'Date' else x)
    return dfEtEx

def EntryExitMaxToMin(dfInterval):
    dfEtEx = dfInterval.groupby('Date')['High'].max().reset_index()
    dfEtEx = pd.merge(dfEtEx, dfInterval[['Date', 'High', 'Datetime']], how='left', on=['Date', 'High']).drop_duplicates(subset=['Date', 'High'], keep='last').reset_index(drop=True).rename(columns={'High': 'Exit', 'Datetime': 'exitDatetime'})
    dfEtEx[['Entry', 'entryDatetime']] = dfEtEx.apply(lambda row: dfInterval.loc[dfInterval[(dfInterval['Date'] == row['Date']) & (dfInterval['Datetime'] <= row['exitDatetime'])]['Low'].idxmin(), ['Low', 'Datetime']], axis=1)
    dfEtEx = pd.merge(dfEtEx, dfInterval[['Date', 'Datetime', 'OpenDay']].drop_duplicates(subset='Date', keep='first').reset_index(drop=True).rename(columns={'Datetime': 'OpenDayDatetime'}), how='left', on='Date')
    dfEtEx = dfEtEx[['Date', 'OpenDayDatetime', 'entryDatetime', 'exitDatetime', 'OpenDay', 'Entry','Exit']]
    dfEtEx['entrytimeDiff'] = ((dfEtEx['entryDatetime'] - dfEtEx['OpenDayDatetime']).dt.total_seconds()/60).astype(int)
    dfEtEx['exittimeDiff'] = ((dfEtEx['exitDatetime'] - dfEtEx['entryDatetime']).dt.total_seconds()/60).astype(int)
    dfEtEx['OpenToEntryLoss'] = ((1-dfEtEx['OpenDay']/dfEtEx['Entry'])*100).round(2)
    dfEtEx['OpenToExitProfit'] = ((1-dfEtEx['OpenDay']/dfEtEx['Exit'])*100).round(2)
    dfEtEx['EtExProfit'] = (((dfEtEx['Exit'] - dfEtEx['Entry'])/dfEtEx['Exit'])*100).round(2)
    dfEtEx = dfEtEx[['Date', 'Entry', 'Exit', 'entrytimeDiff', 'exittimeDiff', 'OpenToEntryLoss', 'OpenToExitProfit', 'EtExProfit']].rename(columns=lambda x: x + '2' if x != 'Date' else x)
    return dfEtEx

def MovingAverage44(df):
    df['44MA'] = df['Close'].rolling(window=44).mean().fillna(0)
    df['44TF'] = df.apply(lambda row: 1 if row['44MA'] <= row['High'] and row['44MA'] >= row['Low'] else 0, axis=1)
    return df

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

def NextDayPrediction(df):
    predTmOpen = round((df['Open']/df['PvClose']-1).mean()*df['Close'].iloc[-1]+df['Close'].iloc[-1], 2)
    predTmEntry1 = round((df['Entry1']/df['Open']-1).mean()*predTmOpen+predTmOpen, 2)
    predTmEntry2 = round((df['Entry2']/df['Open']-1).mean()*predTmOpen+predTmOpen, 2)
    predTmExit1 = round((df['Exit1']/df['Open']-1).mean()*predTmOpen+predTmOpen, 2)
    predTmExit2 = round((df['Exit2']/df['Open']-1).mean()*predTmOpen+predTmOpen, 2)
    predTmClose = round(predTmOpen/(1-(1-df['Open']/df['Close']).mean()), 2)
    predTmMaxhigh = round((df['maxHigh'].mean()/100)*predTmOpen+predTmOpen, 2)
    predTmMaxlow = round((df['maxLow'].mean()/100)*predTmOpen+predTmOpen, 2)
    EtEx1Profit = round((1-predTmEntry1/predTmExit1)*100, 2)
    EtEx2Profit = round((1-predTmEntry2/predTmExit2)*100, 2)
    predPL = round((1-predTmOpen/predTmClose)*100, 2)
    predDct = {
        'predTmOpen': predTmOpen,
        'predTmEntry1': predTmEntry1,
        'predTmExit1': predTmExit1,
        'predTmEntry2': predTmEntry2,
        'predTmExit2': predTmExit2,
        'predTmClose': predTmClose,
        'predTmMaxhigh': predTmMaxhigh,
        'predTmMaxlow': predTmMaxlow,
        'EtEx1Profit': EtEx1Profit,
        'EtEx2Profit': EtEx2Profit,
        'predTmP/L': predPL
        }
    return predDct

def ProbabilityDataProcessing(df, dfInterval, symbol): 
    predDct = NextDayPrediction(df)
    dctSR = {'Support': dfInterval['Support'].value_counts().sort_index().to_dict(), 'Resistance': dfInterval['Resistance'].value_counts().sort_index().to_dict()}
    df['Symbol'] = symbol
    dct = df[['Symbol', 'Date', 'Open', 'High', 'Low', 'Close', 'P/L', 'maxHigh', 'maxLow', 'closeTolerance', 'priceBand']].iloc[-1].to_dict()
    buysellProbability = buy_sell_probability_in_profit_and_loss(df)
    dct = {**dct, **buysellProbability, **predDct, **dctSR}
    return dct

def find_support_resistance(df, window_size=20):
    data = df[['Datetime', 'Low', 'High']]
    supports = []
    resistances = []
    for i in range(window_size, len(data) - window_size):
        window_low = data['Low'].iloc[i - window_size:i + window_size]
        window_high = data['High'].iloc[i - window_size:i + window_size]
        min_val = window_low.min()
        max_val = window_high.max()
        avg_low = window_low.mean()
        avg_high = window_high.mean()
        if data['Low'].iloc[i] < avg_low and data['Low'].iloc[i] == min_val:
            supports.append({'Datetime': data['Datetime'].iloc[i], 'Support': data['Low'].iloc[i]})
        if data['High'].iloc[i] > avg_high and data['High'].iloc[i] == max_val:
            resistances.append({'Datetime': data['Datetime'].iloc[i], 'Resistance': data['High'].iloc[i]})
    dfS = pd.DataFrame(supports)
    dfR = pd.DataFrame(resistances)
    dfSR = pd.merge(df, dfS, how='left', on='Datetime').merge(dfR, how='left', on='Datetime')
    return dfSR


def saveFilesInMachine(symbol, df, subFolder):
    filePath = fr"./Data/{subFolder}/{df['Date'].astype(str).iloc[-1]}/{symbol.split('.')[0]}.xlsx"
    os.makedirs(os.path.dirname(filePath), exist_ok=True)
    df.to_excel(filePath, index=False)
    
def fetching_all_stock_data_based_on_todays(symbol):
    symbol = symbol+'.NS'
    df, dfInterval = yfDownload(symbol, '1y', '5m')
    df = formulaPercentage(df)
    df = MovingAverage44(df)
    dfInterval = MovingAverage44(dfInterval)
    dfInterval = find_support_resistance(dfInterval)
    dfCandle = pd.merge(pd.merge(dfInterval.groupby('Date') ['CandleP/N_OpenDay'].value_counts().unstack(fill_value=0).reset_index().rename(columns={-1: 'nCandleBelowOpen', 1: 'pCandleAboveOpen'}), dfInterval.groupby('Date') ['CandleP/N'].value_counts().unstack(fill_value=0).reset_index().rename(columns={-1: 'nCandle', 1: 'pCandle'}), how='left', on='Date'), dfInterval.groupby('Date') ['44TF'].value_counts().unstack(fill_value=0).reset_index().rename(columns={1: 'Hits44MA'})[['Date', 'Hits44MA']], how='left', on='Date')
    dfEtEx = pd.merge(EntryExitMinToMax(dfInterval), EntryExitMaxToMin(dfInterval), how='left', on='Date')
    dfItCd = pd.merge(dfCandle, dfEtEx,how='left', on='Date')
    df = pd.merge(df, dfItCd, how='left', on='Date')
    saveFilesInMachine(symbol, df, 'Processing')
    dfInterval['Datetime'] = dfInterval['Datetime'].dt.tz_localize(None)
    saveFilesInMachine(symbol, dfInterval, 'intervalData')
    df = df.dropna().reset_index(drop=True)
    dct = ProbabilityDataProcessing(df, dfInterval, symbol)
    return dct


# symbol = 'BLS' + '.NS'
# df, dfInterval = yfDownload(symbol, '1y', '5m')
# df = formulaPercentage(df)
# df = MovingAverage44(df)
# dfInterval = MovingAverage44(dfInterval)
# dfInterval = find_support_resistance(dfInterval)
# dfCandle = pd.merge(pd.merge(dfInterval.groupby('Date') ['CandleP/N_OpenDay'].value_counts().unstack(fill_value=0).reset_index().rename(columns={-1: 'nCandleBelowOpen', 1: 'pCandleAboveOpen'}), dfInterval.groupby('Date') ['CandleP/N'].value_counts().unstack(fill_value=0).reset_index().rename(columns={-1: 'nCandle', 1: 'pCandle'}), how='left', on='Date'), dfInterval.groupby('Date') ['44TF'].value_counts().unstack(fill_value=0).reset_index().rename(columns={1: 'Hits44MA'})[['Date', 'Hits44MA']], how='left', on='Date')
# dfEtEx = pd.merge(EntryExitMinToMax(dfInterval), EntryExitMaxToMin(dfInterval), how='left', on='Date')
# dfItCd = pd.merge(dfCandle, dfEtEx,how='left', on='Date')
# df = pd.merge(df, dfItCd, how='left', on='Date')
# df = df.dropna().reset_index(drop=True)
# dct = ProbabilityDataProcessing(df, dfInterval, symbol)











