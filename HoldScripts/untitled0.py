# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 21:19:08 2024

@author: Hemant
"""

import yfinance as yf
import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime
from stockHelper import *

def fetching_all_stock_data_based_on_todays(symbol):
    symbol = symbol+'.NS'
    df = yfDownload(symbol, '1mo').dropna().reset_index(drop=True)
    df = formulaPercentage(df)
    df['Symbol'] = symbol
    dct = df[['Symbol', 'Date', 'Open', 'High', 'Low', 'Close', 'P/L', 'maxHigh', 'maxLow', 'closeTolerance', 'priceBand']].iloc[-1].to_dict()
    buysellProbability = buy_sell_probability_in_profit_and_loss(df)
    dct = {**dct, **buysellProbability}
    return dct

stockSymbols = pd.read_csv(r'stokeSymbol.csv')['SYMBOL \n'][1:].unique()
dataProbability = [fetching_all_stock_data_based_on_todays(symbol) for symbol in tqdm(stockSymbols)] 
df = pd.DataFrame(dataProbability)
df = df.sort_values(by='ProbabilityOfProfitMT2Percent', ascending=False)
df.to_excel(fr"./Data/Probability/{df['Date'].astype(str)[0]}.xlsx", index=False)


