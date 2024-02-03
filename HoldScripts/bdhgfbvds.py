# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 00:10:13 2024

@author: Hemant
"""
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from stockHelper import *


df = yfDownload('CRAFTSMAN.NS', '1mo')
df = formulaPercentage(df)
buysellProbability = buy_sell_probability_in_profit_and_loss(df)
plot_line_chart(df.tail(20))









# =============================================================================
# 
# =============================================================================
df.iloc[38:].head(20)
df.iloc[103:].head(20)
df.iloc[243:].head(20)


df[df['Day'] == 'Monday']





df['Day'].unique()

df['P/L'].min()

df[df['P/L'] == df['P/L'].min()]
df[df['P/L'] == df['P/L'].max()]


df.sort_values(by='P/L', ascending=False)

df[pd.to_datetime(df['Date']) <= '2023-04-25']['P/L'].max()


df[pd.to_datetime(df['Date']) <= '2023-09-12'].tail(20)
df[pd.to_datetime(df['Date']) >= '2023-09-12'].head(20)


df['P/L'].nlargest(10)




df[pd.to_datetime(df['Date']) <= '2024-01-19'].tail(20)

df.iloc[158:].head(20)
df.iloc[:159].tail(20)
