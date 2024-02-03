# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 22:40:04 2024

@author: Hemant
"""

import warnings
warnings.filterwarnings('ignore')

import time
import pandas as pd
pd.options.display.float_format = '{:.2f}'.format
pd.set_option('display.max_columns', 25)
pd.set_option('display.max_rows', 20)
pd.set_option('expand_frame_repr', True)


import yfinance as yf
import pandas as pd

def download_stock_data(tickers, start_date, end_date):
    try:
        df = yf.download(tickers, start=start_date, end=end_date)
        return df
    except Exception as e:
        print(f"Error downloading stock data: {e}")


tickers = "BLS.NS"  
start_date = "2024-01-01"  
end_date = "2024-02-02"    
df = download_stock_data(tickers, start_date, end_date)
msft = yf.Ticker("COALINDIA.NS")
hist = msft.history(period="1mo")
msft.history_metadata

msft.get_shares_full(start="2024-01-01", end=None)

