# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 14:28:45 2024

@author: Hemant
"""


from tqdm import tqdm
from multiprocessing import Pool, cpu_count

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
pd.options.display.float_format = '{:.2f}'.format
pd.set_option('display.max_columns', 25)
pd.set_option('display.max_rows', 20)
pd.set_option('expand_frame_repr', True)

from dataProcessingHelper import *

def fetch_stock_data(symbol):
    return fetching_all_stock_data_based_on_todays(symbol)

if __name__ == "__main__":
    stockSymbols = pd.read_csv(r'./Data/stokeSymbol.csv')['SYMBOL \n'][1:].unique()
    with Pool(processes=cpu_count()) as pool:
        dataProbability = list(tqdm(pool.imap(fetch_stock_data, stockSymbols), total=len(stockSymbols)))
    df = pd.DataFrame(dataProbability)
    df = df.sort_values(by='ProbabilityOfProfitMT2Percent', ascending=False)
    df.to_excel(fr"./Data/Probability/{df['Date'].astype(str).iloc[0]}.xlsx", index=False)



# stockSymbols = pd.read_csv(r'stokeSymbol.csv')['SYMBOL \n'][1:].unique()
# dataProbability = [fetching_all_stock_data_based_on_todays(symbol) for symbol in tqdm(stockSymbols)] 
# df = pd.DataFrame(dataProbability)
# df = df.sort_values(by='ProbabilityOfProfitMT2Percent', ascending=False)
# df.to_excel(fr"./Data/Probability/{df['Date'].astype(str)[0]}.xlsx", index=False)
