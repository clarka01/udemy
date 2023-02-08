#%%

'''
File to follow along with the Udemy Colurse Importing Free Financial Data with Python

    + Course: https://www.udemy.com/course/importing-free-financial-data-with-python/
    + Teacher: Alexander Hagmann
    + Resources: 
        - https://pypi.org/project/yfinance/
    
    + Definitions:
        - Dividends (payout)
        - Stock Splits (why?)
        - Indexes (Dow Jones, S&P 500)
            -Total Return Index?
        - Mutual Funds
        - ETF (exchange-traded fund)

    + Forms:
        -Balance Sheet
        -P&L
        -Cashflow

    
'''




#%%

# IMPORT STANDARD LIBRARIES
#======================


import sys
import os
import pdb
import datetime
import numpy as np
# import requests
import pandas as pd
# import pickle
import time
import inspect
from datetime import datetime, date, timedelta# import time
import matplotlib.pyplot as plt

import yfinance as yf


pd.options.display.max_rows = 999

# APPEND REPO PATH
#======================

# SIRIUS = os.environ['SIRIUS']
sys.path.append(r'C:\Users\clarka\Documents\SIRIUS\pythlib')


from date_lib import date_utils_hbb as du


# Date Variables
# =======================
today_date = datetime.today()  # with datetime
today = datetime.today().strftime('%Y-%m-%d') 

time_stamp = datetime.today().strftime('%Y-%m-%d @ %H:%M') 
fdolm = du.get_1st_of_last_month().strftime('%Y-%m')
ldolm = f'{du.get_last_of_last_month():%Y-%m-%d}'
fdom12ma = f'{du.get_1st_of_last_n_month(12):%Y-%m-%d}'
l_ym = du.get_last_of_last_month().strftime('%Y %b')

year = du.get_1st_of_this_month().strftime('%Y')
date = f'{du.get_last_of_last_month():%Y-%m-%d}' #change100

#%%

#

ticker = 'HBB'

hbb = yf.download(ticker, period = '1y', interval = '1d', prepost = False) #start = start, end = end

hbb_hr = yf.download(ticker, period = '1y', interval = '1h', prepost = True) #start = start, end = end

hbb_min = yf.download(ticker, period = '5d', interval = '1m') #only works for 5 days


#%%

# Stock Splits and Dividends (Corporate Actions)
# ===================== "actions = True"

ticker = 'AAPL'
AAPL = yf.download(ticker, period = "10y", actions = True)


# Dividends:
## Adjusted close prices are adjusted for dividend payments

# show dates where adj close is affeced by dividend
AAPL[AAPL['Dividends'] > 0]
AAPL.loc['2019-08-05':'2019-08-15']
AAPL.loc['2019-08-05':'2019-08-15'].diff()


# Splits
## must backward adjust for stock splits

AAPL[AAPL['Stock Splits'] > 0]
AAPL.loc['2014-06-05':'2014-06-15']


#%%

# Importing Multiple Stocks
# =====================

## 

ticker = ['GE', 'AAPL', 'FB']
stocks = yf.download(ticker, period = '5y', group_by= 'Ticker')

# Choose Close as column header
stocks = yf.download(ticker, period = '5y').Close

stocks.plot()
plt.show()


#%% ============================================================================================


# Importing Financial Indexes
# =====================

ticker = ['^DJI', '^GSPC']

indexes = yf.download(ticker, period = '5y').Close

# normalize prices starting at 100. 
norm = indexes.div(indexes.iloc[0]).mul(100)

norm.plot()
plt.show()

#%%

# Total REturn Indexes
#========================

ticker = ['^DJITR', '^SP500TR']

indexes = yf.download(ticker, period = '5y').Close


#%% ============================================================================================

# Importing Currencies / FX
#========================

ticker = 'EURUSD=X'
ticker2 = 'USDEUR=X'
ticker3 = 'USDGBP=X'

usd_eur = yf.download(ticker, period = '5y')
eur_usd = yf.download(ticker2, period = '5y')
gbp_usd = yf.download(ticker3, period = '5y')


#%% ============================================================================================

# Importing Crypto Currencies
#========================

ticker = ['BTC-USD', 'ETH-USD']

crypt= yf.download(ticker, period = '5y').Close

crypt.plot()
plt.show()



#%% ============================================================================================

# Importing Funds & ETFs
#========================

ticker = 'TLT' #iShares 20+ Year Treasury Bond ETF
ticker2 = 'OMOIX' #Vivaldi Multi-Strategy Fund Class

yf.download(ticker, period = '5y')
yf.download(ticker2, period = '5y')




#%% ============================================================================================

# Importing Treasury Yields
#========================

ticker1 = '^TNX' # 10Y Treasury Yield
ticker2 = '^FVX' # 5Y Treasury Yield

ticker = ['^TNX','^FVX']


yeilds = yf.download(ticker, period = '5y').Close
yf.download(ticker2, period = '5y')

yeilds.plot()
plt.show()


#%% ============================================================================================

# Creating Ticker Object
#========================

ticker= 'DIS' #disney

dis = yf.Ticker(ticker)

dis.ticker
dis.history(period = '1d', inverval = '1m') #this data is ALREADY ADJUSTED!!!
# -- Alex typically doesn't use the history methods


#%% ============================================================================================

# Stock Fundamentals, Meta Data, Performance Metrics
#========================

dis.info

d = pd.Series(dis.info)
df = pd.Series(dis.info, name = 'DIS').to_frame().T


# add additional rows to df
ticker = ['MSFT', 'FB']
for i in ticker:
    df.loc[i] = pd.Series(yf.Ticker(i).info)



#%% ============================================================================================

# Importing Financials (Balance Sheet, P&L, Cashflows)
#========================

# ticker= 'DIS' #disney

# dis = yf.Ticker(ticker)

# dis.balance_sheet
# dis.financials
# dis.cashflow

ticker= 'HBB'

hbb = yf.Ticker(ticker)

hbb.balance_sheet
hbb.financials #P&L Statement
hbb.cashflow

ticker= ['HBB', 'GE']


for i in ticker:
    yf.Ticker(i).financials.to_csv(f'{i}.csv')

