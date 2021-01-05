#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 11:30:02 2020

@author: franckatteaka
"""

import pandas as pd
import dask
dask.config.set(scheduler="processes")

@dask.delayed
def load_trade(market_name,date,folder_path,tz_exchange = "America/New_York",
               only_regular_trading_hours = True,open_time = "09:30:00",close_time = "16:00:00",is_compressed = False):
    
    stock_path = 'extraction/TRTH/raw/equities/US/trade/'
    
    if is_compressed == False:
        path_name = folder_path + stock_path + market_name + '/' + str(date)[:10] + '-' + market_name + '-trade.csv'
        DF = pd.read_csv(path_name)[['xltime','trade-price']]
    else:
        path_name = folder_path + stock_path + market_name + '/' + str(date)[:10] + '-' + market_name + '-trade.csv.gz'
        DF = pd.read_csv(path_name, compression = 'gzip')[['xltime','trade-price']]
        
    
    if len(DF)==0:
        #if there is no data there is no index so the 'normal' way do not work
        return DF

    
    DF.index = pd.to_datetime(DF["xltime"],unit = "d",origin = "1899-12-30",utc = True)
    DF.index = DF.index.tz_convert(tz_exchange)  
    DF.drop(columns = "xltime",inplace = True)
    
    if only_regular_trading_hours:
        DF = DF.between_time(open_time,close_time)    
    
    DF = DF.rename(columns = {'trade-price': market_name})
    return DF


