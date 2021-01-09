#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 15:28:42 2020

@author: Franck Atte Aka & Corentin Bourdeix
"""


import pandas as pd
import numpy as np
import glob
import datetime


# raw data files paths
allxlsx = glob.glob('/Users/franckatteaka/Desktop/cours/Semester III/Courses Projects/Machine Learning/Data/*.xlsx')
csv = glob.glob('/Users/franckatteaka/Desktop/cours/Semester III/Courses Projects/Machine Learning/Data/*yields_raw.csv')


# yields data cleaning
yields_raw = pd.read_csv(csv[0],sep = ";",skiprows = 3,parse_dates = True,index_col = 0 )

# remove exchange closed days ()
yields = yields_raw[yields_raw["Value"] == yields_raw["Value"] ]

yields = yields.replace('10J0','10J').copy()
dates = pd.Series(yields.index.date)
tab = pd.DataFrame(dates.value_counts() == 12 )
max_date = max(tab[tab[0] == True].index)
yields_clean = yields[yields.index.date > max_date]


# set bonds maturity as columns
yields_clean.reset_index(level = 0, inplace = True)
yields_clean = yields_clean.pivot(index='Date', columns='D0', values='Value')



# economic sentiment data cleaning
df_eco_raw = pd.read_excel(allxlsx[0],parse_dates = True,index_col = 0 )
df_eco_raw.columns = ['sentiment ch','sentiment eu']
df_eco_raw = df_eco_raw.dropna().copy()



# barometer data cleaning
df_baro_raw = pd.read_excel(allxlsx[1],parse_dates = True,index_col = 0 )



# mcp data cleaning
df_mcp_raw = pd.read_excel(allxlsx[2],parse_dates = True,index_col = 0 )
df_mcp_raw.columns = ['kof mpc']
df_mcp_raw = df_mcp_raw.dropna().copy()



#  economic data upsampling: from monthly to daily data
def expand_data(dt):
    '''
        Upsample the dataframe to daily data
        
         parameters
        ------------
        dt(pandas df): dataframe
        
        Return
        ------------
        df(pandas df): dataframe
    '''
    
    df = dt.copy()
    date = max(df.index)
    d = datetime.datetime.strptime(str(date), '%Y-%m-%d %H:%M:%S')
    date = date.replace(month = d.month + 1 , day = 1)
    
    df.loc[date] = np.nan
    df = df.resample('D').last().ffill().copy()
    df = df.drop( date, axis = 0).copy()
    
    return df

df_baro_clean = expand_data(df_baro_raw)
df_mcp_clean = expand_data(df_mcp_raw)
df_eco_clean = expand_data(df_eco_raw)


# clean data 
data = yields_clean.join([df_mcp_clean, df_baro_clean,df_eco_clean] ).dropna()

# ordered columns
col = ['sentiment eu','sentiment ch', 'kofbarometer','kof mpc', '1J', '2J', '3J', '4J', '5J', '6J', '7J', '8J', '9J', '10J', '15J', '20J', '30J']

data = data[col].copy()
data.columns = ['s_eu','s_ch', 'kof_baro','kof_mpc', '1J', '2J', '3J', '4J', '5J', '6J', '7J', '8J', '9J', '10J', '15J', '20J', '30J']
data.iloc[:,4:] = data.iloc[:,4:].copy()/100

data.to_csv('/Users/franckatteaka/Desktop/cours/Semester III/Courses Projects/Machine Learning/Data/data_clean.csv')



