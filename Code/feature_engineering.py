#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 20:08:01 2020

@author: franckatteaka
"""



import pandas as pd
import numpy as np
import datetime


df = pd.read_csv('/Users/franckatteaka/Desktop/cours/Semester III/ML_for_finance/Data/data_clean.csv',sep = ","
                 ,parse_dates = True,index_col = 0 )




def sub_range(df,nb_years = None):
    '''
     
        sub sample the dataframe to the last x years 
            
        parameters
        ------------
        df(pandas df): dataframe
        nb_years(int): number of years of data to return
        
        Return
        ------------
        df(pandas df): dataframe

    '''
    
  
    
    # most recent date
    last = df.index.date.max()
    
    # date x years before most recent date
    first =  last.replace(year = last.year - nb_years)
    
    df2 = df[ df.index.date >= first].copy()

    return df2

def lag(df,lags,col, drop = True):
    
    '''
        return a dataframe with lagged column(s)
        
         parameters
        ------------
        df(pandas df): dataframe
        lags(list): list of lags
        col(str): column to be shifted
        drop (bool): True if the original column should be dropped
        
        Return
        ------------
        df(pandas df): dataframe
    '''
    
    df2 = df.copy()
    
    for l in lags:
        name = col + " (t-" + str(l) + ")"
        df2[name] = df2[col].shift(l)
    
    if drop == True:
        df2.drop(col,axis = 1, inplace = True)
    
    return df2.iloc[:,-len(lags):].join(df2.iloc[:,:-len(lags)])
        
    
def rolling_growth(df,freqs,col, drop = True):
    
    df2 = df.copy()
    
    for freq in freqs:
        name =  col + " (growth_" + str(freq) + ")"
        df2[name]= df2[col].rolling(freq).apply(lambda x: (x.iloc[-1] - x.iloc[0])/x.iloc[0]).copy()
    
    if drop == True:
        df2.drop(col,axis = 1, inplace = True)
    
    return df2.iloc[:,-len(freqs):].join(df2.iloc[:,:-len(freqs)])


def supervised(df,growth_freqs,lead):
    '''
        create supervised learning data for a lead time
        
         parameters
        ------------
        df(pandas df): dataframe economic variable and yields 
        growth:_freqs(list): frequences of growth for econommics data
        lead(int): days between target variables and observed data used to predict
        
        Return
        ------------
        df(pandas df): dataframe
    '''
    
    
    df2 = df.copy()

    cols = list(df2.columns)
    cols.reverse()
    for c in cols:
        if 'J' in c:
            df2 =lag(df2,[lead],c, drop = False)
        else:
            df2 =lag(df2,[lead],c, drop = True)
            
    df2.dropna(inplace = True)
    
    
    eco_cols = list(df2.loc[:,~df2.columns.str.contains('J')].columns)

    
    for c in eco_cols:
         df2 = rolling_growth(df2,growth_freqs,c, drop = True)
        
    
    
    df2 = df2.dropna()
    
    
    
    return df2.iloc[:,:-13] ,df2.iloc[:,-13:]
    
        

    
X,y = supervised(df,growth_freqs = [30,60],lead=1)

