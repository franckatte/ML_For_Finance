#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 20:08:01 2020

@author: franckatteaka
"""



import pandas as pd
import numpy as np
from keras.models import load_model

#df = pd.read_csv('/Users/franckatteaka/Desktop/cours/Semester III/Courses Projects/Machine Learning/Data/data_clean.csv',sep = ","
#                 ,parse_dates = True,index_col = 0 )


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



def denoiser(df,backwards,model_path):
    
    autoencoder = load_model(model_path)
    df2 = df.iloc[:,df.columns.str.contains('J')].copy()
    df3 = df.iloc[:,~df.columns.str.contains('J')].copy()

    for l in backwards:
        name =  "(t-" + str(l) + ")"
        df2.iloc[:,df2.columns.str.contains(name, regex = False)] = autoencoder.predict(df2.iloc[:,df2.columns.str.contains(name, regex = False)].to_numpy())
    
    return df3.join(df2)
        

    
    
def supervised(df,growth_freqs,backwards,denoise = True,model_path = None,scale_eco = True,nb_years = None,binary = False):
    '''
        create supervised learning data for a lead time
        
         parameters
        ------------
        df(pandas df): dataframe economic variable and yields 
        growth:_freqs(list): frequences of growth for econommics data
        backwards(list): days between target variables and observed data used to predict
        denoise(Bool): True if we want to denoise the yields
        model_path(str): path of the keras denoising model 
        scale_eco(Bool): True if we want the economic variables to be scaled
        Return
        ------------
        (X,y) df(pandas df): regressors, target
    '''
    
    
    df2 = sub_range(df,nb_years)
    
    cols = list(df2.columns)
    cols.reverse()
    
    for c in cols:
        if 'J' in c:
            df2 =lag(df2,backwards,c, drop = False)
        else:
            df2 =lag(df2,backwards,c, drop = True)
            
    df2.dropna(inplace = True)
    
    
    eco_cols = list(df2.loc[:,~df2.columns.str.contains('J')].columns)

    
    for c in eco_cols:
         df2 = rolling_growth(df2,growth_freqs,c, drop = True)
        
    
    
    df2 = df2.dropna()
    
    X = df2.iloc[:,:-13].copy()
    
    y = df2.iloc[:,-13:].copy()
    
    if denoise == True:
        
        X = denoiser(X,backwards,model_path)
    
    if scale_eco == True:
        df2.iloc[:,~df2.columns.str.contains('J')] = df2.iloc[:,~df2.columns.str.contains('J')].apply(lambda x: (x - x.min())/(x.max() - x.min())).copy()
    
    if binary == True:
        l = backwards[0]
        columns = list(y.columns)
        for c in columns:
            col = c + " (t-" + str(l) + ")"
            y[c] = (y[c] - X[col]).apply(lambda x: 1 if x > 0 else 0).copy()
            
        
    return X,y
    

def reshape(X,backwards):
    '''
        reshape data as required to feed the lstm model
        
        parameters
        ------------
        X(pandas df): dataframe containing the regressors
        backwards(list): days between target variables and observed data used to predict
        
        Return
        ------------
        A(3 dimensional array numpy array): (samples, timesteps, features)
    '''
    
    backwards2 =  backwards.copy()
    backwards2.sort(reverse = True)
    sequence = []
    
    for i in backwards2:
        ticker = "(t-" + str(i) + ")"
        sequence.append(X.iloc[:,X.columns.str.contains(ticker, regex = False)].values)
    
    nb_obs = sequence[0].shape[0]
    nb_features= sequence[0].shape[1]
    time_steps = len(sequence)
    
    A = np.zeros(( nb_obs,time_steps,nb_features,))
    
    for t_steps in range(len(backwards2)):
        
        A[:,t_steps,:] = sequence[t_steps]
    
    return A


