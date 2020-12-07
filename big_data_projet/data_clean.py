# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 20:03:53 2020

@author: corentin
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data= pd.read_csv('us-stocks.csv',index_col=0).iloc[:,:100]
index = data.index
names_col = data.columns

def number_na(dataframe):
    #return the number of NA value from a DataFrame
    return dataframe.isna().sum()

def hist_na_market(data,names):
    #take a dataset and its columns to give the number of NA per columns
    num_na = np.array([number_na(data.iloc[:,i]) for i in range(len(names))])
    fig = plt.figure(1, figsize=(18, 9))
    plt.bar(names,num_na)
    plt.xticks(rotation=90)
    plt.ylabel('number of NA')
    plt.show()
    return
    
def evolution_na(data,index):
    #take a dataset and plot the evolution of NA througth the index
    num_na = np.array([number_na(data.iloc[i,:]) for i in range(len(index))])
    fig = plt.figure(1, figsize=(18, 9))
    plt.plot(index,num_na)
    plt.xlabel('time')
    plt.ylabel('number of NA')
    plt.show()
    return
    
hist_na_market(data,names_col)
evolution_na(data,index)

def restrain_data(data,index,number_na_limit):
    
    num_na = np.array([number_na(data.iloc[i,:]) for i in range(len(index))])
    filtere = np.argwhere(num_na<number_na_limit).T[0]
    
    return index[filtere],data.iloc[filtere,:]
    
index_clean,data_clean = restrain_data(data,index,20)  
    
hist_na_market(data_clean,names_col)
evolution_na(data_clean,index_clean)

data_clean.to_csv('data_clean.csv')








