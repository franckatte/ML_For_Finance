#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 13:39:15 2020

@author: franckatteaka
"""
import numpy as np
import pandas as pd
def test_date(dates):
    '''test wether one of the dates list is empty'''
    
    a = [len(date) for date in dates]
    
    return 0 in a

def refresh_time(dfs):
    '''
        return list of refresh times 
        parameters
        ------------
        dfs(list): list of dataframes
        
        Return
        ------------
        tau(numpy array): contain refresh times
        
    '''
    dates = np.array([df.index.drop_duplicates(keep = 'last') for df in dfs])
    tau = []
    
    while test_date(dates) == False:
        
        # append refresh date
     
        tau.append(max([min(date) for date in dates]))

        #update dates
        dates = np.array([date[np.where(date > tau[-1])[0]] for date in dates ])
        

      
        
    return tau


def  resample(df,r_times):
    
     index = df.index
     sampled_index = pd.Index([max(index[index<=t] ) for t in r_times])
     df2 = df.loc[sampled_index]
     
     return df2.loc[~df2.index.duplicated(keep = "last")]
     