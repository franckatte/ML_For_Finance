#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 13:39:15 2020

@author: franckatteaka
"""
import numpy as np

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
    dates = np.array([df.loc[~df.index.duplicated(keep='last')].index for df in dfs])
    tau = []
    
    while test_date(dates) == False:
        
        # append refresh date
        tau.append(max([min(date) for date in dates]))

        #update dates
        dates = np.array([date[np.where(date > tau[-1])[0]] for date in dates ])
        
      
        
    return tau