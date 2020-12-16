#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 13:39:15 2020

@author: franckatteaka
"""
import numpy as np

def test_date(dates):
    '''test wether one of the dates list is empty'''
    try:
        return np.apply_along_axis(lambda x: len(x) == 0 ,1, dates.reshape((-1,1))).sum()
    except ValueError:
        return 1

def refresh_time(dfs):
    '''return list of refresh times'''
    dates = np.array([df.loc[~df.index.duplicated(keep='last')].index for df in dfs])
    tau = []
    
    while test_date(dates)< 1:
        
        # append refresh date
        tau.append(max([min(date) for date in dates]))

        #update dates
        dates = np.array([date[np.where(date > tau[-1])[0]] for date in dates ])
    
    return tau