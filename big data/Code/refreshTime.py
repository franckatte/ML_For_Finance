#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 13:39:15 2020

@author: franckatteaka
"""
import numpy as np
import pandas as pd
import dask
dask.config.set(scheduler="processes")



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
    dates = np.array([df.index.drop_duplicates(keep = 'last') for df in dfs],dtype=object)
    tau = []
    
    while test_date(dates) == False:
        
        # append refresh date
     
        tau.append(max([min(date) for date in dates]))

        #update dates
        dates = np.array([date[np.where(date > tau[-1])[0]] for date in dates ],dtype=object)
        
        
    return tau

@dask.delayed
def  resample(df,r_times):
    
     index = df.index
     name = df.columns
     sampled_index = [dask.delayed(max)(index[index<=t] ) for t in r_times]
     sampled_index = pd.Index(dask.compute(*sampled_index))
     
     df2 = df.loc[sampled_index]
     df2 = df2.loc[~df2.index.duplicated(keep = "last")]
     
     return pd.DataFrame(data=df2.values,index=r_times,columns=name)
 
    
 
    
 
    
def synchro_data(dfs):
    '''
        return DataFrame of synchronise data
        ------------
        dfs(list): list of dataframes
        
        Return
        ------------
        list of DF(DataFrame): contain the price at each refresh time
        
    '''
    tau = refresh_time(dfs)
    res = []
    
    for df in dfs:
        res.append(resample(df,tau))
        
    #return dask.compute(*res)
    return dask.compute(res)[0]
        
    
def harmoniz_data(dfs):
    '''
        Concatenate the list of DataFrame from 'synchro_data'
        ------------
        dfs(list): list of dataframes
        
        Return
        ------------
        DF(DataFrame): contain the price at each refresh time
        row : refesh time
        columns : market
        
        
    '''
    data00=synchro_data(dfs)
    test = [data00[i] for i in range(len(data00))]
    tab = pd.concat(test,axis=1)
    
    return tab
    
    
    
'''
    
 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 09:08:52 2020

@author: franckatteaka
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_cleaning import load_trade
from refreshTime import refresh_time, test_date, resample,synchro_data


Date = pd.bdate_range('2010-01-01','2010-12-31')
Market_name = np.array(['AAPL.OQ','AMGN.OQ','AXP.N','BA.N','CAT.N','CSCO.OQ','CVX.N','DOW.N','GS.N','SPY.P','UTX.N','V.N','WMT.N'])

#folder_path = 'D:/GitHub/ML_For_Finance/big data/data/data/'

folder_path = '/Users/franckatteaka/Desktop/cours/Semester III/Financial big data/high freq data/'



aapl = load_trade(Market_name[0],Date[7],folder_path,is_compressed = True)
amgn = load_trade(Market_name[1],Date[7],folder_path,is_compressed = True)
axpn = load_trade(Market_name[2],Date[7],folder_path,is_compressed = True)


tau = refresh_time([aapl.iloc[:1000],amgn.iloc[:1000],axpn.iloc[:1000]])



date1 = aapl.index

df = resample(aapl.iloc[:1000],tau)

df2 = dask.compute(resample(aapl,tau))

df = synchro_data([aapl,amgn,axpn])


test = [df[0].reset_index(drop=True),df[1].reset_index(drop=True),df[2].reset_index(drop=True)]
tab = pd.concat(test)



 
    
'''
    
 
    
 
    
 
    
 
    
 
    
 
    
     