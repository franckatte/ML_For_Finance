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
    '''test whether one of the dates list is empty'''
    
    a = [len(date) for date in dates]
    
    return 0 in a

#This function get the refresh times without dask
def refresh_time_without_dask(dfs):
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

#This function is the same as above but with 'dask delayed'
@dask.delayed
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

#This function will use the above function to compute the refresh times faster
def refresh_time_dask(dfs,date):
    
    #First we set the sets where we search refresh times
    d = date.strftime('%Y/%m/%d')
    d_range_dep=pd.date_range(start=d+" 09:30:00",end=d+" 15:30:00",freq='30T',tz ="America/New_York" )
    d_range_en =  pd.date_range(start=d+" 10:10:00",end=d+" 16:10:00",freq='30T',tz ="America/New_York" )
    
    #For the different set we compute using dask
    n= len(d_range_dep)
    tau_total=[]
    for i in range(n):
        
        datai = [df.iloc[df.index>d_range_dep[i]] for df in dfs]
        datai = [df.iloc[df.index<=d_range_en[i]] for df in datai]
        
        tau_total.append(refresh_time(datai))
        
        
    tau_t=dask.compute(tau_total)[0]
    #Then to put together all sets we need to delete the double and the false refresh times
    limit_d = pd.date_range(start=d+" 10:10:00",end=d+" 15:40:00",freq='30T',tz ="America/New_York" )
    tau_t2=[tau_t[0]]
    for j in range(len(limit_d)):
        #We delete the refresh time found in the first 10min for all sets except the first one
        index = [limit_d[j]<tau_t[j+1][i] for i in range(len(tau_t[j+1]))]
        tau_t2.append(np.array(tau_t[j+1])[index])
    #We concatenate all refresh time and make sur to not have doubles
    tau = np.concatenate(tau_t2,axis=0)
    tau = np.unique(tau)
    return tau







#This function resample the data from the refresh times obtain for one market
@dask.delayed
def  resample(df,r_times):
    
     index = df.index
     name = df.columns
     sampled_index = [dask.delayed(max)(index[index<=t] ) for t in r_times]
     sampled_index = pd.Index(dask.compute(*sampled_index))
     
     df2 = df.loc[sampled_index]
     df2 = df2.loc[~df2.index.duplicated(keep = "last")]
     
     return pd.DataFrame(data=df2.values,index=r_times,columns=name)
     
 
    
 
    
 
#This function use dask to resample all markets at the same time
def synchro_data(dfs,date):
    '''
        return DataFrame of synchronise data
        ------------
        dfs(list): list of dataframes
        
        Return
        ------------
        list of DF(DataFrame): contain the price at each refresh time
        
    '''
    tau = refresh_time_dask(dfs,date)
    
    res = []
    
    for df in dfs:
        res.append(resample(df,tau))
        
    
    return dask.compute(res)[0]
        

def harmoniz_data(dfs,date):
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
    data00=synchro_data(dfs,date)
    
    tab = pd.concat(data00,axis=1)
    
    return tab
    
    
    
'''
TEST 
   
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
    
 
    
 
    
 
    
 
    
 
    
 
    
     