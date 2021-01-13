# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 20:48:35 2020

@author: cbour
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from data_cleaning import load_trade,impor_data
from refreshTime import harmoniz_data
from GMVP import get_GMVP,Louvain_GMVP,get_return_vanilla
import dask

    
    


def nombre_cluster(louvain,df):
    #We add one unit where two assets are in the same cluster
    #df is the matrix where row and columns the asset , we add a 1 when two assets are in the same cluster (similar to adjency matrtix)
    market_name=df.columns
    n=len(market_name)
    
    cluster = louvain.label
    
    for i in range(n):
        index = np.where(cluster[i]==cluster)[0]
        for j in index:
            df.iloc[i,j]+=1
            
    return df
    
    
    
    
    
class daily_back_testing :
    
    def __init__(self,stock_name,path,list_day0):
        #we initialize the value of each strategy
        self.V_vanilla=[100]
        self.V_louvain=[100]
        
        self.number_cluster=[]
        #we save the different days we will backtest the strategy, the personal data path and the assets names
        
        self.date_path=[list_day0[-1]]
        self.path_perso=path
        self.stock_name=stock_name
        n=len(stock_name)
        #We create a list to save the return of each strategy
        self.louvain_return=[]
        self.vanilla_return=[]
        #We initialize the correlation matrix (the average correlation between each asset)
        self.correlation=np.zeros((n,n))
        #we create a matrix to know how many time each asset are together in louvain cluster
        temp1= np.zeros((n,n))
        self.louvain_cluster =pd.DataFrame(data=temp1,columns=stock_name,index=stock_name)
        #we initialize the number of update days
        self.nombre_test=0
        #we import the data for the initialisation of the cluster and portfolios weights
        
        self.data_1 = []
        for d in list_day0 :
            temp1 = impor_data(stock_name,d,path)
            if min([len(df) for df in temp1])>0: 
                self.data_1.append(harmoniz_data(temp1,d))
        
        
    def daily_update(self,day_j):
        '''
        

        Parameters
        ----------
        day_j : date of the day where we want to apply the calibrated stategy

        Returns
        -------
        None.

        '''
        
        #we import the data of the day 2 and make sure it is not empty
        data_j2 = impor_data(self.stock_name,day_j,self.path_perso)
        
        if min([len(df) for df in data_j2])>0:  
            #we concatenate the save synchronised data
            
            
            data_calibrate = pd.concat(self.data_1,axis=0)
            #we calibrate the Louvain strategy weights and compute the return and actualize 
            #the correlation matrix and the number of time assets are in the same cluster
            louvain = Louvain_GMVP(data_calibrate)
            louvain.get_return(data_j2,self.stock_name)
            self.number_cluster.append(max(louvain.label)+1)
            self.louvain_return.append(louvain.retour)
            self.V_louvain.append(self.V_louvain[-1]* (1+louvain.retour))
            
            self.correlation+= louvain.correlation
            self.louvain_cluster = nombre_cluster(louvain,self.louvain_cluster)
            #we add one unit in the number of test
            self.nombre_test+=1
            
            #we calibrate the standard strategy and get the return
            vanilla_w,_,_ = get_GMVP(data_calibrate)
            self.vanilla_return.append(get_return_vanilla(vanilla_w,data_j2))
            self.V_vanilla.append(self.V_vanilla[-1]*(1+self.vanilla_return[-1]) )
            
            #we update the calibration period deleting the first day of the calibration period and adding the synchronised data of the next trading day 
            self.date_path.append(day_j)
            temp=harmoniz_data(data_j2,day_j)
            del self.data_1[0]
            self.data_1.append(temp)
            
        
        
    def plot_value(self,titre):
        plt.figure(figsize=(13,10))
        #we plot the value of each strategy
        plt.plot(self.date_path,self.V_louvain,label='louvain Value')
        
        plt.plot(self.date_path,self.V_vanilla,label='Vanilla Value')
        plt.title('Evolution of each strategy value'+titre)
        plt.legend()
        plt.xlabel('date')
        plt.ylabel('Value')
        #we calibrate the x axis to print the day
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        
        plt.gcf().autofmt_xdate()
        plt.savefig('figures/Value_Strategies'+titre+'.pdf')
        plt.show()
    
    
    

















