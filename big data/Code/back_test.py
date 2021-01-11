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
    #df is the matrix where row and columns the asset and each number, the number of time where the two assets are in the same cluster
    market_name=df.columns
    n=len(market_name)
    
    cluster = louvain.label
    
    for i in range(n):
        index = np.where(cluster[i]==cluster)[0]
        for j in index:
            df.iloc[i,j]+=1
            
    return df
    
    
    
    
    
class daily_back_testing :
    
    def __init__(self,market_name,path,list_day0):
        #we initialize the value of each strategy
        self.V_vanilla=[100]
        self.V_louvain=[100]
        
        self.number_cluster=[]
        #we save the different day we use and the personal path to get the data and the market name
        #self.date_path=[day0]
        self.date_path=[list_day0[-1]]
        self.path_perso=path
        self.market_name=market_name
        n=len(market_name)
        #We create a list to save the return of each strategy
        self.louvain_return=[]
        self.vanilla_return=[]
        #We initialize the correlation matrix (goal to get the average correlation between each asset)
        self.correlation=np.zeros((n,n))
        #we create a matrix to know how many time each asset are together in louvain cluster
        temp1= np.zeros((n,n))
        self.louvain_cluster =pd.DataFrame(data=temp1,columns=market_name,index=market_name)
        #we initialize the number of time we will calculate the different return/strategy value ...
        self.nombre_test=0
        #we import the first data
        #self.data_j1=impor_data(market_name,day0,path)
        self.data_1 = []
        for d in list_day0 :
            temp1 = impor_data(market_name,d,path)
            if min([len(df) for df in temp1])>0: 
                self.data_1.append(harmoniz_data(temp1,d))
        
        
    def daily_update(self,day_j):
        '''
        

        Parameters
        ----------
        day_j : date of the day where we want to apply the stategy calibrate on the previous day 

        Returns
        -------
        None.

        '''
        
        #we import the data of the day 2 and make sure it is not empty
        data_j2 = impor_data(self.market_name,day_j,self.path_perso)
        
        if min([len(df) for df in data_j2])>0:  
            #we resample the stocked data and calibrate the strategies on it
            #data_harmonized_j1 = harmoniz_data(self.data_j1)
            
            data_calibrate = pd.concat(self.data_1,axis=0)
            #we calibrate the Louvain strategy and calcul the return and actualize 
            #the correlation matrix and the number of time assets are in the same cluster
            louvain = Louvain_GMVP(data_calibrate)
            louvain.get_return(data_j2,self.market_name)
            self.number_cluster.append(max(louvain.label)+1)
            self.louvain_return.append(louvain.retour)
            self.V_louvain.append(self.V_louvain[-1]* (1+louvain.retour))
            
            self.correlation+= louvain.correlation
            self.louvain_cluster = nombre_cluster(louvain,self.louvain_cluster)
            #we add one unit in the number of test
            self.nombre_test+=1
            
            #we calibrate the vanilla strategy and get the return
            vanilla_w,_,_ = get_GMVP(data_calibrate)
            self.vanilla_return.append(get_return_vanilla(vanilla_w,data_j2))
            self.V_vanilla.append(self.V_vanilla[-1]*(1+self.vanilla_return[-1]) )
            
            #we replace the a old data day 1 with the data from day 2 to calibrate the future strategy
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
    
    
    

















