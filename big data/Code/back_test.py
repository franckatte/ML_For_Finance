# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 20:48:35 2020

@author: cbour
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from data_cleaning import load_trade
from refreshTime import harmoniz_data
from GMVP import get_GMVP,Louvain_GMVP,get_return_vanilla

    
    
    
def impor_data(market_name,day,path):
    #We creat a list to import the data
    sortie = [load_trade(market,day,path,is_compressed=True) for market in market_name]
    return sortie
    

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
    
    def __init__(self,market_name,path,day0):
        #we initialize the value of each strategy
        self.V_vanilla=[100]
        self.V_louvain=[100]
        #we save the different day we use and the personal path to get the data and the market name
        self.date_path=[day0]
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
        self.data_j1=impor_data(market_name,day0,path)
        
        
    def daily_update(self,day_j2):
        '''
        

        Parameters
        ----------
        day_j2 : date of the day where we want to apply the stategy calibrate on the previous day 

        Returns
        -------
        None.

        '''
        
        #we import the data of the day 2 and make sure it is not empty
        data_j2 = impor_data(self.market_name,day_j2,self.path_perso)
        
        if min([len(df) for df in data_j2])>0:  
            #we resample the stocked data and calibrate the strategies on it
            data_harmonized_j1 = harmoniz_data(self.data_j1)
            
            
            #we calibrate the Louvain strategy and calcul the return and actualize 
            #the correlation matrix and the number of time assets are in the same cluster
            louvain = Louvain_GMVP(data_harmonized_j1)
            louvain.get_return(data_j2,self.market_name)
            self.louvain_return.append(louvain.retour)
            self.V_louvain.append(self.V_louvain[-1]* (1+louvain.retour))
            
            self.correlation+= louvain.correlation
            self.louvain_cluster = nombre_cluster(louvain,self.louvain_cluster)
            #we add one unit in the number of test
            self.nombre_test+=1
            
            #we calibrate the vanilla strategy and get the return
            vanilla_w,_,_ = get_GMVP(data_harmonized_j1)
            self.vanilla_return.append(get_return_vanilla(vanilla_w,data_j2))
            self.V_vanilla.append(self.V_vanilla[-1]*(1+self.vanilla_return[-1]) )
            
            #we replace the a old data day 1 with the data from day 2 to calibrate the future strategy
            
            self.data_j1=data_j2
            self.date_path.append(day_j2)
        
        
    def plot_value(self):
        plt.figure(figsize=(13,10))
        #we plot the value of each strategy
        plt.plot(self.date_path,self.V_louvain,label='louvain Value')
        
        plt.plot(self.date_path,self.V_vanilla,label='Vanilla Value')
        plt.title('Evolution of each strategy value')
        plt.legend()
        plt.xlabel('date')
        plt.ylabel('Value')
        #we calibrate the x axis to print the day
        plt.gca().xaxis.set_major_locator(mdates.DayLocator())
        
        plt.gcf().autofmt_xdate()
        plt.savefig('figures/Value_Strategies.pdf')
        plt.show()
    
    
    

















