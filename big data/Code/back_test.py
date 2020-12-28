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
    sortie = [load_trade(market,day,path,is_compressed=True) for market in market_name]
    return sortie
    

def nombre_cluster(louvain,market_name):
    n=len(market_name)
    M = np.zeros((n,n))
    
    cluster = louvain.label
    for i in range(n-1):
        index = np.where(cluster[i]==cluster[i+1:])
        for j in index:
            M[i,j]+=1
            M[j,i]+=1
            
    return M
    
    
    
    
    
class daily_back_testing :
    
    def __init__(self,market_name,path,day0):
        self.V_vanilla=[100]
        self.V_louvain=[100]
        self.date_path=[day0]
        self.path_perso=path
        self.market_name=market_name
        n=len(market_name)
        self.correlation=np.zeros((n,n))
        self.louvain_cluster = np.zeros((n,n))
        self.nombre_test=0
        self.data_j1=impor_data(market_name,day0,path)
        
        
    def daily_update(self,day_j2):
        
        
        data_harmonized_j1 = harmoniz_data(self.data_j1)
        
        data_j2 = impor_data(self.market_name,day_j2,self.path_perso)
        
        louvain = Louvain_GMVP(data_harmonized_j1)
        self.V_louvain.append(self.V_louvain[-1]* louvain.get_return(data_j2,self.market_name))
        
        self.correlation+= louvain.correlation
        self.louvain_cluster = nombre_cluster(louvain,self.market_name)
        self.nombre+=1
        
        
        vanilla_w,_,_ = get_GMVP(data_harmonized_j1)
        self.V_vanilla.append(self.V_vanilla[-1]*get_return_vanilla(vanilla_w,data_j2) )
        
        self.data_j1=data_j2
        self.date_path.append(day_j2)
        
        
    def plot_value(self):
        plt.figure(figsize=(13,10))
        plt.plot(self.date_path,self.V_louvain,label='louvain Value')
        
        plt.plot(self.date_path,self.V_vanilla,label='Vanilla Value')
    
        plt.legend()
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        
        plt.gcf().autofmt_xdate()
        plt.show()
    
    
    

















