# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 14:33:32 2020

@author: cbour
"""


import pandas as pd
import numpy as np
from scipy.optimize import minimize
from Clustering import get_clusters

def minimize_portfolio(w,Sigma):
    return w.T@Sigma@w

def get_GMVP(data_harmonized):
    '''
    

    function to caculate the GMVP
    ----------
    data_harmosized : Pandas data Frame of harmonized asset

    Returns portfolio weight
    -------

    '''
    return_data = data_harmonized- data_harmonized.shift(1)
    return_data=return_data.dropna()
    Covariance = return_data.cov()
    
    cons = ({'type': 'eq', 'fun': lambda x:  np.sum(x)-1.0})
    n=len(Covariance)
    w0 = np.ones(n)/n
    res= minimize(minimize_portfolio, w0, args=Covariance, method='SLSQP',constraints=cons)
    
    w = res.x
    mean = np.mean(return_data,axis=0)@w
    standart_deviation = w.T@Covariance@w
    
    return w,mean,standart_deviation

def get_profit_vanilla(w,dfs):
    '''
    
    ----------
    w : array weight of the GMVP
    dfs : list of asset price

    profit of the strategy
    -------

    '''
    profit = 0
    i=0
    for df in dfs :
            
        profit+= (df.iloc[-1].values - df.iloc[0].values)/(df.iloc[0].values)*w[i]
        
    return profit
            


def get_Louvain_GMVP(data_harmonized):
    '''
    

    Function where we get the Louvain cluster, then we apply twice the GMVP
    ----------
    data_harmosized : Pandas data Frame of harmonized asset
        DESCRIPTION.

    Returns weight and performance and label_clustering
    -------

    '''
    names = data_harmonized.columns
    date = data_harmonized.index
    q = len(names)/len(date)
    C = data_harmonized.corr()
    
    Cluster = get_clusters(C,q)
    
    nb_cluster = max(Cluster)
    Value_cluster = []
    names_cluster=[]
    weight_louvain=[]
    
    
    for i in range(nb_cluster+1):
        nom = names[Cluster==i]
        datai = data_harmonized[nom]
        wi,_,_ = get_GMVP(datai)
        
        
        WI = pd.DataFrame(data=wi[np.newaxis,:],columns=nom,index=['weight'])
        
        weight_louvain.append(WI)
        Value_portfeuille = datai.values@wi
        Value_cluster.append(Value_portfeuille)
        
        names_cluster.append('portfolio '+str(i))
        
    
    Value_cluster = np.array(Value_cluster).T
    Data_cluster = pd.DataFrame(data = Value_cluster,columns=names_cluster,index=date)
        
    w_clust,mu_cluster,std_cluster = get_GMVP(Data_cluster)
    return w_clust,mu_cluster,std_cluster,Cluster,weight_louvain
        
        
    
class Louvain_GMVP :
    '''
    class to define the weigh and label of the louvain strategy with GMVP
    
    in initialization we get the label, the weights in each cluster and for each cluster
    
    in get_profit we get the profit of the strategy from a list of asset price
    
    '''
    
    def __init__(self,data_harmonized):
        w_clust,mu_cluster,std_cluster,Cluster,weight_louvain = get_Louvain_GMVP(data_harmonized)
        
        self.w_cluster = w_clust
        self.mu_cluster = mu_cluster
        self.std_cluster=std_cluster
        self.label = Cluster
        self.w_louvain = weight_louvain
        
        
    def get_return(self,dfs,names):
        
        nb_clust = self.label
        retour = 0
        
        for df in dfs :
            nom = df.columns[0]
            label = self.label[np.where(nom==names)[0]][0]
            
            retour+= (df.iloc[-1].values - df.iloc[0].values)/(df.iloc[0].values)*self.w_cluster[label]*self.w_louvain[label][nom].values
            
        self.retour = retour
        
        

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_cleaning import load_trade
from refreshTime import refresh_time, test_date, resample,synchro_data,harmoniz_data
from Clustering import get_clusters
from GMVP import get_GMVP,get_Louvain_GMVP

Date = pd.bdate_range('2010-01-01','2010-12-31')
Market_name = np.array(['AAPL.OQ','AMGN.OQ','AXP.N','BA.N','CAT.N','CSCO.OQ','CVX.N','DOW.N','GS.N','SPY.P','UTX.N','V.N','WMT.N'])

#folder_path = 'D:/GitHub/ML_For_Finance/big data/data/data/'

folder_path ='D:/GitHub/ML for Finance project/ML_For_Finance/big data/Data/data/'
#folder_path = '/Users/franckatteaka/Desktop/cours/Semester III/Financial big data/high freq data/'

test2=[]
for i in range(len(Market_name)):
    data0=load_trade(Market_name[i],Date[7],folder_path,is_compressed = True)
    test2.append(data0)

data1=harmoniz_data(test2)

louvain_test = Louvain_GMVP(data1)

louvain_test.w_louvain
louvain_test.get_profit(test2,Market_name)

louvain_test.profit


'''















