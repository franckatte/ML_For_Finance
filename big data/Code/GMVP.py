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

def get_Louvain_GMVP(data_harmonized):
    '''
    

    Function where we get the Louvain cluster, then we apply twice the GMVP
    ----------
    data_harmosized : Pandas data Frame of harmonized asset
        DESCRIPTION.

    Returns weight and performanceand label_clustering
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
    for i in range(nb_cluster+1):
        nom = names[Cluster==i]
        datai = data_harmonized[nom]
        wi,_,_ = get_GMVP(datai)
        Value_portfeuille = datai.values@wi
        Value_cluster.append(Value_portfeuille)
        names_cluster.append('portfolio '+str(i))
        
    
    Value_cluster = np.array(Value_cluster).T
    Data_cluster = pd.DataFrame(data = Value_cluster,columns=names_cluster,index=date)
        
    w_clust,mu_cluster,std_cluster = get_GMVP(Data_cluster)
    return w_clust,mu_cluster,std_cluster,Cluster
        
        
    















