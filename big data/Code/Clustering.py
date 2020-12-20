# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 15:59:19 2020

@author: corentin
"""

#pip install python-louvain
import numpy as np
import pandas as pd
import networkx as nx
import community as community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt

#pip install scikit-network

from sknetwork.clustering import Louvain



def zero_diag_matrix(M):
    n=len(M)
    
    for i in range(n):
        M.iloc[i,i]=0
        
    return M


def louvain_label(corr_matrix):
    '''
        return label of Louvain clustering
        ------------
        corr_matrix(DataFrame): correlation matrix of the market DataFrame
        
        Return
        ------------
        Label(DataFrame): columns : market / row : label
        
    '''
    
    #input : correlation matrix of the market DataFrame
    #
    #return : the graph with weight of each link
    
    Corr = zero_diag_matrix(corr_matrix)
    
    louvain = Louvain()
    labels = louvain.fit_transform(Corr)
    market_name=Corr.columns
    DF = pd.DataFrame(data=labels,index=['labels'],columns = market_name)
    
    return DF




    
    
    
    
    
    
    
    
    
    
    













