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
    #We put zeros on the matrix diagonal
    n=len(M)
    
    for i in range(n):
        M[i,i]=0
        
    return M


def louvain_label(corr_matrix):
    '''
        return label of Louvain clustering
        ------------
        corr_matrix(numpy): correlation matrix of the market array
        
        Return
        ------------
        Label(Numpy): array of the louvain label
        
    '''
    
    Corr = zero_diag_matrix(corr_matrix) #we put 0 on the diagonal to avoid intern edge
    
    louvain = Louvain()
    labels = louvain.fit_transform(Corr)
    
    
    
    return labels




    
    
    
def get_clusters(C,q):
    '''
    

    Parameters
    ----------
    C : correlation matrix
    q : number of stock on the number of refresh time

    Returns
    -------
    cluster : the label of the louvain clustering

    '''
    
    names = C.columns
    #We define the theoritical limite of the correlation
    lbda_p = (1 + np.sqrt(q))**2
    lbda_m = (1 - np.sqrt(q))**2
    
    eigen_val, eigen_vec = np.linalg.eig(C.values)
    
    # random mode
    index = np.where(eigen_val <= lbda_p)[0]
    lbda_random = eigen_val[index]
    mu_random = eigen_vec[index]
    
    C_random = np.sum([lbda_random[i]*mu_random[i].T@mu_random[i] for i in range(len(index))  ])
    
    # market mode
    index = np.where(eigen_val == max(eigen_val))[0]
    lbda_m = eigen_val[index]
    mu_m = eigen_vec[index]
    
    C_market = lbda_m*mu_m.T@mu_m
    
    # We compute the filtered correlation matrix
    C0 = C_random+C_market
    
    C0=np.absolute(C0)
    
    cluster = louvain_label(C0)
    
    return cluster
    
    
    
    
    
    
    
    













