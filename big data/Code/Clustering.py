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


def zero_diag_matrix(M):
    n=len(M)
    
    for i in range(n):
        M.iloc[i,i]=0
        
    return M


def creat_graph(corr_matrix):
    '''
        return DataFrame of synchronise data
        ------------
        corr_matrix(DataFrame): correlation matrix of the market DataFrame
        
        Return
        ------------
        G(networkx Graph): graph unidirect with the market and the weight are the correlation
        
    '''
    
    #input : correlation matrix of the market DataFrame
    #
    #return : the graph with weight of each link
    
    Corr = zero_diag_matrix(corr_matrix)
    
    G = nx.from_numpy_matrix(Corr.values,create_using=nx.Graph())
    label_mapping = {idx: val for idx, val in enumerate(Corr.columns)}
    G = nx.relabel_nodes(G, label_mapping)
    
    return G

def get_clusters(G):
    '''
        return DataFrame with cluster_label
        ------------
        G(networkx Graph): graph unidirect with the market and the weight are the correlation
        
        Return
        ------------
        Label(DataFrame): columns : market / row : label
        
    '''
    partition = community_louvain.best_partition(G)
    
    return partition


    
    
    
    
    
    
    
    
    
    
    













