#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 20:21:43 2020

@author: franckatteaka
"""
from itertools import combinations
import numpy as np


def purge_embargoCV(X,train_folds,test_folds,purge,embargo):

    # train/test set
    X_train =  X.loc[X['group'].isin(train_folds)].drop('group',axis = 1).copy() 

    # purge and embargo after each test fold
    
    for test_fold in test_folds:
        
        test = X.loc[X['group'] == test_fold].drop('group',axis = 1).copy()

        # most recent date of the fold
        max_date = max(test.index)
        
        # index in the full dataset
        max_index = np.where(X.index == max_date)[0][0]
   
        # after which train data should be kept, define embargo and purge period
        index_after = max_index + purge + embargo
        
        # update train/test sets
        X_train = X_train.loc[~X_train.index.isin(X.index[max_index+1:index_after+1])].copy()
       
    X_test = X.loc[~X['group'].isin(train_folds)].drop('group',axis = 1).copy()

    return np.array(X_train.index), np.array(X_test.index)

  
def missing(group,indices):
    
    res = [*filter(lambda x: False if x in group else True,indices)]
    return res


def CPCV(X,Y,n_split,n_folds,purge,embargo):
    
    X2 = X.copy()
 
    subset_size = X2.shape[0]// n_split
   
    # indexes of splitting observation
    indexes = [subset_size * i for i in range(1,n_split)]
    indexes.append(X2.shape[0])
    indexes.reverse()
    
    # subset ids
    splits_ids = np.arange(n_split)
    
    # set which subset and observation belongs to
    groups = np.zeros(X2.shape[0])
    
    for i in range(X2.shape[0]):
        
        for j,index in enumerate(indexes):
            if i < index:
                groups[i] = splits_ids[::-1][j]  
            
    X2['group'] = groups

    train_groups = list(combinations(splits_ids,n_split - n_folds))
    
    
    # saved train test indexes
    results = []
    full_date = np.array(X2.index)

    for train_folds in train_groups:
        
        test_folds = missing(train_folds,splits_ids)
        
        train_index, test_index = purge_embargoCV(X2,train_folds,test_folds,purge,embargo)
        
    
        # convert time index into positional index
        train_pos = [np.where(full_date == d )[0][0] for d in train_index]
        test_pos = [np.where(full_date == d )[0][0] for d in test_index]
        
        results.append((train_pos,test_pos))
    
    return results


# =============================================================================
# class CPCV:
#     
#     def __init__(self,n_split,n_folds,purge,embargo,backwards):
#     
#     def get_splits(X)
#         
# 
# =============================================================================
