#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 20:21:43 2020

@author: franckatteaka
"""
from feature_engineering import reshape
from itertools import combinations
import numpy as np


def purge_embargo(X,y,train_folds,test_folds,purge,embargo):
    
    # train/test set
    X_train =  X.loc[X['group'].isin(train_folds)].drop('group',axis = 1).copy() 
    y_train = y.loc[y['group'].isin(train_folds)].drop('group',axis = 1).copy()
    
    # purge and embargo after each test fold
    for test_fold in test_folds:
        
        test = X.loc[X['group'] == test_fold].drop('group',axis = 1).copy()
        # most recent date of the fold
        max_date = max(test.index)
        # index in the full dataset
        max_index = np.where(X.index == max_date)[0][0]
    
        # most old date of the fold
        min_date = min(test.index)
    
        # after which train data should be kept, define embargo and purge period
        date_after = X.index[max_index + purge + embargo]
        
        # update train/test sets
        X_train = X_train[(X_train.index < min_date) | (X_train.index > date_after)].copy()
        y_train = y_train[(y_train.index < min_date) | (y_train.index > date_after)].copy()
        
    X_test = X.loc[~X['group'].isin(train_folds)].drop('group',axis = 1).copy()
    y_test = y.loc[~y['group'].isin(train_folds)].drop('group',axis = 1).copy()
    
    return X_train, X_test, y_train, y_test

  
def missing(group,indices):
    
    res = [*filter(lambda x: False if x in group else True,indices)]
    return res

def cpcv(x,y,n_split,n_folds,purge,embargo,backwards,model,epochs, batch_size,loss,optimizer):
    
    y2 = y.copy()
    x2 = x.copy()
 
    
    subset_size = x2.shape[0]// n_split
   
    # indexes of splitting observation
    indexes = [subset_size * i for i in range(1,n_split)]
    indexes.append(x2.shape[0])
    indexes.reverse()
    
    # subset ids
    splits_ids = np.arange(n_split)
    
    # set which subset and observation belongs to
    groups = np.zeros(x2.shape[0])
    
    for i in range(x2.shape[0]):
        
        for j,index in enumerate(indexes):
            if i < index:
                groups[i] = splits_ids[::-1][j]  
            
    x2['group'] = groups
    y2['group'] = groups

    train_groups = list(combinations(splits_ids,n_split - n_folds))
    
    # train/test on the different combinations
    test_results = []
    for train_folds in train_groups:
        
        X_train, X_test, y_train, y_test = purge_embargo(x2,y2,train_folds,missing(train_folds,splits_ids),purge,embargo)
        
        X_train, X_test = reshape(X_train,backwards),reshape(X_test,backwards)
        y_train, y_test = y_train.to_numpy(), y_test.to_numpy()
        

        # compile model
        model.compile(optimizer = optimizer, loss = loss)
        #fit model
        model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size, validation_data=(X_test, y_test),verbose=1)
        
        # evaluate the model
        test_results.append(model.evaluate(X_test, y_test))
        
        
    cv_score = np.array(test_results).mean()

    
    return cv_score
    

