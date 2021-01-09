#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 13:19:05 2020

@author: franckatteaka
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from keras.callbacks import ModelCheckpoint
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import RandomizedSearchCV
from tscpcv import CPCV
from model_generators import create_denoising_ae,plot_rmse,plot_yields,yields_rmse
from feature_engineering import sub_range

from keras.models import load_model

# import data
df = pd.read_csv('/Users/franckatteaka/Desktop/cours/Semester III/Courses Projects/Machine Learning/Data/data_clean.csv',sep = "," ,parse_dates = True,index_col = 0 )
df = sub_range(df,5)

train_test_folder = '/Users/franckatteaka/Desktop/cours/Semester III/Courses Projects/Machine Learning/Code/figures/autoencoder/'


X = df.iloc[:,df.columns.str.contains('J')]

X_train_0, X_test_0 = train_test_split(X, test_size = 0.2,shuffle = False)

X_train, X_test = X_train_0.to_numpy(), X_test_0.to_numpy()

## model dimension
input_dim = X_train.shape[1]
output_dim = X_train.shape[1]

    
# Create a KerasRegressor
model = KerasRegressor(build_fn = create_denoising_ae)

# Define the parameters to try out
params = {'input_dim':[input_dim], 'output_dim':[output_dim],'encoding_dim':[2,3],'activation1': ['softmax','linear','tanh','elu'],'activation2':['linear', 'tanh'],'batch_size': [10,50,100,200],
          'learning_rate': [0.01, 0.001],'epochs': [10,50,100,200,500], 'std':[0.001,0.002,0.003]}

# create the CPCV folds indexes
cpcv = CPCV(X_train_0, n_split = 6, n_folds = 2, purge = 0)

# Create a randomize search cv object passing in the parameters to try
random_search = RandomizedSearchCV(model, param_distributions = params, cv = cpcv,n_jobs = -1)


# Search for best combinations
random_search.fit(X_train,X_train)

# results
random_search.best_params_



learning_rate = 0.01
epochs = 500
encoding_dim = 3
std = 0.001
batch_size = 10
activation2 = 'tanh'
activation1 = 'linear'


# create model
autoencoder =  create_denoising_ae(input_dim, output_dim,learning_rate,std,encoding_dim,activation1,activation2)

# summary the model
autoencoder.summary()

# Define a callback
#modelCheckpoint = ModelCheckpoint(filepath = '/Users/franckatteaka/Desktop/cours/Semester III/Courses Projects/Machine Learning/Code/models/best_autoencoder2.hdf5',  save_best_only = True)

# fit the model
history = autoencoder.fit(X_train, X_train, epochs = epochs, batch_size = batch_size, 
                          callbacks = [modelCheckpoint],
                          validation_data=(X_test, X_test),verbose = 1)

# load best model
autoencoder = load_model('/Users/franckatteaka/Desktop/cours/Semester III/Courses Projects/Machine Learning/Code/models/best_autoencoder.hdf5')

# evaluate
print('\n# Evaluate on test data')
results = autoencoder.evaluate(X_test, X_test, batch_size=100)
print('test mse', results)

## Plot RMSE
plot_rmse(history,train_test_folder,'train_test_RMSE')

## RMSE per maturity
yields_rmse(autoencoder,X_train,X_train)

yields_rmse(autoencoder,X_test,X_test)

# prediction train
plot_yields(autoencoder,X_train_0,X_train_0,train_test_folder,"train")

# prediction test
plot_yields(autoencoder,X_test_0,X_test_0,train_test_folder,"test")




