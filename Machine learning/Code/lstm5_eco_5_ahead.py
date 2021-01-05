#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 09:59:57 2020

@author: franckatteaka
"""


from feature_engineering import supervised, reshape
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from keras.callbacks import ModelCheckpoint
from keras.wrappers.scikit_learn import KerasRegressor
from tscpcv import CPCV
from model_generators import vanilla_LSTM,stacked_LSTM,bi_LSTM, plot_rmse, yields_rmse
from keras.models import load_model

from model_generators import plot_yields

# folders paths
df_path = '/Users/franckatteaka/Desktop/cours/Semester III/Courses Projects/Machine Learning/Data/data_clean.csv'
autoencoder_path = '/Users/franckatteaka/Desktop/cours/Semester III/Courses Projects/Machine Learning/Code/models/best_autoencoder.hdf5'
model_folder = '/Users/franckatteaka/Desktop/cours/Semester III/Courses Projects/Machine Learning/Code/models/eco models/'
predictions_folder = '/Users/franckatteaka/Desktop/cours/Semester III/Courses Projects/Machine Learning/Code/figures/LSTM/eco models/predictions/'
train_test_folder = '/Users/franckatteaka/Desktop/cours/Semester III/Courses Projects/Machine Learning/Code/figures/LSTM/eco models/train_test/'

# loading data
df = pd.read_csv(df_path,sep = ",",parse_dates = True,index_col = 0 )
times = [5,6,7,8,9]


# create features and transform the data in the supervised format
X,y = supervised(df,growth_freqs = [20,40,60],backwards = times, scale_eco = True,denoise = True,model_path = autoencoder_path,nb_years = 5)

# split data into test/train first keeping time index
X_train0,X_test0,y_train0,y_test0 = train_test_split(X,y,test_size = 0.2, shuffle = False)

# reshape regressors and convert objective to numpy
X_train,X_test,y_train,y_test = reshape(X_train0,backwards = times),reshape(X_test0,backwards = times),y_train0.to_numpy(),y_test0.to_numpy()

# model dimension
time_steps = len(times)
nb_features = int(X.shape[1]/time_steps)
output_dim = y.shape[1]

#cpcv splits
cpcv = CPCV(X_train0, n_split = 7, n_folds = 2, purge = 60)

### models

## vanilla lstm
# Create a KerasRegressor
lstm1 = KerasRegressor(build_fn = vanilla_LSTM)

# Define the parameters to try out
params1 = {'time_steps':[time_steps],'nb_features':[nb_features],'output_dim':[output_dim],
           'size':[13,50,100],'activation1': ['softmax'],
           'activation2': ['linear', 'tanh']
           ,'batch_size': [50,100],'learning_rate': [0.01, 0.001],
           'epochs': [200,500]}

# Create a randomize search cv object passing in the parameters to try
random_search1 = RandomizedSearchCV(lstm1, param_distributions = params1, cv = cpcv,n_jobs = -1)

# Search for best combinations
random_search1.fit(X_train,y_train)

# results
random_search1.best_params_

## training parameters
learning_rate = 0.01
size = 50
epochs = 200
batch_size = 10
activation1 = 'softmax'
activation2 = 'linear'

# create model
lstm_vanilla =  vanilla_LSTM(time_steps,nb_features,output_dim,learning_rate,size,activation1,activation2)

# checkpoint
modelCheckpoint1 = ModelCheckpoint(filepath = model_folder + 'best_vanilla_lstm 5 ahead.hdf5',  save_best_only = True)

history1 = lstm_vanilla.fit(X_train, y_train,epochs = epochs, batch_size = batch_size, 
                          validation_data=(X_test, y_test),callbacks = [modelCheckpoint1],verbose = 1)

# load best model
lstm_vanilla = load_model(model_folder + 'best_vanilla_lstm 5 ahead.hdf5')

# evaluate
print('\n# Evaluate on test data')
results1 = lstm_vanilla.evaluate(X_test, y_test, batch_size= batch_size)
print('test mse', results1)

# plot loss

plot_rmse(history1,train_test_folder,'vanilla_train_test_RMSE 5 ahead')

# RMSE per Maturity
## train
yields_rmse(lstm_vanilla,X_train,y_train)

##test
yields_rmse(lstm_vanilla,X_test,y_test)

## stacked LSTM
# Create a KerasRegressor
lstm2 = KerasRegressor(build_fn = stacked_LSTM)

# Define the parameters to try out
params2 = {'time_steps':[time_steps],'nb_features':[nb_features],'output_dim':[output_dim],'size1':[13,50],'size2':[13,50],'activation1': ['softmax'],'activation2': ['linear','softmax', 'tanh'],'activation3': ['linear','tanh']
           ,'batch_size': [10,50,100],'learning_rate': [0.01,0.001],'epochs': [200,500]}


# Create a randomize search cv object passing in the parameters to try
random_search2 = RandomizedSearchCV(lstm2, param_distributions = params2, cv = cpcv,n_jobs = -1)

# Search for best combinations
random_search2.fit(X_train,y_train)

# results
random_search2.best_params_


## training parameters

learning_rate = 0.01
size1 = 13
size2 = 13
epochs = 200
batch_size = 10
activation1 = 'softmax'
activation2 = 'softmax'
activation3 = 'linear'
# create model

lstm_stacked =  stacked_LSTM(time_steps,nb_features,output_dim,learning_rate,size1,size2,activation1,activation2,activation3)

# checkpoint
modelCheckpoint2 = ModelCheckpoint(filepath = model_folder + 'best_stacked_lstm 5 ahead.hdf5',  save_best_only = True)


history2 = lstm_stacked.fit(X_train, y_train,epochs = epochs, batch_size = batch_size, 
                          validation_data=(X_test, y_test),callbacks = [modelCheckpoint2],verbose = 1)

# load best model
lstm_stacked = load_model(model_folder + 'best_stacked_lstm 5 ahead.hdf5')

# evaluate
print('\n# Evaluate on test data')
results2 = lstm_stacked.evaluate(X_test, y_test, batch_size = batch_size)
print('test mse', results2)

# plot loss
plot_rmse(history2,train_test_folder,'stacked_train_test_RMSE 5 ahead')

# RMSE per Maturity
## train
yields_rmse(lstm_stacked,X_train,y_train)

##test
yields_rmse(lstm_stacked,X_test,y_test)

## bidirectional LSTM

# Create a KerasRegressor
lstm3 = KerasRegressor(build_fn = bi_LSTM)

# Define the parameters to try out
params3 = {'time_steps':[time_steps],'nb_features':[nb_features],'output_dim':[output_dim],
           'size':[13,50,100],'activation1': ['softmax'],
           'activation2': ['linear', 'tanh']
           ,'batch_size': [10,50,100],'learning_rate': [0.01, 0.001],
           'epochs': [200,500]}

# Create a randomize search cv object passing in the parameters to try
random_search3 = RandomizedSearchCV(lstm3, param_distributions = params3, cv = cpcv,n_jobs = 4)

# Search for best combinations
random_search3.fit(X_train,y_train)

# results
random_search3.best_params_


## training parameters

learning_rate = 0.01
size = 13
epochs = 500
batch_size = 10
activation1 = 'softmax'
activation2 = 'linear'


# create model
lstm_bidirect =  bi_LSTM(time_steps,nb_features,output_dim,learning_rate,size,activation1,activation2)

# checkpoint
modelCheckpoint3 = ModelCheckpoint(filepath = model_folder + 'best_bidirectional_lstm 5 ahead.hdf5',  save_best_only = True)


history3 = lstm_bidirect.fit(X_train, y_train,epochs = epochs, batch_size = batch_size, 
                          validation_data=(X_test, y_test),callbacks = [modelCheckpoint3],verbose=1)

# load best model
lstm_bidirect = load_model(model_folder + 'best_bidirectional_lstm 5 ahead.hdf5')

# evaluate
print('\n# Evaluate on test data')
results3 = lstm_bidirect.evaluate(X_test, y_test, batch_size= batch_size)
print('test mse', results3)

#plot loss
plot_rmse(history3,train_test_folder,'bidirect_train_test_RMSE 5 ahead')

# RMSE per Maturity
## train
yields_rmse(lstm_bidirect,X_train,y_train)

##test
yields_rmse(lstm_bidirect,X_test,y_test)


## yields predictions plots 


plot_yields(lstm_vanilla,X_test,y_test0,predictions_folder,'vanilla 5 ahead')
plot_yields(lstm_stacked,X_test,y_test0,predictions_folder,'stacked 5 ahead')
plot_yields(lstm_bidirect,X_test,y_test0,predictions_folder,'bidirect 5 ahead')


