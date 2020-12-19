# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 10:28:44 2020

@author: corentin
"""

import numpy as np
import pandas as pd
from feature_engineering import supervised, reshape
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.wrappers.scikit_learn import KerasRegressor
from tscpcv import CPCV
from model_generators import vanilla_LSTM,stacked_LSTM,bi_LSTM, plot_rmse, yields_rmse
from keras.models import load_model
from model_generators import plot_yields

from model_custom import LSTM_dropout_bilstm,custom1_LSTM_dropout






df = pd.read_csv('D:/GitHub/ML_For_Finance/Machine Learning/Data/data_clean.csv',sep = ",",parse_dates = True,index_col = 0 )
times = [1,2,3,4,5,6,7,8,9,10]


m_path = 'D:/GitHub/ML_For_Finance/Machine learning/Code/models/best_autoencoder.hdf5'
X,y = supervised(df,growth_freqs = [20,40,60],backwards = times, scale_eco = True,denoise = True,model_path = m_path,nb_years = 5)

# split data into test/train first keeping time index
X_train0,X_test0,y_train0,y_test0 = train_test_split(X,y,test_size = 0.2, shuffle = False)

# reshape regressors and convert objective to numpy
X_train,X_test,y_train,y_test = reshape(X_train0,backwards = times),reshape(X_test0,backwards = times),y_train0.to_numpy(),y_test0.to_numpy()

# model dimension
time_steps = len(times)
nb_features = int(X.shape[1]/time_steps)
output_dim = y.shape[1]


cpcv = CPCV(X_train0, n_split = 6, n_folds = 2, purge = 60, embargo = 1)

# callbacks
earlystopping = EarlyStopping(monitor="val_loss",patience = 10,restore_best_weights = True)

#### Model 1 custom : 1 Bivariate


lstmcustom = KerasRegressor(build_fn = LSTM_dropout_bilstm)

# Define the parameters to try out
params1 = {'time_steps':[time_steps],'nb_features':[nb_features],'output_dim':[output_dim],
           'size':np.arange(10,100,5),'activation1': ['softmax','relu','softsign','tanh'], 'activation2' : ['softmax','relu','softsign','tanh'],
           'activation3': ['linear', 'tanh','selu'],'epochs': [300]
           ,'batch_size': np.arange(10,200,10),'learning_rate': np.arange(0.001,0.3,0.001)
           ,'Dropout1':np.arange(0,0.5,0.05)
           }

# Create a randomize search cv object passing in the parameters to try
random_search1 = RandomizedSearchCV(lstmcustom, param_distributions = params1, cv = cpcv,n_jobs = 3,n_iter=100)

# Search for best combinations
random_search1.fit(X_train,y_train)

# results
random_search1.best_params_

learning_rate = 0.04857878787878789
size = 95
epochs = 350
batch_size = 50
activation1 = 'softmax'
activation2 = 'tanh'
activation3 = 'tanh'
Dropout1=0.2



# create model
lstm_custom1 =  LSTM_dropout_bilstm(time_steps,nb_features,output_dim,learning_rate,size,activation1,activation2,activation3,Dropout1)

# checkpoint
#modelCheckpoint1 = ModelCheckpoint(filepath = '/Users/franckatteaka/Desktop/cours/Semester III/Courses Projects/Machine Learning/Code/models/best_vanilla_lstm10.hdf5',  save_best_only = True)

history1 = lstm_custom1.fit(X_train, y_train,epochs = epochs, batch_size = batch_size, 
                          validation_data=(X_test, y_test)#,callbacks = [modelCheckpoint1]
                          ,verbose = 1)

# load best model
lstm_custom1 = load_model('D:/GitHub/ML_For_Finance/Machine Learning/Code/models/best_vanilla_lstm10.hdf5')

# evaluate
print('\n# Evaluate on test data')
results1 = lstm_custom1.evaluate(X_test, y_test, batch_size=100)
print('test mse', results1)

# plot loss
train_test_folder = 'D:/GitHub/ML_For_Finance/Machine Learning/Code/figures/LSTM//train_test/'

plot_rmse(history1,train_test_folder,'custom1_train_test_RMSE 10')

# RMSE per Maturity
## train
yields_rmse(lstm_custom1,X_train,y_train)

##test
yields_rmse(lstm_custom1,X_test,y_test)



#### Model 2 custom : 1 LSTM Dropout+ 1 Bivariate Dropout


lstmcustom2 = KerasRegressor(build_fn = custom1_LSTM_dropout)

# Define the parameters to try out
params2 = {'time_steps':[time_steps],'nb_features':[nb_features],'output_dim':[output_dim],
           'size':np.arange(10,100,5),'activation1': ['softmax','relu','softsign','tanh'], 'activation2' : ['softmax','relu','softsign','tanh'],
           'activation3': ['linear', 'tanh','selu'],'epochs': [300]
           ,'batch_size': np.arange(10,200,10),'learning_rate': np.arange(0.001,0.3,0.001)
           ,'Dropout1':np.arange(0,0.5,0.05),'Dropout2':np.arange(0,0.5,0.05)
           }

# Create a randomize search cv object passing in the parameters to try
random_search2 = RandomizedSearchCV(lstmcustom2, param_distributions = params2, cv = cpcv,n_jobs = 3,n_iter=100)

# Search for best combinations
random_search2.fit(X_train,y_train)

# results
random_search2.best_params_

learning_rate = 0.01
size = 95
epochs = 350
batch_size = 50
activation1 = 'softmax'
activation2 = 'tanh'
activation3 = 'tanh'
Dropout1=0.2
Dropout2=0.2


# create model
lstm_custom2 =  custom1_LSTM_dropout(time_steps,nb_features,output_dim,learning_rate,size,activation1,activation2,activation3,Dropout1,Dropout2)

# checkpoint
#modelCheckpoint1 = ModelCheckpoint(filepath = '/Users/franckatteaka/Desktop/cours/Semester III/Courses Projects/Machine Learning/Code/models/best_vanilla_lstm10.hdf5',  save_best_only = True)

history2 = lstm_custom2.fit(X_train, y_train,epochs = epochs, batch_size = batch_size, 
                          validation_data=(X_test, y_test)#,callbacks = [modelCheckpoint1]
                          ,verbose = 1)

# load best model
lstm_custom2 = load_model('D:/GitHub/ML_For_Finance/Machine Learning/Code/models/best_vanilla_lstm10.hdf5')

# evaluate
print('\n# Evaluate on test data')
results1 = lstm_custom1.evaluate(X_test, y_test, batch_size=100)
print('test mse', results1)

# plot loss
train_test_folder = 'D:/GitHub/ML_For_Finance/Machine Learning/Code/figures/LSTM//train_test/'

plot_rmse(history2,train_test_folder,'bilstm_drop_train_test_RMSE 10')

# RMSE per Maturity
## train
yields_rmse(lstm_custom2,X_train,y_train)

##test
yields_rmse(lstm_custom2,X_test,y_test)



























