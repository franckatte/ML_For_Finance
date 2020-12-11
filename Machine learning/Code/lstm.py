#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 14:39:20 2020

@author: franckatteaka
"""

from feature_engineering import supervised, reshape
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from keras.layers import Input, Dense, LSTM, Dropout,Bidirectional
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.wrappers.scikit_learn import KerasRegressor
from keras.optimizers import Adam
from tscpcv import CPCV


df = pd.read_csv('/Users/franckatteaka/Desktop/cours/Semester III/Courses Projects/Machine Learning/Data/data_clean.csv',sep = ","
                 ,parse_dates = True,index_col = 0 )

# 
times = [1,2,3]
# create features and transform the data in the supervised format
m_path = '/Users/franckatteaka/Desktop/cours/Semester III/Courses Projects/Machine Learning/Code/models/best_auto_encoder.hdf5'
X,y = supervised(df,growth_freqs = [20,40,60],backwards = times, scale_eco = True,denoise = True,model_path = m_path)


# split data into test/train
X_train0,X_test0,y_train0,y_test0 = train_test_split(X,y,test_size = 0.2, shuffle = False)
X_train,X_test,y_train,y_test = reshape(X_train0,backwards = times),reshape(X_test0,backwards = times),y_train0.to_numpy(),y_test0.to_numpy()

# model dimension
time_steps = len(times)
nb_features = int(X.shape[1]/time_steps)
output_dim = y.shape[1]

##create vanilla LSTM model

def vanilla_LSTM(learning_rate,size,dropout,activation1,activation2):
    # Create an Adam optimizer with the given learning rate
    opt = Adam(lr = learning_rate)
    # input data
    input_layer = Input(shape = (time_steps,nb_features))
    
    ## dropout layer
    Dropout(dropout)(input_layer)
    
    # hidden layers
    hidden =  LSTM(size , activation = activation1)(input_layer)
    
    # output layer
    output_layer = Dense(output_dim, activation = activation2)(hidden)
    
    # model
    lstm = Model(input_layer, output_layer)
    
    # compile the model
    lstm.compile(optimizer = opt, loss = 'mean_squared_error')

    return lstm

def stacked_LSTM(learning_rate,size1,size2,dropout,activation1,activation2,activation3):
    # Create an Adam optimizer with the given learning rate
    opt = Adam(lr = learning_rate)
    # input data
    input_layer = Input(shape = (time_steps,nb_features))
    
    ## dropout layer
    Dropout(dropout)(input_layer)
    
    # hidden layers
    hidden =  LSTM(size1 , activation = activation1, return_sequences=True)(input_layer)
    hidden =  LSTM(size2 , activation = activation2,)(hidden)
    
    # output layer
    output_layer = Dense(output_dim, activation = activation3)(hidden)
    
    # model
    lstm = Model(input_layer, output_layer)
    
     # compile the model
    lstm.compile(optimizer = opt, loss = 'mean_squared_error')
    
    return lstm

def bi_LSTM(learning_rate,size,dropout,activation1,activation2):
    # Create an Adam optimizer with the given learning rate
    opt = Adam(lr = learning_rate)
    # input data
    input_layer = Input(shape = (time_steps,nb_features))
    
    ## dropout layer
    Dropout(dropout)(input_layer)
    
    # hidden layers
    hidden =  Bidirectional(LSTM(size , activation = activation1, return_sequences=True))(input_layer)
    
    # output layer
    output_layer = Dense(output_dim, activation = activation2)(hidden)
    
    # model
    lstm = Model(input_layer, output_layer)
    
     # compile the model
    lstm.compile(optimizer = opt, loss = 'mean_squared_error')

    return lstm



#cpcv splits
cpcv = CPCV(X_train0, n_split = 6, n_folds = 2, purge = 60, embargo = 1)

### vanilla lstm

# Create a KerasRegressor
lstm1 = KerasRegressor(build_fn = vanilla_LSTM)

# Define the parameters to try out
params1 = {'size':[1,10,50,100],'activation1': ['softmax'],'activation2': ['linear', 'tanh']
           ,'batch_size': [10,50, 100, 200],'learning_rate': [0.01, 0.001],'epochs': [10,50,100, 200,500], 'dropout':[0.1,0.2,0.3]}


# Create a randomize search cv object passing in the parameters to try
random_search1 = RandomizedSearchCV(lstm1, param_distributions = params1, cv = cpcv,n_jobs = 4)

# Search for best combinations
random_search1.fit(X_train,y_train)

# results
random_search1.best_params_

## training parameters

learning_rate = 0.01
size = 50
epochs = 200
dropout = 0.2
batch_size = 50
activation1 = 'softmax'
activation2 = 'linear'
#test mse 3.7345090547719054e-06


# create model

lstm_vanilla =  vanilla_LSTM(learning_rate,size,dropout,activation1,activation2)

# checkpoint
modelCheckpoint1 = ModelCheckpoint(filepath = '/Users/franckatteaka/Desktop/cours/Semester III/Courses Projects/Machine Learning/Code/models/best_vanilla_lstm.hdf5',  save_best_only = True)


history1 = lstm_vanilla.fit(X_train, y_train,epochs, batch_size, 
                          validation_data=(X_test, y_test),callbacks = [modelCheckpoint1],verbose=1)

# evaluate
print('\n# Evaluate on test data')
results = lstm_vanilla.evaluate(X_test, y_test, batch_size=100)
print('test mse', results)

plt.figure(figsize = (10,6))
plt.plot(history1.epoch, np.array(history1.history['loss'])**0.5, label="train", color = "blue")
plt.plot(history1.epoch, np.array(history1.history['val_loss'])**0.5, label="test", color = "red")
plt.xlabel("epoch")
plt.ylabel("RMSE")
plt.legend()
plt.show()



### stacked LSTM

# Create a KerasRegressor
lstm2 = KerasRegressor(build_fn = stacked_LSTM)

# Define the parameters to try out
params2 = {'size':[1,10,50,100],'activation1': ['softmax'],'activation2': ['linear','softmax', 'tanh'],'activation3': ['linear','tanh']
           ,'batch_size': [10,50,100,200],'learning_rate': [0.01,0.001],'epochs': [10,50,100,200,500], 'dropout':[0.1,0.2,0.3]}


# Create a randomize search cv object passing in the parameters to try
random_search2 = RandomizedSearchCV(lstm2, param_distributions = params1, cv = cpcv,n_jobs = 4)

# Search for best combinations
random_search2.fit(X_train,y_train)

# results
random_search2.best_params_


## training parameters
learning_rate = 0.01
size1 = 50
size2 = 50
epochs = 200
dropout = 0.2
batch_size = 50
activation1 = 'softmax'
activation2 = 'linear'
activation3 = 
# create model


lstm_stacked =  stacked_LSTM(learning_rate,size1,size2,dropout,activation1,activation2,activation3)

# checkpoint
modelCheckpoint1 = ModelCheckpoint(filepath = '/Users/franckatteaka/Desktop/cours/Semester III/Courses Projects/Machine Learning/Code/models/best_vanilla_lstm.hdf5',  save_best_only = True)


history1 = lstm_vanilla.fit(X_train, y_train,epochs, batch_size, 
                          validation_data=(X_test, y_test),callbacks = [modelCheckpoint1],verbose=1)

# evaluate
print('\n# Evaluate on test data')
results = lstm_vanilla.evaluate(X_test, y_test, batch_size=100)
print('test mse', results)




plt.figure(figsize = (10,6))
plt.plot(history.epoch, np.array(history.history['loss'])**0.5, label="train", color = "blue")
plt.plot(history.epoch, np.array(history.history['val_loss'])**0.5, label="test", color = "red")
plt.xlabel("epoch")
plt.ylabel("RMSE")
plt.legend()
plt.show()
params = {'size':[1,2],'activation1': ['softmax','sigmoid', 'tanh'],'activation2': ['softmax','sigmoid', 'tanh'],'activation3':['linear', 'tanh'],
          'batch_size': [10,50, 100, 200],'learning_rate': [ 0.01, 0.001],'epochs': [10,50,100, 200,500], 'dropout':[0.1,0.2,0.3]}


