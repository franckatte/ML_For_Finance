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
from keras.layers import Input, Dense, LSTM, Dropout
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.wrappers.scikit_learn import KerasRegressor
from keras.optimizers import Adam

from sklearn import preprocessing
from tscpcv import CPCV


df = pd.read_csv('/Users/franckatteaka/Desktop/cours/Semester III/Courses Projects/Machine Learning/Data/data_clean.csv',sep = ","
                 ,parse_dates = True,index_col = 0 )

# scale economic variables
df.iloc[:,~df.columns.str.contains('J')] = df.iloc[:,~df.columns.str.contains('J')].apply(preprocessing.scale).copy()

# 
times = [1,2,3]
# create features and transform the data in the supervised format
X,y = supervised(df,growth_freqs = [30,60],backwards = times)

# reshape the data  
X = reshape(X,backwards = times)
y = y.to_numpy()

# split data into test/train
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, shuffle = False)

# model dimension
time_steps = 3
nb_features = int(X.shape[1]/time_steps)
output_dim = y.shape[1]

##create LSTM model

# input data
input_layer = Input(shape = (time_steps,nb_features))
    
# hidden layers
hidden =  LSTM(50 , activation = 'softmax')(input_layer)

# output layer
output_layer = Dense(output_dim, activation = 'tanh')(hidden)

# model
lstm = Model(input_layer, output_layer)

cv = CPCV(X, y, n_split = 6, n_folds = 2, purge = 20, embargo = 1, backwards = times)


history = lstm.fit(X_train, y_train, epochs = 500, batch_size= 200, 
                          validation_data=(X_test, y_test),verbose=1)



plt.figure(figsize = (10,6))
plt.plot(history.epoch, np.array(history.history['loss'])**0.5, label="train", color = "blue")
plt.plot(history.epoch, np.array(history.history['val_loss'])**0.5, label="test", color = "red")
plt.xlabel("epoch")
plt.ylabel("RMSE")
plt.legend()
plt.show()
