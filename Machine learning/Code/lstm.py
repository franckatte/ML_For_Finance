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
from sklearn.model_selection import RandomizedSearchCV, KFold

from keras.layers import Input, Dense, LSTM, Dropout
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.wrappers.scikit_learn import KerasRegressor
from keras.optimizers import Adam

from sklearn import preprocessing

from cpcv import cpcv


df = df = pd.read_csv('/Users/franckatteaka/Desktop/cours/Semester III/ML_for_finance/Data/data_clean.csv',sep = ","
                 ,parse_dates = True,index_col = 0 )


# scale economic variables
df.iloc[:,~df.columns.str.contains('J')] = df.iloc[:,~df.columns.str.contains('J')].apply(preprocessing.scale).copy()

X,y = supervised(df,growth_freqs = [30,60],backwards = [1,2,3])

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

cpcv(X, y, n_split = 6, n_folds = 2, purge = 0, embargo = 0, backwards = [1,2,3], model = lstm, epochs = 100, batch_size = 50, loss = 'mse', optimizer = 'Adam')

history = lstm.fit(X_train, y_train, epochs = 500, batch_size= 200, 
                          validation_data=(X_test, y_test),verbose=1)



plt.figure(figsize = (10,6))
plt.plot(history.epoch, np.array(history.history['loss'])**0.5, label="train", color = "blue")
plt.plot(history.epoch, np.array(history.history['val_loss'])**0.5, label="test", color = "red")
plt.xlabel("epoch")
plt.ylabel("RMSE")
plt.legend()
plt.show()
