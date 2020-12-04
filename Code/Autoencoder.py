#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 13:19:05 2020

@author: franckatteaka
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from feature_engineering import supervised
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
## 

df = df = pd.read_csv('/Users/franckatteaka/Desktop/cours/Semester III/ML_for_finance/Data/data_clean.csv',sep = ","
                 ,parse_dates = True,index_col = 0 )




X = df.iloc[:,df.columns.str.contains('J')]
X_train_0, X_test_0 = train_test_split(X, test_size = 0.2)

X_train, X_test = X_train_0.to_numpy(), X_test_0.to_numpy()

## model dimension
input_dim = X_train.shape[1]
output_dim = X_train.shape[1]
encoding_dim = 2

## input data
input_layer = Input(shape = (input_dim,))

## "encoded" is the encoded representation of the inputs
encoded = Dense(encoding_dim * 2 , activation = 'tanh')(input_layer)
encoded = Dense(encoding_dim  , activation = 'linear')(input_layer)

## "decoded" is the lossy reconstruction of the input
decoded = Dense(output_dim, activation='tanh')(encoded)
decoded = Dense(output_dim * 2, activation='linear')(decoded)
decoded = Dense(output_dim, activation='linear')(decoded)

## model
autoencoder = Model(input_layer, decoded)

autoencoder.summary()


## training parameters
epochs = 1000
batch_size = 50

## compile the model

# Define a callbacks

monitor_val_loss = EarlyStopping(monitor='val_loss',  patience= 5)
modelCheckpoint = ModelCheckpoint('best_auto_encoder.hdf5', save_best_only = True)

# compile the model
autoencoder.compile(optimizer='adam', loss='mean_squared_error')
# fit the model

history = autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, shuffle=True, 
                          callbacks = [monitor_val_loss, modelCheckpoint],
                          validation_data=(X_test, X_test),verbose=1)



#  plot function
def yields_plot(X,X_pred):
    
    X2 = X.join(X_pred)
    terms = ['1J', '2J','3J', '4J', '5J', '6J','7J', '8J', '9J', '10J', '15J', '20J', '30J']
    for  i,term in enumerate(terms):
    
        X2.loc[:,[term, term + " pred"]].plot()
        plt.show()


# prediction train
X_pred = autoencoder.predict(X_train)
columns_preds = [ i + " pred" for i in list(X_train_0.columns)]
X_pred = pd.DataFrame(X_pred, index = X_train_0.index,columns = columns_preds )

yields_plot(X_train_0,X_pred)

# prediction test

X_pred = autoencoder.predict(X_test)
columns_preds = [ i + " pred" for i in list(X_test_0.columns)]
X_pred = pd.DataFrame(X_pred, index = X_test_0.index,columns = columns_preds )

yields_plot(X_test_0,X_pred)
