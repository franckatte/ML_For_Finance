# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 08:53:44 2020

@author: corentin
"""
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Input, Dense, LSTM, Dropout,Bidirectional, BatchNormalization, GaussianNoise
from keras.models import Model
from keras.layers.core import Activation, Dropout, Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping


# callbacks
earlystopping = EarlyStopping(monitor="val_loss",patience = 10,restore_best_weights = True)


def LSTM_dropout_bilstm(time_steps,nb_features,output_dim,learning_rate,size,activation1,activation2,activation3,Dropout1=0):
    # Create an Adam optimizer with the given learning rate
    opt = Adam(lr = learning_rate)
    # input data
    
    model = Sequential()
    
    model.add(LSTM(size, activation=activation1,return_sequences=True,
                                 input_shape=(time_steps,nb_features)))
    model.add(Dropout(Dropout1))
    
    model.add(Bidirectional(LSTM(size, activation=activation2)))
    
    model.add(Dense(output_dim))
    model.add(Activation(activation3))


    
     # compile the model
    model.compile(optimizer = opt, loss = 'mean_squared_error',metrics=['mse'])

    return model


def custom1_LSTM_dropout(time_steps,nb_features,output_dim,learning_rate,size,activation1,activation2,activation3,Dropout1=0,Dropout2=0):
    # Create an Adam optimizer with the given learning rate
    opt = Adam(lr = learning_rate)
    # input data
    
    model = Sequential()
    
    model.add(Bidirectional(LSTM(size, activation=activation1,return_sequences=True),
                                 input_shape=(time_steps,nb_features)))
    model.add(Dropout(Dropout1))
    
    model.add(Bidirectional(LSTM(size, activation=activation2)))
    model.add(Dropout(Dropout2))
    model.add(Dense(output_dim))
    model.add(Activation(activation3))


    
     # compile the model
    model.compile(optimizer = opt, loss = 'mean_squared_error',metrics=['mse'])

    return model
















