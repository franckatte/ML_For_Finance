#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 12:20:41 2020

@author: franckatteaka
"""


import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Input, Dense, LSTM, Bidirectional, BatchNormalization, GaussianNoise
from keras.models import Model
from keras.optimizers import Adam
import pandas as pd


##create denoising_ae model
def create_denoising_ae(input_dim, output_dim,learning_rate,std,encoding_dim,activation1,activation2):
    # Create an Adam optimizer with the given learning rate
    opt = Adam(lr = learning_rate)
      
    ## input data
    input_layer = Input(shape = (input_dim,))
    
    ## gaussian layer
    GS = GaussianNoise(std)(input_layer)
    
    ## "encoded" is the encoded representation of the inputs
    encoded = Dense(encoding_dim * 2 , activation = activation1)(GS)
    
    ## coded 
    coded = Dense(encoding_dim  , activation = activation1)(encoded)
    
    ## "decoded" is the lossy reconstruction of the input
    decoded = Dense(encoding_dim * 2, activation = activation1)(coded)
    
    # output layer
    output_layer = Dense(output_dim, activation = activation2)(decoded)
    
    ## model
    autoencoder = Model(input_layer, output_layer)
    
    # compile the model
    autoencoder.compile(optimizer = opt, loss = 'mean_squared_error')
    
    
    return autoencoder
    

##create vanilla LSTM model

def vanilla_LSTM(time_steps,nb_features,output_dim,learning_rate,size,activation1,activation2):
    # Create an Adam optimizer with the given learning rate
    opt = Adam(lr = learning_rate)
    # input data
    input_layer = Input(shape = (time_steps,nb_features))
   
    # batch normalization
    BN = BatchNormalization()(input_layer)
    
    
    # hidden layers
    hidden =  LSTM(size , activation = activation1)(BN)
    
    # batch normalization
    BN = BatchNormalization()(hidden)
    
    # output layer
    output_layer = Dense(output_dim, activation = activation2)(BN)
    
    # model
    lstm = Model(input_layer, output_layer)
    
    # compile the model
    lstm.compile(optimizer = opt, loss = 'mean_squared_error')

    return lstm




##create stacked LSTM model
def stacked_LSTM(time_steps,nb_features,output_dim,learning_rate,size1,size2,activation1,activation2,activation3):
    # Create an Adam optimizer with the given learning rate
    opt = Adam(lr = learning_rate)
    # input data
    input_layer = Input(shape = (time_steps,nb_features))
    
    # batch normalization
    BN = BatchNormalization()(input_layer)
    
    
    # hidden layers
    hidden =  LSTM(size1 , activation = activation1, return_sequences=True)(BN)
    
    # bactch normalization
    BN = BatchNormalization()(hidden)
    
    # hidden layers
    hidden =  LSTM(size2 , activation = activation2,)(BN)
    
    # batch normalization
    BN = BatchNormalization()(hidden)
    
    # output layer
    output_layer = Dense(output_dim, activation = activation3)(BN)
    
    # model
    lstm = Model(input_layer, output_layer)
    
     # compile the model
    lstm.compile(optimizer = opt, loss = 'mean_squared_error')
    
    return lstm


##create bivariate LSTM model
def bi_LSTM(time_steps,nb_features,output_dim,learning_rate,size,activation1,activation2):
    # Create an Adam optimizer with the given learning rate
    opt = Adam(lr = learning_rate)
    # input data
    input_layer = Input(shape = (time_steps,nb_features))
    
    # batch normalization
    BN= BatchNormalization()(input_layer)
        
    # hidden layers
    hidden =  Bidirectional(LSTM(size , activation = activation1))(BN)
    
    # batch normalization
    BN = BatchNormalization()(hidden)
    
    # output layer
    output_layer = Dense(output_dim, activation = activation2)(BN)
    

    # model
    lstm = Model(input_layer, output_layer)
    
     # compile the model
    lstm.compile(optimizer = opt, loss = 'mean_squared_error')

    return lstm


# plot loss
def plot_rmse(history,fig_folder,ticker):
    plt.figure(figsize = (10,6))
    plt.plot(history.epoch, np.array(history.history['loss'])**0.5, label="train", color = "blue")
    plt.plot(history.epoch, np.array(history.history['val_loss'])**0.5, label="test", color = "red")
    plt.xlabel("epoch")
    plt.ylabel("RMSE")
    plt.legend()
    plt.savefig(fig_folder  + ticker + ".pdf")
    plt.show()


def plot_yields(model,X,y,folder, ticker):
    
    terms = ['1J', '2J','3J', '4J', '5J', '6J','7J', '8J', '9J', '10J', '15J', '20J', '30J']
    pred_terms = [term +  " pred" for term in terms]
    
    y_pred = pd.DataFrame(model.predict(X),columns = pred_terms, index = y.index)
    y = y.join(y_pred)
    

    for  i,term in enumerate(terms):
        plt.figure(figsize = (10,6))
        y.loc[:,[term, term + " pred"]].plot( color = ['blue','red'])
        plt.ylabel("yields")
        plt.savefig(folder + "obs_pred-" + ticker + "-" + term + ".pdf")
        plt.show()



def yields_rmse(model,X,y):
    
    n = 13
    terms = ['1J', '2J','3J', '4J', '5J', '6J','7J', '8J', '9J', '10J', '15J', '20J', '30J']

    y_preds = model.predict(X)
    res = []
    
    for i in range(n):
        rmse = ((y[:,i] - y_preds[:,i])**2).sum() ** 0.5
        res.append(rmse)
    
    return pd.DataFrame(res,columns = ['RMSE'],index = terms)

