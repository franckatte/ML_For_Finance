#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 13:19:05 2020

@author: franckatteaka
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.wrappers.scikit_learn import KerasRegressor
from keras.optimizers import Adam
from sklearn.model_selection import RandomizedSearchCV
from tscpcv import CPCV

# import data
df = df = pd.read_csv('/Users/franckatteaka/Desktop/cours/Semester III/Courses Projects/Machine Learning/Data/data_clean.csv',sep = ","
                 ,parse_dates = True,index_col = 0 )


X = df.iloc[:,df.columns.str.contains('J')]
X_train_0, X_test_0 = train_test_split(X, test_size = 0.2,shuffle = False)

X_train, X_test = X_train_0.to_numpy(), X_test_0.to_numpy()

## model dimension
input_dim = X_train.shape[1]
output_dim = X_train.shape[1]



def create_denoising_ae(learning_rate, dropout,encoding_dim,activation1,activation2):
    # Create an Adam optimizer with the given learning rate
    opt = Adam(lr = learning_rate)
      
    ## input data
    input_layer = Input(shape = (input_dim,))
    ## dropout layer
    Dropout(dropout)(input_layer)
    
    ## "encoded" is the encoded representation of the inputs
    encoded = Dense(encoding_dim * 2 , activation = activation1)(input_layer)
    
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
    
    
# Create a KerasRegressor
model = KerasRegressor(build_fn = create_denoising_ae)

# Define the parameters to try out
params = {'encoding_dim':[1,2],'activation1': ['sigmoid', 'tanh'],'activation2':['linear', 'tanh'],'batch_size': [10,50, 100, 200],
          'learning_rate': [0.01, 0.001],'epochs': [10,50,100, 200,500], 'dropout':[0.4,0.5]}

# Create a randomize search cv object passing in the parameters to try
#random_search = RandomizedSearchCV(model, param_distributions = params, cv = KFold(5))

# create the CPCV folds indexes
cpcv = CPCV(X_train_0, n_split = 6, n_folds = 2, purge = 0, embargo = 0)

# Create a randomize search cv object passing in the parameters to try
random_search = RandomizedSearchCV(model, param_distributions = params, cv = cpcv,n_jobs = 4,n_iter = 20)


# Search for best combinations
random_search.fit(X_train,X_train)

# results
random_search.best_params_

## training parameters

learning_rate = 0.001
epochs = 500
encoding_dim = 2
dropout = 0.5
batch_size = 100
activation2 = 'linear'
activation1 = 'tanh'
# 4.329401294704847e-07

# create model
autoencoder =  create_denoising_ae(learning_rate,dropout,encoding_dim,activation1,activation2)

# summary the model
autoencoder.summary()

# Define a callback

modelCheckpoint = ModelCheckpoint(filepath = '/Users/franckatteaka/Desktop/cours/Semester III/Courses Projects/Machine Learning/Code/models/best_auto_encoder.hdf5',  save_best_only = True)


# fit the model

history = autoencoder.fit(X_train, X_train, epochs = epochs, batch_size = batch_size, shuffle = True, 
                          callbacks = [modelCheckpoint],
                          validation_data=(X_test, X_test),verbose = 1)


## Plot RMSE
fig_folder = '/Users/franckatteaka/Desktop/cours/Semester III/Courses Projects/Machine Learning/Code/figures/autoencoder'
plt.figure(figsize = (10,6))
plt.plot(history.epoch, np.array(history.history['loss'])**0.5, label="train", color = "blue")
plt.plot(history.epoch, np.array(history.history['val_loss'])**0.5, label="test", color = "red")
plt.xlabel("epoch")
plt.ylabel("RMSE")
plt.legend()
plt.savefig(fig_folder  + '/train_test_RMSE.pdf')
plt.show()

# evaluate
print('\n# Evaluate on test data')
results = autoencoder.evaluate(X_train, X_train, batch_size=100)
print('test mse', results)


#  plot function
def yields_plot(X,X_pred,folder, ticker):
    
    X2 = X.join(X_pred)
    terms = ['1J', '2J','3J', '4J', '5J', '6J','7J', '8J', '9J', '10J', '15J', '20J', '30J']
    
    for  i,term in enumerate(terms):
        plt.figure(figsize = (10,6))
        X2.loc[:,[term, term + " pred"]].plot( color = ['blue','red'])
        plt.ylabel("yields")
        plt.savefig(folder + "/" + "obs_pred-" + ticker + "-" + term + ".pdf")
        plt.show()


# prediction train

X_pred = autoencoder.predict(X_train)
columns_preds = [ i + " pred" for i in list(X_train_0.columns)]
X_pred = pd.DataFrame(X_pred, index = X_train_0.index,columns = columns_preds )

yields_plot(X_train_0,X_pred,fig_folder,"train")

# prediction test
X_pred = autoencoder.predict(X_test)
columns_preds = [ i + " pred" for i in list(X_test_0.columns)]
X_pred = pd.DataFrame(X_pred, index = X_test_0.index,columns = columns_preds )

yields_plot(X_test_0,X_pred,fig_folder,"test")



