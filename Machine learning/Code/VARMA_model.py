# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 09:34:21 2020

@author: corentin Franck
"""


import numpy as np
import pandas as pd
import statsmodels.api as sm


   # dta = sm.datasets.webuse('lutkepohl2', 'https://www.stata-press.com/data/r12/')
   # dta.index = dta.qtr
   # dta.index.freq = dta.index.inferred_freq
   # endog = dta.loc['1960-04-01':'1977-10-01', ['dln_inv', 'dln_inc', 'dln_consump']]
    
    
   # mod = sm.tsa.VARMAX(endog[['dln_inv', 'dln_inc']], order=(8,0))
   # res = mod.fit(maxiter=10000, disp=False)
    #print(res.summary())
    
   # print(res.predict())
   # print('\n')
   # print(res.predict(dta.index[-1]))


class VARMAmodel:
    # We init the model with the train data set in pandas and 
    #the order p and q we want
    
    #get_summary is to have every detail on the fit
    
    #predict need the index of the data we want to predict
    
    
    
    def __init__(self,Xtrain,p,q):
        self.model = sm.tsa.VARMAX(Xtrain,order=(p,q)).fit(maxiter=10000, disp=False)
    
    def get_summury(self):
        print(self.model.summary())
        return
    def model_predit(self,indice_test):
        return self.model.predict(indice_test)
    
        
    
