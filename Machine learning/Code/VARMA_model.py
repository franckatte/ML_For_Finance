# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 09:34:21 2020

@author: corentin Franck
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa import stattools
from statsmodels.tsa.vector_ar.var_model import VAR


df = pd.read_csv('D:/GitHub/ML_For_Finance/Machine learning/Data/data_clean.csv',sep = ","
                 ,parse_dates = True,index_col = 0 )

X_day = df.iloc[:,df.columns.str.contains('J')]


X_day2=X_day.diff().dropna()


model_var= VAR(X_day)
print(model_var.select_order(20))
#We focus on BIC to do not take too many variable then we get order 3
exog = df[['s_eu','s_ch','kof_baro','kof_mpc']]

    
    







    
def VAR_EXOG_predict(data,index_train,index_test,order):
    
    names=data.columns
    X_train=data.loc[index_train]
    X_test=data.loc[index_test]
    
    model = VAR(endog=X_train,exog=exog.loc[index_train]).fit(order)
    X_pred= pd.DataFrame(data=model.forecast(model.y,steps=len(index_test),exog_future=exog.loc[index_test] ) ,columns=names,index=index_test )
    
    
    
        
    return X_pred

#Calculate of the different differenciation
X_day2=X_day.diff().dropna()
X_day3=X_day2.diff().dropna()


### Differenciation 2







model_var= VAR(X_day3)
print(model_var.select_order(20))


index_train=X_day3.index[:2800]
index_test=X_day3.index[2800:]
X_pred_diff=VAR_EXOG_predict(X_day3,index_train,index_test,7)
        
index=X_day.index    
names=X_day.columns


X_pred = np.zeros(np.shape(X_pred_diff))
X_pred[0] = -X_day.loc[index_train[-2]] + X_pred_diff.loc[index_test[0]] +2*X_day.loc[index_train[-1]]
X_pred[1] = -X_day.loc[index_train[-1]] + X_pred_diff.loc[index_test[1]] +2*X_pred[0]
for i in range(2,len(index_test)):
    X_pred[i] = -X_pred[i-2] + X_pred_diff.loc[index_test[i]] +2*X_pred[i-1]

X_pred = pd.DataFrame(data= X_pred,columns=names,index=index_test)
   
for nom in names:
        plt.plot(index,X_day[nom],label='True '+ nom)
        plt.plot(index_test,X_pred[nom],label='predict '+ nom )
        plt.legend()
        plt.title('forcast VARIMA(7,2,0)')
        plt.show()
        
        
        
### Diffeciation 1


model_var= VAR(X_day2)
print(model_var.select_order(20))

index_train=X_day2.index[:2800]
index_test=X_day2.index[2800:]
X_pred_diff=VAR_EXOG_predict(X_day2,index_train,index_test,3)


X_pred = np.zeros(np.shape(X_pred_diff))  
X_pred[0] = X_day.loc[index_train[-1]] + X_pred_diff.loc[index_test[0]] 
    
for i in range(1,len(index_test)):
    X_pred[i] = X_pred[i-1] + X_pred_diff.loc[index_test[i]] 

X_pred = pd.DataFrame(data= X_pred,columns=names,index=index_test)  
    
for nom in names:
        plt.plot(index,X_day[nom],label='True '+ nom)
        plt.plot(index_test,X_pred[nom],label='predict '+ nom )
        plt.legend()
        plt.title('forcast VARIMA(3,1,0)')
        plt.show()
    
    
    
