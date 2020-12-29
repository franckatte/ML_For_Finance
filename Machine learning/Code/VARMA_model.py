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


df = pd.read_csv('D:/GitHub/ML for Finance project/ML_For_Finance/Machine learning/Data/data_clean.csv',sep = ","
                 ,parse_dates = True,index_col = 0 )

X_day = df.iloc[:,df.columns.str.contains('J')]

X_day = X_day.loc['2015-01-05':]

train_size=0.8



exog = df[['s_eu','s_ch','kof_baro','kof_mpc']]

    
    







    
def VAR_EXOG_predict(data,index_train,index_test,order):
    '''
    

    Parameters
    ----------
    data : time series total
    index_train : index of the train section
    index_test : index of the test section
    order : order of the model VAR

    Returns
    -------
    X_pred : the forcarst value for the index_test 

    '''
    names=data.columns
    X_train=data.loc[index_train]
    X_test=data.loc[index_test]
    
    #calibrate the model
    model = VAR(endog=X_train,exog=exog.loc[index_train]
                ).fit(order)
    #calculation of the forcast value
    X_pred= pd.DataFrame(data=model.forecast(model.y,steps=len(index_test),exog_future=exog.loc[index_test]
                                             ) ,columns=names,index=index_test )
    
    
    
        
    return X_pred

#Calculate of the different differenciation
X_day2=X_day.diff().dropna()
X_day3=X_day2.diff().dropna()


### Differenciation 2






#we determine the good order in minimizing the AIC criterion
model_var= VAR(X_day3)
print(model_var.select_order(20))

n=len(X_day3)
index_train=X_day3.index[:int(n*train_size)]
index_test=X_day3.index[int(n*train_size):]

#we select p=7 and get the predicted value
X_pred_diff=VAR_EXOG_predict(X_day3,index_train,index_test,17)
        
index=X_day.index    
names=X_day.columns

#we calculate the predicted value for d=2

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
        plt.title('forcast VARIX(17,2,0) ' +nom)
        plt.savefig('figures/VARIX/forcast of VARIX(17,2,0) '+nom+'.pdf')
        plt.show()
        
        
        
### Differenciation 1

#we determine the good order in minimizing the AIC criterion
model_var= VAR(X_day2)
n=len(X_day2)
print(model_var.select_order(20))
#we define the train and test part
index_train=X_day2.index[:int(n*train_size)]
index_test=X_day2.index[int(n*train_size):]
X_pred_diff=VAR_EXOG_predict(X_day2,index_train,index_test,6)

#we calculate the predicted value for d=1

X_pred = np.zeros(np.shape(X_pred_diff))  
X_pred[0] = X_day.loc[index_train[-1]] + X_pred_diff.loc[index_test[0]] 
    
for i in range(1,len(index_test)):
    X_pred[i] = X_pred[i-1] + X_pred_diff.loc[index_test[i]] 

X_pred = pd.DataFrame(data= X_pred,columns=names,index=index_test)  
    
for nom in names:
        plt.plot(index,X_day[nom],label='True '+ nom)
        plt.plot(index_test,X_pred[nom],label='predict '+ nom )
        plt.legend()
        plt.title('forcast VARIMA(6,1,0) ' + nom)
        plt.savefig('figures/VARIX/forcast of VARIX(6,1,0) '+nom+'.pdf')
        plt.show()
    
    
    
### Differenciation 0    
    
model_var= VAR(X_day)

print(model_var.select_order(20))
n=len(X_day)
index_train=X_day.index[:int(n*train_size)]
index_test=X_day.index[int(n*train_size):]

#we calculate the predicted value for d=0

X_pred=VAR_EXOG_predict(X_day,index_train,index_test,2)



X_pred = pd.DataFrame(data= X_pred,columns=names,index=index_test)  
    
for nom in names:
        plt.plot(index,X_day[nom],label='True '+ nom)
        plt.plot(index_test,X_pred[nom],label='predict '+ nom )
        plt.legend()
        plt.title('forcast VARIMA(2,0,0) ' + nom)
        plt.savefig('figures/VARIX/forcast of VARIX(2,0,0) '+nom+'.pdf')
        plt.show()
    
    
