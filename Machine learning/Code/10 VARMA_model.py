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

#path = 'D:/GitHub/ML_For_Finance/Machine learning/Data' #coco desktop
path = 'D:/GitHub/ML for Finance project/ML_For_Finance/Machine learning/Data' #coco laptop

df = pd.read_csv(path+'/data_clean.csv',sep = ","
                 ,parse_dates = True,index_col = 0 )

X_day = df.iloc[:,df.columns.str.contains('J')]

X_day = X_day.loc['2015-01-05':]
names=X_day.columns
index = X_day.index
train_size=0.8



exogenous = df[['s_eu','s_ch','kof_baro','kof_mpc']].loc['2015-01-05':]

   




    
def VAR_EXOG_predict(data,len_test,order):
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
    X_pred_1j=[]
    X_pred_5j=[]
    index=data.index
    dep = int(len(index)- len_test)
    
    for i in range(dep,int(len(index)-5)):
        
    #calibrate the model
        indice_train = index[i-500:i]
        
        
        model = VAR(endog=data.loc[indice_train],exog=exogenous.loc[indice_train]).fit(order,verbose=False)
    #calculation of the forcast value
        
        ind_test=index[i:i+5]
        X_predi=model.forecast(model.y,steps=5,exog_future=exogenous.loc[ind_test]) 
        
        X_pred_1j.append(X_predi[0])
        X_pred_5j.append(X_predi[-1])
        
        
    
        
    X_pred_1j = pd.DataFrame(data =X_pred_1j ,columns=names,index=index[dep:-5])
    X_pred_5j = pd.DataFrame(data = X_pred_5j,columns=names,index=index[dep+5:])
    return X_pred_1j,X_pred_5j


    




#Calculate of the different differenciation
X_day2=X_day.diff().dropna()  
        
### Differenciation 1

#we determine the good order in minimizing the AIC criterion
model_var= VAR(X_day2)
n=len(X_day2)
print(model_var.select_order(20))
#we define the train and test part

len_test=int((1-train_size)*len(X_day2))
X_pred_diff1,X_pred_diff5=VAR_EXOG_predict(X_day2,len_test,6)

index_test1=X_pred_diff1.index


#we calculate the predicted value for d=1

X_pred1 = np.zeros(np.shape(X_pred_diff1))  


    
for i in range(len(index_test1)):
    index_av = index[np.argwhere(index==index_test1[i])[0]-1]
    X_pred1[i] = X_day.loc[index_av] + X_pred_diff1.loc[index_test1[i]] 

X_pred1 = pd.DataFrame(data= X_pred1,columns=names,index=index_test1)  
    
X_pred5 = np.zeros(np.shape(X_pred_diff5))  
index_test5=X_pred_diff5.index   

for i in range(len(index_test5)):
    index_av = index[np.argwhere(index==index_test5[i])[0]-1]
    X_pred5[i] = X_day.loc[index_av] + X_pred_diff5.loc[index_test5[i]] 

X_pred5 = pd.DataFrame(data= X_pred5,columns=names,index=index_test5)  
    

path_fig='D:/GitHub/ML for Finance project/ML_For_Finance/Machine learning/Code/figures/'

for nom in names:
        plt.plot(index,X_day[nom],label='True '+ nom)
        plt.plot(index_test1,X_pred1[nom],label='predict 1 day after '+ nom )
        plt.plot(index_test5,X_pred5[nom],label='predict 5 day after'+ nom )
        plt.legend()
        plt.title('forcast VARIX(6,1) ' + nom)
        plt.savefig(path_fig+'VARIX/forcast_of_VARIX(6,1)_'+nom+'.pdf')
        plt.show()
    
    
temp1 = (X_day.loc[X_pred1.index]-X_pred1)**2
RMSE_1_d1 = np.sqrt(np.mean(temp1,axis=0))
    
temp5 = (X_day.loc[X_pred5.index]-X_pred5)**2
RMSE_5_d1 = np.sqrt(np.mean(temp5,axis=0)) 
    

    
    
    
### Differenciation 0    
    
model_var= VAR(X_day)

print(model_var.select_order(20))
n=len(X_day)

#we calculate the predicted value for d=0
len_test=int((1-train_size)*len(X_day))
X_pred1,X_pred5=VAR_EXOG_predict(X_day,len_test,2)
 
    
for nom in names:
        plt.plot(index,X_day[nom],label='True '+ nom)
        plt.plot(X_pred1.index,X_pred1[nom],label='predict 1 day after '+ nom )
        plt.plot(X_pred5.index,X_pred5[nom],label='predict 5 day after'+ nom )
        plt.legend()
        plt.title('forcast VARIX(2,0) ' + nom)
        plt.savefig(path_fig+'VARIX/forcast of VARIX(2,0) '+nom+'.pdf')
        plt.show()
    
temp1 = (X_day.loc[X_pred1.index]-X_pred1)**2
RMSE_1_d0 = np.sqrt(np.mean(temp1,axis=0))
    
temp5 = (X_day.loc[X_pred5.index]-X_pred5)**2
RMSE_5_d0 = np.sqrt(np.mean(temp5,axis=0)) 



print('RMSE VARIX(6,1) 1day predicted : ',RMSE_1_d1) 
print('RMSE VARIX(6,1) 5day predicted : ',RMSE_5_d1) 

print('RMSE VARIX(2,0) 1day predicted : ',RMSE_1_d0) 
print('RMSE VARIX(2,0) 5day predicted : ',RMSE_5_d0) 



RMSE = pd.DataFrame(data=[RMSE_1_d1.values,RMSE_5_d1.values,RMSE_1_d0.values,RMSE_5_d0.values],
                    index= ['RMSE 1 day VARIX(6,1)','RMSE 5 day VARIX(6,1)',
                              'RMSE 1 day VARIX(2,0)','RMSE 5 day VARIX(2,0)'],
                    columns = names)



print(RMSE.to_latex()) 


