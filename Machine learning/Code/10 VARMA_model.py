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
from feature_engineering import sub_range

#We define the right paths to get and save datas

path = 'D:/GitHub/ML_For_Finance/Machine learning/Data' #coco desktop
#path = 'D:/GitHub/ML for Finance project/ML_For_Finance/Machine learning/Data' #coco laptop

#path_fig='D:/GitHub/ML for Finance project/ML_For_Finance/Machine learning/Code/figures/' #coco laptop
path_fig='D:/GitHub/ML_For_Finance/Machine learning/Code/00 figures/' #coco desktop

#We load and select the right data

df = pd.read_csv(path+'/data_clean.csv',sep = ","
                 ,parse_dates = True,index_col = 0 )

df = sub_range(df,nb_years = 5)
#We select the yield curves
X_day = df.iloc[:,df.columns.str.contains('J')]

names=X_day.columns
index = X_day.index
train_size=0.8
#We select the exonegeous data
exogenous = df[['s_eu','s_ch','kof_baro','kof_mpc']]#.loc['2015-01-05':]

   




    
def VAR_EXOG_predict(data,len_test,order):
    '''
    

    Parameters
    ----------
    data : time series total
    len_test : len of the test size
    order : order of the model VAR

    Returns
    -------
    X_pred : the forcarst value for the index_test 

    '''
    #We set the variables to stock the predicted values
    names=data.columns
    X_pred_1j=[]
    X_pred_5j=[]
    index=data.index
    #we set the index of the train part and the first index of the test part
    dep = int(len(index)- len_test)
    indice_train = index[:dep]
    #we train our model
    model = VAR(endog=data.loc[indice_train],exog=exogenous.loc[indice_train]
                ).fit(order)
    
    for i in range(dep,int(len(index)-5)):
        
        
    #we set the variables used to forecast 
        
        ind_test=index[i:i+5]
        ind_forecast = index[i-order:i]
        X_predi=model.forecast(y=data.loc[ind_forecast].values,steps=5,exog_future=exogenous.loc[ind_test]
                               ) 
        #we save the wanted predicted values
        X_pred_1j.append(X_predi[0])
        X_pred_5j.append(X_predi[-1])
        
        
    
    #we create a DataFrame to coorectly save the predicted values
    X_pred_1j = pd.DataFrame(data =X_pred_1j ,columns=names,index=index[dep:-5])
    X_pred_5j = pd.DataFrame(data = X_pred_5j,columns=names,index=index[dep+5:])
    return X_pred_1j,X_pred_5j


    






        
### Differenciation 1

#We calculate the differential data
X_day2=X_day.diff().dropna()  
#we determine the right order in minimizing the AIC criterion
model_var= VAR(X_day2)
n=len(X_day2)
print(model_var.select_order(20))
#we define the train and test part

len_test=int((1-train_size)*len(X_day2))
X_pred_diff1,X_pred_diff5=VAR_EXOG_predict(X_day2,len_test,5)

index_test1=X_pred_diff1.index


#we calculate the predicted value for d=1

X_pred1 = np.zeros(np.shape(X_pred_diff1))  


#we calculate the right predicted values from the differential predicted values

#for the 1 day forecast
for i in range(len(index_test1)):
    index_av = index[np.argwhere(index==index_test1[i])[0]-1]
    X_pred1[i] = X_day.loc[index_av] + X_pred_diff1.loc[index_test1[i]] 

X_pred1 = pd.DataFrame(data= X_pred1,columns=names,index=index_test1)  
    
#for the 5 day forecast
X_pred5 = np.zeros(np.shape(X_pred_diff5))  
index_test5=X_pred_diff5.index   

for i in range(len(index_test5)):
    index_av = index[np.argwhere(index==index_test5[i])[0]-1]
    X_pred5[i] = X_day.loc[index_av] + X_pred_diff5.loc[index_test5[i]] 

X_pred5 = pd.DataFrame(data= X_pred5,columns=names,index=index_test5)  
    


#we plot the figures of the True and predicted values
for nom in names:
        plt.plot(X_pred1.index,X_day[nom].loc[X_pred1.index],label='True '+ nom)
        plt.plot(index_test1,X_pred1[nom],label='predict 1 day  '+ nom )
        
        plt.legend()
        #plt.title('forcast VARIX(5,1) ' + nom)
        plt.savefig(path_fig+'VARIX/forcast 1 day of VARIX(5,1) '+nom+'.pdf')
        plt.show()
        
for nom in names:
        plt.plot(X_pred1.index,X_day[nom].loc[index_test5],label='True '+ nom)
        
        plt.plot(index_test5,X_pred5[nom],label='predict 5 day '+ nom )
        plt.legend()
        #plt.title('forcast VARIX(5,1) ' + nom)
        plt.savefig(path_fig+'VARIX/forcast 5 day of VARIX(5,1) '+nom+'.pdf')
        plt.show()
    
#We calculate the RMSE for 1 and 5 day prediction
temp1 = (X_day.loc[X_pred1.index]-X_pred1)**2
RMSE_1_d1 = np.sqrt(np.mean(temp1,axis=0))
    
temp5 = (X_day.loc[X_pred5.index]-X_pred5)**2
RMSE_5_d1 = np.sqrt(np.mean(temp5,axis=0)) 
    

    
    
    
### Differenciation 0    
    
#we determine the right order in minimizing the AIC criterion
model_var= VAR(X_day)

print(model_var.select_order(20))
n=len(X_day)

#we calculate the predicted value for d=0
len_test=int((1-train_size)*len(X_day))
X_pred1,X_pred5=VAR_EXOG_predict(X_day,len_test,2)
 
#we plot the figures of the True and predicted values
for nom in names:
        plt.plot(X_pred1.index,X_day[nom].loc[X_pred1.index],label='True '+ nom)
        plt.plot(X_pred1.index,X_pred1[nom],label='predict 1 day  '+ nom )
       
        plt.legend()
        #mplt.title('forcast VARIX(2,0) ' + nom)
        plt.savefig(path_fig+'VARIX/forcast 1 day of VARIX(2,0) '+nom+'.pdf')
        plt.show()
        
for nom in names:
        plt.plot(X_pred1.index,X_day[nom].loc[X_pred5.index],label='True '+ nom)
        
        plt.plot(X_pred5.index,X_pred5[nom],label='predict 5 day '+ nom )
        plt.legend()
        #plt.title('forcast VARIX(2,0) ' + nom)
        plt.savefig(path_fig+'VARIX/forcast 5 day of VARIX(2,0) '+nom+'.pdf')
        plt.show()
    
#We calculate the RMSE for 1 and 5 day prediction
temp1 = (X_day.loc[X_pred1.index]-X_pred1)**2
RMSE_1_d0 = np.sqrt(np.mean(temp1,axis=0))
    
temp5 = (X_day.loc[X_pred5.index]-X_pred5)**2
RMSE_5_d0 = np.sqrt(np.mean(temp5,axis=0)) 


#We print the RMSE for each model for 1 and 5 day forecast
print('RMSE VARIX(6,1) 1day predicted : ',RMSE_1_d1) 
print('RMSE VARIX(6,1) 5day predicted : ',RMSE_5_d1) 

print('RMSE VARIX(2,0) 1day predicted : ',RMSE_1_d0) 
print('RMSE VARIX(2,0) 5day predicted : ',RMSE_5_d0) 


#We create a DataFrame to print all RMSE and calculate the mean of the RMSE for each model
data_temp= np.array([RMSE_1_d1.values,RMSE_5_d1.values,RMSE_1_d0.values,RMSE_5_d0.values])

data_temp2 = np.zeros((4,14))
data_temp2[:,:-1] = np.array([RMSE_1_d1.values,RMSE_5_d1.values,RMSE_1_d0.values,RMSE_5_d0.values])
data_temp2[:,-1] = np.mean(data_temp,axis=1)

RMSE = pd.DataFrame(data=data_temp2,
                    index= ['RMSE 1 day VARIX(6,1)','RMSE 5 day VARIX(6,1)',
                              'RMSE 1 day VARIX(2,0)','RMSE 5 day VARIX(2,0)'],
                    columns = list(names)+['Mean RMSE'])


print(RMSE)
print(RMSE.to_latex()) 
























