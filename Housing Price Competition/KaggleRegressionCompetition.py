#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 15:35:31 2017

@author: Ankit
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
train = pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
X = train.iloc[:, :-1].values
y = train.iloc[:, -1].values

#obtain all the numeric features in both datasets
#here numric features is also a dataframe
numeric_features_train=train.select_dtypes(include=[np.number])
numeric_features_train.describe()

numeric_features_test=test.select_dtypes(include=[np.number])
numeric_features_test.describe()

#obtain categorical variables in both datasets
#Use one -hot encoding to encode these variables to numerical values
categoricals_train = train.select_dtypes(exclude=[np.number])
categoricals_train.describe()

categoricals_test = test.select_dtypes(exclude=[np.number])
categoricals_test.describe()



#obtain null values for each column in both datasets
nulls_train = pd.DataFrame(train.isnull().sum().sort_values(ascending=False)[:])
nulls_train.columns = ['Null Count']
nulls_train.index.name = 'Feature'
nulls_train[:16]

nulls_test = pd.DataFrame(test.isnull().sum().sort_values(ascending=False)[:])
nulls_test.columns = ['Null Count']
nulls_test.index.name = 'Feature'
nulls_test[:16]


      
    

#Delete the columns haivng most null values in Train as well as in test dataset
for elem in nulls_train.index[:16]:
    train=train.drop(elem,1)
    test=test.drop(elem,1)






#examine the correlation between attributes in train dataset
corr=numeric_features_train.corr()
#corr['SalePrice'].sort_values(ascending='False').index


#Delete all the columns which has the very less correlation with Target Variable
#Removing attributes which have correlation coeff b/w -0.2 to 0.4
del_corr=[]
for elem in corr['SalePrice'].sort_values(ascending='False').index:
    val=corr['SalePrice'][elem]
    if(val<0.400000 and val>(-0.20000)):
        del_corr.append(elem)
        
       
#check if label are present in dataset or not
for label in del_corr:
    if(label in train.columns):
       train=train.drop(label,axis=1)
       test=test.drop(label,axis=1)

    

        
categoricals_train = train.select_dtypes(exclude=[np.number])
categoricals_train.describe()

categoricals_test = test.select_dtypes(exclude=[np.number])
categoricals_test.describe()
        
 
#Remove the categorcial attributes which have categories<=6
#this is beacauese the wont affect much the dependent variable  

    
for column in categoricals_train.columns:
    if(len(train[column].unique())<=6):
        train=train.drop(column,axis=1)
        test=test.drop(column,axis=1)
        

          
  
#UP till here 
#Removed Null variable
#Removed less correlated variables
#Removed some categorical variables       
                   
                    
 


#Split Categorical variables into Dummy Variables with Corresponding Values as  0 or 1 
#Depending on whether that variable need to present for that particular record
l=[0,1,2,3,7,8,9,16,20] #index of categorical variables
from sklearn.preprocessing import LabelEncoder ,OneHotEncoder
labelencoder_train= LabelEncoder()
labelencoder_test= LabelEncoder()
for i in l:
    train.iloc[:,i]=labelencoder_train.fit_transform(train.iloc[:,i].factorize()[0])
    test.iloc[:,i]=labelencoder_test.fit_transform(test.iloc[:,i].factorize()[0])

#Encode the dataset to get dummy categories
#train=pd.get_dummies(train,columns=['Neighborhood','Condition1','Condition2','HouseStyle','RoofMatl','Exterior1st','Exterior2nd','FullBath','TotRmsAbvGrd','Functional','Fireplaces','GarageCars','SaleType']) 
#test=pd.get_dummies(test,columns=['Neighborhood','Condition1','Condition2','HouseStyle','RoofMatl','Exterior1st','Exterior2nd','FullBath','TotRmsAbvGrd','Functional','Fireplaces','GarageCars','SaleType'])    

train.isnull().any()

train['MasVnrArea']=train['MasVnrArea'].factorize()[0]
train.isnull().any()
test.isnull().any()
test['MasVnrArea']=test['MasVnrArea'].factorize()[0]
test['TotalBsmtSF']=test['TotalBsmtSF'].factorize()[0]
test['GarageCars']=test['GarageCars'].factorize()[0]
test['GarageArea']=test['GarageArea'].factorize()[0]
test.isnull().any()

#Run Random forest regression
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor.fit(train.iloc[:,:-1], train.iloc[:,-1])
final_prediction=regressor.predict(test)

#Run decision tree regression
from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor()
regressor.fit(train.iloc[:,:-1], train.iloc[:,-1])
final_prediction=regressor.predict(test)



#Run support vector regression
from sklearn.svm import SVR
regressor = SVR(kernel='linear')
regressor.fit(train.iloc[:,:-1], train.iloc[:,-1])
final_prediction=regressor.predict(test)

#run linear regression
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(train.iloc[:,:-1], train.iloc[:,-1])
final_prediction=regressor.predict(test)









submission=open('/Users/arjita/ML/KaggleRegressionCompetition/LinearRegressionSubmission.csv','w')
submission.write('Id'+','+'SalePrice'+'\n')
for i in range(len(final_prediction)):
    submission.write(str(i+1461)+','+str(format(final_prediction[i],'0.9f'))+'\n')







