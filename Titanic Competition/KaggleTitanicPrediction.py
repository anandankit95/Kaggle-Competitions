#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 16:45:43 2017

@author: Ankit
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')
X_train = train.iloc[:,[1,3,4,5,6,10]]  #taking only relevat columns
y_train = train.iloc[:, -1].values

X_test = test.iloc[:,[1,3,4,5,6,10]]    #taking only relevant columnns
#here passenrger ID,Name,Ticket,Fare,Cabin has been dropped
#reasons being as Follows:
    #Paassenger ID,Name, TIcket and Fare are Irrelevant to the Problem
    #Cabin has lot of Null values hence does not contribute much
    
#fill in the missing null values inside the Age column
X_train['Age'].fillna((X_train['Age'].mean()),inplace='True') 
X_test['Age'].fillna((X_test['Age'].mean()),inplace='True')

#Fill null values in Embarked column
X_train['Embarked'].fillna('S',inplace='True')
X_test['Embarked'].fillna('S',inplace='True')





#Encode all the categorcal variables
#Encode Features 'Sex' and 'Embarked' into numerical values
from sklearn.preprocessing import LabelEncoder ,OneHotEncoder
labelencoder_train= LabelEncoder()
labelencoder_test= LabelEncoder()
X_train.iloc[:,1]=labelencoder_train.fit_transform(X_train.iloc[:,1])
X_train.iloc[:,5]=labelencoder_train.fit_transform(X_train.iloc[:,5].factorize()[0])

X_test.iloc[:,1]=labelencoder_test.fit_transform(X_test.iloc[:,1])
X_test.iloc[:,5]=labelencoder_test.fit_transform(X_test.iloc[:,5].factorize()[0])

X_train=pd.get_dummies(X_train,columns=['Embarked'])
X_test=pd.get_dummies(X_test,columns=['Embarked'])


#Fit the SVM  Model
from sklearn.svm import SVC
classifier = SVC(C=10,kernel='rbf',random_state=0,gamma=0.1)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
#This model gives accuracy score of 0.805951


#Fit the Random forest Model
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=100,criterion='gini',random_state=0)
classifier.fit(X_train,y_train)
y_pred= classifier.predict(X_test)
#This model gives best accuracy score of 0.80717483

#Fit the Naive Bayes Model
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
#This model gives best accuracy of 0.765551015




#Fit the Decision Tree Model
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(X_train,y_train)
y_pred= classifier.predict(X_test)
#gini gives accuracy of 0.785129
#entroy gives accuracy of  0.788022


#obtain the accuracy score
from sklearn.model_selection import cross_val_score
accuracies= cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10)
accuracies.mean()


"""
#Applying gridSearchCV for random forest
from sklearn.model_selection import GridSearchCV
parameters = [{'n_estimators': [1, 10, 100, 1000], 'criterion': ['gini']},
              {'n_estimators': [1, 10, 100, 1000], 'criterion': ['entropy']}]
#also add n_jobs parameter = -1 if having large dataset
grid_search=GridSearchCV(estimator=classifier,param_grid=parameters,scoring='accuracy',cv=10,n_jobs=-1)
grid_search=grid_search.fit(X_train,y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

"""
"""
#Applying gridSearchCV for SVC
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
#also add n_jobs parameter = -1 if having large dataset
grid_search=GridSearchCV(estimator=classifier,param_grid=parameters,scoring='accuracy',cv=10,n_jobs=-1)
grid_search=grid_search.fit(X_train,y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

#got best parameters for SVC as follows
# C= 10
# gamma= 0.1
#kernel = rbf
"""

#Write the preidictionns inside a file
submission=open('/Users/arjita/ML/KaggleTitanicCompetition/submission2.csv','w')
submission.write('PassengerId'+','+'Survived'+'\n')
for i in range(len(y_pred)):
    submission.write(str(i+892)+','+str(y_pred[i])+'\n')
#these predicitions are based on training set 
#Do these for the test set using cross validation


