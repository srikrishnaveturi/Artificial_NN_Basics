# -*- coding: utf-8 -*-
"""
Created on Thu May 21 16:00:47 2020

@author: vetur
"""
#import the required libraries
import pandas as pd
import numpy as np

#import the dataset and seperate the independent and the dependent variables
dataset = pd.read_excel("Folds5x2_pp.xlsx")
X = dataset.iloc[:,0:4].values
y = dataset.iloc[:,4].values

#temperature,pressure,vaccuum,humidity,electrical energy 

#split the depencent and independent variables into training and testing sets 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#since there are continuous values in the dependent variables,  we scale all the columns in the training and test sets
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#import keras and some other requirements for making an ANN model
import keras
from keras.models import Sequential #this is the class of the model
from keras.layers import Dense #we use this to add levels into the model

classifier = Sequential()

classifier.add(Dense(activation="relu", input_dim=4, units=6, kernel_initializer="uniform")) 

classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))

classifier.add(Dense(units=1)) #since this is a regression model, we are not giving any activation functions, for classification models, "segmoid" function is used most commonly

classifier.compile(optimizer = "adam", loss = "mean_squared_error") #since this is a regression model, we are using "mean_squared_error" as a way to calculate the loss, for classification we mostly use 

classifier.fit(X_train,y_train,batch_size = 32,epochs = 100)

y_pred = classifier.predict(X_test)

