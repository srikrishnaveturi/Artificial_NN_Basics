# -*- coding: utf-8 -*-
"""
Created on Thu May 21 16:00:47 2020

@author: vetur
"""

import pandas as pd
import numpy as np

dataset = pd.read_excel("Folds5x2_pp.xlsx")
X = dataset.iloc[:,0:4].values
y = dataset.iloc[:,4].values
#temperature,pressure,vaccuum,humidity,electrical energy 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(activation="relu", input_dim=4, units=6, kernel_initializer="uniform"))

classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))

classifier.add(Dense(units=1))

classifier.compile(optimizer = "adam", loss = "mean_squared_error") 

classifier.fit(X_train,y_train,batch_size = 32,epochs = 100)

y_pred = classifier.predict(X_test)

