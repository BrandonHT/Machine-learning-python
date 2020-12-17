#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import datasets

X, y = datasets.fetch_covtype(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Scaling data
scaler = StandardScaler() 
scaler.fit(X_train)  
X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)  

n_neighbors = 5
modelo = KNeighborsClassifier(n_neighbors)
modelo.fit(X_train,y_train)
print(modelo.predict(X_test))

# Bagging 

modeloB = BaggingClassifier(KNeighborsClassifier(n_neighbors), max_samples=0.3, max_features=0.3)
modeloB.fit(X_train,y_train)
print(modeloB.predict(X_test))

# Bagging 2
modeloB2 = BaggingClassifier(n_estimators=10, max_samples=0.3, max_features=0.3)
modeloB2.fit(X_train,y_train)
print(modeloB2.predict(X_test))

print(y_test)

print(modelo.score(X_test,y_test))
print(modeloB.score(X_test,y_test))
print(modeloB2.score(X_test,y_test))
