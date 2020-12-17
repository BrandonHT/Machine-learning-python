# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 13:59:51 2020

@author: brand
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import datasets

X, y = datasets.load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
scaler=StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

n_neighbors = 5
modelo = KNeighborsClassifier(n_neighbors)
modelo.fit(X_train, y_train)

modeloRF = RandomForestClassifier(n_estimators=1000, max_depth=5, random_state=123)
modeloRF.fit(X_train, y_train)

print(modelo.score(X_test, y_test))
print(modeloRF.score(X_test, y_test))
print(modeloRF.feature_importances_)