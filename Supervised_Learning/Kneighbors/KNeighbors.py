#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 00:26:44 2020

@author: brandonhdz
"""
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import neighbors

X, y = datasets.load_wine(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler=StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

n_neighbors=3

modelo=neighbors.KNeighborsClassifier(n_neighbors)

modelo.fit(X_train, y_train)
modelo.predict(X_test)

print(modelo.score(X_test,y_test))