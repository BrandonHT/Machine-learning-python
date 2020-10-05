#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 12:58:29 2020

@author: brandonhdz
"""
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import tree
X, y = datasets.load_wine(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Scaling data
scaler=StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test=scaler.transform(X_test)

#modelo = tree.DecisionTreeClassifier(criterion='gini')
modelo = tree.DecisionTreeClassifier(criterion='entropy')

fScores = cross_val_score(modelo, X_train, y_train, cv=10)

print("Promedio: ", fScores.mean(), "\tStd: ", fScores.std())

modelo.fit(X_train, y_train)

y_res = modelo.predict(X_test)

print(sum(y_res-y_test))