#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 13:20:18 2020

@author: brandonhdz
"""
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

X, y = datasets.load_wine(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Scaling data
scaler=StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test=scaler.transform(X_test)

parametros=[{'kernel':('linear','rbf'),'C':[1,5,10,15]}]

mySVM = GridSearchCV(SVC(), parametros, cv=10)
mySVM.fit(X_train, y_train)

print("Mejores parametros: ")
print()
print(mySVM.best_params_)
print()

print("Informe: ")
y_pred = mySVM.predict(X_test)
print(classification_report(y_test, y_pred))