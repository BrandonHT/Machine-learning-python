#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier

X, y = datasets.load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Scaling data
scaler = StandardScaler() 
scaler.fit(X_train)  
X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)  

n_neighbors = 200
modelo = KNeighborsClassifier(n_neighbors)
modelo.fit(X_train,y_train)

# AdaBoostClassifier
modeloAB = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5),n_estimators=1000)
modeloAB.fit(X_train,y_train)

print(modelo.score(X_test,y_test))
print(modeloAB.score(X_test,y_test))


