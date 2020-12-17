#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: brandonhdz
"""
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)

score=0
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    scaler=StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    modelo=SVC(kernel='linear', C=0.9)
               
    modelo.fit(X_train, y_train)
    
    score+=modelo.score(X_test, y_test)

score=score/10

print("SVM score: ",score)