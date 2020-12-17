#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: brandonhdz
"""
import matplotlib.pyplot as plt
from sklearn import neural_network, model_selection, preprocessing
from sklearn.datasets import load_iris

X, y=load_iris(return_X_y=True)

score=0
for i in range(10):
    X_train, X_test, y_train, y_test=model_selection.train_test_split(X,y,test_size=0.2)
    
    scaler=preprocessing.StandardScaler()
    scaler.fit(X_train)
    X_train=scaler.transform(X_train)
    X_test=scaler.transform(X_test)
    
    
    modelo=neural_network.MLPClassifier(
        hidden_layer_sizes=(4,9),
        activation='relu',
        learning_rate_init=0.1,
        max_iter=1000,
        solver='sgd'
        )
    
    modelo.fit(X_train, y_train)
    
    score+=modelo.score(X_test, y_test)

score=score/10

print("NN score: ",score)