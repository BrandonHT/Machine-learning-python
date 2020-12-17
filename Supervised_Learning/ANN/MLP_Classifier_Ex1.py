#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 12:01:14 2020

@author: brandonhdz
"""
import math
#import numpy as np
import matplotlib.pyplot as plt
from sklearn import neural_network, datasets, model_selection, preprocessing

X, y=datasets.load_wine(return_X_y=True)

X_train, X_test, y_train, y_test=model_selection.train_test_split(X,y,test_size=0.2,random_state=0)

scaler=preprocessing.StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)


modelo=neural_network.MLPClassifier(
    hidden_layer_sizes=(13,20,3),
    activation='relu',
    learning_rate_init=0.1,
    max_iter=1000,
    solver='sgd'
    )

modelo.fit(X_train, y_train)

plt.plot(modelo.predict_proba(X_test[:3]))

print(modelo.n_layers_)
print(modelo.n_outputs_)