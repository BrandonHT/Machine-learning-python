#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 12:43:26 2020

@author: brandonhdz
"""

import matplotlib.pyplot as plt
from sklearn import neural_network, datasets, model_selection, preprocessing

#X, y=datasets.load_wine(return_X_y=True)

X,y=datasets.fetch_covtype(return_X_y=True)

X_train, X_test, y_train, y_test=model_selection.train_test_split(X,y,test_size=0.2)

scaler=preprocessing.StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)


modelo=neural_network.MLPClassifier(
    hidden_layer_sizes=(54,27),
    activation='relu',
    learning_rate_init=0.1,
    max_iter=100,
    solver='sgd'
    )

modelo.fit(X_train, y_train)

plt.plot(modelo.predict_proba(X_test[:3]))
print(modelo.score(X_test, y_test))
print(modelo.n_layers_)
print(modelo.n_outputs_)