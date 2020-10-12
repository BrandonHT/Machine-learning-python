#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 11:41:58 2020

@author: brandonhdz
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import sklearn.neural_network

p=np.arange(-2,2,0.1)
print(p.shape)
p=p.reshape(-1,1)
print(p.shape)
t=1+np.sin(math.pi*p/4)

modelo=sklearn.neural_network.MLPRegressor(
    hidden_layer_sizes=2,
    activation='logistic',
    learning_rate_init=0.1,
    max_iter=1000,
    solver='sgd')

modelo.fit(p,t)

nuevoT=modelo.predict(p)
nuevoT=nuevoT.reshape(-1,1)
plt.plot(nuevoT,label='nuevoT')
plt.plot(nuevoT-t, label='nuevoT-t')
plt.plot(t)
plt.legend()
plt.show()

plt.plot(modelo.loss_curve_)