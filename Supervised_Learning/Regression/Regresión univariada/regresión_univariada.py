#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 13:38:31 2020

@author: brandonhdz
"""

import matplotlib.pyplot as plt
import numpy as np
from math import isclose
from sklearn import linear_model

m=10
np.random.seed(2)
x=np.random.random(m)
print(x)
y=1.5*x + np.random.random(m)

#inicializar
a=0
b=0
alpha=0.1 #tasa de aprendizaje

converge = False
maxIteraciones = 10000

i = 1

loga = []
logb = []
while not converge and i<= maxIteraciones:
    
    #Observar / evaluar
    
    yF = a*x + b

    
    #Actualizar
    nuevaA = a - alpha * (1/m) * sum((yF-y)*x)
    nuevaB = b - alpha * (1/m) * sum(yF-y)

    if (isclose(a,nuevaA,rel_tol=1e-6)) and  (isclose(b, nuevaB, rel_tol=1e-9)):
        converge=True
    
    a = nuevaA
    b = nuevaB
    
    loga.append(a)
    logb.append(b)
    
    i += 1

plt.plot(x,y,'o')
plt.plot(x,a*x+b)
plt.show()

plt.plot(loga, label='a')
plt.plot(logb, label='b')
plt.legend()
plt.show()

print('a: ', a)
print('b: ', b)

print ('*******************+')


regM = linear_model.LinearRegression()
xx = x.reshape(-1,1)
yy = y.reshape(-1,1)
regM.fit(xx,yy)

print('a: ',regM.coef_)
print('b: ',regM.intercept_)









