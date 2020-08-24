#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 16:34:43 2020

@author: brandonhdz
"""

import matplotlib.pyplot as plt
import numpy as np
from math import isclose
from sklearn import linear_model, datasets

diabetes=datasets.load_diabetes()
x=diabetes.data
y=diabetes.target
m,n=x.shape


#Réplica de resultados
np.random.seed(27)

#Inicializar
w=np.random.random(n)
b=np.random.random()
alpha=0.1

converge=False
maxIteraciones=100000

i=0

logw0=[]
logw1=[]
logw2=[]
logw3=[]
logw4=[]
logw5=[]
logw6=[]
logw7=[]
logw8=[]
logw9=[]

waux=[0]*n
b0=sum(y)/m
Eab=[]

errorAnt=0
while not converge and i < maxIteraciones:
    #Observar y evaluar
    yF=np.sum(w*x,axis=1)+b
    aux=yF-y
    
    #Actualizar
    for c in range(n):
        waux[c]=w[c]-alpha*sum(aux*x[:,c])
    
    error=(1/(2*m))*sum(pow(aux,2))
    Eab.append(error)
        
    if ((isclose(w[0],waux[0],rel_tol=1e-9)) and (isclose(w[1], waux[1], rel_tol=1e-9))
        and (isclose(w[2], waux[2], rel_tol=1e-9)) and (isclose(w[3], waux[3], rel_tol=1e-9)) 
        and (isclose(w[4], waux[4], rel_tol=1e-9)) and (isclose(w[5], waux[5], rel_tol=1e-9))
        and (isclose(w[6], waux[6], rel_tol=1e-9)) and (isclose(w[7], waux[7], rel_tol=1e-9))
        and (isclose(w[8], waux[8], rel_tol=1e-9)) and (isclose(w[9], waux[9], rel_tol=1e-9))) or (isclose(error,errorAnt, rel_tol=1e-9)):
        converge=True
    
    errorAnt=error
    c=0
    for c in range(n):
        w[c]=waux[c]

        
    logw0.append(w[0])
    logw1.append(w[1])
    logw2.append(w[2])
    logw3.append(w[3])
    logw4.append(w[4])
    logw5.append(w[5])
    logw6.append(w[6])
    logw7.append(w[7])
    logw8.append(w[8])
    logw9.append(w[9])
    
    i+=1

plt.plot(Eab)
plt.title('Error E(w)')
plt.show()

plt.plot(logw0,label='w0')
plt.plot(logw1,label='w1')
plt.plot(logw2,label='w2')
plt.plot(logw3,label='w3')
plt.plot(logw4,label='w4')
plt.plot(logw5,label='w5')
plt.plot(logw6,label='w6')
plt.plot(logw7,label='w7')
plt.plot(logw8,label='w8')
plt.plot(logw9,label='w9')
plt.legend()
plt.title('Convergencia de w\'s')
plt.show()


print('w:\n',w)
print('intercepto',b0)

regM = linear_model.LinearRegression()
regM.fit(x,y)

print('coeff: ',regM.coef_)
print('\nb: ',regM.intercept_)