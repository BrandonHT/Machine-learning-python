#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 18:03:10 2020

@author: brandonhdz
"""
import matplotlib.pyplot as plt
import numpy as np

#inicializar

np.random.seed(27)
w=np.random.random(2)
b=np.random.random()
p=[[2,0], [2,2], [0,2], [-2,2], [-2,0], [-2,-2], [0,-2], [2,-2]]
p=np.asarray(p)
y=[1, 1, 0, 0, 0, 0, 1, 1]
y=np.asarray(y)
c,d=p.shape

colors=['red' if it==1 else 'blue' if it==0 else 'green' for it in y]

puntos=[-2,-1,0,1,2]
puntos=np.asarray(puntos)

converge=False
maxIter=100
logE=[]
i=0

while not converge and i<maxIter:
    countZeroE=0
    for j in range(c):
        pactual=p[j]
        n=np.dot(w,pactual)+b
        a=0
        if n>0:
           a=1
        e=y[j]-a
        logE.append(e)
        if e==1:
            w+=pactual
        elif e==-1:
            w-=pactual
        else: 
            countZeroE+=1
        b+=e
    if countZeroE==8:
        converge=True
    i+=1

#test
xtest=[[1,0], [-1,1], [-1,-1], [1,1]]
ytest=[1,0,0,1]
xtest=np.asarray(xtest)
ytest=np.asarray(ytest)

resultados=[]
for i in range(4):
    n=np.dot(w,xtest[i])+b
    if n>0:
        a=1
    else:
        a=0
    resultados.append(a)

resultados=np.asarray(resultados)

print("Pesos esperados: \t", ytest)
print("Pesos obtenidos:\t", resultados)

ax=plt.gca()
ax.scatter(p[:,0], p[:,1], c=colors, label='x')
plt.plot(puntos,(-(w[0]/w[1]))*puntos+b, label='w*p+b')
plt.scatter(xtest[:,0], xtest[:,1], c='black', label='xtest')
plt.legend()
plt.show()
