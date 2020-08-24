#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 12:20:20 2020

@author: brandonhdz
"""

import numpy as np
from math import isclose, floor
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

m=100
np.random.seed(3)
x=np.random.random([m,2])
y=np.random.choice([-1,1],size=(m,))

for i in range(m):
    if x[i,0] >= 0.7:
        y[i] = -1
 
colors=['red' if i==-1 else 'blue' if i==1 else 'lightgreen' for i in y]
ax=plt.gca()
ax.scatter(x[:, 0], x[:, 1], c=colors)
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

#Inicializar 
theta0=0
theta1=0
theta2=0
alpha=0.1

converge=False
maxIteraciones=1000
i=0

logE=[]
Eini=0

while not converge and i<=maxIteraciones:
    #Observar
    yF = theta0 + theta1*x[:,0] + theta2*x[:,1]
    E=(1/(2*m))*sum((yF-y)**2)
    
    #Actualizar 
    nuevaT0= theta0 - alpha * (1/m) * sum(yF-y)
    nuevaT1= theta1 - alpha * (1/m) * sum((yF-y)*x[:,0])
    nuevaT2= theta2 - alpha * (1/m) * sum((yF-y)*x[:,1])
    
    if (isclose(Eini,E,rel_tol=1e-5)):
        converge=True
        
    theta0=nuevaT0
    theta1=nuevaT1
    theta2=nuevaT2
    
    Eini=E
    logE.append(E)
    
    i+=1

#### nuevos puntos
m = floor(m/10)
x = np.random.random([m,2])
y = theta0 + theta1*x[:,0] + theta2*x[:,1]
colors=['red' if i<=0 else 'blue' if i>0 else 'lightgreen' for i in y]
ax=plt.gca()
ax.scatter(x[:, 0], x[:, 1], c=colors)

 

ax.scatter(-1*(theta0/theta1), 0, marker='x')
ax.scatter(0, -1*(theta0/theta2), marker='x')

 

plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

