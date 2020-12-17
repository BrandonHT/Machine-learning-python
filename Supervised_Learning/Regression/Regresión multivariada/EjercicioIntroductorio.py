#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 11:45:28 2020

@author: brandonhdz
"""

import numpy as np
from math import isclose
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

m=2
np.random.seed(3)
x=np.random.random([m,2])
y=1.5*x[:,0]+np.random.random()

#Inicializar 
theta0=0
theta1=0
theta2=0
alpha=0.1

converge=False
maxIteraciones=100000
i=0

logE=[]
Eini=0

while not converge and i<maxIteraciones:
    yF= theta0 + theta1*x[:,0] + theta2 * x[:,1]
    E=(1/(2*m))*sum((yF-y)**2)
    
    #Actualizar
    nuevaT0= theta0 - alpha * (1/m) * sum(yF-y)
    nuevaT1= theta1 - alpha * (1/m) * sum((yF-y)*x[:,0])
    nuevaT2= theta2 - alpha * (1/m) * sum((yF-y)*x[:,1])
    
    if (isclose(Eini,E,rel_tol=1e-9)):
        converge=True
        
    theta0=nuevaT0
    theta1=nuevaT1
    theta2=nuevaT2
    
    Eini=E
    logE.append(E)
    
    i+=1


fig = plt.figure()
ax = plt.axes(projection='3d')
x1line=x[:,0]
x2line=x[:,1]
yline=y
ax.scatter3D(x1line,x2line,yline)
x1,x2 = np.meshgrid(x[:,0],x[:,1])
Y= theta0 + theta1 * x1 + theta2 * x2
ax.plot_wireframe(x1,x2,Y)
plt.figure()

plt.plot(logE)