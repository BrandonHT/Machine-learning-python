#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 11:16:53 2020

@author: brandonhdz
"""

import numpy as np
from math import isclose
import matplotlib.pyplot as plt

m=10
np.random.seed(3)
x=np.sort(np.random.random(m))
y=1.5*x+np.random.random()

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

while not converge and i<maxIteraciones:
    yF= theta0 + theta1*x + theta2 * (x**2)
    E=(1/(2*m))*sum((yF-y)**2)
    
    #Actualizar
    nuevaT0= theta0 - alpha * (1/m) * sum(yF-y)
    nuevaT1= theta1 - alpha * (1/m) * sum((yF-y)*x)
    nuevaT2= theta2 - alpha * (1/m) * sum((yF-y)*x)
    
    if ((isclose(nuevaT0,theta0,rel_tol=1e-5) and (isclose(nuevaT1,theta1,rel_tol=1e-5)) and
        (isclose(nuevaT2,theta2,rel_tol=1e-5))) or (isclose(Eini,E,rel_tol=1e-5))):
        converge=True
        
    theta0=nuevaT0
    theta1=nuevaT1
    theta2=nuevaT2
    
    Eini=E
    logE.append(E)
    
    i+=1

plt.plot(x,y,'o',label='Datos')
plt.plot(x,yF)
plt.legend()
plt.show()
plt.plot(logE)