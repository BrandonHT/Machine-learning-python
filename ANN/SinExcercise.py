#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 18:35:25 2020

@author: brandonhdz
"""
import math
import numpy as np
import matplotlib.pyplot as plt

def seno(p):
    pi4=math.pi/4
    return 1+math.sin(pi4*p)

def logsig(n):
    denominador=1+math.pow(math.e, -n)
    return 1/denominador

def derivadalog(a):
    return (1-a)*(a)

def feedforward(w1,p,b1,w2,b2):
    n1=np.transpose(w1*p+b1)
    a1=np.array([logsig(n1[0]), logsig(n1[1])])
    a2=np.dot(w2,a1)+b2
    return a1,a2

#inicializar
w1=np.transpose(np.asarray([-0.27, -0.41]))
w2=np.array([0.09, -0.17])
b1=np.transpose(np.array([-0.48, -0.13]))
b2=0.48
p=1
alpha=0.01

datos=np.random.uniform(-1.5,1.5,2500)

esperado=[]
for it in range(len(datos)):
    esperado.append(seno(datos[it]))
esperado=np.array(esperado)

converge=False
maxEpocas=1000

logE=[]
logw11=[]
logw12=[]
logw21=[]
logw22=[]
logb11=[]
logb12=[]
logb2=[]

epocas=0
while not converge and epocas<maxEpocas:
  for j in range(len(datos)):
    logw11.append(w1[0])
    logw12.append(w1[1])
    logw21.append(w2[0])
    logw22.append(w2[1])
    logb11.append(b1[0])
    logb12.append(b1[1])
    logb2.append(b2)
    
    #propagación hacia delante
    a1,a2=feedforward(w1, datos[j], b1, w2, b2)
        
    #calculo del error
    error=esperado[j]-a2
    logE.append(error)
        
    #Propagación hacia atrás
    s2=(-2)*(1)*error
    s00=derivadalog(a1[0])
    s11=derivadalog(a1[1])
    derivada1=np.array([[s00,0], [0,s11]])
    derivada2=w2*s2
    s1=np.matmul(derivada1,derivada2)
        
    #actualización de pesos
    w2=w2-alpha*s2*a1
    b2=b2-alpha*s2
    w1=w1-alpha*s1*datos[j]
    b1=b1-alpha*s1
    error=abs(error)
    if error<1e-06:
      converge=True
  epocas+=1
  print(epocas)

plt.figure()
plt.gca()
plt.plot(logE, label='error')
plt.title('Historia del error')
plt.legend()
plt.show()

plt.figure()
plt.gca()
plt.plot(logw11, label='w11_1')
plt.plot(logw12, label='w21_1')
plt.plot(logw21, label='w11_2')
plt.plot(logw22, label='w12_2')
plt.plot(logb2, label='b2')
plt.plot(logb11, label='b2')
plt.plot(logb12, label='b2')
plt.title('Actualización de pesos y bias')
plt.legend()
plt.show()