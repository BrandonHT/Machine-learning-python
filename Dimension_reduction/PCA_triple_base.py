# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 12:01:06 2020

@author: brand
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import datasets

#Dos clases
#X,y = datasets.load_breast_cancer(return_X_y=True)

#Tres clases
#X,y = datasets.load_wine(return_X_y=True)
X,y = datasets.load_iris(return_X_y=True)
#X,y = datasets.fetch_covtype(return_X_y=True)

#Centrar
xNew = StandardScaler().fit_transform(X)

#Calcular matriz de covarianza
matriz_cov = np.cov(xNew.T)

#Calcular valores y vectores propios
eig_val, eig_vecs = np.linalg.eig(matriz_cov)

X_proy1 = xNew.dot(eig_vecs.T[0])
X_proy2 = xNew.dot(eig_vecs.T[1])
X_proy3 = xNew.dot(eig_vecs.T[2])

ax=plt.axes(projection='3d')
ax.set_xlabel('pc1')
ax.set_xlabel('pc2')
ax.set_xlabel('pc3')
ax.scatter3D(X_proy1, X_proy2, X_proy3, c=y, cmap='Paired')
plt.figure()

print()
print("************************************************")
print("******************Scikit learn******************")
print("************************************************")

from sklearn import decomposition
pca = decomposition.PCA(n_components=3)
sklearn_pca_x = pca.fit_transform(X)

ax=plt.axes(projection='3d')
ax.scatter(sklearn_pca_x[:,0],sklearn_pca_x[:,1],sklearn_pca_x[:,2],c=y, cmap='Paired')
plt.xlabel('pc1')
plt.ylabel('pc2')
plt.show()