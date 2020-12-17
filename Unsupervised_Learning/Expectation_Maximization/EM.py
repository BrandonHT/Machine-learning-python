# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 02:18:12 2020

@author: brand
"""

import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.datasets.samples_generator import make_blobs


X,Y = make_blobs(centers=4, random_state=123, n_samples=1000)
plt.scatter(X[:,0], X[:,1], c='green', s=50)
plt.show()

GMM = GaussianMixture(n_components=3).fit(X)
color = 0
colores = ['blue', 'red', 'black', 'orange', 'gray', 'navy', 'turquoise', 'green']
muestras = 100.0
for m,c in zip(GMM.means_, GMM.covariances_):
    multi_normal = multivariate_normal(mean=m, cov=c)
    puntos = multi_normal.rvs(size=muestras, random_state=0)
    plt.scatter(puntos[:,0], puntos[:,1], c=colores[color], s=10)
    plt.scatter(m[0],m[1],c=colores[color],zorder=10,s=200)
    color+=1

plt.show()