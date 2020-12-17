# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 02:36:22 2020

@author: brand
"""

import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

X,Y = make_blobs(centers=4, random_state=123, n_samples=1000)
X = StandardScaler().fit_transform(X)

plt.scatter(X[:,0],X[:,1], c='green', s=50)
plt.show()

modelo = DBSCAN(eps=0.3, min_samples=10).fit(X)
y2 = modelo.labels_


plt.scatter(X[:,0], X[:,1], c=y2, cmap='Paired', s=50)
plt.show()