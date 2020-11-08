import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import datasets

#Dos clases
#X,y = datasets.load_breast_cancer(return_X_y=True)

#Tres clases
X,y = datasets.load_wine(return_X_y=True)

#Centrar
xNew = StandardScaler().fit_transform(X)

#Calcular matriz de covarianza
matriz_cov = np.cov(xNew.T)

#Calcular valores y vectores propios
eig_val, eig_vecs = np.linalg.eig(matriz_cov)

X_proy = xNew.dot(eig_vecs.T[0])
eje_dummy = np.zeros(y.shape[0])

ax=plt.gca()
ax.scatter(X_proy,eje_dummy,c=y, cmap='Paired')
plt.xlabel('pc1')
plt.ylabel('y')
plt.show()
plt.figure()

print()
print("************************************************")
print("******************Scikit learn******************")
print("************************************************")

from sklearn import decomposition
pca = decomposition.PCA(n_components=1)
sklearn_pca_x = pca.fit_transform(X)

ax=plt.gca()
ax.scatter(sklearn_pca_x,eje_dummy,c=y, cmap='Paired')
plt.xlabel('pc1')
plt.ylabel('y')
plt.show()