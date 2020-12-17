#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: ivan
"""

from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

X, y = datasets.load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Scaling data
scaler = StandardScaler() 
scaler.fit(X_train)  
X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)  

myModel = GaussianNB()

skf = StratifiedKFold(n_splits=5)

i = 0
for train_index, test_index in skf.split(X_train, y_train):
    #print("Particion: ", i, "Entrenamiento: ", train_index, "Pruebas: ", test_index)
    print("Particion: ", i, "Pruebas: ", test_index)
    X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
    y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]
    
    prob = myModel.fit(X_train_fold, y_train_fold).predict_proba(X_test_fold)
    
    # Calcular ROC  y area bajo la curva
    fpr, tpr, _ = roc_curve(y_test_fold, prob[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr,label='ROC particion %d (AUC = %0.2f)' % (i, roc_auc))
    
    i+=1

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('ROC Ejemplo')
plt.legend(loc="lower right")
plt.show()

print(" Informe ")
y_pred = myModel.predict(X_test)
print(classification_report(y_test, y_pred)) 

# y_res = myModel.predict(X_test)
# sum(y_res - y_test)
