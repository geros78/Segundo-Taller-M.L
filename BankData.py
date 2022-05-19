# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 18:25:46 2022

@author: alejo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from warnings import simplefilter
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
# Bank data

url = './BankMarketing/bank-full.csv'
data = pd.read_csv(url)

# Limpiar la data

data.job.replace(['blue-collar','management','technician','admin.','services','retired',
'self-employed', 'entrepreneur','unemployed','housemaid','student','unknown'], 
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], inplace=True)

data.education.replace(['unknown', 'primary', 'secondary', 'tertiary'], [0, 1, 2, 3], inplace=True)
data.default.replace(['no', 'yes'], [0, 1], inplace=True)
data.marital.replace(['married', 'single', 'divorced'], [0, 1, 2], inplace=True)
data.housing.replace(['no', 'yes'], [0, 1], inplace=True)
data.loan.replace(['no', 'yes'], [0, 1], inplace=True)
data.poutcome.replace(['unknown', 'failure', 'other', 'success'], [0, 1, 2, 3], inplace=True)
data.y.replace(['no', 'yes'], [0, 1], inplace=True)

#Datos en rango

rangosAge = [20, 30, 40, 50, 70, 100]
nombresAge = ['1', '2', '3', '4', '5']
data.age = pd.cut(data.age, rangosAge, labels=nombresAge)


rangoCam = [10, 20, 30, 40, 50, 70]
nombreCam = ['1', '2', '3', '4', '5']
data.campaign = pd.cut(data.campaign, rangoCam, labels=nombreCam)

#Datos desechados

data.drop(['balance','contact','day','month','duration',
           'pdays','previous','campaign'], axis= 1, inplace = True)

data.dropna(axis=0,how='any', inplace=True)

#Dividir la data

data_train = data[:22605]
data_test = data[22605:]


x = np.array(data_train.drop(['y'], 1))
y = np.array(data_train.y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2)

x_test_out = np.array(data_test.drop(['y'], 1))
y_test_out = np.array(data_test.y)




# REGRESIÓN LOGÍSTICA CON VALIDACIÓN CRUZADA

kfold = KFold(n_splits=10)

acc_scores_train_train = []
acc_scores_test_train = []
logreg = LogisticRegression(solver='lbfgs', max_iter = 7600)


for train, test in kfold.split(x, y):
    logreg.fit(x[train], y[train])
    scores_train_train = logreg.score(x[train], y[train])
    scores_test_train = logreg.score(x[test], y[test])
    acc_scores_train_train.append(scores_train_train)
    acc_scores_test_train.append(scores_test_train)
    
y_pred = logreg.predict(x_test_out)

#Metricas del modelo

print('-'*60)
print('Regresión Logística Validación cruzada')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {np.array(acc_scores_train_train).mean()}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {np.array(acc_scores_test_train).mean()}')

# Accuracy de Validación
print(f'accuracy de Validación: {logreg.score(x_test_out, y_test_out)}')


# Matriz de confusión
print(f'Matriz de confusión: {confusion_matrix(y_test_out, y_pred)}')

matriz_confusion = confusion_matrix(y_test_out, y_pred)
plt.figure(figsize = (6, 6))
sns.heatmap(matriz_confusion)
plt.title("Mariz de confución")

precision = precision_score(y_test_out, y_pred, average=None).mean()
print(f'Precisión: {precision}')

recall = recall_score(y_test_out, y_pred, average=None).mean()
print(f'Re-call: {recall}')

f1_score = f1_score(y_test_out, y_pred, average=None).mean()

print(f'f1: {f1_score}')






















# MAQUINA DE SOPORTE VECTORIAL

#Entreno el modelo
kfoldSvc = KFold(n_splits=10)

acc_scores_train_train_svc = []
acc_scores_test_train_svc = []
svc = SVC(gamma='auto')


for train, test in kfoldSvc.split(x, y):
    svc.fit(x[train], y[train])
    scores_train_train = svc.score(x[train], y[train])
    scores_test_train = svc.score(x[test], y[test])
    acc_scores_train_train_svc.append(scores_train_train)
    acc_scores_test_train_svc.append(scores_test_train)
    
y_pred = svc.predict(x_test_out)

#Metricas del modelo

print('-'*60)
print('Maquina de soporte vectorial')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {np.array(acc_scores_train_train_svc).mean()}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {np.array(acc_scores_test_train_svc).mean()}')

# Accuracy de Validación
print(f'accuracy de Validación: {svc.score(x_test_out, y_test_out)}')


# Matriz de confusión
print(f'Matriz de confusión: {confusion_matrix(y_test_out, y_pred)}')

matriz_confusion = confusion_matrix(y_test_out, y_pred)
plt.figure(figsize = (6, 6))
sns.heatmap(matriz_confusion)
plt.title("Mariz de confución")

precision = precision_score(y_test_out, y_pred, average=None).mean()
print(f'Precisión: {precision}')

recall = recall_score(y_test_out, y_pred, average=None).mean()
print(f'Re-call: {recall}')

f1_score = f1_score(y_test_out, y_pred, average=None).mean()

print(f'f1: {f1_score}')



















# ARBOL DE DESCISION

# Seleccionar un Modelo
arbol = DecisionTreeClassifier()

#Entreno el modelo
arbol.fit(x_train, y_train)

#Metricas del modelo

print('-'*60)
print('Decision Tree')

#accuracy de test de Entrenamiento de entrenamiento

print(f'accuracy de Entrenamiento de Entrenamiento: {arbol.score (x_train, y_train)}')

#accuracy de test de Entrenamiento

print(f'accuracy de test Entrenamiento: {arbol.score (x_test, y_test)}')

#accuracy de Validacio

print(f'accuracy de Validacion: {arbol.score (x_test_out, y_test_out)}')






















