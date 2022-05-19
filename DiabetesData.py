# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 22:10:12 2022

@author: alejo
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

url = './DiabetesDatabase/diabetes.csv'
data = pd.read_csv(url)



# Limpiar la data



data.Glucose.replace(np.nan, 120, inplace=True)
rangos = [ 70, 100 ,120, 150, 170, 200]
nombres = ['1', '2', '3', '4', '5']
data.Glucose = pd.cut(data.Glucose, rangos, labels=nombres)

rangos = [ 20, 30, 40, 50, 70, 100]
nombres = ['1', '2', '3', '4', '5']
data.Age = pd.cut(data.Age, rangos, labels=nombres)

data.BMI.replace(np.nan, 32, inplace=True)
rangos = [ 10, 20, 30, 40, 50, 70]
nombres = ['1', '2', '3', '4', '5']
data.BMI = pd.cut(data.BMI, rangos, labels=nombres)

rangos = [ 0.05, 0.25, 0.50, 1, 1.50, 2.50]
nombres = ['1', '2', '3', '4', '5']
data.DiabetesPedigreeFunction = pd.cut(data.DiabetesPedigreeFunction, rangos, labels=nombres)

rangos = [ 0, 20, 40, 60, 80, 100, 130]
nombres = ['1', '2', '3', '4', '5', '6']
data.BloodPressure = pd.cut(data.BloodPressure, rangos, labels=nombres)

rangos = [ 0, 20, 40, 60, 80, 100]
nombres = ['1', '2', '3', '4', '5']
data.SkinThickness = pd.cut(data.SkinThickness, rangos, labels=nombres)

rangos = [ 0, 100, 200, 300, 400, 500, 700, 900]
nombres = ['1', '2', '3', '4', '5', '6', '7']
data.Insulin = pd.cut(data.Insulin, rangos, labels=nombres)




#Dropear los datos

data.drop(['Pregnancies'], axis= 1, inplace = True)

data.dropna(axis=0,how='any', inplace=True)





#Dividir la data

data_train = data[:383]
data_test = data[383:]


x = np.array(data_train.drop(['Outcome'], 1))
y = np.array(data_train.Outcome)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2)

x_test_out = np.array(data_test.drop(['Outcome'], 1))
y_test_out = np.array(data_test.Outcome)





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
























# MAQUINA DE SOPORTE VECTORIAL

#Entreno el modelo
kfoldSvc = KFold(n_splits=10)

acc_scores_train_train_svc = []
acc_scores_test_train_svc = []
svc = SVC(gamma='auto')


for train, test in kfoldSvc.split(x, y):
    svc.fit(x[train], y[train])
    scores_train_train_svc = svc.score(x[train], y[train])
    scores_test_train_svc = svc.score(x[test], y[test])
    acc_scores_train_train_svc.append(scores_train_train_svc)
    acc_scores_test_train_svc.append(scores_test_train_svc)
    
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

matriz_confusion_svc = confusion_matrix(y_test_out, y_pred)
plt.figure(figsize = (6, 6))
sns.heatmap(matriz_confusion_svc)
plt.title("Mariz de confución")

precision_svc = precision_score(y_test_out, y_pred, average=None).mean()
print(f'Precisión: {precision_svc}')

recall_svc = recall_score(y_test_out, y_pred, average=None).mean()
print(f'Re-call: {recall_svc}')





















# ARBOL DE DESCISION

# Seleccionar un Modelo
arbol = DecisionTreeClassifier()

#Entreno el modelo
kfold_arbol = KFold(n_splits=10)

acc_scores_train_train_arbol = []
acc_scores_test_train_arbol = []
arbol = LogisticRegression(solver='lbfgs', max_iter = 7600)


for train, test in kfold_arbol.split(x, y):
    arbol.fit(x[train], y[train])
    scores_train_train_arbol = arbol.score(x[train], y[train])
    scores_test_train_arbol = arbol.score(x[test], y[test])
    acc_scores_train_train_arbol.append(scores_train_train_arbol)
    acc_scores_test_train_arbol.append(scores_test_train_arbol)
    
y_pred = arbol.predict(x_test_out)


#Metricas del modelo

print('-'*60)
print('Decision Tree')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {np.array(acc_scores_train_train_arbol).mean()}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {np.array(acc_scores_test_train_arbol).mean()}')

# Accuracy de Validación
print(f'accuracy de Validación: {arbol.score(x_test_out, y_test_out)}')


# Matriz de confusión
print(f'Matriz de confusión: {confusion_matrix(y_test_out, y_pred)}')

matriz_confusion_arbol = confusion_matrix(y_test_out, y_pred)
plt.figure(figsize = (6, 6))
sns.heatmap(matriz_confusion_arbol)
plt.title("Mariz de confución")

precision_arbol = precision_score(y_test_out, y_pred, average=None).mean()
print(f'Precisión: {precision}')

recall_arbol = recall_score(y_test_out, y_pred, average=None).mean()
print(f'Re-call: {recall}')


