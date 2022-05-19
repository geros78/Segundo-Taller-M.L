# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 22:10:12 2022

@author: alejo
"""


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

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

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.1)

x_test_out = np.array(data_test.drop(['Outcome'], 1))
y_test_out = np.array(data_test.Outcome)





# MODELO DE REGRESION LOGICA
logreg = LogisticRegression(solver= 'lbfgs', max_iter= 7600)

#Entrenamiento del Modelo
logreg.fit(x_train, y_train)

#Metricas del modelo

print('-'*60)
print('Regresion Logistica')

#Presicion del test de Entrenamiento de entrenamiento

print(f'accuracy de Entrenamiento de Entrenamiento: {logreg.score (x_train, y_train)}')

#Presicion del test de Entrenamiento

print(f'accuracy de test Entrenamiento: {logreg.score (x_test, y_test)}')

#Presicion del Validacio

print(f'accuracy de Validacion: {logreg.score (x_test_out, y_test_out)}')




# MAQUINA DE SOPORTE VECTORIAL

# Seleccionar un Modelo
svc = SVC(gamma='auto')

#Entreno el modelo
svc.fit(x_train, y_train)

#Metricas del modelo

print('-'*60)
print('Maquina de soporte vectorial')

#accuracy de test de Entrenamiento de entrenamiento

print(f'accuracy de Entrenamiento de Entrenamiento: {svc.score (x_train, y_train)}')

#accuracy de test de Entrenamiento

print(f'accuracy de test Entrenamiento: {svc.score (x_test, y_test)}')

#accuracy de Validacio

print(f'accuracy de Validacion: {svc.score (x_test_out, y_test_out)}')





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
























