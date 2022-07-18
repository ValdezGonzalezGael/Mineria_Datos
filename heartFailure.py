import numpy as np
import time

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import pandas as pd

import pickle

df = pd.read_csv("mineriadatos/heart.csv")
# print('Los datos del dataframe son = ')
# print(df)

""" print(df.shape)
print(df.info)
print(df.describe) """

# Ajustar valores para enviar strings y normalizar datos

d = {'F' : 1, 'M': 0}
#Utilizando un lamda para el reemplazo en una sola linea
df['Sex'] = df['Sex'].apply(lambda x:d[x])
# print(df)

d = {'ATA' : 0, 'NAP': 1, 'ASY':2, 'TA':3}
#Utilizando un lamda para el reemplazo en una sola linea
df['ChestPainType'] = df['ChestPainType'].apply(lambda x:d[x])
# print(df)

d = {'Normal' : 1, 'ST': 0, 'LVH': 2}
#Utilizando un lamda para el reemplazo en una sola linea
df['RestingECG'] = df['RestingECG'].apply(lambda x:d[x])
# print(df)

d = {'N' : 1, 'Y': 0}
#Utilizando un lamda para el reemplazo en una sola linea
df['ExerciseAngina'] = df['ExerciseAngina'].apply(lambda x:d[x])
# print(df)

d = {'Up' : 2, 'Down': 1, 'Flat':0}
#Utilizando un lamda para el reemplazo en una sola linea
df['ST_Slope'] = df['ST_Slope'].apply(lambda x:d[x])


# Seleccionar las columnas a procesar
df1 = df[['Age','Sex','HeartDisease']]

# Crear un cruce entre columnas y filas
""" ct = pd.crosstab([df1['Sex']], df1['HeartDisease']).plot(kind='bar')
plt.title('Grafica para cruce de Daño y Genero')
plt.ylabel("Género")
plt.xlabel("Daño cardiaco")

for barra in ct.containers:
    print(barra)
    ct.bar_label(barra, label_type='edge') """

#plt.show()

# Transformar a array numpy
all_cols = df.to_numpy()

# Label Data is stored into y (prediction)
y = all_cols[:,11]
y = np.array(y)

# Information is stored in x (predictors)
x = all_cols [:,0:10]
x = np.array(x)
print('x=')
print(x)

# Generar grfica de 
plt.scatter(x[:,0], y)
# plt.show()

#Definition of the model
x_train, x_test,y_train, y_test= train_test_split(x,y,test_size=0.3,random_state=int(time.time()))

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
print(x_train)
#Analisar el modelo entrenado
model=LogisticRegression(solver='liblinear',C=0.05,multi_class='ovr',random_state=0)
model.fit(x_train,y_train)

#LogisticRegression(C=0.05,multi_class='orv',random_state=0,solver='liblinear')

np.set_printoptions(precision=3, suppress=True)
print('intercept (b): ', model.intercept_)
print()
print('slope(s):', model.coef_)

print('Los siguientes valores son las probabilidades de cada dato de ser 0 o 1')
y_prob= model.predict_proba(x)
print('predicted response:', y_prob,sep='\n')

#x_test=scaler.transform(x_test)

#print('Los siguientes valores son los valores asignados de 0 a 1 por el modelo')
#y_pred= model.predict(x_test)
#print ('predicted response:', y_pred,sep='\n')

#print('El modelo usado los datos de entrenamiento=', model.score(x_train, y_train))
#print('El modelo usado los datos de prueba=', model.score(x_test, y_test))