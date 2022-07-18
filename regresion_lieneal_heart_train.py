import numpy as np
import time

from sklearn.linear_model import LinearRegression
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
ct = pd.crosstab([df1['Sex']], df1['HeartDisease']).plot(kind='bar')
plt.title('Grafica para cruce de Daño y Genero')
plt.ylabel("Género")
plt.xlabel("Daño cardiaco")

for barra in ct.containers:
    print(barra)
    ct.bar_label(barra, label_type='edge')

#plt.show()

# Transformar a array numpy
all_cols = df.to_numpy()

# Label Data is stored into y (prediction)
y = all_cols[:,11]
y = np.array(y)

# Information is stored in x (predictors)
x = all_cols [:,0:11]
x = np.array(x)

# Generar grfica de 
plt.scatter(x[:,0], y)
plt.show()

# Crear modelo de regresión lineal
model = LinearRegression()
model.fit(x, y)

# Coefficient of determination
r_sq = model.score(x,y)
print('Coefficient of determination', r_sq)
print('----------- Resultados del modelo matemático de regresión ---------')
print('intercept (b):', model.intercept_)
print('slope(s):', model.coef_)

#Save the model
open('heart.pkl','wb')
pickle.dump(model,open('heart.pkl','wb'))
