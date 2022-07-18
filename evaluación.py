import pandas as pd
import matplotlib.pyplot as plt

#Leer archivos
df = pd.read_csv('mineriadatos/train.csv')

#Consultar si está conectada al dataset/ mostrando las primero 5 filas
#print(df.shape)

#info del dataset
#print (df.types)

#Me muestra datos que son nulos, el tipo de dato, el nombre de la columna y el rango que tienen del index
#print (df.info())

#Saber si hay algun registro duplicado
#print (df.duplicated().sum)

#Llenara los campos que son nulos

df['Name'] = df ['Name'].fillna ('Desconocido')

df['Sex'] = df['Sex'].fillna('N/P')

df['Cabin'] = df['Cabin'].fillna('Inexistente')

#Crear crosstab 
ct = pd.crosstab(df['Sex'], df['Parch']).plot(kind='bar')
plt.xlabel ('Sexualidad')
plt.ylabel ('Parched')

ct = pd.crosstab(df['Pclass'], df['Survived']).plot(kind='bar')
plt.xlabel ('PClase')
plt.ylabel ('Sobreviviente')

ct = pd.crosstab(df['SibSp'], df['Embarked']).plot(kind='bar')
plt.xlabel ('SibSp')
plt.ylabel ('Embarcada')

plt.show()

        

