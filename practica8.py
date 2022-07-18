import pandas as pd
import matplotlib.pyplot as plt

# Agregar el archivo para análisis con pandas
df = pd.read_csv('mineriadatos/titanic.csv')

# Consultar de manera rapida si está conectado con el dataset
# print(df.head(6))
# Conocer la dimensión del dataset
# print(df.shape)

# Conse si hay datos duplicados
# print(df.duplicated().sum())

# Conocer Info de dataset
# print(df.info)

# Conocer la descripción del dataset
# print(df.describe())

# Contar el numero de registros por columna
# print(df.count())

# Cambiar datos null por un 2 para desconocido
df['Survived'] = df['Survived'].fillna('2')

# Cambar datos null por S/C en columna cabina
df['Cabin'] = df['Cabin'].fillna('S/C')


print(df.count())

# Cambiar un diccionario con los valores originales por valores de remplazo
d = {'male': 'M', 'female': 'F'}
# Utilizados un lambda el remplazo en una sola linea
df['Sex'] = df['Sex'].apply(lambda x: d[x])

# Conocer el dataset con valores cambiados
# print(df['Sex'])


# obtener los numeros de las columnas en una lista
col_names = df.columns.tolist()

# iterar sobre la lista
# for column in control_names:
# print("Valores nulos en <" + column + ">:"+str(df[column].isnull))

# Cruce de tabla o de informacion
ct = pd.crosstab(df['Survived'], df['Sex']).plot(kind='bar')
plt.xlabel("Sobrevivio")
plt.ylabel("Cantidad de sobrevivientes por género")

#
#Crear crosstab para survived y pclass
ct1 = pd.crosstab(df['Survived'], df['Pclass']).plot(kind='bar')
plt.xlabel("Sobrevivio")
plt.ylabel("Cantidad de sobrevivientes por Pclass")

#Crear crosstab para survived y cabin
ct2 = pd.crosstab(df['Survived'], df['Cabin']).plot(kind='bar')
plt.xlabel("Sobrevivio")
plt.ylabel("Cantidad de sobrevivientes por Cabin")
#Crear crosstab para survived y age
ct3 = pd.crosstab(df['Survived'], df['Age']).plot(kind='bar')
plt.xlabel("Sobrevivio")
plt.ylabel("Cantidad de sobrevivientes por Age")
#

for barra in ct.containers:
    ct.bar_label(barra, label_type='edge')
    ct1.bar_label(barra, label_type='edge')
    ct2.bar_label(barra, label_type='edge')
    ct3.bar_label(barra, label_type='edge')

    plt.show()
