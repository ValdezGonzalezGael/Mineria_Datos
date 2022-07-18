# Importar librerias de Pandas
import pandas as pd
from pyparsing import col

# Empezar a leer el archivo csv
df = pd.read_csv('mineriadatos/users_data2.csv')
# Muestrame las primeras 5 filas
# print(df.head())

# Muestrame la dimensión de el dataset/ dataframe
# print(df.shape)

# Muestra los tipos de datos que recibe o acepta la columna
# print(df.dtypes)

# Muestra datos del archivo, peso, nombre de columnas y rango de index
#print (df.info())

# Describe DataFrame
# Muestra datos de cada columna
# print(df.describe())

# Conocer la cantidad de datos faltantes por cada columno
# Muestra la cantidad de filas faltantes
# print(df.count())


# Conse si hay datos duplicados
# print(df.duplicated().sum())

# obtener los nombres de las columnas en una lista
col_names = df.columns.tolist()
# print(col_names)
# # iterar sobre la lista
for column in col_names:
    # Conocer valores nulos
    print("Valores nulos en <" + column +
          ">: " + str(df[column].isnull().sum()))
    # Conocer tipo de dato por columna
    print("Tipo de valor de <" + column + ">: " + str(df[column].dtypes))

    # llenar la columna avatar con una url por default
    df['avatar'] = df['avatar'].fillna('default.png')
    # llenar la columna gender con una url por D (Desconocido)
    df['gender'] = df['gender'].fillna('D')
    # llenar la columna lang con una url por desconoocido
    df['lenguage'] = df['lenguage'].fillna('Desconocido')

    df.to_csv('mineriadatos/users_modify.csv', index=False)
