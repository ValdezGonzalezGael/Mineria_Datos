import matplotlib.pyplot as plt
import pandas as pd
# Empezar a leer el archivo csv
df = pd.read_csv('mineriadatos/usuarios_completo.csv')
# Muestrame la dimensión de el dataset/ dataframe
# print(df.shape)
# Muestrame las primeras 5 filas
# print(df.head())
# Conocer la cantidad de datos faltantes por cada columno
# Muestra la cantidad de filas faltantes
# print(df.count())
# obtener los nombres de las columnas en una lista
#col_names = df.columns.tolist()
# print(col_names)
# # iterar sobre la lista
# for column in col_names:
# Conocer valores nulos
#  print("Valores nulos en <" + column +
#      ">: " + str(df[column].isnull().sum()))
# Conocer tipo de dato por columna
# print("Tipo de valor de <" + column + ">: " + str(df[column].dtypes))

# Conse si hay datos duplicados
# print(df.duplicated().sum())

# llenar la columna avatar con una url por default
#df['company'] = df['company'].fillna('Desconocido')
# llenar la columna gender con una url por D (Desconocido)
#df['car'] = df['car'].fillna('Desconocido')
# llenar la columna lang con una url por desconoocido
#df['favorite_app'] = df['favorite_app'].fillna('Desconocido')
# llenar la columna lang con una url por desconoocido
#df['avatar'] = df['avatar'].fillna('default.png')
# llenar la columna lang con una url por desconoocido
#df['active'] = df['active'].fillna('none')
# llenar la columna lang con una url por desconoocido
#df['is_admin'] = df['is_admin'].fillna('none')
# llenar la columna lang con una url por desconoocido
#df['department'] = df['department'].fillna('Desconocido')
# llenar la columna lang con una url por desconoocido
#df['gender'] = df['gender'].fillna('D')

#df.to_csv('mineriadatos/usuarios_completo.csv', index=False)

# Graficar la columna gender
# df['gender'].value_counts().plot(kind='bar')
# Graficar la columna favorite_app
df['favorite_app'].value_counts().plot(kind='bar')
# Mostrar la grafica
plt.show()
