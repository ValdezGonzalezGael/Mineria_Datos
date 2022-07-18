import pandas as pd
# Empezar a leer el archivo csv
df = pd.read_csv('mineriadatos/users_modify.csv')

# print(df.head(15))

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

# Quitar los datos duplicados manteniendo el ultimo
df = df.drop_duplicates(keep='last', subset=['first_name'])
# Conse si hay datos duplicados
print(df.duplicated().sum())
print(df.shape)
df.to_csv('mineriadatos/users_modify.csv', index=False)
