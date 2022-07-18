import matplotlib.pyplot as plt
import pandas as pd

# grup by
# Agregar el csv al dataframe
df = pd.read_csv('mineriadatos/usuarios_completo.csv')
# seleccionar columnas para análisis
df = df[['gender', 'avatar']]
# print(df.head(6))

# Agrupar geder y role del dataframe
group = df.groupby(["gender", "avatar"])
print(group.size().reset_index(name='counts'))
