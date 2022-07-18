import matplotlib.pyplot as plt
import pandas as pd

# grup by
# Agregar el csv al dataframe
df = pd.read_csv('mineriadatos/usuarios_completo.csv')

# seleccionar para de análisis
df1 = df[['gender', 'avatar']]
df2 = df[['email', 'car']]
# print(df.head(6))

# Agrupar geder y role del dataframe
group = df.groupby(["gender", "avatar"])
group2 = df2.groupby(["email", "car"])
print(group.size().reset_index(name='counts'))
print(group2.size().reset_index(name='counts'))
