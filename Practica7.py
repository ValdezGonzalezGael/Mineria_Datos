import pandas as pd
import matplotlib.pyplot as plt

# crosstab
# Agregar el csv al dataframe
df = pd.read_csv('mineriadatos/usuarios_completo.csv')
# seleccionar columnas para análisis
#df1 = df[['gender', 'car']]
#df2 = df[['active', 'company']]
#df3 = df[['avatar', 'department']]
df4 = df[['gender', 'first_name']]
# print(df.head(6))

#pd.crosstab(df1['gender'], df1['car']).plot(kind='bar')
#pd.crosstab(df2['active'], df2['company']).plot(kind='bar')
#pd.crosstab(df3['avatar'], df3['department']).plot(kind='bar')
pd.crosstab(df4['gender'], df4['first_name']).plot(kind='bar')
plt.show()
