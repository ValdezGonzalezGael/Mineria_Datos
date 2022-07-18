import pandas as pd
import matplotlib.pyplot as plt

# crosstab
# Agregar el csv al dataframe
df = pd.read_csv('mineriadatos/usuarios_completo.csv')
# seleccionar columnas para análisis
df = df[['gender', 'car']]
# print(df.head(6))

ct = pd.crosstab(df['gender'], df['car']).plot(kind='bar')
plt.title('Grafica para cruce de género y carros')
plt.xlabel('genero')
plt.ylabel('Cantidad de carros')
plt.legend(loc='lower right')

# print(ct.containers)
for barra in ct.containers:
    print(barra)
    ct.bar_label(barra, label_type='edge')

plt.savefig('grafica_gender.png')
plt.show()
