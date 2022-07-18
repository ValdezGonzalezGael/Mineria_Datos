import pickle
""" from pyexpat import model """
""" from re import X
 """""" from statistics import mode """ 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("mineriadatos/heart.csv")
print('Los datos del dataframe son =')
print(df)

""" print(df.shape)
print(df.info)
print(df.describe) """

d={'F':1,'M':0 }

#Utiliza un lamba para el reemplazo en una sola linea
df['Sex'] = df['Sex'].apply(lambda x:d[x])
print(df)

""" d={'NORMAL':1,'ST':0, 'LVH':2 }
#Utiliza un lamba para el reemplazo en una sola linea
df['RestingECG'] = df['RestingECG'].apply(lambda x:d[x]) """

""" d={'N':1, 'Y':0 }
#Utiliza un lamba para el reemplazo en una sola linea
df['RestingECG'] = df['RestingECG'].apply(lambda x:d[x]) """

d={'southwest':1,'northwest':2,'southeast':3, 'northeast':4}
#Utiliza un lamba para el reemplazo en una sola linea
df['region'] = df['region'].apply(lambda x:d[x])

d={'Up':2,'Down':1,'Flat':0}
#Utiliza un lamba para el reemplazo en una sola linea
df['ST_Slope'] = df['ST_Slope'].apply(lambda x:d[x])

#Seleccionar las columnas a procesar
df1= df[['Age','sex','HeartDisease']]

#Crear un cruce entre columnas y filas
ct=pd.crosstab([df1['sex']],df1['HeartDisease']).plot(kind='bar')
plt.title('Grafica para cruce de Daño y Género')
plt.xlabel("Daño cardiaco")
plt.ylabel("Género")
plt.show()

all_cols=df.to_numpy()

#Label data is tored into y (prediction)
y=all_cols[:,6]
y=np.array(y)
print('y= ',y)

x=all_cols[:,0:6]
x=np.array(x)

plt.scatter(x[:,0],y)
plt.show()

model = LinearRegression()
model.fit(x,y)

LinearRegression()

r_sq = model.score(x,y)
print()
print()
print('coefficient of determination', r_sq)
print()
print('-----------Resultados del modelo matemático de regresión----------')
print()
print('intercept(b)', model.intercept_)
print('slope(s):', model.coef_)
print() 

print('Insertar los valores de las variables independientes -x- medidas para predecir la variable independiente')
print ('la variable independiente -changes-')
x_pred = np.array([20.0,1.0,20.60,0.0,0.0,4.0]).reshape((-1,1))
print(x_pred.T)

y_pred = model.predict(x_pred.T)
print('predicted response:', y_pred, sep='\n')

# Save the model
open('medicalcosts.pkl', 'wb')
pickle.dump(model, open('medicalcosts.pkl', 'wb'))

for barra  in ct.containers:
    print(barra)
    ct.bar_label(barra, label_type='edge')

#plt.show()

#Transformar array en nupy
all_cols = df.to_numpy()

# Label data is stored into y (prediction)
y = all_cols[:,11]
y = np.array(y)
#print ('y= ',y)

#Information is stored in x (predictors)
x= all_cols[:,0:11]
x= np.array(x)
#print('x= ')
#print(x)

#Generar grafica de puntos
plt.scatter(x[:,0],y)
plt.show()

