# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 18:38:03 2020

@author: RikSo
"""

# Biblioteca so
import os

# Biblioteca para realizar solicitudes HTTP
import requests

# Biblioteca para exploración y análisis de datos
import pandas as pd

# Biblioteca con métodos numéricos y representaciones matriciales
import numpy as np

# Biblioteca para hacer graficos
import matplotlib.pyplot as plt

# Biblioteca para construir un modelo basado en la técnica Gradient Boosting
#import xgboost as xgb

# Paquetes scikit-learn para preprocesamiento de datos
# "SimpleImputer" es una transformación para completar los valores faltantes en conjuntos de datos
from sklearn.impute import SimpleImputer

# Paquetes de scikit-learn para entrenamiento de modelos y construcción de pipelines

# Método para separar el conjunto de datos en muestras de testes y entrenamiento
from sklearn.model_selection import train_test_split

# Clase para crear una pipeline de machine-learning
from sklearn.pipeline import Pipeline

from sklearn.base import BaseEstimator, TransformerMixin

mainpath = "D:\\Participaciones\\Hackaton\\BehindTheCode\\desafio-2-2020-master\\Assets\\Data"
filename = "dataset-tortuga-desafio-2.csv"
fullpath = os.path.join(mainpath,filename)

df = pd.read_csv(fullpath,sep=',')

#df.plot.box()


class GeneralTransformer(BaseEstimator, TransformerMixin):
    #   Clase constructor
    def __init__(self, tipo, columns):
        self.tipo = tipo
        self.columns = columns
        
    #   Return self nothing else to do here
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Primero copiamos el dataframe de datos de entrada 'X'
        result = X.copy()
        #Check if needed
        if (self.tipo == 'D'):        
        # Devolvemos un nuevo dataframe de datos sin las columnas no deseadas
            return result.drop(labels=self.columns, axis='columns')




class NumericalTransformer(BaseEstimator, TransformerMixin):
    #Class Constructor
    def __init__( self, tipo, columns ):
        self.tipo = tipo
        self.columns = columns

        
    #Return self, nothing else to do here
    def fit( self, X, y = None ):
        return self
    
    
    def transform(self, X, y = None):
        
        #Copiar el dataframe
        result = X.copy()
        
        for item in self.columns:
            
            if(self.tipo == 'I0'):
                result[item].fillna(0)
                
            if(self.tipo == 'I1'):
                promedio = X[item].mean()
                result[item].fillna(value = promedio) 
                
            if(self.tipo == 'I2'):
                mediana = X[item].median()
                result[item].fillna(mediana) 
                
            if(self.tipo == 'I3'):
                mode = X[item].mode()
                result[item].fillna(mode)

            if (self.tipo == 'N'):
                max_value = X[item].max()
                min_value = X[item].min()
                result[item] = (X[item] - min_value) / (max_value - min_value)

        #returns a df
        return result


print(df["HOURS_BACKEND"].mean())

# Creacion de instancias de una transformacion Seleccion
gen_transf = GeneralTransformer(tipo = 'D', columns = ["NAME", "Unnamed: 0","USER_ID"])

# Creación de instancias de una transformación Imputacion
num_transf_imp = NumericalTransformer(tipo = 'I0', columns=["AVG_SCORE_FRONTEND"])          

# Creación de instancias de una transformación Normalizacion
#num_transf_nor = NumericalTransformer(tipo = 'N', columns=["HOURS_DATASCIENCE"])  



# Ver las columnas del conjunto de datos original
print("Columnas del conjunto de datos original: \n")
print(df.columns)

# Aplicar la transformación ``DropColumns`` al conjunto de datos base
gen_transf.fit(X=df)


# Reconstruyendo un DataFrame de Pandas con el resultado de la transformación General
df_1 = pd.DataFrame.from_records(
    data = gen_transf.transform(
        X=df
    ),
)

# Ver las columnas del conjunto de datos transformado
print("Columnas del conjunto de datos después de la transformación ``DropColumns``: \n")
print(df_1.columns)

# Ver los datos faltantes del conjunto de datos antes de la primera transformación (df_data_2)
print("Valores nulos antes de la transformación SimpleImputer: \n\n{}\n".format(df_1.isnull().sum(axis = 0)))


# Aplicar la transformacion a variables numericas
num_transf_imp.fit(X=df_1)
#num_transf_nor.fit(X=df_1)


# Reconstruyendo un DataFrame de Pandas con el resultado de la transformación Numerica
df_2 = pd.DataFrame.from_records(
    data = num_transf_imp.transform(
        X=df_1
    ),
)

# Ver los datos faltantes del conjunto de datos despues de la primera transformación (df_data_2)
print("Valores nulos despues de la transformación SimpleImputer: \n\n{}\n".format(df_2.isnull().sum(axis = 0)))

"""
df_3 = pd.DataFrame.from_records(
    data = num_transf_nor.transform(
        X=df_2
    ),
)
"""


print("Columnas del conjunto de datos después de la transformación ``Normalizar``: \n")
print(df_2.loc[:1])

# Ver los datos faltantes del conjunto de datos después de la segunda transformación (SimpleImputer) (df_data_3)
#print("Valores nulos en el conjunto de datos después de la transformación SimpleImputer: \n\n{}\n".format(df_2.isnull().sum(axis = 0)))




# Import necessary modules
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical


"""Definición de features del modelo"""


# Definición de las columnas que seran features (Notese que la columna NOMBRE no esta presente)
features = [
    "HOURS_DATASCIENCE", "HOURS_BACKEND", "HOURS_FRONTEND",
    "NUM_COURSES_BEGINNER_DATASCIENCE", "NUM_COURSES_BEGINNER_BACKEND", "NUM_COURSES_BEGINNER_FRONTEND",
    "NUM_COURSES_ADVANCED_DATASCIENCE", "NUM_COURSES_ADVANCED_BACKEND", "NUM_COURSES_ADVANCED_FRONTEND",
    "AVG_SCORE_DATASCIENCE", "AVG_SCORE_BACKEND", "AVG_SCORE_FRONTEND"
]

# Definición de variable objetivo
target = ['PROFILE']

# Preparación de los argumentos para los métodos de la biblioteca ``scikit-learn``
X = df_2[features]
y = df_2[target]

#El conjunto de entrada (X):
X.head()

#La variable objetivo (y):
y.head()

# Separación de datos en conjunto de entrenamiento y conjunto de pruebas
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=337)


#You can find the dataset in a pandas dataframe called df. 
#For convenience, everything in df except for the target has been converted 
#to a NumPy matrix called predictors



def asignar_clases(dataframe):
    lista_clases_int = []
    lista_clases_string = dataframe.PROFILE.unique().tolist()
    for x in range(0,len(lista_clases_string)):
        lista_clases_int.append(x)
    
    dict_listas = dict(zip(lista_clases_string,lista_clases_int))
    return dict_listas
    
print(asignar_clases(df_2))

dict_clases = asignar_clases(df_2)



def generar_targetFinal(row,diccionario):
    result = 0
    valor_target = row['PROFILE']
    for target in diccionario.keys():
        if valor_target == str(target):
            result = diccionario[target]
            
    return result

df_2["TARGET"] = df_2.apply(generar_targetFinal, args = [dict_clases], axis=1)   

predictors = df_2[features].to_numpy()

y_1 = df_2['TARGET']

#Separacion de datos manual X=Dataframe de predictores, y_1=Dataframe de target
X_train, X_test, y_1_train, y_1_test = train_test_split(X, y_1, test_size=0.3)


Xtrain_np = X_train.to_numpy()
Xtest_np = X_test.to_numpy()


# Convert the target to categorical: target (One hot Encoding)
ytrain_enc = to_categorical(y_1_train)
ytest_enc = to_categorical(y_1_test)


# Save the number of columns in predictors: n_cols

#Forma (shape): se trata de una tupla de enteros que describen cuántas dimensiones tiene el tensor en cada eje.
#Un vector tiene un shape con un único elemento, por ejemplo “(5,)”, mientras que un escalar 
#tiene un shape vacío “( )”. En la librería Numpy este atributo se llama shape.

n_cols = predictors.shape[1]
print(n_cols)

input_shape = (n_cols,)



def get_new_model(input_shape = input_shape ):
    # Set up the model
    model = Sequential()
    
    # Add the first layer
    model.add(Dense(100,activation='relu',input_shape = input_shape))
    
    # Add the second layer
    model.add(Dense(50,activation='relu'))
    
    # Add the output layer
    model.add(Dense(6,activation='sigmoid'))
    
    # Descripcion del modelo
    model.summary()
    
    return(model)



#predictors es un Numpy.array matrix 
#target es un Numpy.array vector
# Import the SGD optimizer
from keras.optimizers import SGD 

# Create list of learning rates: lr_to_test
lr_to_test = [0.01]


for lr in lr_to_test:
    print('\n\nTesting model with learning rate: %f\n'%lr )
    
    # Build new model to test, unaffected by previous models
    model = get_new_model()
    
    # Create SGD optimizer with specified learning rate: my_optimizer
    my_optimizer = SGD(lr=lr)
    
    # Compile the model
    model.compile(optimizer=my_optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
     
    # Fit the model
    #model.fit(predictors,target)
    
    # Fit the model
    #model.fit(X_train, target, epochs=50, batch_size=10)
    
    # validacion "manual"
    model.fit(Xtrain_np, ytrain_enc, validation_data=(Xtest_np,ytest_enc), epochs=15)


import pickler as pickleador
import keras
import pickle
 
pickleador.make_keras_picklable()


with open('D:\\Diplomado\\Model\\testPickle.pickle', 'wb') as handle:
    pickle.dump(model,handle, protocol=pickle.HIGHEST_PROTOCOL)