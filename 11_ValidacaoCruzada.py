#####    Validacao Cruzada     #####

import pandas as pd

# Problema:
  # Definir Bom/Mal Pagador de Credito

dados = pd.read_csv('credit-data.csv')
dados.loc[dados.age < 0, 'age'] = 40.92

previsores = dados.iloc[:, 1:4].values
classe = dados.iloc[:, 4].values

# Tratamento de Valores Faltantes
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(previsores[:, 1:4])
previsores[:, 1:4] = imputer.transform(previsores[:, 1:4])

# Tratamento de Escalonamento de Atributos
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

# Criacao do modelo
from sklearn.naive_bayes import GaussianNB
modelo = GaussianNB()



# =====   Utilizando o cross_val_score   ===== #
# Teste e Avaliacao do modelo utilizando validação cruzada e algoritmo NaiveBayes
from sklearn.model_selection import cross_val_score
precisao = cross_val_score(modelo, previsores, classe, cv=10)
  # cv: numero de folds a dividir a base de dados
precisao.mean()
precisao.std() # desvio padrao



# =====   Utilizando o StratifiedKFold   ===== #
# Processo de Stratificação
# cada um dos folds vai ter quantidade proporcional de cada classe
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
  # n_splits: numero de folds a dividir a base

import numpy as np
b = np.zeros(shape=(previsores.shape[0], 1)) # vetor de n linhas e 1 coluna

precisao = []
matrizes = []
for indice_train, indice_test in kfold.split(previsores, b):
    # Criacao do modelo
    modelo = GaussianNB()
    modelo.fit(previsores[indice_train], classe[indice_train])
    
    # Teste do modelo
    previsoes = modelo.predict(previsores[indice_test])
    
    # Avaliacao do modelo
    score = accuracy_score(classe[indice_test], previsoes)
    matrizes.append(confusion_matrix(classe[indice_test], previsoes))
    precisao.append(score)

matriz_final = np.mean(matrizes, axis=0)
precisao = np.array(precisao)
precisao.mean()
precisao.std() # desvio padrao
