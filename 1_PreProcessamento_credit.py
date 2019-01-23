# -*- coding: utf-8 -*-

# Pre Processamento de dados de uma base de dados de Credito

import pandas as pd

# Importacao de dados
base = pd.read_csv('credit-data.csv')
pd.set_option('display.max_columns', None)
base.describe()


# ===== Tratamento de Valores Inconsistente ===== #
base.loc[base['age'] < 0] # localiza registros com idade negativa

# Opção 1. Apagar coluna
base.drop('age', 1, inplace=True)
    # 1: apagar a coluna inteira
    # inplace: apagar sem retornar nada

# Opção 2. Apagar somente os registros com problema
base.drop(base[base.age < 0].index, inplace=True)

# Opção 3. Preencher os valores com a media
media = base['age'][base.age > 0].mean()
base.loc[base.age < 0, 'age'] = media



# ===== Tratamento de Valores Faltantes ===== #
base.loc[pd.isnull(base.age)] # localiza registros com idade em branco

# divide os dados entre atributos previsores e classe
previsores = base.iloc[:, 1:4].values # colunas 1, 2, 3
classe = base.iloc[:, 4].values

# faz a inclusao dos dados faltantes
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(previsores[:, :])
previsores[:] = imputer.transform(previsores[:]) # atualiza os dados


# ===== Tratamento de Escalonamento de Atributos ===== #
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)
