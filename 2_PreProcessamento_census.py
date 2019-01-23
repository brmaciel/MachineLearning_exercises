# -*- coding: utf-8 -*-

# Pre Processamento de dados de uma base de dados de Censo
# fonte dos dados: archive.ics.uci.edu/ml/datasets/Adult

import pandas as pd

# Define o nome de todas as colunas da base de dados resultante
def descobre_titulos(colunas):
    nome_col = dados.columns.tolist()
    
    # Cria e insere dados na variavel titulos
    # cada linha contem os valores de um atributo 'original'
    titulos = []
    for col in colunas:
        titulos.append([])

    for i in range(len(previsores)):
        for n, col in enumerate(colunas):
            if not previsores[i][col] in titulos[n]:
                titulos[n].insert(0, previsores[i][col])
        
    for i in range(len(previsores)):
        if not previsores[i][colunas[5]] in titulos[5]:
            titulos[5].insert(0, previsores[i][colunas[5]])
    
    # Coloca os atributos de cada linha em ordem alfabetica
    for n in titulos:
        n.sort()

    # Coloca todos os nomes das colunas em uma unica lista
    nome_colunas = []
    for titulo in titulos:
        for t in titulo:
            nome_colunas.append(t)

    # Add as colunas restantes que nao serão 'sub-divididas'
    for i in range(len(nome_col)-1):
        if not i in colunas:
            nome_colunas.append(nome_col[i])
    
    return nome_colunas
            
# Importacao de dados
dados = pd.read_csv('census.csv')


# ===== Transformacao de Variaveis Categoricas ===== #
previsores = dados.iloc[:, 0:14].values
classe = dados.iloc[:, 14].values

colunas = [1, 3, 5, 6, 7, 8, 9, 13]
nome_colunas = descobre_titulos(colunas)


# Altera as variaveis categorias por numericas
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_previsores = LabelEncoder()
#labels = labelencoder_previsores.fit_transform(previsores[:,1])
previsores[:, 1] = labelencoder_previsores.fit_transform(previsores[:, 1])
previsores[:, 3] = labelencoder_previsores.fit_transform(previsores[:, 3])
previsores[:, 5] = labelencoder_previsores.fit_transform(previsores[:, 5])
previsores[:, 6] = labelencoder_previsores.fit_transform(previsores[:, 6])
previsores[:, 7] = labelencoder_previsores.fit_transform(previsores[:, 7])
previsores[:, 8] = labelencoder_previsores.fit_transform(previsores[:, 8])
previsores[:, 9] = labelencoder_previsores.fit_transform(previsores[:, 9])
previsores[:, 13] = labelencoder_previsores.fit_transform(previsores[:, 13])

# Altera todos os atributos para n novos atributos contendo 1/0 para identificar o atributo
onehotencoder = OneHotEncoder(categorical_features=colunas)
previsores = onehotencoder.fit_transform(previsores).toarray()
previsores2 = pd.DataFrame(previsores, columns = nome_colunas)


# Exemplo: alterando a raca para 5 novos atributos contendo 1 ou 0 para identificar a raca
raca = dados.iloc[:, 8:9].values
raca[:,0] = labelencoder_previsores.fit_transform(raca[:,0])
onehotencoder = OneHotEncoder(categorical_features=[0])
raca = onehotencoder.fit_transform(raca).toarray()
titulos_raca = [' Amer-Indian-Eskimo', ' Asian-Pac-Islander', ' Black', ' Other', ' White']
raca = pd.DataFrame(raca, columns=titulos_raca)
