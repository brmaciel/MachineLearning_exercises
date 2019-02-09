#####    Regras de Associacao - APRIORI     #####

import pandas as pd

# Objetivo
  # Definir quais Produtos sao Comprados em Conjunto


# =====   Pequena Base de Dados de um Mercado   ===== #
dados = pd.read_csv('mercado.csv', header=None)
transacoes = []
for i in range(len(dados)):
    transacoes.append([str(dados.values[i, j]) for j in range(len(dados.columns))])

# Criacao das regras
from apyori import apriori
regras = apriori(transacoes, min_support=0.3, min_confidence=0.8,
                 min_lift=2, min_length=2)

# Visualização das regras
resultados = list(regras)
resultados

# Melhorar a visualização das regras
resultados2 = [list(x) for x in resultados]
resultado_formatado = []
for j in range(len(resultados)):
    resultado_formatado.append([list(x) for x in resultados2[j][2]])
resultado_formatado



# =====   Grande Base de Dados de um Mercado   ===== #
dados = pd.read_csv('mercado2.csv', header=None)
transacoes = []
for i in range(len(dados)):
    produtos = []
    for j in range(len(dados.columns)):
        if str(dados.values[i, j]) != 'nan':
            produtos.append(str(dados.values[i, j]))
    transacoes.append(produtos)

# Criacao das regras
from apyori import apriori
regras = apriori(transacoes, min_support=0.003, min_confidence=0.4,
                 min_lift=3, min_length=2)

# Visualização das regras
resultados = list(regras)

# Melhorar a visualização das regras
resultados2 = [list(x) for x in resultados]
resultado_formatado = []
for j in range(len(resultados)):
    resultado_formatado.append([list(x) for x in resultados2[j][2]])

resultado_formatado
