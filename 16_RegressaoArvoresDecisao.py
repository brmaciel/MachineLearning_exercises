#####    Regressao com Arvores de Decisao     #####

import pandas as pd


# =====   Definir Preco do Plano de Saude   ===== #
dados = pd.read_csv('plano-saude2.csv')

# Divisao dos atributos em previsores e classe 
x = dados.iloc[:, 0:1].values  # idade
y = dados.iloc[:, 1].values    # preco

# Criacao do modelo
from sklearn.tree import DecisionTreeRegressor
modelo = DecisionTreeRegressor()
modelo.fit(x, y)

# Teste do modelo
# cliente a avaliar: idades '40', '50'
clientes = [[40], [50]]
previsao = modelo.predict(clientes)

# Avaliacao do modelo
score = modelo.score(x, y)

# Plota os dados e a reta de regressao
import matplotlib.pyplot as plt
plt.scatter(x, y)
plt.plot(x, modelo.predict(x), color='red')
plt.title('Regressao com Arvores de Decisao')
plt.xlabel('Idade')
plt.ylabel('Preco')

# Exibe os splits feitos pela arvore
import numpy as np
x_teste = np.arange(min(x), max(x), 0.1)
x_teste = x_teste.reshape(-1,1)
plt.scatter(x, y)
plt.plot(x_teste, modelo.predict(x_teste), color='red')



# =====   Definir Preco das Casas   ===== #
dados = pd.read_csv('house-prices.csv')

# Divisao dos atributos em previsores e classe 
previsores = dados.iloc[:, 3:19].values
classe = dados.iloc[:, 2].values

# Divisao entre dados de Treino e de Teste
from sklearn.model_selection import train_test_split
dados_train_test = train_test_split(previsores, classe, test_size=0.3, random_state=0)
previsores_train = dados_train_test[0]
previsores_test = dados_train_test[1]
classe_train = dados_train_test[2]
classe_test = dados_train_test[3]

# Criacao do modelo
from sklearn.tree import DecisionTreeRegressor
modelo = DecisionTreeRegressor()
modelo.fit(previsores_train, classe_train)

# Teste do modelo
previsoes = modelo.predict(previsores_test)

# Avaliacao do modelo
score_train = modelo.score(previsores_train, classe_train)
score_test = modelo.score(previsores_test, classe_test)

# Margem de erro
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(classe_test, previsoes)
