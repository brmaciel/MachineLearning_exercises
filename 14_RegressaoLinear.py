#####    Regressao Linear Simples & Multipla     #####

import pandas as pd


# =====   Definir Preco do Plano de Saude   ===== #
dados = pd.read_csv('plano-saude.csv')

# Divisao dos atributos em previsores e classe 
x = dados.iloc[:, 0].values  # idade
y = dados.iloc[:, 1].values  # preco

# Correlacao entre as variaveis
# 0 ------- 0.5 ------- 0.7 ------- 1.0
#    fraca      moderada     forte
# valores baixos de correlacao podem indicar que o modelo de regressao linear
# pode nao se adaptar muito bem
import numpy as np
correlacao = np.corrcoef(x, y)

x = x.reshape(-1,1)

# Criacao do modelo
from sklearn.linear_model import LinearRegression
modelo = LinearRegression()
modelo.fit(x, y)

print(modelo.intercept_) # coeficiente B0
print(modelo.coef_) # coeficientes B1 (declive da curva)

# Teste do modelo
# cliente a avaliar: idade '40'
clientes = [[40]]
previsao1 = modelo.predict(clientes)
previsao2 = modelo.intercept_ + modelo.coef_ * clientes[0]

# Avaliacao do modelo
score = modelo.score(x, y)

# Plota os dados e a reta de regressao
import matplotlib.pyplot as plt
plt.scatter(x, y)
plt.plot(x, modelo.predict(x), color='red')
plt.title('Regressao Linear Simples')
plt.xlabel('Idade')
plt.ylabel('Preco')

# Visualiza os residuais (erros dos pontos em relacao a reta de regressao) 
from yellowbrick.regressor import ResidualsPlot
visualizador = ResidualsPlot(model=modelo)
visualizador.fit(x, y)
visualizador.poof()



# =====   Definir Preco das Casas (1 atributo)   ===== #
dados = pd.read_csv('house-prices.csv')

# Divisao dos atributos em previsores e classe 
x = dados.iloc[:, 5:6].values  # metragem
y = dados.iloc[:, 2].values  # preco

# Divisao entre dados de Treino e de Teste
from sklearn.model_selection import train_test_split
dados_train_test = train_test_split(x, y, test_size=0.3, random_state=0)
x_train = dados_train_test[0]
x_test = dados_train_test[1]
y_train = dados_train_test[2]
y_test = dados_train_test[3]

# Criacao do modelo
from sklearn.linear_model import LinearRegression
modelo = LinearRegression()
modelo.fit(x_train, y_train)

# Teste do modelo
previsoes = modelo.predict(x_test)
resultado = y_test - previsoes # diferenca entre os valores real e previsto

# Avaliacao do modelo
score = modelo.score(x_train, y_train)

# Plota os dados e a reta de regressao
import matplotlib.pyplot as plt
plt.scatter(x_train, y_train)
plt.plot(x_train, modelo.predict(x_train), color='red')



# =====   Definir Preco das Casas (Multiplos atributos)   ===== #
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
from sklearn.linear_model import LinearRegression
modelo = LinearRegression()
modelo.fit(previsores_train, classe_train)

print(modelo.intercept_) # coeficiente B0
print(modelo.coef_) # coeficientes Bn

# Teste do modelo
previsoes = modelo.predict(previsores_test)

# Avaliacao do modelo
score_train = modelo.score(previsores_train, classe_train)
score_test = modelo.score(previsores_test, classe_test)

# Margem de erro
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(classe_test, previsoes)
