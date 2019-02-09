#####    Regressao com Redes Neurais     #####

import pandas as pd


# =====   Definir Preco do Plano de Saude   ===== #
dados = pd.read_csv('plano-saude2.csv')

# Divisao dos atributos em previsores e classe 
x = dados.iloc[:, 0:1].values  # idade
y = dados.iloc[:, 1:2].values  # preco

# Tratamento de Escalonamento de Atributos
from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
x_scaled = scaler_x.fit_transform(x)
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)

# Criacao do modelo
from sklearn.neural_network import MLPRegressor
modelo = MLPRegressor()
modelo.fit(x_scaled, y_scaled)

# Teste do modelo
# cliente a avaliar: idades '40', '50'
clientes = [[40], [50]]
previsao = modelo.predict(scaler_x.transform(clientes))
previsao = scaler_y.inverse_transform(previsao)

# Avaliacao do modelo
score = modelo.score(x_scaled, y_scaled)

# Plota os dados e a reta de regressao
import matplotlib.pyplot as plt
plt.scatter(x, y)
plt.plot(x, scaler_y.inverse_transform(modelo.predict(x_scaled)), color='red')
plt.title('Regressao com Rede Neural')
plt.xlabel('Idade')
plt.ylabel('Preco')



# =====   Definir Preco das Casas   ===== #
dados = pd.read_csv('house-prices.csv')

# Divisao dos atributos em previsores e classe 
previsores = dados.iloc[:, 3:19].values
classe = dados.iloc[:, 2:3].values

# Tratamento de Escalonamento de Atributos
from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
previsores = scaler_x.fit_transform(previsores)
scaler_y = StandardScaler()
classe = scaler_y.fit_transform(classe)

# Divisao entre dados de Treino e de Teste
from sklearn.model_selection import train_test_split
dados_train_test = train_test_split(previsores, classe, test_size=0.3, random_state=0)
previsores_train = dados_train_test[0]
previsores_test = dados_train_test[1]
classe_train = dados_train_test[2]
classe_test = dados_train_test[3]

# Criacao do modelo
from sklearn.neural_network import MLPRegressor
modelo = MLPRegressor(hidden_layer_sizes=(9,9))
modelo.fit(previsores_train, classe_train)

# Avaliacao do modelo
score_train = modelo.score(previsores_train, classe_train)
score_test = modelo.score(previsores_test, classe_test)

# Teste do modelo
previsoes = modelo.predict(previsores_test)
previsoes = scaler_y.inverse_transform(previsoes)
classe_test = scaler_y.inverse_transform(classe_test)

# Margem de erro
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(classe_test, previsoes)
