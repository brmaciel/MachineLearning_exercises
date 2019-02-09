#####    Regressao Linear Polinomial     #####

import pandas as pd


# =====   Definir Preco do Plano de Saude   ===== #
dados = pd.read_csv('plano-saude2.csv')

# Divisao dos atributos em previsores e classe 
x = dados.iloc[:, 0:1].values  # idade
y = dados.iloc[:, 1].values    # preco

# Ajuste na dimensao dos dados
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=3) # polinomio de 3 grau
x_poly = poly.fit_transform(x)

# Criacao do modelo Linear Polinomial
from sklearn.linear_model import LinearRegression
modelo = LinearRegression()
modelo.fit(x_poly, y)

print(modelo.intercept_) # coeficiente B0
print(modelo.coef_) # coeficientes B1 (declive da curva)

# Teste do modelo
# cliente a avaliar: idades '40', '50'
clientes = [[40], [50]]
previsao = modelo.predict(poly.transform(clientes))

# Avaliacao do modelo
score = modelo.score(x_poly, y)

# Plota os dados e a reta de regressao
import matplotlib.pyplot as plt
plt.scatter(x, y)
plt.plot(x, modelo.predict(x_poly), color='red')
plt.title('Regressao Linear Polinomial')
plt.xlabel('Idade')
plt.ylabel('Preco')



# =====   Definir Preco das Casas   ===== #
dados = pd.read_csv('house-prices.csv')

# Divisao dos atributos em previsores e classe 
previsores = dados.iloc[:, 3:19].values
classe = dados.iloc[:, 2].values

# Ajuste na dimensao dos dados
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2) # polinomio de 3 grau

# Divisao entre dados de Treino e de Teste
from sklearn.model_selection import train_test_split
dados_train_test = train_test_split(previsores, classe, test_size=0.3, random_state=0)
previsores_train = poly.fit_transform(dados_train_test[0])
previsores_test = poly.transform(dados_train_test[1])
classe_train = dados_train_test[2]
classe_test = dados_train_test[3]

# Criacao do modelo Linear Polinomial
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
