#####    Deteccao de Outliers     #####


import pandas as pd

# Base de Dados de Credito
dados = pd.read_csv('credit-data.csv')
dados = dados.dropna() # Remove registros com informações invalidas (nan)


# =====   Usando Boxplot   ===== #
import matplotlib.pyplot as plt
# Plota o Boxplot
# Atributo idade
plt.boxplot(dados['age'], showfliers=True)
outliers_age = dados[(dados.age < -20)]

# Atributo Loan
plt.boxplot(dados['loan'])
outliers_loan = dados[(dados.loan > 13300)]



# =====   Usando Grafico de Dispersao   ===== #
import matplotlib.pyplot as plt
# Base de Dados de Credito
# Analise renda x idade
plt.scatter(dados['income'], dados['age'])

# Analise renda x loan
plt.scatter(dados['income'], dados['loan']) # nao há outliers

# Analise idade x loan
plt.scatter(dados['age'], dados['loan'])


# Base de Dados do Censu
dados_census = pd.read_csv('census.csv')
# Analise idade x final-weight
plt.scatter(dados_census['age'], dados_census['final-weight'])



# =====   Usando PyOD   ===== #
from pyod.models.knn import KNN
detector = KNN()
detector.fit(dados.iloc[:, 1:4])

previsoes = detector.labels_  # valor 1 significa um outlier
confianca_previsoes = detector.decision_scores_

# Identifica os registros da base de dado que sao outliers
outliers = []
for i in range(len(previsoes)):
    if previsoes[i] == 1:
        outliers.append(i)
registros_outliers = dados.iloc[outliers, :]
