#####    Clustering - DBSCAN     #####



# =====   Dados Didaticos   ===== #
import numpy as np
x = [20, 27, 21, 37, 46, 53, 55, 47, 52, 32, 39, 41, 39, 48, 48]
y = [1000,1200,2900,1850,900,950,2000,2100,3000,5900,4100,5100,7000,5000,6500]
dados = []
for i, j in zip(x, y):
    dados.append([i,j])
dados = np.array(dados)

# Tratamento de Escalonamento de Atributos
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
dados = scaler.fit_transform(dados)

# Criacao dos clusters (de forma automatica)
from sklearn.cluster import DBSCAN
n_clusters = -1
eps = 0.5
while n_clusters == -1 and eps < 10:
    cluster = DBSCAN(eps=eps, min_samples=3)
    cluster.fit_predict(dados)
    
    previsoes = cluster.labels_

    # garante que todos os pontos pertencem a um cluster
    n_clusters = 0
    for valor in previsoes:
        # -1 indica que o ponto nao pertence a nenhum cluster
        if valor == -1:
            n_clusters = valor
    
    eps += 0.05

dados = scaler.inverse_transform(dados)
# Plota os dados de cada cluster
import matplotlib.pyplot as plt
colors = ['red', 'blue', 'green', 'orange', 'black']
for i in range(max(previsoes)+1):
    plt.scatter(dados[previsoes == i, 0], dados[previsoes == i, 1], s=50,
                c=colors[i], label="Cluster {}".format(i))
plt.xlabel('Idade')
plt.ylabel('Salario')
plt.legend()



# =====   Base de Dados de Cartao de Credito (2 Atributos)   ===== #
import pandas as pd

base = pd.read_csv('credit-card-clients.csv', header=1)
base['bill_total'] = base['BILL_AMT1'] + base['BILL_AMT2'] + base['BILL_AMT3']
base['bill_total'] += base['BILL_AMT4'] + base['BILL_AMT5'] + base['BILL_AMT6']
# Divisao dos atributos em previsores e classe 
dados = base.iloc[:, [1,25]].values

# Tratamento de Escalonamento de Atributos
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
dados = scaler.fit_transform(dados)

# Criacao dos clusters
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.37, min_samples=4)
previsoes = dbscan.fit_predict(dados)
print(max(previsoes))

# Exibe quantos dados tem em cada cluster
import numpy as np
clusters, cl_quantidade = np.unique(previsoes, return_counts=True)

dados = scaler.inverse_transform(dados)
# Plota os dados de cada cluster
import matplotlib.pyplot as plt
color = ['red', 'blue', 'green', 'orange', 'black']
for i in range(4):
    plt.scatter(dados[previsoes == i, 0], dados[previsoes == i, 1], s=50, c=color[i], label='Cluster')
plt.xlabel('Limite')
plt.ylabel('Gastos')
plt.legend()

# Add o cluster correspondente ao registro na base de dados
# e reordena pelo cluster
lista_clientes = np.column_stack((base, previsoes))
lista_clientes = lista_clientes[lista_clientes[:,26].argsort()]