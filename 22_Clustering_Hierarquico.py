#####    Clustering - Hierarquico     #####



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

# Define o numero de clusters a usar (Criacao do Dendograma)
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
dendrograma = dendrogram(linkage(dados, method='ward'))
plt.xlabel('Registros')
plt.ylabel('Distancia')

# Criacao dos clusters
from sklearn.cluster import AgglomerativeClustering
num_cluster = 3
cl_hier = AgglomerativeClustering(n_clusters=num_cluster, affinity='euclidean', linkage='ward')
previsoes = cl_hier.fit_predict(dados)

dados = scaler.inverse_transform(dados)
# Plota os dados de cada cluster
colors = ['red', 'blue', 'green', 'orange', 'black']
for i in range(num_cluster):
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
from sklearn.cluster import AgglomerativeClustering
num_cluster = 3
cl_hier = AgglomerativeClustering(n_clusters=num_cluster, affinity='euclidean', linkage='ward')
previsoes = cl_hier.fit_predict(dados)

dados = scaler.inverse_transform(dados)
# Plota os dados de cada cluster
import matplotlib.pyplot as plt
colors = ['red', 'blue', 'green', 'orange', 'black']
for i in range(num_cluster):
    plt.scatter(dados[previsoes == i, 0], dados[previsoes == i, 1],
                s=50, c=colors[i], label="Cluster {}".format(i))
plt.xlabel('Limite')
plt.ylabel('Gastos')
plt.legend()

# Add o cluster correspondente ao registro na base de dados
# e reordena pelo cluster
import numpy as np
lista_clientes = np.column_stack((base, previsoes))
lista_clientes = lista_clientes[lista_clientes[:,26].argsort()]
