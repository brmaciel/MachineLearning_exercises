#####    Clustering - Kmeans     #####



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

# Criacao dos clusters
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
kmeans.fit(dados)

centroid = kmeans.cluster_centers_  # centroid de cada cluster
rotulos = kmeans.labels_  # qual cluster a que pertence cada registro

# Retorna os dados para os valores originais nao-escalonados
dados = scaler.inverse_transform(dados)
centroid = scaler.inverse_transform(centroid)

# Plota os dados de cada cluster e o centroid
import matplotlib.pyplot as plt
cores = ["g.", "r.", "b."]
for i in range(len(dados)):
    plt.plot(dados[i][0], dados[i][1], cores[rotulos[i]], markersize=15)
plt.scatter(centroid[:, 0], centroid[:, 1], marker="x")
plt.xlabel('Idade')
plt.ylabel('Salario')



# =====   Dados Aleatorios Didaticos   ===== #
from sklearn.datasets.samples_generator import make_blobs
dados, centroid = make_blobs(n_samples=200, centers=4)

# Criacao dos clusters
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4)
kmeans.fit(dados)

# Identifica a qual cluster pertence cada registro
previsoes = kmeans.predict(dados) # equivalente ao cluster_centers_

# Plota os dados de cada cluster
import matplotlib.pyplot as plt
plt.scatter(dados[:, 0], dados[:, 1], c=previsoes)



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

# Define o numero de clusters a usar (Elbow Method)
from sklearn.cluster import KMeans
wcss = []
for n in range(1, 11):
    kmeans = KMeans(n_clusters=n, random_state=0)
    kmeans.fit(dados)
    wcss.append(kmeans.inertia_)
import matplotlib.pyplot as plt
plt.plot(range(1,11), wcss)
plt.xlabel('Numero de Cluster')
plt.ylabel('WCSS')

# Criacao dos clusters
num_cluster = 4
kmeans = KMeans(n_clusters=num_cluster, random_state=0)
previsoes = kmeans.fit_predict(dados)

dados = scaler.inverse_transform(dados)
# Plota os dados de cada cluster
import matplotlib.pyplot as plt
colors = ['red', 'orange', 'green', 'blue', 'black']
for i in range(num_cluster):
    plt.scatter(dados[previsoes == i, 0], dados[previsoes == i, 1], s=50,
                c=colors[i], label="Cluster {}".format(i))
plt.xlabel('Limite')
plt.ylabel('Gastos')
plt.legend()

# Add o cluster correspondente ao registro na base de dados
# e reordena pelo cluster
import numpy as np
lista_clientes = np.column_stack((base, previsoes))
lista_clientes = lista_clientes[lista_clientes[:,26].argsort()]



# =====   Base de Dados de Cartao de Credito (Multiplos Atributos)   ===== #
import pandas as pd

base = pd.read_csv('credit-card-clients.csv', header=1)
base['bill_total'] = base['BILL_AMT1'] + base['BILL_AMT2'] + base['BILL_AMT3']
base['bill_total'] += base['BILL_AMT4'] + base['BILL_AMT5'] + base['BILL_AMT6']
# Divisao dos atributos em previsores e classe 
atributos = [1, 2, 3, 4, 5, 25]
dados = base.iloc[:, atributos].values

# Tratamento de Escalonamento de Atributos
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
dados = scaler.fit_transform(dados)

# Define o numero de clusters a usar (Elbow Method)
from sklearn.cluster import KMeans
wcss = []
for n in range(1, 11):
    kmeans = KMeans(n_clusters=n, random_state=0)
    kmeans.fit(dados)
    wcss.append(kmeans.inertia_)
import matplotlib.pyplot as plt
plt.plot(range(1,11), wcss)
plt.xlabel('Numero de Cluster')
plt.ylabel('WCSS')

# Criacao dos clusters
kmeans = KMeans(n_clusters=4, random_state=0)
previsoes = kmeans.fit_predict(dados)

# Add o cluster correspondente ao registro na base de dados
# e reordena pelo cluster
import numpy as np
lista_clientes = np.column_stack((base, previsoes))
lista_clientes = lista_clientes[lista_clientes[:,26].argsort()]

#obs.: Por utilizar varios atributos, nao permite uma análise grafica dos dados
# sendo preciso fazer uma analise manual do resultado obtido
