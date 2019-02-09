#####    Clustering - Comparacao     #####

# Comparacao entre Kmean, Hierarquico e DBSCAN

from sklearn import datasets
dados, cluster = datasets.make_moons(n_samples = 1500, noise = 0.09)

# Plota os dados originais
import matplotlib.pyplot as plt
plt.scatter(dados[:, 0], dados[:, 1], s=5)

import numpy as np
cores = np.array(['green', 'blue', 'magenta'])


# Clustering Kmeans
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 2)
previsoes = kmeans.fit_predict(dados)
plt.scatter(dados[:, 0], dados[:, 1], s=5, color=cores[previsoes])
plt.title('K-Means')


# Clustering Hierarquico
from sklearn.cluster import AgglomerativeClustering
hierarquico = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
previsoes = hierarquico.fit_predict(dados)
plt.scatter(dados[:, 0], dados[:, 1], s=5, color=cores[previsoes])
plt.title('Hierarquico')


# Clustering DBSCAN
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.1)
previsoes = dbscan.fit_predict(dados)
plt.scatter(dados[:, 0], dados[:, 1], s=5, color=cores[previsoes])
plt.title('DBSCAN')