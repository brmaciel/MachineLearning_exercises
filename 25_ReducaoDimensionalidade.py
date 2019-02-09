#####    Reducao de Dimensionalidade     #####

import pandas as pd

# Problema:
  # Definir se Pessoa Ganhara + ou - de 50k


# =====   PCA - Principal Component Analysis   ===== #
dados = pd.read_csv('census.csv')

# Divisao dos atributos em previsores e classe 
previsores = dados.iloc[:, 0:14].values
classe = dados.iloc[:, 14].values

# Transforma variaveis categorias em numericas
from sklearn.preprocessing import LabelEncoder
labelencoder_previsores = LabelEncoder()
colunas = [1, 3, 5, 6, 7, 8, 9, 13]
for c in colunas:
    previsores[:, c] = labelencoder_previsores.fit_transform(previsores[:, c])

# Tratamento de Escalonamento de Atributos
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

# Divisao entre dados de Treino e de Teste
from sklearn.model_selection import train_test_split
dados_train_test = train_test_split(previsores, classe, test_size=0.15, random_state=0)
previsores_train = dados_train_test[0]
previsores_test = dados_train_test[1]
classe_train = dados_train_test[2]
classe_test = dados_train_test[3]

# Reducao de Dimensionalidade
from sklearn.decomposition import PCA
pca = PCA(n_components=6) # 6 atributos mais impotantes
previsores_train = pca.fit_transform(previsores_train)
previsores_test = pca.transform(previsores_test)

componentes = pca.explained_variance_ratio_ # importancia de cada atributo

# Criacao do modelo
from sklearn.ensemble import RandomForestClassifier
modelo = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state=0)
modelo.fit(previsores_train, classe_train)

# Teste do modelo
previsoes = modelo.predict(previsores_test)

# Avaliacao do modelo
from sklearn.metrics import accuracy_score
precisao = accuracy_score(classe_test, previsoes)



# =====   Kernel PCA   ===== #
dados = pd.read_csv('census.csv')

# Divisao dos atributos em previsores e classe 
previsores = dados.iloc[:, 0:14].values
classe = dados.iloc[:, 14].values

# Transforma variaveis categorias em numericas
from sklearn.preprocessing import LabelEncoder
labelencoder_previsores = LabelEncoder()
colunas = [1, 3, 5, 6, 7, 8, 9, 13]
for c in colunas:
    previsores[:, c] = labelencoder_previsores.fit_transform(previsores[:, c])

# Tratamento de Escalonamento de Atributos
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

# Divisao entre dados de Treino e de Teste
from sklearn.model_selection import train_test_split
dados_train_test = train_test_split(previsores, classe, test_size=0.15, random_state=0)
previsores_train = dados_train_test[0]
previsores_test = dados_train_test[1]
classe_train = dados_train_test[2]
classe_test = dados_train_test[3]

# Reducao de Dimensionalidade
from sklearn.decomposition import KernelPCA
kernel_pca = KernelPCA(n_components=6, kernel='rbf')
previsores_train = kernel_pca.fit_transform(previsores_train)
previsores_test = kernel_pca.transform(previsores_test)

# Criacao do modelo
from sklearn.ensemble import RandomForestClassifier
modelo = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state=0)
modelo.fit(previsores_train, classe_train)

# Teste do modelo
previsoes = modelo.predict(previsores_test)

# Avaliacao do modelo
from sklearn.metrics import accuracy_score
precisao = accuracy_score(classe_test, previsoes)



# =====   LDA - Linear Discriminant Analysis   ===== #
dados = pd.read_csv('census.csv')

# Divisao dos atributos em previsores e classe 
previsores = dados.iloc[:, 0:14].values
classe = dados.iloc[:, 14].values

# Transforma variaveis categorias em numericas
from sklearn.preprocessing import LabelEncoder
labelencoder_previsores = LabelEncoder()
colunas = [1, 3, 5, 6, 7, 8, 9, 13]
for c in colunas:
    previsores[:, c] = labelencoder_previsores.fit_transform(previsores[:, c])

# Tratamento de Escalonamento de Atributos
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

# Divisao entre dados de Treino e de Teste
from sklearn.model_selection import train_test_split
dados_train_test = train_test_split(previsores, classe, test_size=0.15, random_state=0)
previsores_train = dados_train_test[0]
previsores_test = dados_train_test[1]
classe_train = dados_train_test[2]
classe_test = dados_train_test[3]

# Reducao de Dimensionalidade
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components=6)
previsores_train = lda.fit_transform(previsores_train, classe_train)
previsores_test = lda.transform(previsores_test)

# Criacao do modelo
from sklearn.ensemble import RandomForestClassifier
modelo = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state=0)
modelo.fit(previsores_train, classe_train)

# Teste do modelo
previsoes = modelo.predict(previsores_test)

# Avaliacao do modelo
from sklearn.metrics import accuracy_score
precisao = accuracy_score(classe_test, previsoes)
