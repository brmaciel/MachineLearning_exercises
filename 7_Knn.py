#####    Aprendizagem baseado em Instancias - KNN     #####

import pandas as pd


# =====   Definir Bom/Mal Pagador de Credito   ===== #
dados = pd.read_csv('credit-data.csv')
dados.loc[dados.age < 0, 'age'] = 40.92

# Divisao dos atributos em previsores e classe
previsores = dados.iloc[:, 1:4].values
classe = dados.iloc[:, 4].values

# Tratamento de Valores Faltantes
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(previsores[:, 1:4])
previsores[:, 1:4] = imputer.transform(previsores[:, 1:4])

# Tratamento de Escalonamento de Atributos
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

# Divisao entre dados de Treino e de Teste
from sklearn.model_selection import train_test_split
dados_train_test = train_test_split(previsores, classe, test_size=0.25, random_state=0)
previsores_train = dados_train_test[0]
previsores_test = dados_train_test[1]
classe_train = dados_train_test[2]
classe_test = dados_train_test[3]

# Criacao do modelo
from sklearn.neighbors import KNeighborsClassifier
modelo = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
modelo.fit(previsores_train, classe_train)

# Teste do modelo
previsoes = modelo.predict(previsores_test)

# Avaliacao do modelo
from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_test, previsoes)
matriz = confusion_matrix(classe_test, previsoes)

# Baseline por Majority Learner
import collections
min_precisao = collections.Counter(classe_test)
min_precisao = max(min_precisao.values())
print('Precisao minima exigida: {:.4f} = ({}/{})'.format(min_precisao/len(classe_test), min_precisao, len(classe_test)))



# =====   Definir se Pessoa Ganhara + ou - de 50k   ===== #
dados = pd.read_csv('census.csv')

# Divisao dos atributos em previsores e classe 
previsores = dados.iloc[:, 0:14].values
classe = dados.iloc[:, 14].values
                
# Transforma variaveis categorias em numericas
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_previsores = LabelEncoder()
colunas = [1, 3, 5, 6, 7, 8, 9, 13]
for c in colunas:
    previsores[:, c] = labelencoder_previsores.fit_transform(previsores[:, c])

onehotencoder = OneHotEncoder(categorical_features = colunas)
previsores = onehotencoder.fit_transform(previsores).toarray()

labelencoder_classe = LabelEncoder()
classe = labelencoder_classe.fit_transform(classe)

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

# Criacao do modelo
from sklearn.neighbors import KNeighborsClassifier
modelo = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
modelo.fit(previsores_train, classe_train)

# Teste do modelo
previsoes = modelo.predict(previsores_test)

# Avaliacao do modelo
from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_test, previsoes)
matriz = confusion_matrix(classe_test, previsoes)

# Baseline por Majority Learner
import collections
min_precisao = collections.Counter(classe_test)
min_precisao = max(min_precisao.values())
print('Precisao minima exigida: {:.4f} = ({}/{})'.format(min_precisao/len(classe_test), min_precisao, len(classe_test)))
