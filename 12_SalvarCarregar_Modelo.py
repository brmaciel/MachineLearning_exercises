#####    Salvar/Carregar Modelo Final     #####

import pandas as pd

# Problema:
  # Definir Bom/Mal Pagador de Credito

dados = pd.read_csv('credit-data.csv')
dados.loc[dados.age < 0, 'age'] = 40.92

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

# Criacao dos Modelo
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

modelo_svm = SVC(kernel='rbf', C=2.0, probability=True)
modelo_svm.fit(previsores, classe)

modelo_forest = RandomForestClassifier(n_estimators=40, criterion='entropy')
modelo_forest.fit(previsores, classe)

modelo_MLP = MLPClassifier(max_iter=1000, tol=0.000001, solver='adam',
                           hidden_layer_sizes=(100), activation='relu',
                           batch_size=200, learning_rate_init=0.001,
                           verbose=True)
modelo_MLP.fit(previsores, classe)



# =====   Salvamento do Modelo em disco   ===== #
import pickle
pickle.dump(modelo_svm, open('modelo_svm.sav', 'wb'))
pickle.dump(modelo_forest, open('modelo_forest.sav', 'wb'))
pickle.dump(modelo_MLP, open('modelo_MLP.sav', 'wb'))



# =====   Carregamento do Modelo em disco   ===== #
import pickle
svm = pickle.load(open('modelo_svm.sav', 'rb'))
random_forest = pickle.load(open('modelo_forest.sav', 'rb'))
mlp = pickle.load(open('modelo_MLP.sav', 'rb'))

# Avaliação dos modelos carregados
resultado_svm = svm.score(previsores, classe)
resultado_random_forest = random_forest.score(previsores, classe)
resultado_mlp = mlp.score(previsores, classe)

# Teste de Classificação de um novo registro
novo_registro = [[50000, 40, 5000]]
import numpy as np
novo_registro = np.array(novo_registro)
novo_registro = novo_registro.reshape(-1, 1) # muda registro para 1 coluna
novo_registro = scaler.fit_transform(novo_registro) # faz o scalonamento
novo_registro = novo_registro.reshape(-1, 3) # retorna para 3 colunas

resposta_svm = svm.predict(novo_registro)
resposta_forest = random_forest.predict(novo_registro)
resposta_mlp = mlp.predict(novo_registro)