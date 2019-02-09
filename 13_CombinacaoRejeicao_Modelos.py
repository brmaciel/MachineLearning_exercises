#####    Combinacao de Modelos Classificadores     #####



# Carregamento dos modelos
import pickle
svm = pickle.load(open('modelo_svm.sav', 'rb'))
random_forest = pickle.load(open('modelo_forest.sav', 'rb'))
mlp = pickle.load(open('modelo_MLP.sav', 'rb'))

# Avaliacao de um novo registro
novo_registro = [[50000, 40, 5000]]
import numpy as np
novo_registro = np.array(novo_registro)
novo_registro = novo_registro.reshape(-1, 1) # muda registro para 1 coluna
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
novo_registro = scaler.fit_transform(novo_registro) # faz o scalonamento
novo_registro = novo_registro.reshape(-1, 3) # retorna para 3 colunas

# Preve classe do novo registro
respostas = []
respostas.append(svm.predict(novo_registro))
respostas.append(random_forest.predict(novo_registro))
respostas.append(mlp.predict(novo_registro))

# Preve a confianca na classificacao do novo registro
probabilidades = []
probabilidades.append(svm.predict_proba(novo_registro))
probabilidades.append(random_forest.predict_proba(novo_registro))
probabilidades.append(mlp.predict_proba(novo_registro))

confianca = []
for i in range(len(probabilidades)):    
    confianca.append(probabilidades[i].max())

# Faz a combinacao das avaliacoes dos modelos
# rejeitando os que nao superam a confianca minima estabelecidade
bom_pagador = 0
mal_pagador = 0
confianca_minima = 0.95

for i in range(len(confianca)):
    if confianca[i] >= confianca_minima:
        if respostas[i] == 1:
            bom_pagador += 1
        else:
            mal_pagador += 1

if bom_pagador > mal_pagador:
    print('Cliente pagara o emprestimo')
elif bom_pagador == mal_pagador:
    print('Resultado empatado')
else:
    print('Cliente nao pagara o emprestimo')
