#####    Aprendizagem por Regras     #####

import Orange


# =====   Definir risco do Credito   ===== #
dados = Orange.data.Table('risco-credito2.csv')
  #obs.: no arquivo de dados, é preciso por [c#] antes do nome da classe
dados.domain # exibe o titulo das colunas

# Criacao do modelo
cn2_learner = Orange.classification.rules.CN2Learner() # cria as regras
modelo = cn2_learner(dados)

# Exibe as regras
for regra in modelo.rule_list:
    print(regra)

# Teste do modelo
# cliente1 a avaliar: historico 'boa', divida 'alta', garantias 'nenhuma', renda '>35'
# cliente2 a avaliar: historico 'boa', divida 'alta', garantias 'adequada', renda '<15'
clientes = [['boa', 'alta', 'nenhuma', 'acima_35'],
            ['boa', 'alta', 'adequada', '0_15']]
previsoes = modelo(clientes)

# transforma valores de resposta em variavel categorica
for i in previsoes:
    print(dados.domain.class_var.values[i])
    


# =====   Definir Bom/Mal Pagador de Credito   ===== #
dados = Orange.data.Table('credit-data2.csv')
  #obs.: no arquivo de dados, é preciso por [i#] antes do primeiro atributo para ignora-lo
dados.domain

# Divisao entre dados de Treino e de Teste
dados_train_test = Orange.evaluation.testing.sample(dados, n=0.25)
dados_train = dados_train_test[1]
dados_test = dados_train_test[0]

# Criacao do modelo
cn2_learner = Orange.classification.rules.CN2Learner() # cria as regras
modelo = cn2_learner(dados_train)

# Exibe as regras
for regra in modelo.rule_list:
    print(regra)

# Avaliacao do Modelo
previsoes = Orange.evaluation.testing.TestOnTestData(dados_train, dados_test, [modelo])
print(Orange.evaluation.CA(previsoes))



# =====   Definir se Pessoa Ganhara + ou - de 50k   ===== #
# feito usando a interface orange



# =====   Bom/Mal Pagador com Majority Learner   ===== #
# Classifica o registro com base na maioria dos dados cadastrados
# Define uma linha base de eficiencia que um algoritmo de machine learning deve ter
# caso contrario, vale mais apena utilizar o MajorityLearner
dados = Orange.data.Table('credit-data2.csv')
dados.domain

# Divisao entre dados de Treino e de Teste
dados_train_test = Orange.evaluation.testing.sample(dados, n=0.25)
dados_train = dados_train_test[1]
dados_test = dados_train_test[0]

# Criacao do modelo
modelo = Orange.classification.MajorityLearner()

# Avaliacao do Modelo
previsoes = Orange.evaluation.testing.TestOnTestData(dados_train, dados_test, [modelo])
print(Orange.evaluation.CA(previsoes))

from collections import Counter
print(Counter(str(d.get_class()) for d in dados_test))