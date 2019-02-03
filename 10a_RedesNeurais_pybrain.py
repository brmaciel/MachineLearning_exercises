#####    Redes Neurais - pybrain     #####


# =====   Construcao Manual da Rede Neural   ===== #
from pybrain.structure import FeedForwardNetwork, FullConnection
from pybrain.structure import LinearLayer, SigmoidLayer, BiasUnit


# Cria as camadas da rede neural com:
camadaEntrada = LinearLayer(2)  # 2 neuronios na camada de entrada
camadaOculta = SigmoidLayer(3)  # 3 neuronios na camada oculta
camadaSaida = SigmoidLayer(1)   # 1 neuronio na camada de saida
bias1 = BiasUnit()
bias2 = BiasUnit()

# Criacao da Rede Neural
rede = FeedForwardNetwork()
rede.addModule(camadaEntrada)
rede.addModule(camadaOculta)
rede.addModule(camadaSaida)
rede.addModule(bias1)
rede.addModule(bias2)

# Cria as conexoes entre as camadas
entrada2Oculta = FullConnection(camadaEntrada, camadaOculta)
oculta2Saida = FullConnection(camadaOculta, camadaSaida)
biasOculta = FullConnection(bias1, camadaOculta)
biasSaida = FullConnection(bias2, camadaSaida)

rede.sortModules()

print(rede)
# Pesos
print(entrada2Oculta.params)
print(oculta2Saida.params)
print(biasOculta.params)
print(biasSaida.params)



# =====   Construcao Automatica da Rede   ===== #
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer, SigmoidLayer

# Insercao da base de dados de entrada
dados = SupervisedDataSet(2, 1)  # 2 atributos previsores, 1 classe
dados.addSample((0, 0), (0, ))
dados.addSample((0, 1), (1, ))
dados.addSample((1, 0), (1, ))
dados.addSample((1, 1), (0, ))

print(dados['input'])
print(dados['target'])

# Criacao e Treinamento da Rede Neural
rede = buildNetwork(2, 3, 1)
    # numero de neuronios de cada camada (entrada, oculta, saida)
treinamento = BackpropTrainer(rede, dataset=dados, learningrate=0.01, momentum=0.06)
for i in range(1, 25000):
    erro = treinamento.train()
    if i % 1000 == 0:
        print('Erro: {}'.format(erro))

# Teste da rede
print(rede.activate([0, 0]))
print(rede.activate([1, 0]))
print(rede.activate([0, 1]))
print(rede.activate([1, 1]))


# Exemplo de criacao da rede e seus parametros
rede = buildNetwork(2, 3, 1, outclass=SoftmaxLayer, hiddenclass=SigmoidLayer,
                    bias = False)

# Informações sobre a rede
print(rede['in'])
print(rede['hidden0'])  # funcao de ativacao da camada oculta
print(rede['out'])  # funcao de ativacao da camada de saida
print(rede['bias'])